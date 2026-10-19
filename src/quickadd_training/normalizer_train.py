"""Same-data normalization model, supervised copy/reorder sequence rather than span heads."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .data import digest, load
from .span_data import validate
from .span_model import encode_rows
from .normalizer_model import Normalizer, canonical, predict, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-normalization-compare-v1'))
    parser.add_argument('--teacher', type=Path, default=Path('artifacts/zh-g6-pretrained'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/zh-normalizer-v1'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    if not args.allow_synthetic or args.output.exists():
        parser.error('Require explicit synthetic opt-in and fresh output directory')
    torch.set_num_threads(3)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    splits = {s: load(args.data / (s + '.jsonl')) for s in ['train', 'dev']}
    validate(splits)
    hashes = {s: digest(args.data / (s + '.jsonl')) for s in splits}
    upstream = json.loads((args.teacher / 'training-report.json').read_text())
    config = AutoConfig.from_pretrained(upstream['pretrainedModel'], revision=upstream['revision'], local_files_only=True)
    model = Normalizer(config)
    weights = torch.load(args.teacher / 'teacher.pt', weights_only=True)
    model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in weights.items() if k.startswith('encoder.')})
    tokenizer = AutoTokenizer.from_pretrained(args.teacher / 'tokenizer', local_files_only=True)
    encoded, offsets, _, _ = encode_rows(tokenizer, splits['train'], labels=False)
    valid = offsets[:, :, 1] > offsets[:, :, 0]
    actions, labels = targets(splits['train'], offsets)
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=3e-5),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('encoder.')], lr=5e-4),
    ], weight_decay=.01)
    args.output.mkdir(parents=True)
    config.save_pretrained(args.output / 'encoder-config')
    tokenizer.save_pretrained(args.output / 'tokenizer')
    history, best = [], -1.
    source = args.output / 'training-source'
    source.mkdir()
    import shutil
    for name in ['normalizer_model.py', 'normalizer_train.py', 'span_model.py', 'span_data.py']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    for epoch in range(args.epochs):
        losses = []
        model.train()
        for batch in torch.randperm(len(labels)).split(24):
            length = int((labels[batch] != -100).sum(-1).max())
            logits = model(**{k: v[batch] for k, v in encoded.items()}, copy_valid=valid[batch], actions=actions[batch, :length])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[batch, :length].reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions = predict(model, tokenizer, splits['dev'])
        positives = [(r, p) for r, p in zip(splits['dev'], predictions) if r['expected']['decision'] == 'accept']
        negatives = [p for r, p in zip(splits['dev'], predictions) if r['expected']['decision'] == 'defer']
        exact = sum(p['normalized'] == canonical(r) for r, p in positives) / len(positives)
        confirmed = sum(p['normalized'] in [None, '需要确认'] for p in negatives) / len(negatives)
        score = .9 * exact + .1 * confirmed
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), devPositiveExactNormalization=exact,
                            devNegativeConfirmation=confirmed, checkpointScore=score))
        if score > best:
            best, best_epoch = score, epoch + 1
            torch.save(model.state_dict(), args.output / 'normalizer.pt')
        print(json.dumps(history[-1]), flush=True)
    assert all(digest(args.data / (s + '.jsonl')) == hashes[s] for s in splits)
    report = dict(architecture='tiny-BERT-plus-two-layer-autoregressive-copy-decoder', seed=42, epochs=args.epochs,
                  bestEpoch=best_epoch, history=history, trainSha256=hashes['train'], devSha256=hashes['dev'],
                  warmStartSha256=digest(args.teacher / 'teacher.pt'), modelSha256=digest(args.output / 'normalizer.pt'),
                  parameters=sum(p.numel() for p in model.parameters()), learningRate=dict(encoder=3e-5, decoder=5e-4),
                  selection='dev positive exact canonical string 90% + negative confirmation 10%; no test read',
                  humanReviewed=False, deployment='offline normalization comparison, not native client integration',
                  sourceSha256={p.name: digest(p) for p in source.glob('*.py')})
    (args.output / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
