"""Same-data BIO control for attributing data/schema gains versus span architecture."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .data import digest, load
from .span_data import ROLES, validate
from .span_decode import span_metrics
from .span_model import LENGTH, encode_rows


class BioModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config, attn_implementation='eager')
        self.request = nn.Linear(config.hidden_size, 2)
        self.tokens = nn.Linear(config.hidden_size, 1 + 2 * len(ROLES))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        return self.request(hidden[:, 0]), self.tokens(hidden)


def bio_targets(rows, offsets):
    targets = torch.full(offsets.shape[:2], -100)
    for i, row in enumerate(rows):
        for j, (a, b) in enumerate(offsets[i].tolist()):
            if a == b:
                continue
            targets[i, j] = 0
            for span in row['spans']:
                if a >= span['start'] and b <= span['end']:
                    targets[i, j] = 1 + 2 * ROLES.index(span['label']) + int(a != span['start'])
                    break
                if a < span['end'] and span['start'] < b:
                    raise ValueError('Mixed-role token in BIO control')
    return targets


def predict(model, tokenizer, rows, batch_size=32):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded, offsets, _, _ = encode_rows(tokenizer, batch, labels=False)
            request, tokens = model(**encoded)
            probabilities = tokens.softmax(-1)
            for i, row in enumerate(batch):
                labels = probabilities[i].argmax(-1).tolist()
                selected, current, malformed = [], None, False
                for j, (a, b) in enumerate(offsets[i].tolist()):
                    if a == b:
                        continue
                    label = labels[j]
                    if label == 0:
                        if current:
                            selected.append(current)
                        current = None
                        continue
                    role = ROLES[(label - 1) // 2]
                    confidence = float(probabilities[i, j, label])
                    # Same span decoder threshold uses logits; convert token confidence to odds.
                    score = float(np.log(max(confidence, 1e-7) / max(1 - confidence, 1e-7)))
                    if label % 2:
                        if current:
                            selected.append(current)
                        current = dict(role=role, start=a, end=b, score=score)
                    elif current and current['role'] == role:
                        # Raw offset range preserves any number of internal whitespace characters.
                        current['end'] = b
                        current['score'] = min(current['score'], score)
                    else:
                        malformed = True
                        if current:
                            selected.append(current)
                        current = None
                if current:
                    selected.append(current)
                predictions.append(dict(requestProbabilities=request[i].softmax(-1).tolist(),
                                        candidates=selected, malformedBIO=malformed))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-core-spans-v1'))
    parser.add_argument('--teacher', type=Path, default=Path('artifacts/zh-g6-pretrained'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/zh-core-bio-control-v1'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    if not args.allow_synthetic or args.output.exists():
        parser.error('Require --allow-synthetic and fresh output directory')
    torch.set_num_threads(3)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    splits = {s: load(args.data / (s + '.jsonl')) for s in ['train', 'dev']}
    validate(splits)
    source = json.loads((args.teacher / 'training-report.json').read_text())
    config = AutoConfig.from_pretrained(source['pretrainedModel'], revision=source['revision'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher / 'tokenizer', local_files_only=True)
    model = BioModel(config)
    weights = torch.load(args.teacher / 'teacher.pt', weights_only=True)
    model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in weights.items() if k.startswith('encoder.')})
    encoded, offsets, _, _ = encode_rows(tokenizer, splits['train'], labels=False)
    labels = bio_targets(splits['train'], offsets)
    requests = torch.tensor([0 if r['expected']['decision'] == 'accept' else 1 for r in splits['train']])
    opt = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=3e-5),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('encoder.')], lr=5e-4),
    ], weight_decay=.01)
    args.output.mkdir(parents=True)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    config.save_pretrained(args.output / 'encoder-config')
    hashes = {s: digest(args.data / (s + '.jsonl')) for s in splits}
    history, best = [], -1.
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in torch.randperm(len(requests)).split(24):
            request_logits, logits = model(**{k: v[batch] for k, v in encoded.items()})
            loss = nn.functional.cross_entropy(request_logits, requests[batch]) + 2 * nn.functional.cross_entropy(
                logits.reshape(-1, 1 + 2 * len(ROLES)), labels[batch].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        raw = predict(model, tokenizer, splits['dev'])
        fields = span_metrics(splits['dev'], raw)
        accuracy = float(np.mean([int(np.argmax(p['requestProbabilities'])) == (0 if r['expected']['decision'] == 'accept' else 1)
                                  for r, p in zip(splits['dev'], raw)]))
        score = .9 * float(np.mean([fields[k]['f1'] or 0 for k in ['TITLE', 'TIME', 'LOCATION', 'NOTE']])) + .1 * accuracy
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), devSpanMetrics=fields,
                            devRequestAccuracy=accuracy, checkpointScore=score))
        if score > best:
            best, best_epoch = score, epoch + 1
            torch.save(model.state_dict(), args.output / 'bio.pt')
        print(json.dumps(dict(control='same-data-BIO', epoch=epoch + 1, loss=history[-1]['loss'], devScore=score)), flush=True)
    assert all(digest(args.data / (s + '.jsonl')) == hashes[s] for s in hashes)
    report = dict(architecture='same-encoder-11-label-BIO', seed=42, epochs=args.epochs, bestEpoch=best_epoch,
                  trainSha256=hashes['train'], devSha256=hashes['dev'], warmStartSha256=digest(args.teacher / 'teacher.pt'),
                  history=history, weightsSha256=digest(args.output / 'bio.pt'),
                  selection='same dev macro span F1 90% + request accuracy 10%', thresholds=dict(span=0., request=.8),
                  humanReviewed=False, deployment='offline architectural control, not shipped',
                  parameters=sum(p.numel() for p in model.parameters()), sourceSha256=digest(Path(__file__)))
    (args.output / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
