"""Explicit synthetic screening training; no calibration or final partitions read."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .data import digest, load
from .joint_data import KINDS, ROLES
from .joint_model import JointModel, encode, targets, predict, span_quality, span_loss
from .joint_adapter import network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-joint-v2'))
    parser.add_argument('--teacher', type=Path, default=Path('artifacts/zh-g6-pretrained'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', choices=['joint', 'TASK', 'HABIT'], default='joint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--allow-synthetic', action='store_true')
    parser.add_argument('--adapter', action='store_true')
    parser.add_argument('--type-heads', action='store_true')
    parser.add_argument('--ratio', choices=['40:40:20', '60:20:20'], default='40:40:20')
    args = parser.parse_args()
    if not args.allow_synthetic or args.output.exists():
        parser.error('Explicit synthetic opt-in and a new output directory required')
    torch.set_num_threads(3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, dev = [load(args.data / (s + '.jsonl')) for s in ['train', 'dev']]
    if args.domain != 'joint':
        rows = [r for r in rows if r['domain'] == args.domain]
        dev = [r for r in dev if r['domain'] == args.domain]
    upstream = json.loads((args.teacher / 'training-report.json').read_text())
    config = AutoConfig.from_pretrained(args.teacher / 'encoder-config', local_files_only=True) if (args.teacher / 'encoder-config').exists() else AutoConfig.from_pretrained(
        upstream['pretrainedModel'], revision=upstream['revision'], local_files_only=True)
    if args.adapter:
        config.quickadd_adapter = True
    if args.type_heads:
        config.quickadd_type_heads = True
    model = network(config)
    weights = torch.load(args.teacher / 'teacher.pt', weights_only=True)
    model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in weights.items() if k.startswith('encoder.')})
    tokenizer = AutoTokenizer.from_pretrained(args.teacher / 'tokenizer', local_files_only=True)
    inputs, offsets, valid = encode(tokenizer, rows)
    args.output.mkdir(parents=True)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    config.save_pretrained(args.output / 'encoder-config')
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=3e-5),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('encoder.')], lr=5e-4)], weight_decay=.01)
    hashes = {s: digest(args.data / (s + '.jsonl')) for s in ['train', 'dev']}
    history, best, started = [], -1., time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = torch.randperm(len(rows))
        if args.ratio == '60:20:20':
            sample_weights = torch.tensor([2. if r['expected']['decision'] == 'defer' else
                                            3. if r['expected']['proposal']['kind'] == 'TASK' else 1. for r in rows])
            order = torch.multinomial(sample_weights, len(rows), replacement=True)
        for batch in order.split(24):
            batch_rows = [rows[i] for i in batch.tolist()]
            intent_target, span_target = targets(batch_rows, offsets[batch])
            arguments = {k: v[batch] for k, v in inputs.items()}
            if args.type_heads:
                arguments['training_kind'] = torch.tensor([int(r['expected']['proposal']['kind'] == 'HABIT')
                    if r['expected']['decision'] == 'accept' else int(any(s['label'] in ['DAYS', 'GOAL_WINDOW', 'COUNT'] for s in r['spans'])) for r in batch_rows])
            il, sl, _ = model(**arguments)
            loss = torch.nn.functional.cross_entropy(il, intent_target) + span_loss(sl, span_target, valid[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        raw = predict(model, tokenizer, dev, with_vectors=False)
        fields = span_quality(dev, raw)
        accuracy = np.mean([np.argmax(p['intentProbabilities']) == (KINDS.index(r['expected']['proposal']['kind'])
                            if r['expected']['decision'] == 'accept' else 2) for r, p in zip(dev, raw)])
        score = .9 * np.mean([v for v in fields.values() if v is not None]) + .1 * accuracy
        entry = dict(epoch=epoch + 1, loss=float(np.mean(losses)), devIntentAccuracy=float(accuracy),
                     devSpanF1=fields, checkpointScore=float(score), elapsedSeconds=time.monotonic() - started)
        history.append(entry)
        if score > best:
            best, best_epoch = score, epoch + 1
            torch.save(model.state_dict(), args.output / 'joint.pt')
        print(json.dumps(entry), flush=True)
    assert hashes == {s: digest(args.data / (s + '.jsonl')) for s in hashes}
    report = dict(domain=args.domain, seed=args.seed, trainRows=len(rows), devRows=len(dev), epochs=args.epochs,
                  bestEpoch=best_epoch, history=history, dataSha256=hashes, warmStartSha256=digest(args.teacher / 'teacher.pt'),
                  modelSha256=digest(args.output / 'joint.pt'), roles=ROLES, intentClasses=KINDS,
                  parameters=sum(p.numel() for p in model.parameters()), elapsedSeconds=time.monotonic() - started,
                  batchSize=24, learningRate=dict(encoder=3e-5, heads=5e-4), humanReviewed=False,
                  adapter=args.adapter, typeHeads=args.type_heads, ratio=args.ratio,
                  selection='dev macro role F1 90% + intent accuracy 10%; shared layouts, synthetic screening only',
                  sourceSha256={p.name: digest(p) for p in Path(__file__).parent.glob('joint_*.py')})
    (args.output / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
