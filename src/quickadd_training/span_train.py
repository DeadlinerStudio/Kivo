"""Offline core-span experiment; dev-only checkpoint selection, fresh span heads."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import digest, load
from .span_data import ROLES, validate
from .span_decode import decode, span_metrics
from .span_model import LENGTH, encode_rows, initialize, predict, span_loss


def train(args):
    if args.output.exists():
        raise ValueError('Refusing to replace an experiment checkpoint')
    torch.set_num_threads(3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = {s: load(args.data / (s + '.jsonl')) for s in ['train', 'dev']}
    validate(splits)
    hashes = {s: digest(args.data / (s + '.jsonl')) for s in splits}
    model, upstream = initialize(args.teacher)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher / 'tokenizer', local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    encoded, _, valid, targets = encode_rows(tokenizer, splits['train'])
    requests = torch.tensor([0 if r['expected']['decision'] == 'accept' else 1 for r in splits['train']])
    opt = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=3e-5),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('encoder.')], lr=5e-4),
    ], weight_decay=.01)
    args.output.mkdir(parents=True)
    tokenizer.save_pretrained(args.output / 'tokenizer')
    model.encoder.config.save_pretrained(args.output / 'encoder-config')
    history, best = [], -1.
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in torch.randperm(len(requests)).split(args.batch_size):
            request_logits, logits = model(**{k: v[batch] for k, v in encoded.items()})
            loss = span_loss(logits, targets[batch], valid[batch]) + torch.nn.functional.cross_entropy(request_logits, requests[batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        raw = predict(model, tokenizer, splits['dev'])
        fields = span_metrics(splits['dev'], raw)
        request_accuracy = np.mean([int(np.argmax(p['requestProbabilities'])) == (0 if r['expected']['decision'] == 'accept' else 1)
                                    for r, p in zip(splits['dev'], raw)])
        # Every role receives equal weight; missing location/note cannot hide behind token accuracy.
        quality = float(np.mean([fields[k]['f1'] or 0 for k in ['TITLE', 'TIME', 'LOCATION', 'NOTE']]))
        quality = .9 * quality + .1 * float(request_accuracy)
        record = dict(epoch=epoch + 1, loss=float(np.mean(losses)), devSpanMetrics=fields,
                      devRequestAccuracy=float(request_accuracy), checkpointScore=quality)
        history.append(record)
        if quality > best:
            best = quality
            torch.save(model.state_dict(), args.output / 'span.pt')
            best_epoch = epoch + 1
        print(json.dumps(dict(epoch=epoch + 1, loss=record['loss'], devScore=quality,
                              requestAccuracy=float(request_accuracy),
                              spanF1={k: v['f1'] for k, v in fields.items()})), flush=True)
    if any(digest(args.data / (s + '.jsonl')) != hashes[s] for s in hashes):
        raise RuntimeError('Training data changed during experiment')
    model.load_state_dict(torch.load(args.output / 'span.pt', weights_only=True))
    model.eval()
    sample = tokenizer('明天下午四点在教学楼开会', padding='max_length', max_length=LENGTH, return_tensors='pt')
    inputs = tuple(sample[k].to(torch.int32) for k in ['input_ids', 'attention_mask'])
    torch.onnx.export(model, inputs, args.output / 'core-spans.onnx',
                      input_names=['input_ids', 'attention_mask'], output_names=['request_logits', 'span_logits'],
                      opset_version=13, dynamo=False)
    import onnxruntime as ort
    runtime = ort.InferenceSession(str(args.output / 'core-spans.onnx'), providers=['CPUExecutionProvider'])
    actual = runtime.run(None, {k: v.numpy() for k, v in zip(['input_ids', 'attention_mask'], inputs)})
    with torch.inference_mode():
        expected = model(*inputs)
    errors = [float(np.abs(x - y.numpy()).max()) for x, y in zip(actual, expected)]
    if max(errors) > .001:
        raise RuntimeError('ONNX export differs from checkpoint')
    report = dict(seed=args.seed, epochs=args.epochs, bestEpoch=best_epoch, history=history,
                  trainSha256=hashes['train'], devSha256=hashes['dev'],
                  encoderSource=upstream['pretrainedModel'], encoderRevision=upstream['revision'],
                  warmStartSha256=digest(args.teacher / 'teacher.pt'),
                  weightsSha256=digest(args.output / 'span.pt'), onnxSha256=digest(args.output / 'core-spans.onnx'),
                  onnxCheckpointMaxErrors=errors, parameters=sum(p.numel() for p in model.parameters()),
                  length=LENGTH, roles=ROLES, thresholds=dict(span=0., request=.8),
                  learningRate=dict(encoder=3e-5, heads=5e-4), batchSize=args.batch_size,
                  humanReviewed=False, syntheticOptIn=True, deployment='offline span prototype; no native client integration',
                  selection='best dev macro span F1 (90%) + request accuracy (10%); no test read',
                  sourceSha256={p.name: digest(p) for p in Path(__file__).parent.glob('span_*.py')})
    (args.output / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('best', best_epoch, 'ONNX errors', errors, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-core-spans-v1'))
    parser.add_argument('--teacher', type=Path, default=Path('artifacts/zh-g6-pretrained'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/zh-core-spans-v1'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    if not args.allow_synthetic:
        parser.error('Explicit --allow-synthetic required for this experimental corpus')
    train(args)


if __name__ == '__main__':
    main()
