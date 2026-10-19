"""Frozen core-field comparison using predicted spans and production KMP dates only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .data import digest, load
from .metrics import accepted, correct
from .native_adapter import evaluate
from .prepare_adapter import source_digest
from .span_data import validate
from .span_decode import decode, finish, span_metrics
from .span_model import SpanModel, predict, predict_onnx


def calibrate(binary, rows, raw, directory):
    """Fixed dev grid, complete business-field scoring; test/regression never consulted."""
    cache, grid = {}, []
    for threshold in [0., 1., 2., 3.]:
        for request_threshold in [.8, .9, .95, .98]:
            drafts = [decode(r, p, threshold, request_threshold) for r, p in zip(rows, raw)]
            missing = {}
            for row, draft in zip(rows, drafts):
                if draft['fields']['TIME']:
                    key = (' '.join(draft['fields']['TIME']), row['now'], row['timezone'])
                    if key not in cache:
                        missing[key] = dict(id=f'cal-{len(missing)}', text=key[0] + ' 片段时间占位',
                                            preferredKind='TASK', now=key[1], timezone=key[2])
            if missing:
                native = evaluate(binary, None, list(missing.values()), directory / f'{threshold}-{request_threshold}')
                cache.update({key: native[row['id']]['rules'] for key, row in missing.items()})
            results = [finish(d, cache.get((' '.join(d['fields']['TIME']), r['now'], r['timezone'])))
                       for r, d in zip(rows, drafts)]
            summary = summarize(rows, results)
            grid.append(dict(span=threshold, request=request_threshold,
                             **{k: summary[k] for k in ['correct', 'accepted', 'wrongAccepted', 'wrongAcceptanceRate', 'correctCoverage']}))
    feasible = [v for v in grid if v['accepted'] and v['wrongAcceptanceRate'] <= .01]
    best = max(feasible, key=lambda v: (v['correct'], -v['wrongAccepted'], v['span'], v['request'])) if feasible else max(
        grid, key=lambda v: (v['correct'] - 10 * v['wrongAccepted'], v['span'], v['request']))
    return dict(span=best['span'], request=best['request'], grid=grid,
                selection='dev only: maximize correct with <=1% observed wrong acceptance; otherwise correct-10*wrong')


def normalize_drafts(binary, rows, drafts, directory):
    payload = []
    for row, draft in zip(rows, drafts):
        if draft['fields']['TIME']:
            # ONLY predicted text; never inject gold spans, answers or dates here.
            payload.append(dict(id=row['id'], text=' '.join(draft['fields']['TIME']) + ' 片段时间占位',
                                preferredKind='TASK', now=row['now'], timezone=row['timezone']))
    normalized = evaluate(binary, None, payload, directory) if payload else {}
    return [finish(draft, normalized.get(row['id'], {}).get('rules')) for row, draft in zip(rows, drafts)]


def summarize(rows, results):
    eligible = [r for r in rows if r['expected']['decision'] == 'accept']
    right = wrong = taken = negative_wrong = 0
    fields = dict.fromkeys(['title', 'dueAt', 'note'], 0)
    failures, states, reasons = [], Counter(), Counter()
    for row, result in zip(rows, results):
        p = result['proposal']
        is_accepted = accepted(p)
        is_correct = correct(row['expected'], p)
        taken += is_accepted
        right += is_correct
        wrong += is_accepted and not is_correct
        negative_wrong += is_accepted and row['expected']['decision'] == 'defer'
        states[result['state']] += 1
        reasons.update(result.get('reasons', []))
        if row['expected']['decision'] == 'accept':
            for field in fields:
                fields[field] += p.get(field) == row['expected']['proposal'][field]
        if (row['expected']['decision'] == 'accept' and not is_correct) or (is_accepted and not is_correct):
            failures.append(dict(id=row['id'], text=row['text'], category=row['category'],
                                 expected=row['expected'], actual=result))
    return dict(total=len(rows), eligible=len(eligible), accepted=taken, correct=right, wrongAccepted=wrong,
                negativeWrongAccepted=negative_wrong, correctCoverage=right / len(eligible) if eligible else None,
                wrongAcceptanceRate=wrong / taken if taken else None, states=dict(states), reasons=dict(reasons),
                retainedFieldExact={k: v / len(eligible) if eligible else None for k, v in fields.items()}, failures=failures)


def legacy(binary, rows, directory):
    import onnxruntime as ort
    model_dir = Path('artifacts/zh-v3')
    manifest = json.loads((model_dir / 'quick-add-model.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    path = model_dir / 'quick-add-zh-v3.onnx'
    if digest(path) != manifest['artifacts']['android']['sha256']:
        raise ValueError('v3 artifact hash differs from manifest')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    vocabulary = set(manifest['codec']['vocabulary'])
    payload = []
    for row in rows:
        value = {k: row[k] for k in ['id', 'text', 'preferredKind', 'now', 'timezone']}
        text = row['text']
        if len(text) <= 128 and sum(c not in vocabulary for c in text) / len(text) <= .1:
            encoded = tokenizer(text, padding='max_length', max_length=128, return_tensors='np', truncation=False)
            if encoded['input_ids'].shape[1] == 128:
                ids = encoded['input_ids'].astype(np.int32)
                mask = encoded['attention_mask'].astype(np.int32)
                il, sl = runtime.run(None, dict(input_ids=ids, attention_mask=mask))
                value.update(inputIds=ids.reshape(-1).tolist(), attentionMask=mask.reshape(-1).tolist(),
                             intent=il.reshape(-1).tolist(), slots=sl.reshape(-1).tolist())
        payload.append(value)
    predictions = evaluate(binary, manifest, payload, directory)
    return [dict(proposal=predictions[r['id']]['hybrid'], state='COMPLETE' if accepted(predictions[r['id']]['hybrid']) else 'CONFIRM',
                 reasons=[], rawText=r['text']) for r in rows]


def backend_parity(model, tokenizer, rows, model_dir):
    """All evaluation inputs, not one selected export example; padded tokens included."""
    import onnxruntime as ort
    from .span_model import encode_rows
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    runtime = ort.InferenceSession(str(model_dir / 'core-spans.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    errors, same = [0., 0.], True
    model.eval()
    with torch.inference_mode():
        for row in rows:
            encoded, _, _, _ = encode_rows(tokenizer, [row], labels=False)
            expected = model(**encoded)
            actual = runtime.run(None, {k: v.numpy().astype(np.int32) for k, v in encoded.items()})
            for i, (a, e) in enumerate(zip(actual, expected)):
                errors[i] = max(errors[i], float(np.abs(a - e.numpy()).max()))
                same &= bool(np.array_equal(a.argmax(-1), e.numpy().argmax(-1)))
    if max(errors) > .001:
        raise ValueError('Evaluation ONNX differs from PyTorch')
    return dict(examples=len(rows), maxAbsoluteErrors=errors, lastAxisArgmaxAgreement=same,
                note='Numerical checks only; new span codec not integrated on devices')


def verify_existing(args):
    import onnxruntime as ort
    report = json.loads((args.output / 'report.json').read_text())
    if digest(args.model / 'core-spans.onnx') != report['onnxSha256']:
        raise ValueError('Evaluation and ONNX model hashes differ')
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    if digest(binary) != report['kmpDateAdapterSha256']:
        raise ValueError('Native date adapter changed since evaluation')
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    runtime = ort.InferenceSession(str(args.model / 'core-spans.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    thresholds = report['thresholds']['span']
    parity = {}
    for split in ['dev', 'test', 'regression']:
        saved = json.loads((args.output / (split + '-predictions.json')).read_text())
        rows = saved['rows']
        raw = predict_onnx(runtime, tokenizer, rows)
        drafts = [decode(r, p, thresholds['span'], thresholds['request']) for r, p in zip(rows, raw)]
        actual = normalize_drafts(binary, rows, drafts, args.output / (split + '-onnx-dates'))
        different = [r['id'] for r, a, old in zip(rows, actual, saved['results']['span'])
                     if any(a[k] != old[k] for k in ['proposal', 'state', 'fields', 'reasons', 'residual'])]
        parity[split] = dict(examples=len(rows), completeProposalDifferences=different,
                             metrics={k: v for k, v in summarize(rows, actual).items() if k != 'failures'})
        if different:
            raise ValueError('ONNX complete proposal differs: ' + str(different))
    parity['sourceSha256'] = {p.name: digest(p) for p in Path(__file__).parent.glob('span_*.py')}
    (args.output / 'onnx-proposal-parity.json').write_text(json.dumps(parity, ensure_ascii=False, indent=2))
    print(json.dumps(parity, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-core-spans-v1'))
    parser.add_argument('--model', type=Path, default=Path('artifacts/zh-core-spans-v1'))
    parser.add_argument('--bio-model', type=Path, default=Path('artifacts/zh-core-bio-control-v1'))
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--output', type=Path, default=Path('reports/zh-core-spans-v1'))
    parser.add_argument('--verify-existing', action='store_true', help='Check all frozen ONNX proposals against saved PyTorch results')
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.verify_existing:
        verify_existing(args)
        return
    splits = {s: load(args.data / (s + '.jsonl')) for s in ['train', 'dev', 'test']}
    validate(splits)
    if args.output.exists():
        raise ValueError('Refusing to replace exposed evaluation results')
    args.output.mkdir(parents=True)
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    model = SpanModel(AutoConfig.from_pretrained(args.model / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(args.model / 'span.pt', weights_only=True))
    from .bio_control import BioModel, predict as bio_predict
    bio = BioModel(AutoConfig.from_pretrained(args.bio_model / 'encoder-config', local_files_only=True))
    bio.load_state_dict(torch.load(args.bio_model / 'bio.pt', weights_only=True))
    dev_raw = predict(model, tokenizer, splits['dev'])
    dev_bio_raw = bio_predict(bio, tokenizer, splits['dev'])
    calibrated = {name: calibrate(binary, splits['dev'], values, args.output / ('calibrate-' + name))
                  for name, values in [('span', dev_raw), ('same-data-BIO', dev_bio_raw)]}
    report = dict(humanReviewed=False, realUserData=False,
                  scope='Chinese offline core-field prototype, not device integration or production release',
                  kmpDateAdapterSha256=digest(binary), kmpSourceSha256=source_digest(args.core),
                  modelSha256=digest(args.model / 'span.pt'), onnxSha256=digest(args.model / 'core-spans.onnx'),
                  bioSha256=digest(args.bio_model / 'bio.pt'), thresholds=calibrated,
                  sourceSha256={p.name: digest(p) for p in Path(__file__).parent.glob('span_*.py')}, sets={})
    all_parity = []
    for split in ['dev', 'test', 'regression']:
        rows = load(args.data / (split + '.jsonl'))
        all_parity.extend(rows)
        raw = predict(model, tokenizer, rows)
        bio_raw = bio_predict(bio, tokenizer, rows)
        variants = {}
        for name, values in [('span', raw), ('same-data-BIO', bio_raw)]:
            thresholds = calibrated[name]
            drafts = [decode(r, p, thresholds['span'], thresholds['request']) for r, p in zip(rows, values)]
            variants[name] = normalize_drafts(binary, rows, drafts, args.output / f'{split}-{name}-dates')
        variants['v3'] = legacy(binary, rows, args.output / (split + '-v3'))
        metrics = {}
        for name, results in variants.items():
            metrics[name] = summarize(rows, results)
            groups = defaultdict(list)
            for i, row in enumerate(rows):
                groups[row['category']].append(i)
            metrics[name]['categories'] = {k: {kk: vv for kk, vv in summarize([rows[i] for i in ids], [results[i] for i in ids]).items() if kk != 'failures'}
                                           for k, ids in groups.items()}
        metrics['span']['spanMetrics'] = span_metrics(rows, raw, calibrated['span']['span'])
        metrics['same-data-BIO']['spanMetrics'] = span_metrics(rows, bio_raw, calibrated['same-data-BIO']['span'])
        report['sets'][split] = dict(datasetSha256=digest(args.data / (split + '.jsonl')), modes=metrics)
        (args.output / (split + '-predictions.json')).write_text(json.dumps(dict(rows=rows, rawSpan=raw, rawBIO=bio_raw, results=variants), ensure_ascii=False, indent=2))
        print(split, json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ['total', 'eligible', 'correct', 'accepted', 'wrongAccepted', 'correctCoverage', 'states']}
                                for k, v in metrics.items()}, ensure_ascii=False), flush=True)
    report['onnxParity'] = backend_parity(model, tokenizer, all_parity, args.model)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    training = json.loads((args.model / 'training-report.json').read_text())
    (args.model / 'prototype.json').write_text(json.dumps(dict(
        schema='experimental-core-spans-v1', locale='zh', maxTokens=96, roles=training['roles'],
        onnxSha256=training['onnxSha256'],
        thresholds={k: v for k, v in calibrated['span'].items() if k in ['span', 'request']},
        calibrationDatasetSha256=report['sets']['dev']['datasetSha256'],
        status='offline-prototype-not-production-manifest'), ensure_ascii=False, indent=2))
    print('ONNX parity', report['onnxParity'], flush=True)


if __name__ == '__main__':
    main()
