"""Independent normalization -> fixed-format rules -> shared production date parser."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from .normalizer_rules import parse_normalized
from .data import digest, load
from .normalizer_model import Normalizer, canonical, predict as normalize
from .span_decode import decode, finish
from .span_evaluate import legacy, normalize_drafts, summarize
from .span_model import SpanModel, predict as extract
from .bio_control import BioModel, predict as bio_predict


def calibrate(binary, rows, predictions, directory):
    drafts = [parse_normalized(r, p) for r, p in zip(rows, predictions)]
    base = normalize_drafts(binary, rows, drafts, directory)
    grid = []
    for threshold in [0., .2, .4, .6, .8, .9]:
        results = []
        for old, p in zip(base, predictions):
            value = dict(old, proposal=dict(old['proposal']), reasons=list(old['reasons']))
            if p['minimumProbability'] < threshold:
                value['reasons'].append('normalization-low-confidence')
                value['proposal'].update(ready=False, needsAI=True)
                value.update(needsConfirmation=True, state='PARTIAL' if value['fields']['TITLE'] else 'CONFIRM')
            results.append(value)
        metrics = summarize(rows, results)
        grid.append(dict(minimumProbability=threshold, **{k: metrics[k] for k in [
            'correct', 'accepted', 'wrongAccepted', 'wrongAcceptanceRate', 'correctCoverage']}))
    feasible = [v for v in grid if v['accepted'] and v['wrongAcceptanceRate'] <= .01]
    best = max(feasible, key=lambda v: (v['correct'], -v['wrongAccepted'], v['minimumProbability'])) if feasible else max(
        grid, key=lambda v: (v['correct'] - 10 * v['wrongAccepted'], v['minimumProbability']))
    return dict(minimumProbability=best['minimumProbability'], grid=grid, qualityTargetMet=bool(feasible),
                selection='dev only: maximize correct with <=1% observed wrong acceptance; if no feasible grid point, use correct-10*wrong and mark target unmet')


def latency(model, tokenizer, rows, prediction_function):
    for row in rows[:3]:
        prediction_function(model, tokenizer, [row], batch_size=1)
    values = []
    for row in rows[:30]:
        start = time.perf_counter()
        prediction_function(model, tokenizer, [row], batch_size=1)
        values.append((time.perf_counter() - start) * 1000)
    return dict(samples=len(values), medianMs=statistics.median(values), p95Ms=sorted(values)[int(.95 * (len(values) - 1))],
                runtime='host PyTorch CPU, 3 threads, tokenization/decoding included, no KMP/UI, not phone latency')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-normalization-compare-v1'))
    parser.add_argument('--model', type=Path, default=Path('artifacts/zh-normalizer-v1'))
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--output', type=Path, default=Path('reports/zh-normalization-compare-v1'))
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.output.exists():
        raise ValueError('Refusing to replace comparison results')
    args.output.mkdir(parents=True)
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    model = Normalizer(AutoConfig.from_pretrained(args.model / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(args.model / 'normalizer.pt', weights_only=True))
    span_dir, bio_dir = Path('artifacts/zh-core-spans-v1'), Path('artifacts/zh-core-bio-control-v1')
    span = SpanModel(AutoConfig.from_pretrained(span_dir / 'encoder-config', local_files_only=True))
    span.load_state_dict(torch.load(span_dir / 'span.pt', weights_only=True))
    bio = BioModel(AutoConfig.from_pretrained(bio_dir / 'encoder-config', local_files_only=True))
    bio.load_state_dict(torch.load(bio_dir / 'bio.pt', weights_only=True))
    original = json.loads(Path('reports/zh-core-spans-v1/report.json').read_text())
    training = json.loads((args.model / 'training-report.json').read_text())
    for s in ['train', 'dev']:
        key = 'trainSha256' if s == 'train' else 'devSha256'
        assert digest(args.data / (s + '.jsonl')) == training[key]
        for directory in [span_dir, bio_dir]:
            assert json.loads((directory / 'training-report.json').read_text())[key] == training[key]
    dev_rows = load(args.data / 'dev.jsonl')
    dev_raw = normalize(model, tokenizer, dev_rows)
    threshold = calibrate(binary, dev_rows, dev_raw, args.output / 'dev-calibration-dates')
    report = dict(scope='same-data Chinese normalization-vs-extraction experiment', humanReviewed=False,
                  modelSha256=digest(args.model / 'normalizer.pt'), dataProvenance=json.loads((args.data / 'provenance.json').read_text()),
                  kmpDateAdapterSha256=digest(binary), calibration=threshold,
                  spanCalibration=original['thresholds']['span'], bioCalibration=original['thresholds']['same-data-BIO'],
                  rules='fixed canonical format parser, followed by the same production KMP date normalizer',
                  sets={})
    for split in ['dev', 'test', 'regression']:
        rows = load(args.data / (split + '.jsonl'))
        raw = dev_raw if split == 'dev' else normalize(model, tokenizer, rows)
        normalization_drafts = [parse_normalized(r, p, threshold['minimumProbability']) for r, p in zip(rows, raw)]
        results = {'normalize-then-rules': normalize_drafts(binary, rows, normalization_drafts, args.output / (split + '-normalized-dates'))}
        for name, predictor, network, settings in [('span', extract, span, original['thresholds']['span']),
                                                    ('same-data-BIO', bio_predict, bio, original['thresholds']['same-data-BIO'])]:
            predictions = predictor(network, tokenizer, rows)
            drafts = [decode(r, p, settings['span'], settings['request']) for r, p in zip(rows, predictions)]
            results[name] = normalize_drafts(binary, rows, drafts, args.output / (split + '-' + name + '-dates'))
        results['v3'] = legacy(binary, rows, args.output / (split + '-v3'))
        metrics = {name: summarize(rows, predictions) for name, predictions in results.items()}
        for name, predictions in results.items():
            groups = defaultdict(list)
            for i, row in enumerate(rows):
                groups[row['category']].append(i)
            metrics[name]['categories'] = {key: {k: v for k, v in summarize([rows[i] for i in ids], [predictions[i] for i in ids]).items() if k != 'failures'}
                                          for key, ids in groups.items()}
        positives = [(r, p) for r, p in zip(rows, raw) if r['expected']['decision'] == 'accept']
        metrics['normalize-then-rules']['exactNormalization'] = sum(p['normalized'] == canonical(r) for r, p in positives) / len(positives)
        report['sets'][split] = dict(datasetSha256=digest(args.data / (split + '.jsonl')), modes=metrics)
        (args.output / (split + '-predictions.json')).write_text(json.dumps(dict(rows=rows, normalization=raw, results=results), ensure_ascii=False, indent=2))
        print(split, json.dumps({name: {k: v for k, v in values.items() if k in ['correct', 'eligible', 'accepted', 'wrongAccepted', 'correctCoverage', 'exactNormalization']}
                                for name, values in metrics.items()}, ensure_ascii=False), flush=True)
    test_rows = load(args.data / 'test.jsonl')
    report['hostLatency'] = dict(normalizer=latency(model, tokenizer, test_rows, normalize), span=latency(span, tokenizer, test_rows, extract))
    report['parameters'] = dict(normalizer=sum(p.numel() for p in model.parameters()), span=sum(p.numel() for p in span.parameters()))
    report['sourceSha256'] = {p.name: digest(p) for p in Path(__file__).parent.glob('normalizer_*.py')}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.model / 'prototype.json').write_text(json.dumps(dict(schema='experimental-normalizer-v1', minimumProbability=threshold['minimumProbability'],
                                                             modelSha256=report['modelSha256'], calibrationSha256=digest(args.data / 'dev.jsonl'),
                                                             devQualityTargetMet=threshold['qualityTargetMet']), indent=2))
    print('host latency', report['hostLatency'], flush=True)


if __name__ == '__main__':
    main()
