"""Frozen-weight comparison: normalization without/with source fidelity checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from .bio_control import BioModel, predict as bio_predict
from .data import digest, load
from .metrics import accepted, correct
from .normalizer_fidelity import guard_normalized, policy_sha256
from .normalizer_model import Normalizer, predict
from .normalizer_rules import parse_normalized
from .span_decode import decode
from .span_evaluate import legacy, normalize_drafts, summarize
from .span_model import SpanModel, predict as span_predict


def confidence_filter(results, raw, threshold):
    values = []
    for old, prediction in zip(results, raw):
        value = dict(old, proposal=dict(old['proposal']), reasons=list(old['reasons']))
        if prediction['minimumProbability'] < threshold:
            value['reasons'] = list(dict.fromkeys(value['reasons'] + ['normalization-low-confidence']))
            value['proposal'].update(ready=False, needsAI=True)
            value.update(needsConfirmation=True, state='PARTIAL' if any(value['fields'].values()) else 'CONFIRM')
        values.append(value)
    return values


def calibrate(rows, guarded, raw):
    grid = []
    for threshold in [0., .2, .4, .6, .8, .9]:
        metrics = summarize(rows, confidence_filter(guarded, raw, threshold))
        grid.append(dict(minimumProbability=threshold, **{k: metrics[k] for k in [
            'correct', 'accepted', 'wrongAccepted', 'wrongAcceptanceRate', 'correctCoverage']}))
    feasible = [v for v in grid if v['accepted'] and v['wrongAcceptanceRate'] <= .01]
    best = max(feasible, key=lambda v: (v['correct'], -v['wrongAccepted'], v['minimumProbability'])) if feasible else max(
        grid, key=lambda v: (v['correct'] - 10 * v['wrongAccepted'], v['minimumProbability']))
    return dict(minimumProbability=best['minimumProbability'], grid=grid, qualityTargetMet=bool(feasible),
                selection='dev only: most correct with <=1% observed wrong acceptance; fallback correct-10*wrong, target unmet')


def paired(rows, baseline, guarded):
    blocked_wrong, lost_correct, remaining_wrong = [], [], []
    for row, old, new in zip(rows, baseline, guarded):
        if accepted(new['proposal']):
            assert accepted(old['proposal']), 'Fixed-threshold guard must only reduce acceptance'
            assert old['proposal'] == new['proposal'], 'Guard must not repair or change accepted fields'
        old_right = correct(row['expected'], old['proposal'])
        new_right = correct(row['expected'], new['proposal'])
        if accepted(old['proposal']) and not old_right and not accepted(new['proposal']):
            blocked_wrong.append(row['id'])
        if old_right and not new_right:
            lost_correct.append(row['id'])
        if accepted(new['proposal']) and not new_right:
            remaining_wrong.append(row['id'])
    return dict(blockedWrongIds=blocked_wrong, lostCorrectIds=lost_correct, remainingWrongIds=remaining_wrong)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-fidelity-compare-v1'))
    parser.add_argument('--model', type=Path, default=Path('artifacts/zh-normalizer-v1'))
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--output', type=Path, default=Path('reports/zh-fidelity-compare-v1'))
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.output.exists():
        raise ValueError('Refusing to replace frozen results')
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
    settings = json.loads((args.model / 'prototype.json').read_text())
    assert digest(args.model / 'normalizer.pt') == settings['modelSha256']
    for directory in [args.model, span_dir, bio_dir]:
        training = json.loads((directory / 'training-report.json').read_text())
        for split in ['train', 'dev']:
            assert digest(args.data / (split + '.jsonl')) == training[split + 'Sha256']
    sources = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')
               if p.name.startswith(('fidelity_', 'normalizer_', 'span_', 'bio_'))}
    report = dict(scope='fixed weights; source-copy fidelity post-validation; third entity holdout',
                  humanReviewed=False, dataProvenance=json.loads((args.data / 'provenance.json').read_text()),
                  modelSha256=dict(normalizer=digest(args.model / 'normalizer.pt'), span=digest(span_dir / 'span.pt'),
                                   bio=digest(bio_dir / 'bio.pt')),
                  kmpDateAdapterSha256=digest(binary), policySha256=policy_sha256(), sourceSha256=sources,
                  baselineMinimumProbability=settings['minimumProbability'],
                  extractionThresholds=original['thresholds'], sets={})
    for split in ['dev', 'test', 'regression']:
        rows = load(args.data / (split + '.jsonl'))
        raw = predict(model, tokenizer, rows)
        drafts = [parse_normalized(r, p) for r, p in zip(rows, raw)]
        guarded = [guard_normalized(r, p, d, tokenizer) for r, p, d in zip(rows, raw, drafts)]
        base_results = normalize_drafts(binary, rows, drafts, args.output / (split + '-raw-dates'))
        guarded_results = normalize_drafts(binary, rows, guarded, args.output / (split + '-guard-dates'))
        if split == 'dev':
            report['calibration'] = calibrate(rows, guarded_results, raw)
            # Save the policy, model/data hashes and thresholds BEFORE any test inference.
            (args.output / 'frozen-protocol.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        results = {
            'normalize-raw': confidence_filter(base_results, raw, settings['minimumProbability']),
            'normalize-guard-fixed': confidence_filter(guarded_results, raw, settings['minimumProbability']),
            'normalize-guard-calibrated': confidence_filter(guarded_results, raw, report['calibration']['minimumProbability']),
        }
        for name, network, predictor in [('span', span, span_predict), ('same-data-BIO', bio, bio_predict)]:
            threshold = original['thresholds'][name]
            predictions = predictor(network, tokenizer, rows)
            decoded = [decode(r, p, threshold['span'], threshold['request']) for r, p in zip(rows, predictions)]
            results[name] = normalize_drafts(binary, rows, decoded, args.output / (split + '-' + name + '-dates'))
        results['v3'] = legacy(binary, rows, args.output / (split + '-v3'))
        comparison = paired(rows, results['normalize-raw'], results['normalize-guard-fixed'])
        metrics = {name: summarize(rows, values) for name, values in results.items()}
        report['sets'][split] = dict(datasetSha256=digest(args.data / (split + '.jsonl')), modes=metrics,
                                    fixedThresholdPaired=comparison,
                                    auditErrors=dict(Counter(e for d in guarded for e in d['fidelity'].get('errors', []))))
        (args.output / (split + '-predictions.json')).write_text(json.dumps(
            dict(rows=rows, normalization=raw, results=results), ensure_ascii=False, indent=2))
        print(split, json.dumps({name: {k: v[k] for k in ['correct', 'eligible', 'accepted', 'wrongAccepted']}
                                 for name, v in metrics.items()}, ensure_ascii=False), flush=True)
        if split == 'test':
            elapsed = []
            for row, prediction, draft in zip(rows[:60], raw[:60], drafts[:60]):
                start = time.perf_counter()
                guard_normalized(row, prediction, draft, tokenizer)
                elapsed.append((time.perf_counter() - start) * 1000)
            report['hostGuardLatency'] = dict(samples=len(elapsed), medianMs=statistics.median(elapsed),
                                             p95Ms=sorted(elapsed)[56],
                                             scope='host CPU; guard plus tokenization only, no model/KMP/UI; not phone latency')
    assert sources == {name: digest(Path(__file__).parent / name) for name in sources}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    prototype = dict(schema='experimental-normalizer-fidelity-v1', modelSha256=settings['modelSha256'],
                     minimumProbability=report['calibration']['minimumProbability'],
                     devQualityTargetMet=report['calibration']['qualityTargetMet'], policySha256=policy_sha256(),
                     fidelitySourceSha256=digest(Path(__file__).parent / 'normalizer_fidelity.py'),
                     calibrationSha256=digest(args.data / 'dev.jsonl'))
    (args.output / 'prototype.json').write_text(json.dumps(prototype, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
