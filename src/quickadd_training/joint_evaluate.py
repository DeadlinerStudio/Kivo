"""Train frozen-encoder heads, calibrate by predicted kind, then open frozen synthetic holdouts."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .data import digest, load
from .joint_model import JointModel, predict
from .joint_adapter import network
from .joint_heads import prepare, choose, train_rank, train_quality, quality_logits, save_heads, load_heads, exact
from .joint_native import NativeFields
from .joint_calibration import fit, confidence, select, evaluate, reliability
from .joint_decode import apply_gate
from .joint_diagnostics import diagnose
from .joint_rules import baseline


def plain(value):
    result = {k: v for k, v in value.items() if k not in ['feature', 'qualityFeature']}
    if 'spans' in result:
        result['spans'] = [{k: v for k, v in span.items() if k not in ['vector', 'edge']} for span in result['spans']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/zh-joint-v2'))
    parser.add_argument('--model', type=Path, default=Path('artifacts/zh-joint-v2-seed42'))
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--output', type=Path, default=Path('reports/zh-joint-v2/seed42'))
    parser.add_argument('--heads-only', action='store_true', help='Train/compare on dev without opening calibration or final data')
    parser.add_argument('--resume-heads', action='store_true', help='Evaluate the dev-selected champion from its saved heads')
    parser.add_argument('--base-rank', action='store_true', help='Preplanned ranking ablation: use frozen span scores, train only correctness heads')
    args = parser.parse_args()
    torch.set_num_threads(3); torch.manual_seed(4201)
    if args.output.exists() and not args.resume_heads:
        raise ValueError('Refusing to replace frozen screening evidence')
    args.output.mkdir(parents=True, exist_ok=args.resume_heads)
    if args.resume_heads and (args.output / 'report.json').exists():
        raise ValueError('Refusing to replace final evidence')
    settings = json.loads((args.model / 'training-report.json').read_text())
    assert digest(args.model / 'joint.pt') == settings['modelSha256']
    assert digest(args.data / 'train.jsonl') == settings['dataSha256']['train']
    provenance = json.loads((args.data / 'provenance.json').read_text())
    for split, partition in provenance['partitions'].items():
        assert digest(args.data / (split + '.jsonl')) == partition['sha256'], 'Frozen data changed: ' + split
    model = network(AutoConfig.from_pretrained(args.model / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(args.model / 'joint.pt', weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    native = NativeFields(binary, args.output / ('native-fields-' + str(time.time_ns())))
    started = time.monotonic()
    if args.resume_heads:
        report = json.loads((args.output / 'heads-report.json').read_text())
        assert report['modelSha256'] == digest(args.model / 'joint.pt')
        assert report['nativeSha256'] == digest(binary)
        for name, checksum in report['criticalSourceSha256'].items():
            assert digest(Path(__file__).parent / name) == checksum, 'Head feature policy changed'
        rankers, quality = load_heads(args.output / 'rank.pt'), load_heads(args.output / 'quality.pt')
        dev_rows = load(args.data / 'dev.jsonl')
        dev_selected = choose(prepare(dev_rows, predict(model, tokenizer, dev_rows), native), rankers)
        dev_logits = quality_logits(quality, dev_selected)
        (args.output / 'dev-predictions.json').write_text(json.dumps(dict(rows=dev_rows, logits=dev_logits.tolist(),
            selected=[plain(v) for v in dev_selected]), ensure_ascii=False, indent=2))
    else:
        report, rankers, quality, dev_rows, dev_selected = train_heads(args, model, tokenizer, native, settings, binary)
    if args.heads_only:
        return
    report['devRulesBaseline'], _ = baseline(dev_rows, binary, native.directory / 'rules-dev')
    stopped = False
    for split in ['cal_fit', 'cal_gate', 'test', 'protection', 'regression']:
        if stopped and split in ['test', 'protection']:
            continue
        rows = load(args.data / (split + '.jsonl'))
        groups = prepare(rows, predict(model, tokenizer, rows), native)
        selected = choose(groups, rankers)
        del groups
        gc.collect()
        logits = quality_logits(quality, selected)
        if split == 'cal_fit':
            report['calibration'] = fit(rows, selected, logits)
        probabilities = confidence(selected, logits, report['calibration'])
        if split == 'cal_gate':
            report['thresholdSelection'] = select(rows, selected, probabilities)
            report['headSha256'] = {name: digest(args.output / name) for name in ['rank.pt', 'quality.pt']}
            source_paths = list(Path(__file__).parent.glob('joint_*.py')) + [Path(__file__).parent / name for name in
                ['span_model.py', 'normalizer_fidelity.py', 'corpus.py', 'data.py', 'native_adapter.py']]
            report['sourceSha256'] = {p.name: digest(p) for p in source_paths}
            report['modelAssetSha256'] = {str(p.relative_to(args.model)): digest(p)
                for folder in ['tokenizer', 'encoder-config'] for p in (args.model / folder).rglob('*') if p.is_file()}
            stopped = not all(v['screeningTargetMet'] for v in report['thresholdSelection']['choices'].values())
            report['finalHoldoutsOpened'] = not stopped
            report['screeningStatus'] = 'stopped-no-qualified-cutoff' if stopped else 'calibration-screening-qualified'
            (args.output / 'frozen-protocol.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        choices = report.get('thresholdSelection', {}).get('choices', {})
        gated = [apply_gate(v, c, choices.get(v['proposal']['kind'], {}).get('threshold')) for v, c in zip(selected, probabilities)]
        diagnostic = [apply_gate(v, c, .90) for v, c in zip(selected, probabilities)]
        metrics, metrics90 = evaluate(rows, gated), evaluate(rows, diagnostic)
        rules_metrics, rule_values = baseline(rows, binary, native.directory / ('rules-' + split))
        failures = [dict(id=r['id'], text=r['text'], expected=r['expected'], actual=plain(v))
                    for r, v in zip(rows, gated) if (r['expected']['decision'] == 'accept' and not (v['localTrusted'] and exact(r, v)))
                    or (v['localTrusted'] and not exact(r, v))]
        report['sets'][split] = dict(sha256=digest(args.data / (split + '.jsonl')), selectedPolicy=metrics,
                                    fixed90Diagnostic=metrics90, reliability=reliability(rows, selected, probabilities),
                                    proposalDiagnostics=diagnose(rows, selected),
                                    rulesBaseline=rules_metrics,
                                    hardReasons=dict(Counter(reason for v in selected for reason in v['hardReasons'])), failures=failures)
        payload = dict(rows=rows, rawQualityLogits=logits.tolist(), results=[plain(v) for v in gated],
                       rules=[v['proposal'] for v in rule_values],
                       fixed90Diagnostic=[plain(v) for v in diagnostic])
        (args.output / (split + '-predictions.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(split, json.dumps(dict(selected=metrics, diagnostic90=metrics90), ensure_ascii=False), flush=True)
    report['elapsedSeconds'] = time.monotonic() - started
    assert report['nativeSha256'] == digest(binary), 'Native adapter changed during evaluation'
    assert report['sourceSha256'] == {name: digest(Path(__file__).parent / name) for name in report['sourceSha256']}
    assert report['modelAssetSha256'] == {name: digest(args.model / name) for name in report['modelAssetSha256']}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    prototype = dict(schema='experimental-task-habit-confidence-v2', modelDirectory=str(args.model.resolve()),
                     dataDirectory=str(args.data.resolve()),
                     modelSha256=report['modelSha256'], nativeSha256=report['nativeSha256'],
                     headSha256=report['headSha256'], calibration=report['calibration'],
                     modelAssetSha256=report['modelAssetSha256'],
                     thresholds=report['thresholdSelection']['choices'], humanReviewed=False, formalGatePassed=False,
                     policySources=report['sourceSha256'])
    (args.output / 'prototype.json').write_text(json.dumps(prototype, ensure_ascii=False, indent=2))


def train_heads(args, model, tokenizer, native, settings, binary):
    dev_rows = load(args.data / 'dev.jsonl')
    dev_groups = prepare(dev_rows, predict(model, tokenizer, dev_rows), native)
    rank_groups = []
    if args.base_rank:
        rankers, rank_history = {}, dict(policy='sum of frozen span scores; no learned reranker')
    else:
        rank_rows = load(args.data / 'rank.jsonl')
        rank_groups = prepare(rank_rows, predict(model, tokenizer, rank_rows), native)
        print('prepared rank/dev candidates', len(rank_groups), len(dev_groups), flush=True)
        rankers, rank_history = train_rank(rank_rows, rank_groups, dev_rows, dev_groups)
    save_heads(args.output / 'rank.pt', rankers)
    dev_selected = choose(dev_groups, rankers)
    dev_diagnostics = {}
    for kind in ['TASK', 'HABIT']:
        positive = [(r, g) for r, g in zip(dev_rows, dev_groups) if r['expected']['decision'] == 'accept' and r['expected']['proposal']['kind'] == kind]
        dev_diagnostics[kind] = dict(eligible=len(positive), correctInCandidatePool=sum(any(v['draftValid'] and exact(r, v) for v in g) for r, g in positive),
                                    topCandidateCorrect=sum(v['draftValid'] and exact(r, v) for r, v in zip(dev_rows, dev_selected)
                                                            if r['expected']['decision'] == 'accept' and r['expected']['proposal']['kind'] == kind))
    del rank_groups, dev_groups
    gc.collect()
    quality_rows = load(args.data / 'quality.jsonl')
    quality_groups = prepare(quality_rows, predict(model, tokenizer, quality_rows), native)
    quality_selected = choose(quality_groups, rankers)
    del quality_groups
    gc.collect()
    quality, quality_history = train_quality(quality_rows, quality_selected, dev_rows, dev_selected)
    save_heads(args.output / 'quality.pt', quality)
    del quality_selected
    gc.collect()
    print('trained rank and correctness heads', dev_diagnostics, flush=True)
    report = dict(scope=f'seed{settings["seed"]} synthetic engineering screening', humanReviewed=False, formalGatePassed=False,
                  modelSha256=settings['modelSha256'], nativeSha256=digest(binary), seed=settings['seed'],
                  candidateSelection='base-span-scores' if args.base_rank else 'learned-reranker',
                  dataProvenance=json.loads((args.data / 'provenance.json').read_text()),
                  rankHistory=rank_history, qualityHistory=quality_history, devCandidateDiagnostics=dev_diagnostics, sets={})
    report['devProposalDiagnostics'] = diagnose(dev_rows, dev_selected)
    logits = quality_logits(quality, dev_selected)
    grid = []
    for cutoff in [.5, .8, .9, .95, .98, .99, .995, .999]:
        metrics = evaluate(dev_rows, [apply_gate(v, float(1 / (1 + np.exp(-z))), cutoff) for v, z in zip(dev_selected, logits)])
        grid.append(dict(rawScoreCutoff=cutoff, modes=metrics))
    feasible = [v for v in grid if all(m['accepted'] and m['wrongAcceptanceRate'] <= .005 for m in v['modes'].values())]
    best = max(feasible, key=lambda v: (min(m['correctCoverage'] for m in v['modes'].values()),
                                        np.mean([m['correctCoverage'] for m in v['modes'].values()]))) if feasible else None
    report['devSelection'] = best
    report['devUncalibratedGrid'] = grid
    report['criticalSourceSha256'] = {name: digest(Path(__file__).parent / name) for name in [
        'joint_model.py', 'joint_adapter.py', 'joint_decode.py', 'joint_heads.py', 'joint_native.py']}
    report['headSha256'] = {name: digest(args.output / name) for name in ['rank.pt', 'quality.pt']}
    regression_rows = load(args.data / 'regression.jsonl')
    regression_selected = choose(prepare(regression_rows, predict(model, tokenizer, regression_rows), native), rankers)
    report['knownRegression'] = dict(scope='Known development examples, replayed in v3 training; not independent accuracy',
        positiveRows=sum(r['expected']['decision'] == 'accept' for r in regression_rows),
        validCorrect=sum(v['draftValid'] and exact(r, v) for r, v in zip(regression_rows, regression_selected)),
        negativeRows=sum(r['expected']['decision'] == 'defer' for r in regression_rows),
        negativeValid=sum(v['draftValid'] for r, v in zip(regression_rows, regression_selected) if r['expected']['decision'] == 'defer'),
        diagnostics=diagnose(regression_rows, regression_selected))
    (args.output / 'known-regression-predictions.json').write_text(json.dumps(dict(rows=regression_rows,
        selected=[plain(v) for v in regression_selected], logits=quality_logits(quality, regression_selected).tolist()),
        ensure_ascii=False, indent=2))
    (args.output / 'heads-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output / 'dev-predictions.json').write_text(json.dumps(dict(rows=dev_rows, logits=logits.tolist(),
        selected=[plain(v) for v in dev_selected]), ensure_ascii=False, indent=2))
    print('head training complete, dev selection', best, flush=True)
    return report, rankers, quality, dev_rows, dev_selected


if __name__ == '__main__':
    main()
