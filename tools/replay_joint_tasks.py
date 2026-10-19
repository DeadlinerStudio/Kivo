"""Replay already exposed legacy TASK positives without changing frozen model or gold."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import quickadd_training
from quickadd_training.data import digest, load
from quickadd_training.joint_adapter import network
from quickadd_training.joint_model import predict
from quickadd_training.joint_heads import prepare, choose, load_heads, quality_logits, exact
from quickadd_training.joint_native import NativeFields
from quickadd_training.joint_calibration import confidence, evaluate
from quickadd_training.joint_decode import apply_gate
from quickadd_training.joint_diagnostics import diagnose
from quickadd_training.joint_evaluate import plain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prototype', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace legacy replay evidence')
    torch.set_num_threads(3)
    config = json.loads(args.prototype.read_text())
    model_dir, head_dir = Path(config['modelDirectory']), args.prototype.parent
    sources = Path(quickadd_training.__file__).parent
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    assert digest(binary) == config['nativeSha256']
    assert digest(model_dir / 'joint.pt') == config['modelSha256']
    for folder, values in [(sources, config['policySources']), (head_dir, config['headSha256']),
                           (model_dir, config['modelAssetSha256'])]:
        assert all(digest(folder / name) == checksum for name, checksum in values.items())
    all_rows = load(args.data)
    rows = [r for r in all_rows if r['expected']['decision'] == 'accept' and r['expected']['proposal']['kind'] == 'TASK']
    args.output.mkdir(parents=True)
    model = network(AutoConfig.from_pretrained(model_dir / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(model_dir / 'joint.pt', weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    rankers, quality = [load_heads(head_dir / name) for name in ['rank.pt', 'quality.pt']]
    native = NativeFields(binary, args.output / 'native')
    groups = prepare(rows, predict(model, tokenizer, rows), native)
    candidate_correct = sum(any(v['draftValid'] and exact(r, v) for v in group) for r, group in zip(rows, groups))
    selected = choose(groups, rankers)
    probabilities = confidence(selected, quality_logits(quality, selected), config['calibration'])
    values = [apply_gate(v, c, config['thresholds'][v['proposal']['kind']]['threshold']) for v, c in zip(selected, probabilities)]
    wrong = [dict(text=r['text'], expected=r['expected']['proposal'], actual=plain(v))
             for r, v in zip(rows, values) if v['localTrusted'] and not exact(r, v)]
    report = dict(scope='Already exposed legacy TASK-positive development replay after final freeze; no retuning or new independent accuracy claim',
                  dataSha256=digest(args.data), prototypeSha256=digest(args.prototype), positives=len(rows),
                  toolSha256=digest(Path(__file__)), correctCandidateAvailable=candidate_correct,
                  excludedRows=len(all_rows) - len(rows), excludedReason='Old confirm labels included then-unsupported HABIT; not reinterpreted',
                  metrics=evaluate(rows, values), diagnostics=diagnose(rows, values),
                  wrongAcceptanceExamples=wrong, formalGatePassed=False)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output / 'predictions.json').write_text(json.dumps(dict(rows=rows, results=[plain(v) for v in values]), ensure_ascii=False, indent=2))
    print(json.dumps(dict(metrics=report['metrics'], wrong=len(wrong)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
