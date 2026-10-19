"""Dev-only matched candidate diagnostics; no oracle routing and no final-set selection."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from .data import load, digest
from .joint_model import JointModel, predict
from .joint_adapter import network
from .joint_heads import prepare, choose, exact
from .joint_native import NativeFields
from .joint_evaluate import plain


def diagnose(rows, groups):
    selected = choose(groups, None)
    result = {}
    for kind in ['TASK', 'HABIT']:
        ids = [i for i, r in enumerate(rows) if r['expected']['decision'] == 'accept' and r['expected']['proposal']['kind'] == kind]
        result[kind] = dict(eligible=len(ids), rawTopCorrect=sum(selected[i]['draftValid'] and exact(rows[i], selected[i]) for i in ids),
                            correctCandidateAvailable=sum(any(v['draftValid'] and exact(rows[i], v) for v in groups[i]) for i in ids))
    return result, selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', required=True, type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    p.add_argument('--data', type=Path, default=Path('data/zh-joint-v2'))
    args = p.parse_args()
    torch.set_num_threads(3)
    if args.output.exists():
        raise ValueError('Refusing to overwrite dev comparison')
    args.output.mkdir(parents=True)
    rows = load(args.data / 'dev.jsonl')
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    native, reports = NativeFields(binary, args.output / 'native'), {}
    for directory in args.models:
        training = json.loads((directory / 'training-report.json').read_text())
        assert digest(directory / 'joint.pt') == training['modelSha256']
        model = network(AutoConfig.from_pretrained(directory / 'encoder-config', local_files_only=True))
        model.load_state_dict(torch.load(directory / 'joint.pt', weights_only=True))
        tokenizer = AutoTokenizer.from_pretrained(directory / 'tokenizer', local_files_only=True)
        groups = prepare(rows, predict(model, tokenizer, rows), native)
        metrics, selected = diagnose(rows, groups)
        reports[directory.name] = dict(training=training, metrics=metrics)
        (args.output / (directory.name + '-predictions.json')).write_text(json.dumps(
            dict(rows=rows, selected=[plain(v) for v in selected]), ensure_ascii=False, indent=2))
        print(directory.name, metrics, flush=True)
        del groups, selected, model
    (args.output / 'report.json').write_text(json.dumps(dict(scope='dev only; matched base beam selection, no correctness calibration',
        devSha256=digest(args.data / 'dev.jsonl'), nativeSha256=digest(binary), models=reports), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
