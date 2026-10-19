"""Real three-backend graph checks, plus Core codec fixtures from Core ML/MS logits."""
import argparse
import json
from pathlib import Path

import numpy as np
import coremltools as ct
import onnxruntime as ort
from transformers import AutoTokenizer

from quickadd_training.data import load
from quickadd_training.integration_bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--core-checks', type=Path, required=True)
    parser.add_argument('--core-results', type=Path, required=True)
    parser.add_argument('--verify-mindspore', action='store_true')
    args = parser.parse_args()
    root = args.directory / 'parity'
    frozen = json.loads(args.core_checks.read_text())
    if args.verify_mindspore:
        metadata = json.loads((root / 'checks.json').read_text())
        differences = {}
        for role, info in metadata['graphs'].items():
            errors = {name: 0. for name in info['outputs']}
            for i in range(info['examples']):
                for name in errors:
                    expected = np.fromfile(root / role / f'onnx-{i}-{name}.bin', dtype=np.float32)
                    actual = np.fromfile(root / role / f'mindspore-{i}-{name}.bin', dtype=np.float32)
                    if expected.shape != actual.shape or not np.isfinite(actual).all():
                        raise ValueError('MindSpore output invalid: ' + role + '/' + name)
                    errors[name] = max(errors[name], float(np.max(np.abs(expected - actual))))
            if max(errors.values()) > .001:
                raise ValueError('MindSpore numerical mismatch: ' + str(errors))
            differences[role] = errors
        metadata['mindsporeMaxErrors'] = differences
        for backend in ['coreml', 'mindspore']:
            cases = []
            for i, row in enumerate(frozen['cases']):
                outputs = []
                for tensor in row['encoderOutputs']:
                    values = np.fromfile(root / 'encoder' / f'{backend}-{i}-{tensor["name"]}.bin', dtype=np.float32)
                    outputs.append(dict(name=tensor['name'], shape=tensor['shape'], values=values.tolist()))
                cases.append(dict(row, encoderOutputs=outputs))
            result = dict(frozen, cases=cases)
            (root / f'{backend}-core-checks.json').write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
        (root / 'three-format-report.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(metadata, indent=2))
        return
    if root.exists():
        raise ValueError('Refusing to overwrite parity evidence')
    root.mkdir()
    tokenizer = AutoTokenizer.from_pretrained('artifacts/kivo-tiny-v1-preview/tokenizer', local_files_only=True)
    extra = load(Path('data/zh-joint-v3/dev.jsonl'))[:50]
    rows = [r['input'] for r in frozen['cases']] + extra
    core = json.loads(args.core_results.read_text())
    features = np.asarray([r['qualityFeature'] for r in core], dtype=np.float32)
    options = ort.SessionOptions(); options.intra_op_num_threads = 3
    metadata = dict(experimental=True, formalGatePassed=False,
                    scope='Host CPU ONNX, Core ML and official Linux MindSpore Lite; not phone performance', graphs={})
    for role in ['encoder', 'TASK', 'HABIT']:
        folder = root / role; folder.mkdir()
        ort_model = ort.InferenceSession(str(args.directory / f'kivo-{role}.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        ml_model = ct.models.MLModel(str(args.directory / f'kivo-{role}.mlmodel'), compute_units=ct.ComputeUnit.CPU_ONLY)
        names = [v.name for v in ort_model.get_outputs()]
        errors = {name: 0. for name in names}
        examples = len(rows) if role == 'encoder' else len(features)
        for i in range(examples):
            if role == 'encoder':
                token = tokenizer(rows[i]['text'], padding='max_length', max_length=96, truncation=False)
                inputs = {n: np.asarray([token[n]], dtype=np.int32) for n in ['input_ids', 'attention_mask']}
                inputs['preferred_kind'] = np.asarray([int(rows[i]['preferredKind'] == 'HABIT')], dtype=np.int32)
            else:
                inputs = dict(features=features[i:i + 1])
            for name, values in inputs.items():
                values.tofile(folder / f'input-{i}-{name}.bin')
            outputs = ort_model.run(names, inputs)
            actual = ml_model.predict(inputs)
            for name, expected in zip(names, outputs):
                value = np.asarray(actual[name], dtype=np.float32)
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError('Core ML output invalid: ' + role + '/' + name + str(value.shape))
                errors[name] = max(errors[name], float(np.max(np.abs(expected - value))))
                expected.astype(np.float32).tofile(folder / f'onnx-{i}-{name}.bin')
                value.tofile(folder / f'coreml-{i}-{name}.bin')
            if max(errors.values()) > .001:
                raise ValueError('Core ML numerical mismatch: ' + str(errors))
        metadata['graphs'][role] = dict(examples=examples, outputs=names, coremlMaxErrors=errors,
            assets={ext: dict(sha256=digest(args.directory / f'kivo-{role}.{ext}'), bytes=(args.directory / f'kivo-{role}.{ext}').stat().st_size)
                    for ext in ['onnx', 'mlmodel', 'ms']})
        print('Verified Core ML ' + role + ': ' + str(errors), flush=True)
    (root / 'checks.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
