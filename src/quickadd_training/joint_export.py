"""Experimental ONNX graphs and complete proposal parity, never a production manifest."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer

from .data import load, digest
from .joint_adapter import network
from .joint_model import encode, from_logits, predict
from .joint_heads import load_heads, prepare, choose, quality_logits
from .joint_native import NativeFields
from .joint_calibration import confidence
from .joint_decode import apply_gate


class OrtHead:
    def __init__(self, path, options):
        self.runtime = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    def eval(self):
        return self

    def __call__(self, x):
        return torch.from_numpy(self.runtime.run(None, dict(features=x.numpy().astype(np.float32)))[0])


def infer(runtime, tokenizer, rows):
    results = []
    for row in rows:
        inputs, offsets, valid = encode(tokenizer, [row])
        out = runtime.run(None, {k: v.numpy().astype(np.int32) for k, v in inputs.items()})
        results += from_logits(*(torch.from_numpy(v) for v in out), offsets, valid)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prototype', required=True, type=Path)
    p.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(3)
    config = json.loads(args.prototype.read_text())
    model_dir, heads_dir = Path(config['modelDirectory']), args.prototype.parent
    if args.output.exists():
        raise ValueError('Refusing to overwrite ONNX evidence')
    args.output.mkdir(parents=True)
    assert digest(model_dir / 'joint.pt') == config['modelSha256']
    for name, checksum in config['policySources'].items():
        assert digest(Path(__file__).parent / name) == checksum, 'Frozen inference source changed'
    for name, checksum in config['headSha256'].items():
        assert digest(heads_dir / name) == checksum, 'Frozen scoring weights changed'
    for name, checksum in config['modelAssetSha256'].items():
        assert digest(model_dir / name) == checksum, 'Frozen tokenizer/config changed'
    model = network(AutoConfig.from_pretrained(model_dir / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(model_dir / 'joint.pt', weights_only=True)); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    data = Path(config['dataDirectory'])
    rows = load(data / 'dev.jsonl') + load(data / 'regression.jsonl')
    source_report = json.loads((heads_dir / 'report.json').read_text())
    if source_report['finalHoldoutsOpened']:
        rows += load(data / 'test.jsonl') + load(data / 'protection.jsonl')
    inputs, _, _ = encode(tokenizer, rows[:1])
    names = ['input_ids', 'attention_mask', 'preferred_kind']
    torch.onnx.export(model, tuple(inputs[k].to(torch.int32) for k in names), args.output / 'encoder-spans.onnx',
                      input_names=names, output_names=['intent_logits', 'span_logits', 'token_features'], opset_version=13, dynamo=False)
    rankers, qualities = load_heads(heads_dir / 'rank.pt'), load_heads(heads_dir / 'quality.pt')
    for name, heads in [('rank', rankers), ('quality', qualities)]:
        for kind, head in heads.items():
            head.eval()
            torch.onnx.export(head, torch.zeros((2, head.mean.numel())), args.output / (name + '-' + kind + '.onnx'),
                              input_names=['features'], output_names=['score'], dynamic_axes={'features': {0: 'candidates'}, 'score': {0: 'candidates'}},
                              opset_version=13, dynamo=False)
    options = ort.SessionOptions(); options.intra_op_num_threads = 3
    runtime = ort.InferenceSession(str(args.output / 'encoder-spans.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    errors = [0., 0., 0.]
    torch_ms, ort_ms = [], []
    with torch.inference_mode():
        for row in rows[:100]:
            inputs, _, _ = encode(tokenizer, [row])
            start_time = time.perf_counter()
            expected = model(**inputs)
            torch_ms.append((time.perf_counter() - start_time) * 1000)
            start_time = time.perf_counter()
            actual = runtime.run(None, {k: v.numpy().astype(np.int32) for k, v in inputs.items()})
            ort_ms.append((time.perf_counter() - start_time) * 1000)
            for i, (a, b) in enumerate(zip(actual, expected)):
                errors[i] = max(errors[i], float(np.max(np.abs(a - b.numpy()))))
    assert max(errors) <= .001
    onnx_rank = {k: OrtHead(args.output / ('rank-' + k + '.onnx'), options) for k in rankers}
    onnx_quality = {k: OrtHead(args.output / ('quality-' + k + '.onnx'), options) for k in ['TASK', 'HABIT']}
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    assert digest(binary) == config['nativeSha256']
    native = NativeFields(binary, args.output / 'parity-native')
    differences, max_confidence_error = [], 0.
    for start in range(0, len(rows), 80):
        batch = rows[start:start + 80]
        expected = choose(prepare(batch, predict(model, tokenizer, batch), native), rankers)
        actual = choose(prepare(batch, infer(runtime, tokenizer, batch), native), onnx_rank)
        pc = confidence(expected, quality_logits(qualities, expected), config['calibration'])
        oc = confidence(actual, quality_logits(onnx_quality, actual), config['calibration'])
        for row, a, b, x, y in zip(batch, expected, actual, pc, oc):
            max_confidence_error = max(max_confidence_error, abs((x or 0.) - (y or 0.)))
            a = apply_gate(a, x, config['thresholds'][a['proposal']['kind']]['threshold'])
            b = apply_gate(b, y, config['thresholds'][b['proposal']['kind']]['threshold'])
            if any(a[k] != b[k] for k in ['proposal', 'fields', 'hardReasons', 'localTrusted']):
                differences.append(row['id'])
    report = dict(examples=len(rows), numericalExamples=min(100, len(rows)), maxAbsoluteOutputErrors=errors,
                  completeProposalDifferences=differences, maxConfidenceError=max_confidence_error,
                  graphs={v.name: dict(sha256=digest(v), bytes=v.stat().st_size) for v in args.output.glob('*.onnx')},
                  prototypeSha256=digest(args.prototype), nativeSha256=digest(binary),
                  encoderHostMilliseconds={k: dict(samples=len(v[5:]), p50=float(np.percentile(v[5:], 50)),
                                                     p95=float(np.percentile(v[5:], 95)))
                                           for k, v in [('torch', torch_ms), ('onnx', ort_ms)]},
                  runtime='host ONNX Runtime CPU; not client codec, CoreML/MindSpore or phone verification')
    (args.output / 'parity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if differences:
        raise ValueError('ONNX proposal parity failed: ' + str(differences[:10]))
    print(json.dumps({k: v for k, v in report.items() if k != 'graphs'}), flush=True)


if __name__ == '__main__':
    main()
