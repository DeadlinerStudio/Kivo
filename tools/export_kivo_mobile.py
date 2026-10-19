"""Export fixed-batch mobile graphs from the unchanged frozen Kivo checkpoint."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
import coremltools as ct
import onnxruntime as ort
import quickadd_training

from quickadd_training.integration_bundle import verify_bundle, digest, checked
from quickadd_training.joint_adapter import network
from quickadd_training.joint_heads import load_heads
from quickadd_training.joint_model import encode
from quickadd_training.data import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prototype', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite mobile exports')
    frozen = verify_bundle(args.bundle)
    if digest(args.prototype) != frozen['prototypeSha256']:
        raise ValueError('Frozen prototype differs')
    config = json.loads(args.prototype.read_text())
    for name, checksum in config['policySources'].items():
        checked(Path(quickadd_training.__file__).parent, name, checksum)
    directory = Path(config['modelDirectory'])
    checked(directory, 'joint.pt', config['modelSha256'])
    for n, h in config['headSha256'].items():
        checked(args.prototype.parent, n, h)
    torch.set_num_threads(3)
    model = network(AutoConfig.from_pretrained(args.bundle / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(directory / 'joint.pt', weights_only=True))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / 'tokenizer', local_files_only=True)
    row = dict(text='明天下午4点要在逸夫楼开会', preferredKind='TASK')
    data, _, _ = encode(tokenizer, [row])
    names = ['input_ids', 'attention_mask', 'preferred_kind']
    inputs = tuple(data[n].to(torch.int32) for n in names)
    quality = load_heads(args.prototype.parent / 'quality.pt')
    graphs = {'encoder': (model, inputs, names, ['intent_logits', 'span_logits', 'token_features'])}
    for kind in ['TASK', 'HABIT']:
        graphs[kind] = (quality[kind], (torch.zeros((1, 1530)),), ['features'], ['score'])
    args.output.mkdir(parents=True)
    results = {}
    for role, (module, example, input_names, output_names) in graphs.items():
        module.eval()
        stem = 'kivo-' + role
        onnx_file = args.output / (stem + '.onnx')
        if role == 'encoder':
            shutil.copyfile(args.bundle / 'encoder-spans.onnx', onnx_file)
        else:
            torch.onnx.export(module, example, onnx_file, input_names=input_names,
                              output_names=output_names, opset_version=13, dynamo=False)
        trace = torch.jit.trace(module, example, strict=False)
        ml = ct.convert(trace,
            inputs=[ct.TensorType(name=n, shape=x.shape, dtype=np.int32 if x.dtype == torch.int32 else np.float32)
                    for n, x in zip(input_names, example)],
            outputs=[ct.TensorType(name=n) for n in output_names], convert_to='neuralnetwork',
            minimum_deployment_target=ct.target.iOS14)
        ml.short_description = 'Kivo Tiny frozen development preview: ' + role
        path = args.output / (stem + '.mlmodel')
        ml.save(str(path))
        results[role] = dict(onnx=dict(sha256=digest(onnx_file), bytes=onnx_file.stat().st_size),
                             coreml=dict(sha256=digest(path), bytes=path.stat().st_size))
        print('Exported ' + role, flush=True)
    (args.output / 'export.json').write_text(json.dumps(dict(
        frozenPrototypeSha256=digest(args.prototype), frozenModelSha256=config['modelSha256'],
        frozenHeadsSha256=config['headSha256'], fixedBatch=1, graphs=results,
        scope='Mobile graph export; backend numerical and proposal parity still required'), indent=2) + '\n')


if __name__ == '__main__':
    main()
