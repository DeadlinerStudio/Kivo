"""Export exposed regression, quality-feature and Unicode tokenizer references."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from transformers import AutoConfig, AutoTokenizer
import quickadd_training
from quickadd_training.data import load
from quickadd_training.integration_bundle import checked, digest, verify_bundle
from quickadd_training.joint_adapter import network
from quickadd_training.joint_model import encode, predict
from quickadd_training.joint_heads import prepare, choose, load_heads, quality_logits
from quickadd_training.joint_native import NativeFields
from quickadd_training.joint_calibration import confidence
from quickadd_training.joint_decode import apply_gate
from quickadd_training.joint_evaluate import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototype', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--core-runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace fixtures')
    manifest = verify_bundle(args.bundle)
    if manifest['prototypeSha256'] != digest(args.prototype):
        raise ValueError('Bundle/prototype mismatch')
    config = json.loads(args.prototype.read_text())
    directory = Path(config['modelDirectory'])
    for name, checksum in config['policySources'].items():
        checked(Path(quickadd_training.__file__).parent, name, checksum)
    checked(directory, 'joint.pt', config['modelSha256'])
    for name, checksum in config['headSha256'].items():
        checked(args.prototype.parent, name, checksum)
    binary = checked(args.core_runtime, 'shared/build/bin/macosArm64/debugTest/test.kexe', config['nativeSha256'])
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / 'tokenizer', local_files_only=True)
    torch.set_num_threads(3)
    model = network(AutoConfig.from_pretrained(args.bundle / 'encoder-config', local_files_only=True))
    model.load_state_dict(torch.load(directory / 'joint.pt', weights_only=True))
    rows = load(Path(config['dataDirectory']) / 'regression.jsonl')
    with tempfile.TemporaryDirectory(prefix='kivo-golden-') as temporary:
        selected = choose(prepare(rows, predict(model, tokenizer, rows), NativeFields(binary, temporary)), {})
    logits = quality_logits(load_heads(args.prototype.parent / 'quality.pt'), selected)
    probabilities = confidence(selected, logits, config['calibration'])
    cases = []
    for row, value, score, c in zip(rows, selected, logits, probabilities):
        threshold = config['thresholds'][value['proposal']['kind']]['threshold']
        cases.append(dict(input={k: row[k] for k in ['id', 'text', 'preferredKind', 'now', 'timezone']},
                          result=plain(apply_gate(value, c, threshold)),
                          qualityFeature=value['qualityFeature'].tolist(), rawQualityLogit=float(score)))
    token_rows = rows + [dict(id='unicode-' + str(i), text=t, preferredKind='TASK') for i, t in enumerate([
        '明天16:00在逸夫楼👩‍💻开会', '明天开会𠮷', '明天  Café\u0301  开会', '明天\u3000下午4点\u00a0开会'])]
    inputs, offsets, _ = encode(tokenizer, token_rows)
    token_cases = []
    for i, row in enumerate(token_rows):
        mapping = [0]
        for character in row['text']:
            mapping.append(mapping[-1] + len(character.encode('utf-16-le')) // 2)
        token_cases.append(dict(id=row['id'], text=row['text'], preferredKind=row['preferredKind'],
                               codePointToUtf16=mapping, offsets=offsets[i].tolist(),
                               **{k: v[i].tolist() for k, v in inputs.items()}))
    result = dict(scope='Previously exposed development regressions; not independent accuracy evidence',
                  prototypeSha256=digest(args.prototype), nativeSha256=config['nativeSha256'],
                  tokenizerSha256=manifest['assets']['tokenizer/tokenizer.json']['sha256'],
                  offsetUnit=manifest['offsetUnit'], cases=cases, tokenizerCases=token_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), tokenizerCases=len(token_cases), sha256=digest(args.output))))


if __name__ == '__main__':
    main()
