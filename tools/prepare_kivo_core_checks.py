"""Prepare actual ONNX logits and frozen references for the Kotlin Native codec."""
import argparse
import json
from pathlib import Path
import unicodedata

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from quickadd_training.integration_bundle import digest, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite Core checks')
    manifest = verify_bundle(args.bundle)
    reference = json.loads(args.reference.read_text())
    if manifest['prototypeSha256'] != reference['prototypeSha256']:
        raise ValueError('Reference/prototype mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / 'tokenizer', local_files_only=True)
    vocabulary = [''] * len(tokenizer.get_vocab())
    for token, index in tokenizer.get_vocab().items():
        vocabulary[index] = token
    normalization = {}
    for number in range(0x110000):
        if 0xd800 <= number <= 0xdfff:
            continue
        char = chr(number)
        value = ''.join(v for v in unicodedata.normalize('NFD', char.lower()) if unicodedata.category(v) != 'Mn')
        if value != char:
            normalization[char] = value
    config = dict(vocabulary=vocabulary, normalization=normalization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / 'tokenizer-config.json').write_text(json.dumps(config, ensure_ascii=False))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    runtime = ort.InferenceSession(str(args.bundle / 'encoder-spans.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    token_cases = {r['id']: r for r in reference['tokenizerCases']}
    cases = []
    for case in reference['cases']:
        row = token_cases[case['input']['id']]
        values = dict(input_ids=np.asarray([row['input_ids']], dtype=np.int32),
                      attention_mask=np.asarray([row['attention_mask']], dtype=np.int32),
                      preferred_kind=np.asarray([row['preferred_kind']], dtype=np.int32))
        outputs = runtime.run(None, values)
        cases.append(dict(case, encoderOutputs=[dict(name=o.name, shape=list(v.shape), values=v.flatten().tolist())
                                                for o, v in zip(runtime.get_outputs(), outputs)]))
    result = dict(tokenizer=config, calibration=manifest['calibration'], thresholds=manifest['thresholds'],
                  tokenizerCases=reference['tokenizerCases'], cases=cases,
                  prototypeSha256=manifest['prototypeSha256'], referenceSha256=digest(args.reference))
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    print(json.dumps(dict(normalizationEntries=len(normalization), cases=len(cases), bytes=args.output.stat().st_size)))


if __name__ == '__main__':
    main()
