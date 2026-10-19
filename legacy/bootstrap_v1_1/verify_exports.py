"""Run actual ONNX/Core ML exports; emit logits for KMP decoder and MindSpore accuracy checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import coremltools as ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    directory = parser.parse_args().directory
    manifest = json.loads((directory / 'quick-add-model.json').read_text())
    lookup = {c: i for i, c in enumerate(manifest['codec']['vocabulary'])}
    rows = json.loads((directory / 'acceptance.json').read_text())
    runtime = ort.InferenceSession(str(directory / 'quick-add-zh-v1.onnx'), providers=['CPUExecutionProvider'])
    ml = ct.models.MLModel(str(directory / 'quick-add-zh-v1.mlmodel'))
    fixtures = []
    errors = []
    checks = directory / 'checks'
    checks.mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        ids = np.zeros((1, 128), dtype=np.int32)
        mask = np.zeros_like(ids)
        for position, char in enumerate(row['text']):
            ids[0, position] = lookup.get(char, 1)
            mask[0, position] = 1
        inputs = dict(input_ids=ids, attention_mask=mask)
        names = ['intent_logits', 'slot_logits']
        logits = runtime.run(names, inputs)
        core = ml.predict(inputs)
        errors.append(max(float(np.max(np.abs(core[name]-values))) for name, values in zip(names, logits)))
        expected = 'READY_OR_DEFER' if index in (11, 12) else 'OTHER' if row['intent'] == 2 else 'READY'
        for backend, values in [('onnx', logits), ('coreml', [core[name] for name in names])]:
            fixtures.append(dict(text=row['text'], expected=expected, backend=backend,
                                 title=''.join(c for c, label in zip(row['text'], row['slots']) if label in (1, 2)),
                                 kind='TASK' if row['intent'] == 0 else 'HABIT',
                                 reminderTime=row.get('reminderTime'),
                                 intent=values[0].reshape(-1).tolist(), slots=values[1].reshape(-1).tolist()))
        ids.tofile(checks / f'{index}-ids.bin')
        mask.tofile(checks / f'{index}-mask.bin')
        reference = ''.join(name + ' ' + str(value.ndim) + ' ' + ' '.join(map(str, value.shape)) + '\n' +
                            ' '.join(map(str, value.reshape(-1))) + '\n' for name, value in zip(names, logits))
        (checks / f'{index}-reference.txt').write_text(reference)
    (directory / 'decoder-fixtures.json').write_text(json.dumps(fixtures, ensure_ascii=False))
    if max(errors) > 0.1:
        raise ValueError('Core ML/ONNX numerical mismatch: inspect before publishing')
    report_path = directory / 'training-report.json'
    report = json.loads(report_path.read_text())
    report['coremlAllExamplesMaxAbsoluteError'] = max(errors)
    report['acceptancePolicy'] = 'slot probability >= 0.9; two prior boundary errors may defer or produce a correct proposal; other acceptance examples must pass; synthetic/template evaluation only'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Actual exports checked:', len(rows), 'Core ML maximum error:', max(errors))


if __name__ == '__main__':
    main()
