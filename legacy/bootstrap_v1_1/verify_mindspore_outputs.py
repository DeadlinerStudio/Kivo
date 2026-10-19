"""Compare real MindSpore float tensors and append them to KMP production decoder fixtures."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
directory = parser.parse_args().directory
path = directory / 'decoder-fixtures.json'
fixtures = [row for row in json.loads(path.read_text()) if row['backend'] != 'mindspore']
references = [row for row in fixtures if row['backend'] == 'onnx']
errors = []
argmax_equal = []
for index, reference in enumerate(references):
    values = {}
    for field, name, count in [('intent', 'intent_logits', 3), ('slots', 'slot_logits', 128*9)]:
        output = np.fromfile(directory / 'checks' / f'{index}-ms-{name}.bin', dtype=np.float32)
        if len(output) != count or not np.isfinite(output).all():
            raise ValueError('Invalid MindSpore output ' + name)
        expected = np.array(reference[field], dtype=np.float32)
        errors.append(float(np.max(np.abs(output-expected))))
        classes = 3 if field == 'intent' else 9
        length = 1 if field == 'intent' else len(reference['text'])
        argmax_equal.append(bool(np.array_equal(output.reshape(-1, classes)[:length].argmax(-1),
                                              expected.reshape(-1, classes)[:length].argmax(-1))))
        values[field] = output.tolist()
    fixtures.append(dict(reference, backend='mindspore', **values))
if not all(argmax_equal):
    raise ValueError('MindSpore changed a text-token class: inspect outputs before publishing')
if max(errors) > 0.001:
    raise ValueError('MindSpore/ONNX numerical mismatch: check stale or incompatible weights')
path.write_text(json.dumps(fixtures, ensure_ascii=False))
report_path = directory / 'training-report.json'
report = json.loads(report_path.read_text())
report['mindsporeArm64MaxAbsoluteError'] = max(errors)
report['mindsporeArm64TextTokenArgmaxEqual'] = all(argmax_equal)
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('Real MindSpore outputs:', len(references), 'maximum absolute error:', max(errors), 'class agreement:', all(argmax_equal))
