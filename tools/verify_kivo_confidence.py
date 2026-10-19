"""Finish backend parity with the features actually constructed by the Core port."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import coremltools as ct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--checks', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    parity = args.directory / 'parity'
    frozen = json.loads(args.checks.read_text())
    ms_core = json.loads((parity / 'mindspore-core-results.json').read_text())
    if args.prepare:
        for kind in ['TASK', 'HABIT']:
            folder = parity / ('quality-after-core-' + kind)
            folder.mkdir(exist_ok=False)
            for i, case in enumerate(ms_core):
                np.asarray([case['qualityFeature']], dtype=np.float32).tofile(folder / f'input-{i}-features.bin')
        print('Prepared actual MindSpore-encoder/Core-feature inputs')
        return
    heads = {kind: ct.models.MLModel(str(args.directory / f'kivo-{kind}.mlmodel'), compute_units=ct.ComputeUnit.CPU_ONLY)
             for kind in ['TASK', 'HABIT']}
    report = {}
    for backend in ['coreml', 'mindspore']:
        values = json.loads((parity / f'{backend}-core-results.json').read_text())
        maximum = 0.
        for i, (actual, case) in enumerate(zip(values, frozen['cases'], strict=True)):
            kind = actual['proposal']['kind']
            if backend == 'coreml':
                score = float(np.asarray(heads[kind].predict(dict(features=np.asarray([actual['qualityFeature']], dtype=np.float32)))['score']).reshape(-1)[0])
            else:
                score = float(np.fromfile(parity / f'quality-after-core-{kind}/mindspore-{i}-score.bin', dtype=np.float32).item())
            cal = frozen['calibration'][kind]
            c = 1 / (1 + math.exp(-max(-700, min(700, cal['slope'] * score + cal['bias']))))
            expected = case['result']
            maximum = max(maximum, abs(c - expected['confidence']))
            trusted = actual['proposal']['ready'] and c >= frozen['thresholds'][kind]['threshold']
            if (not trusted) != expected['proposal']['needsAI']:
                raise ValueError('Backend changes AI gate: ' + case['input']['id'])
            for key in ['kind', 'title', 'dueAt', 'note', 'period', 'targetCount', 'weekdaysMask', 'reminderTime', 'ready']:
                if actual['proposal'][key] != expected['proposal'][key]:
                    raise ValueError('Backend changes business field: ' + key)
        if maximum > 1e-5:
            raise ValueError('Confidence tolerance exceeded: ' + str(maximum))
        report[backend] = dict(completeProposalCases=len(values), differences=0, maxConfidenceError=maximum,
                               pipeline='actual backend encoder -> actual shared Core decoder/features -> actual backend quality head')
    report.update(formalGatePassed=False, phoneVerified=False, prototypeSha256=frozen['prototypeSha256'])
    (parity / 'complete-proposal-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
