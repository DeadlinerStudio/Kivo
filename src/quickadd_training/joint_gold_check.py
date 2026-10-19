"""Development-only gold-to-Core/provenance diagnostic; never neural inference accuracy."""
import argparse
import json
from pathlib import Path

from .data import load, digest
from .joint_data import ROLES
from .joint_decode import draft
from .joint_native import NativeFields, attach


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostics')
    args.output.mkdir(parents=True)
    binary = (args.core / 'shared/build/bin/macosArm64/debugTest/test.kexe').resolve()
    native = NativeFields(binary, args.output / 'native')
    report = dict(scope='gold development diagnostic, not model predictions; no final data opened',
                  nativeSha256=digest(binary), sets={})
    for split in ['train', 'dev', 'regression']:
        rows = [r for r in load(args.data / (split + '.jsonl')) if r['expected']['decision'] == 'accept']
        drafts = []
        for row in rows:
            kind = row['expected']['proposal']['kind']
            raw = dict(intentProbabilities=[float(kind == 'TASK'), float(kind == 'HABIT'), 0.])
            spans = [dict(role=s['label'], start=s['start'], end=s['end'], score=1.) for s in row['spans'] if s['label'] != 'CONNECTOR']
            drafts.append(draft(row, raw, spans, kind))
        results = [attach(d, n) for d, n in zip(drafts, native.normalize(rows, drafts))]
        failures = []
        for row, actual in zip(rows, results):
            if not actual['draftValid'] or any(actual['proposal'][k] != row['expected']['proposal'][k]
                                              for k in row['expected']['proposal']):
                failures.append(dict(id=row['id'], text=row['text'], expected=row['expected'],
                                     actual=actual['proposal'], reasons=actual['hardReasons']))
        report['sets'][split] = dict(positiveRows=len(rows), failures=failures)
        print(split, len(rows), 'gold-path failures', len(failures), flush=True)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if any(v['failures'] for v in report['sets'].values()):
        raise ValueError('Authored gold fields cannot pass actual Core/provenance; inspect diagnostic')


if __name__ == '__main__':
    main()
