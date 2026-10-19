"""Check frozen joint-data hashes, authored spans and cross-partition isolation."""
import json
from pathlib import Path

from quickadd_training.data import digest, load
from quickadd_training.joint_data import validate


def main():
    for directory in [Path('data/zh-joint-v2'), Path('data/zh-joint-v3')]:
        provenance = json.loads((directory / 'provenance.json').read_text())
        splits = {}
        for name, record in provenance['partitions'].items():
            path = directory / (name + '.jsonl')
            if digest(path) != record['sha256']:
                raise ValueError('Frozen corpus differs: ' + str(path))
            rows = load(path)
            if len(rows) != record['rows']:
                raise ValueError('Corpus row count differs: ' + str(path))
            splits[name] = rows
        validate(splits)
        print(str(directory) + ': frozen hashes and isolation passed')


if __name__ == '__main__':
    main()
