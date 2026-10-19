"""Create the explicit developer schema-2 manifest after real backend/Core checks."""
import argparse
import json
from pathlib import Path

from quickadd_training.integration_bundle import checked, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--prototype', type=Path, required=True)
    p.add_argument('--tokenizer-config', type=Path, required=True)
    p.add_argument('--release-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite client manifest')
    config = json.loads(args.prototype.read_text())
    exported = json.loads((args.directory / 'export.json').read_text())
    if exported['frozenPrototypeSha256'] != digest(args.prototype) or exported['frozenModelSha256'] != config['modelSha256']:
        raise ValueError('Export/prototype mismatch')
    if exported['frozenHeadsSha256'] != config['headSha256']:
        raise ValueError('Frozen scoring heads differ')
    parity_file = args.directory / 'parity/complete-proposal-report.json'
    parity = json.loads(parity_file.read_text())
    if parity['prototypeSha256'] != digest(args.prototype):
        raise ValueError('Core parity/prototype mismatch')
    for backend in ['coreml', 'mindspore']:
        if parity[backend]['differences'] or parity[backend]['completeProposalCases'] != 14:
            raise ValueError('Complete proposal parity incomplete')
    platforms = {}
    for platform, extension in [('android', 'onnx'), ('ios', 'mlmodel'), ('ohos', 'ms')]:
        files = {}
        for role in ['encoder', 'TASK', 'HABIT']:
            name = f'kivo-{role}.{extension}'
            path = args.directory / name
            frozen_graph = json.loads((args.directory / 'parity/three-format-report.json').read_text())['graphs'][role]['assets'][extension]
            checked(args.directory, name, frozen_graph['sha256'])
            if path.stat().st_size != frozen_graph['bytes']:
                raise ValueError('Graph byte size differs')
            files[role] = dict(url=args.release_url.rstrip('/') + '/' + name, sha256=digest(path), sizeBytes=path.stat().st_size)
        platforms[platform] = files
    manifest = dict(schemaVersion=2, modelId='kivo-tiny-zh-v0.1.0-preview.1', protocol='kivo-joint-v1',
                    experimental=True, productionEligible=False, platforms=platforms,
                    tokenizer=json.loads(args.tokenizer_config.read_text()),
                    calibration={k: {n: v[n] for n in ['slope', 'bias']} for k, v in config['calibration'].items()},
                    thresholds={k: v['threshold'] for k, v in config['thresholds'].items()})
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(bytes=args.output.stat().st_size, sha256=digest(args.output))))


if __name__ == '__main__':
    main()
