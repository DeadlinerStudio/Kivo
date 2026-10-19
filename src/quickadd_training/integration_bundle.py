"""Package frozen Kivo ONNX assets for local client experiments; no publishing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


GRAPHS = {'encoder-spans.onnx', 'quality-TASK.onnx', 'quality-HABIT.onnx'}
SCHEMA = 'kivo-experimental-joint-preview-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(root, name, checksum):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Asset outside root: ' + name)
    if digest(path) != checksum:
        raise ValueError('Frozen asset differs: ' + name)
    return path


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'kivo-preview.json').read_text())
    if manifest['schema'] != SCHEMA or manifest['enabledByDefault'] or manifest['formalGatePassed']:
        raise ValueError('Only disabled experimental bundles are supported')
    assets = manifest['assets']
    if not GRAPHS.issubset(assets) or not any(n.startswith('tokenizer/') for n in assets):
        raise ValueError('Missing runtime assets')
    for name, value in assets.items():
        path = checked(directory, name, value['sha256'])
        if path.stat().st_size != value['bytes']:
            raise ValueError('Asset size differs: ' + name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != set(assets) | {'kivo-preview.json'}:
        raise ValueError('Unexpected or missing bundle files')
    return manifest


def build_bundle(prototype, onnx_dir, runtime, output):
    prototype, onnx_dir, runtime, output = map(Path, (prototype, onnx_dir, runtime, output))
    if output.exists():
        raise ValueError('Refusing to replace existing bundle')
    config = json.loads(prototype.read_text())
    parity_file = onnx_dir / 'parity.json'
    parity = json.loads(parity_file.read_text())
    if (parity['prototypeSha256'] != digest(prototype) or parity['nativeSha256'] != config['nativeSha256']
            or parity['completeProposalDifferences'] or parity['examples'] <= 0
            or set(parity['graphs']) != GRAPHS):
        raise ValueError('ONNX evidence does not match frozen prototype')
    model_dir = Path(config['modelDirectory'])
    checked(model_dir, 'joint.pt', config['modelSha256'])
    for name, checksum in config['headSha256'].items():
        checked(prototype.parent, name, checksum)
    for name, checksum in config['policySources'].items():
        checked(runtime / 'src/quickadd_training', name, checksum)
    checked(runtime, 'shared/build/bin/macosArm64/debugTest/test.kexe', config['nativeSha256'])
    files = {}
    for name, record in parity['graphs'].items():
        path = checked(onnx_dir, name, record['sha256'])
        if path.stat().st_size != record['bytes']:
            raise ValueError('ONNX size differs: ' + name)
        files[name] = path
    for name, checksum in config['modelAssetSha256'].items():
        files[name] = checked(model_dir, name, checksum)
    manifest = dict(
        schema=SCHEMA, brand='Kivo', model='Kivo Tiny', locale='zh', experimental=True,
        enabledByDefault=False, formalGatePassed=False, productionEligible=False,
        protocol='eight-role-spans-with-correctness-heads; incompatible with legacy BIO manifest',
        prototypeSha256=digest(prototype), nativeSha256=config['nativeSha256'],
        modelSha256=config['modelSha256'], headSha256=config['headSha256'],
        policySources=config['policySources'], calibration=config['calibration'], thresholds=config['thresholds'],
        parity=dict(sha256=digest(parity_file), examples=parity['examples'], completeProposalDifferences=0,
                    scope=parity['runtime']),
        roles=['TITLE', 'TIME', 'LOCATION', 'NOTE', 'CONNECTOR', 'DAYS', 'GOAL_WINDOW', 'COUNT'],
        intents=['TASK', 'HABIT', 'CONFIRM'], preferredKind={'TASK': 0, 'HABIT': 1},
        sequenceLength=96, hiddenSize=128, qualityFeatureSize=1530,
        offsetUnit='Unicode code points; start inclusive, end exclusive',
        integrationGuide='docs/kivo-integration-v1.md',
        assets={n: dict(sha256=digest(p), bytes=p.stat().st_size) for n, p in sorted(files.items())})
    # Validate everything before publishing even a local directory; failed copies leave no partial bundle.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.kivo-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'bundle'
        staged.mkdir()
        for name, path in files.items():
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (staged / 'kivo-preview.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        verify_bundle(staged)
        if output.exists():
            raise ValueError('Bundle destination appeared during packaging')
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototype', type=Path)
    parser.add_argument('--onnx-dir', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_bundle(args.verify)
    else:
        if not all((args.prototype, args.onnx_dir, args.runtime, args.output)):
            parser.error('Packaging needs --prototype, --onnx-dir, --runtime and --output')
        result = build_bundle(args.prototype, args.onnx_dir, args.runtime, args.output)
    print(json.dumps(dict(model=result['model'], assets=len(result['assets']),
                          bytes=sum(v['bytes'] for v in result['assets'].values()), formalGatePassed=False)))


if __name__ == '__main__':
    main()
