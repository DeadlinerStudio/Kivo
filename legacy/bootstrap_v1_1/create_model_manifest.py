#!/usr/bin/env python3
"""Build a release manifest from delivered single-file weights; never fabricate production weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--vocabulary', type=Path, required=True, help='JSON list starting with [PAD], [UNK]')
    parser.add_argument('--base-url', required=True, help='HTTPS directory hosting the supplied model files')
    parser.add_argument('--android', type=Path, help='Single-file ONNX weights')
    parser.add_argument('--ios', type=Path, help='Single-file Core ML .mlmodel weights')
    parser.add_argument('--ohos', type=Path, help='Single-file MindSpore .ms weights')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', args.model_id):
        parser.error('model-id must use 1..80 safe filename characters')
    url = urlparse(args.base_url)
    if url.scheme != 'https' or not url.netloc or url.query or url.fragment:
        parser.error('base-url must be an HTTPS directory without query or fragment')
    vocab = json.loads(args.vocabulary.read_text())
    if not isinstance(vocab, list) or len(vocab) < 3 or vocab[:2] != ['[PAD]', '[UNK]'] or not all(isinstance(s, str) for s in vocab):
        parser.error('invalid vocabulary')
    if len(vocab) != len(set(vocab)) or len(vocab) > 100000 or any(len(s) != 1 for s in vocab[2:]):
        parser.error('vocabulary must contain unique Unicode characters')
    if not 8 <= args.max_length <= 512:
        parser.error('max-length must be 8..512')
    artifacts = {}
    for platform, extension in [('android', '.onnx'), ('ios', '.mlmodel'), ('ohos', '.ms')]:
        path = getattr(args, platform)
        if path is None:
            continue
        if path.suffix != extension or not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024 * 1024:
            parser.error('invalid ' + platform + ' single-file weights')
        with path.open('rb') as file:
            sha = hashlib.file_digest(file, 'sha256').hexdigest()
        artifacts[platform] = dict(url=args.base_url.rstrip('/') + '/' + quote(path.name),
                                   sha256=sha, sizeBytes=path.stat().st_size)
    if not artifacts:
        parser.error('supply at least one platform weight file')
    manifest = dict(schemaVersion=1, modelId=args.model_id, artifacts=artifacts,
                    codec=dict(vocabulary=vocab, maxLength=args.max_length,
                               intentThreshold=0.85, slotThreshold=0.9, intentMargin=0.25, maxUnknownRatio=0.1))
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    if len(text.encode('utf-8')) > 8 * 1024 * 1024:
        parser.error('manifest exceeds 8 MiB')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print('Manifest:', args.output)
    print('Validate tensor names, shapes, logits and exported numerical parity before publishing.')


if __name__ == '__main__':
    main()
