"""Actual ONNX logits + production KMP proposals, fixed clock, no Python parser."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from .data import load, validate, digest
from .metrics import summarize, streaming


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--adapter-metadata',type=Path,help='Verified build-once native adapter fingerprint')
    args = parser.parse_args()
    splits = {s: load(args.data_dir / (s + '.jsonl')) for s in ('train', 'dev', 'test')}
    validate(splits)
    originals = splits[args.split]
    if not originals:
        raise SystemExit('Empty evaluation set; no benchmark or release claim can be made')
    rows = list(originals)
    if args.streaming:
        for row in originals:
            for length in range(1, len(row['text']) + 1):
                rows.append(dict(row, id=f"{row['id']}:prefix:{length}", text=row['text'][:length],
                                 parentId=row['id'], prefixLength=length, isFinal=length == len(row['text'])))
    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(rows=rows)
    model_hash = None
    if args.model_dir:
        import numpy as np
        import onnxruntime as ort
        manifest_path = args.model_dir / 'quick-add-model.json'
        manifest = json.loads(manifest_path.read_text())
        from urllib.parse import urlparse, unquote
        model_path = args.model_dir / Path(unquote(urlparse(manifest['artifacts']['android']['url']).path)).name
        if any(r.get('locale', manifest['codec'].get('locale', 'zh')) != manifest['codec'].get('locale', 'zh') for r in originals):
            raise ValueError('Corpus and model locale differ')
        model_hash = digest(model_path)
        if model_hash != manifest['artifacts']['android']['sha256']:
            raise ValueError('Model SHA differs from manifest')
        config = manifest['codec']; length = config['maxLength']
        vocabulary = {c: i for i, c in enumerate(config['vocabulary'])}
        tokenizer = None
        if config.get('wordPiece'):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir / 'tokenizer', trust_remote_code=False)
        runtime = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        # Warm-up excluded. Timings are desktop proxies, never phone performance claims.
        blank = np.zeros((1, length), dtype=np.int32)
        runtime.run(None, dict(input_ids=blank, attention_mask=blank))
        for row in rows:
            text = row['text']
            row['inferenceMs'] = 0.0
            if not text or len(text) > length or sum(c not in vocabulary for c in text) / len(text) > config['maxUnknownRatio']:
                continue
            ids = blank.copy(); mask = blank.copy()
            if tokenizer:
                encoded = tokenizer(text, padding='max_length', max_length=length, truncation=False, return_tensors='np')
                if encoded['input_ids'].shape[1] != length: continue
                ids = encoded['input_ids'].astype(np.int32); mask = encoded['attention_mask'].astype(np.int32)
                row['inputIds'] = ids.reshape(-1).tolist(); row['attentionMask'] = mask.reshape(-1).tolist()
            else:
                for i, c in enumerate(text):
                    ids[0, i] = vocabulary.get(c, 1); mask[0, i] = 1
            started = time.perf_counter()
            intent, slots = runtime.run(['intent_logits', 'slot_logits'], dict(input_ids=ids, attention_mask=mask))
            row['inferenceMs'] = (time.perf_counter() - started) * 1000
            row['intent'] = intent.reshape(-1).tolist(); row['slots'] = slots.reshape(-1).tolist()
        payload['manifest'] = manifest
    input_path = (args.output / 'kmp-input.json').resolve()
    input_path.write_text(json.dumps(payload, ensure_ascii=False))
    env = dict(os.environ, QUICK_ADD_BENCHMARK_INPUT=str(input_path))
    predictions = {}
    adapter = None
    if args.adapter_metadata:
        from .prepare_adapter import source_digest
        adapter = json.loads(args.adapter_metadata.read_text())
        if Path(adapter['core']).resolve()!=args.core.resolve() or adapter['sourceSha256']!=source_digest(args.core) or digest(adapter['binary'])!=adapter['binarySha256']:
            raise ValueError('Stale or mismatched native adapter; run prepare_adapter again')
        command=[adapter['binary'],'--ktest_filter=com.deadliner.domain.quickadd.model.TrainingBenchmarkAdapterTest.*']
    else:
        command=[str(args.core.resolve() / 'gradlew'), ':shared:macosArm64Test', '--rerun',
            '--tests', 'com.deadliner.domain.quickadd.model.TrainingBenchmarkAdapterTest.*',
            '-x', ':shared:generateKmpBuildMetadata', '--offline', '--no-daemon', '--console=plain']
    if adapter:
        from .native_adapter import evaluate
        predictions=evaluate(adapter['binary'],payload.get('manifest'),rows,args.output/'native-batches')
        (args.output / 'kmp-build.log').write_text('Verified native adapter, evaluated in bounded batches.\n')
        outputs=[]
    else:
        result = subprocess.run(command,cwd=args.core,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (args.output / 'kmp-build.log').write_text(result.stdout)
        if result.returncode:
            raise SystemExit('KMP adapter failed: inspect ' + str(args.output / 'kmp-build.log'))
        outputs=[out.text or '' for xml in (args.core / 'shared/build/test-results/macosArm64Test').glob('*.xml') for out in ET.parse(xml).iter('system-out')]
    for output in outputs:
        for line in output.splitlines():
            if line.startswith('QUICK_ADD_BENCHMARK '):
                row = json.loads(line.removeprefix('QUICK_ADD_BENCHMARK ')); predictions[row['id']] = row
    if set(predictions) != {r['id'] for r in rows}:
        raise ValueError('Missing or stale KMP results')
    (args.output / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    report = dict(adapterBinarySha256=adapter['binarySha256'] if adapter else None,split=args.split, datasetSha256=digest(args.data_dir / (args.split + '.jsonl')),
                  modelSha256=model_hash, manifestSha256=digest(args.model_dir / 'quick-add-model.json') if args.model_dir else None,
                  runtime='desktop ONNX CPU + macOS KMP, warm-up excluded, single-run samples',
                  coreHead=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.core, text=True).strip(),
                  coreQuickAddSourceSha256=__import__('hashlib').sha256(b''.join(
                      str(p.relative_to(args.core)).encode()+p.read_bytes()
                      for p in sorted((args.core / 'shared/src').rglob('*.kt'))
                      if 'quickadd' in str(p))).hexdigest(),
                  evaluatorSha256=__import__('hashlib').sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
                  provenance=sorted({r['provenance'] for r in originals}),
                  modes={m: summarize(originals, predictions, m) for m in ('rules', 'model', 'hybrid')},
                  streaming={m: streaming(rows, predictions, m) for m in ('rules', 'model', 'hybrid')})
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({m: {k: v for k,v in metrics.items() if k not in ('failures','sources')} for m,metrics in report['modes'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
