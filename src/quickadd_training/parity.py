"""Export parity on arbitrary corpus; prepares inputs for official MindSpore verifier."""
import argparse
import json
from pathlib import Path
from .data import load, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--coreml',action='store_true')
    parser.add_argument('--mindspore',action='store_true',help='Read outputs previously produced by official native verifier')
    parser.add_argument('--adapter-metadata',type=Path,help='Production KMP executable fingerprint for complete proposal parity')
    parser.add_argument('--stem',help='Export stem; defaults to training-report modelStem or legacy filename')
    args=parser.parse_args()
    metadata=args.directory/'training-report.json'
    stem=args.stem or (json.loads(metadata.read_text()).get('modelStem','quick-add-zh-v1') if metadata.exists() else 'quick-add-zh-v1')
    import numpy as np
    import onnxruntime as ort
    vocabulary=json.loads((args.directory/'vocabulary.json').read_text())
    lookup={c:i for i,c in enumerate(vocabulary)}
    tokenizer=None
    manifest=json.loads((args.directory/'quick-add-model.json').read_text())
    if manifest['codec'].get('wordPiece'):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.directory/'tokenizer',trust_remote_code=False)
    rows=load(args.data)
    if not rows or any(len(r['text'])>128 for r in rows):
        raise ValueError('Parity corpus must contain examples <=128 Unicode characters')
    runtime=ort.InferenceSession(str(args.directory/(stem+'.onnx')),providers=['CPUExecutionProvider'])
    ml=None
    if args.coreml:
        import coremltools as ct
        ml=ct.models.MLModel(str(args.directory/(stem+'.mlmodel')),compute_units=ct.ComputeUnit.CPU_ONLY)
    checks=args.directory/'checks';checks.mkdir(exist_ok=True)
    errors=dict(coreml=[],mindspore=[]);agreement=dict(coreml=[],mindspore=[])
    fixtures=[];backend_fixtures=dict(onnx=[],coreml=[],mindspore=[]);names=['intent_logits','slot_logits']
    for index,row in enumerate(rows):
        ids=np.zeros((1,128),dtype=np.int32);mask=ids.copy()
        if tokenizer:
            encoded=tokenizer(row['text'],padding='max_length',max_length=128,truncation=False,return_tensors='np')
            ids=encoded['input_ids'].astype(np.int32);mask=encoded['attention_mask'].astype(np.int32)
            if ids.shape!=(1,128):raise ValueError('Text exceeds model token context')
        else:
            for i,c in enumerate(row['text']):ids[0,i]=lookup.get(c,1);mask[0,i]=1
        # Read mode never overwrites MindSpore inputs: detect stale data instead.
        for suffix,array in [('ids',ids),('mask',mask)]:
            path=checks/f'{index}-{suffix}.bin'
            if args.mindspore:
                if path.read_bytes()!=array.tobytes():raise ValueError('Stale MindSpore inputs')
            else:array.tofile(path)
        values=runtime.run(names,dict(input_ids=ids,attention_mask=mask))
        if [list(v.shape) for v in values]!=[[1,3],[1,128,9]] or not all(np.isfinite(v).all() for v in values):
            raise ValueError('Invalid ONNX output contract')
        backend_fixtures['onnx'].append(dict(row,intent=values[0].reshape(-1).tolist(),slots=values[1].reshape(-1).tolist()))
        fixtures.append(dict(id=row['id'],text=row['text'],intent=values[0].reshape(-1).tolist(),slots=values[1].reshape(-1).tolist()))
        backends={}
        if ml:
            prediction=ml.predict(dict(input_ids=ids,attention_mask=mask));backends['coreml']=[prediction[n] for n in names]
        if args.mindspore:
            backends['mindspore']=[np.fromfile(checks/f'{index}-ms-{n}.bin',dtype=np.float32).reshape(v.shape) for n,v in zip(names,values)]
        for backend,actual in backends.items():
            backend_fixtures[backend].append(dict(row,intent=actual[0].reshape(-1).tolist(),slots=actual[1].reshape(-1).tolist()))
            for output_index,(a,b) in enumerate(zip(actual,values)):
                if a.shape!=b.shape or not np.isfinite(a).all():raise ValueError('Invalid '+backend+' tensor')
                errors[backend].append(float(np.abs(a-b).max()))
                left,right=(a,b) if output_index==0 else (a[:,:int(mask.sum())],b[:,:int(mask.sum())])
                agreement[backend].append(bool(np.array_equal(left.argmax(-1),right.argmax(-1))))
    results={b:dict(maxAbsoluteError=max(e),argmaxAgreement=all(agreement[b])) for b,e in errors.items() if e}
    report=dict(dataSha256=digest(args.data),examples=len(rows),models={n:digest(args.directory/(stem+'.'+ext)) for n,ext in [('onnx','onnx'),('coreml','mlmodel'),('mindspore','ms')] if (args.directory/(stem+'.'+ext)).exists()},backends=results,threeFormatParityPassed=False)
    proposal_agreement={}
    if args.adapter_metadata:
        from .prepare_adapter import source_digest
        from .native_adapter import evaluate
        adapter=json.loads(args.adapter_metadata.read_text());core=Path(adapter['core'])
        if source_digest(core)!=adapter['sourceSha256'] or digest(adapter['binary'])!=adapter['binarySha256']:raise ValueError('Stale KMP parity adapter')
        expected=evaluate(adapter['binary'],manifest,backend_fixtures['onnx'],args.directory/'parity-kmp-onnx')
        for backend in results:
            actual=evaluate(adapter['binary'],manifest,backend_fixtures[backend],args.directory/('parity-kmp-'+backend))
            differences=[r['id'] for r in rows if any(expected[r['id']][mode]!=actual[r['id']][mode] for mode in ['model','hybrid'])]
            proposal_agreement[backend]=dict(completeProposalAgreement=not differences,differentIds=differences)
        report['productionKmp']=dict(adapterBinarySha256=adapter['binarySha256'],sourceSha256=adapter['sourceSha256'],backends=proposal_agreement)
    passed=all(v['maxAbsoluteError'] <= (.1 if b=='coreml' else .001) and v['argmaxAgreement'] for b,v in results.items())
    passed=passed and all(v['completeProposalAgreement'] for v in proposal_agreement.values())
    report['threeFormatParityPassed']=passed and set(results)=={'coreml','mindspore'} and set(proposal_agreement)=={'coreml','mindspore'}
    (args.directory/'parity-report.json').write_text(json.dumps(report,indent=2))
    (args.directory/'parity-onnx-fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False))
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit('Export parity failed')

if __name__=='__main__':main()
