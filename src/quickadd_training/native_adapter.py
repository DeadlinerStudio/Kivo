"""Bound native JSON memory while preserving exact production decoder evaluation."""
import json, os, subprocess
from pathlib import Path

def evaluate(binary, manifest, rows, directory, batch_size=80):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);predictions={}
 for start in range(0,len(rows),batch_size):
  batch=rows[start:start+batch_size];path=directory/f'batch-{start:05}.json'
  payload=dict(rows=batch)
  if manifest is not None:payload['manifest']=manifest
  path.write_text(json.dumps(payload,ensure_ascii=False))
  result=subprocess.run([str(binary),'--ktest_filter=com.deadliner.domain.quickadd.model.TrainingBenchmarkAdapterTest.*'],env=dict(os.environ,QUICK_ADD_BENCHMARK_INPUT=str(path.resolve())),capture_output=True,text=True)
  if result.returncode:
   (directory/f'batch-{start:05}.log').write_text(result.stdout+result.stderr)
   raise RuntimeError(f'Native adapter failed with exit {result.returncode}, batch {start}; inspect {directory}')
  for line in result.stdout.splitlines():
   if line.startswith('QUICK_ADD_BENCHMARK '):
    row=json.loads(line.split(' ',1)[1]);predictions[row['id']]=row
 if set(predictions)!={r['id'] for r in rows}:raise ValueError('Missing or stale KMP results')
 return predictions
