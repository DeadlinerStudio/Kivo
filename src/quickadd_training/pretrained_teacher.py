"""Pinned pretrained BERT training. Deployment requires separate export and production KMP validation."""
import argparse, json, os, subprocess, time
from pathlib import Path
from .data import load, supervised, digest

MODELS={'zh':'uer/roberta-tiny-word-chinese-cluecorpussmall','en':'prajjwal1/bert-tiny'}

def tokenize_rows(tokenizer, rows, with_labels=True):
 import torch
 encoded=tokenizer([r['text'] for r in rows],padding='max_length',max_length=128,truncation=True,return_offsets_mapping=True,return_tensors='pt')
 offsets=encoded.pop('offset_mapping');labels=torch.full(encoded['input_ids'].shape,-100)
 for i,r in enumerate(rows):
  chars=supervised(r)['slots'] if with_labels else [0]*len(r['text'])
  for j,(a,b) in enumerate(offsets[i].tolist()):
   if b>a:
    covered=chars[a:b];groups={0 if c==0 else (c+1)//2 for c in covered}
    if len(groups)==1:labels[i,j]=covered[0]
 return encoded, offsets, labels

def character_logits(token_logits, offsets, text):
 import numpy as np
 out=np.full((128,9),-30.,np.float32);out[:,0]=0
 for j,(a,b) in enumerate(offsets):
  if b<=a:continue
  out[a]=token_logits[j]
  for index in range(a+1,min(b,len(text))):
   # One token's B entity becomes I after its first Unicode character.
   out[index]=token_logits[j]
   for start in [1,3,5,7]:out[index,start+1]=np.logaddexp(token_logits[j,start],token_logits[j,start+1]);out[index,start]=-30
 # Spaces not represented by wordpiece tokens keep the adjacent same entity continuous.
 for i,c in enumerate(text):
  if c.isspace() and 0<i<len(text)-1:
   left=int(out[i-1].argmax());right=int(out[i+1].argmax())
   if left and right and (left-1)//2==(right-1)//2:
    out[i]=out[i-1].copy()
    for start in [1,3,5,7]:out[i,start+1]=np.logaddexp(out[i,start],out[i,start+1]);out[i,start]=-30
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--locale',choices=['zh','en'],required=True);ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model-id');ap.add_argument('--warm-start',type=Path);ap.add_argument('--epochs',type=int,default=4);ap.add_argument('--core',type=Path,required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--evaluate-only',action='store_true');ap.add_argument('--extra-evaluation',type=Path);ap.add_argument('--evaluation-splits',default='dev,test,regression');args=ap.parse_args()
 import torch,numpy as np
 from torch import nn
 from transformers import AutoTokenizer,AutoModel
 from huggingface_hub import model_info
 torch.set_num_threads(3);torch.manual_seed(args.seed);np.random.seed(args.seed)
 model_id=args.model_id or MODELS[args.locale];revision=json.loads((args.output/'training-report.json').read_text())['revision'] if args.evaluate_only else model_info(model_id).sha
 tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False,use_fast=True)
 encoder=AutoModel.from_pretrained(model_id,revision=revision,trust_remote_code=False,attn_implementation='eager')
 class Teacher(nn.Module):
  def __init__(self):
   super().__init__();self.encoder=encoder;self.intent=nn.Linear(encoder.config.hidden_size,3);self.slots=nn.Linear(encoder.config.hidden_size,9)
  def forward(self,inputs):
   hidden=self.encoder(**inputs).last_hidden_state
   return self.intent(hidden[:,0]),self.slots(hidden)
 model=Teacher();
 if args.warm_start:model.load_state_dict(torch.load(args.warm_start,weights_only=True))
 rows=load(args.data_dir/'train.jsonl');inputs,offsets,slots=tokenize_rows(tokenizer,rows);intents=torch.tensor([supervised(r)['intent'] for r in rows]);opt=torch.optim.AdamW(model.parameters(),lr=5e-5)
 history=[]
 for epoch in range(0 if args.evaluate_only else args.epochs):
  losses=[];model.train()
  for batch in torch.randperm(len(rows)).split(32):
   il,sl=model({k:v[batch] for k,v in inputs.items()});loss=nn.functional.cross_entropy(il,intents[batch])+2*nn.functional.cross_entropy(sl.reshape(-1,9),slots[batch].reshape(-1),ignore_index=-100)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
  history.append(sum(losses)/len(losses));print(f'{args.locale} pretrained epoch {epoch+1}: {history[-1]:.4f}',flush=True)
 args.output.mkdir(parents=True,exist_ok=True);weights=args.output/'teacher.pt';model.load_state_dict(torch.load(weights,weights_only=True)) if args.evaluate_only else torch.save(model.state_dict(),weights);tokenizer.save_pretrained(args.output/'tokenizer');model.eval()
 characters={c for r in rows for c in r['text']}
 for token in tokenizer.get_vocab():
  if not token.startswith('['):characters.update(token.replace('##','').replace('▁',''))
 vocab=['[PAD]','[UNK]']+sorted(characters)
 manifest=dict(schemaVersion=1,modelId='offline-pretrained-'+args.locale,artifacts={'android':dict(url='https://example.invalid/teacher.onnx',sha256=digest(weights),sizeBytes=weights.stat().st_size)},codec=dict(locale=args.locale,vocabulary=vocab,maxLength=128,intentThreshold=.85,slotThreshold=.9,intentMargin=.25,maxUnknownRatio=.1))
 report=dict(pretrainedModel=model_id,revision=revision,modelUrl='https://huggingface.co/'+model_id,epochs=args.epochs,seed=args.seed,loss=history,trainSha256=digest(args.data_dir/'train.jsonl'),weightsSha256=digest(weights),parameters=sum(p.numel() for p in model.parameters()),warmStartSha256=digest(args.warm_start) if args.warm_start else None,deployment='offline PyTorch teacher only; not exported or shipped',thresholds='same fixed v2 thresholds; no test calibration')
 if args.evaluate_only:report=json.loads((args.output/'training-report.json').read_text())
 report['vocabularySource']='business train plus pretrained tokenizer; no dev/test characters';report['pretrainedCard']=model_info(model_id).card_data.to_dict() if model_info(model_id).card_data else {}
 (args.output/'training-report.json').write_text(json.dumps(report,indent=2))
 evaluations=[('dev',args.data_dir/'dev.jsonl'),('test',args.data_dir/'test.jsonl')]
 if args.extra_evaluation:evaluations.append(('regression',args.extra_evaluation))
 for split,evaluation_path in evaluations:
  if split not in args.evaluation_splits.split(','):continue
  evaluation=load(evaluation_path);payload=[]
  for r in evaluation:
   encoded,off,_=tokenize_rows(tokenizer,[r],with_labels=False);started=time.perf_counter()
   with torch.no_grad():il,sl=model(encoded)
   r=dict(r,intent=il[0].numpy().tolist(),slots=character_logits(sl[0].numpy(),off[0].tolist(),r['text']).reshape(-1).tolist(),inferenceMs=(time.perf_counter()-started)*1000);payload.append(r)
  path=args.output/(split+'-kmp-input.json');path.write_text(json.dumps(dict(manifest=manifest,rows=payload),ensure_ascii=False))
  binary=args.core/'shared/build/bin/macosArm64/debugTest/test.kexe'
  from .native_adapter import evaluate
  predictions=evaluate(binary,manifest,payload,args.output/(split+'-batches'))
  from .metrics import summarize
  metrics={m:summarize(evaluation,predictions,m) for m in ['rules','model','hybrid']}
  (args.output/(split+'-predictions.json')).write_text(json.dumps(predictions,ensure_ascii=False,indent=2))
  (args.output/(split+'-report.json')).write_text(json.dumps(dict(modes=metrics,datasetSha256=digest(evaluation_path),teacher=report,adapterBinarySha256=digest(binary),humanReviewed=False),ensure_ascii=False,indent=2))
  print(split,{m:{k:v for k,v in vals.items() if k not in ['failures','sources']} for m,vals in metrics.items()},flush=True)
if __name__=='__main__':main()
