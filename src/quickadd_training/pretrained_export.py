"""Pinned tiny BERT export; shared WordPiece contract, unchanged rejection thresholds."""
import argparse,json
from pathlib import Path
from .data import load,digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--teacher',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--locale',required=True,choices=['zh','en']);a=ap.parse_args()
 import torch,numpy as np,onnxruntime as ort
 from torch import nn
 from transformers import AutoModel,AutoTokenizer
 torch.set_num_threads(3)
 report=json.loads((a.teacher/'training-report.json').read_text())
 tokenizer=AutoTokenizer.from_pretrained(a.teacher/'tokenizer',trust_remote_code=False,use_fast=True)
 encoder=AutoModel.from_pretrained(report['pretrainedModel'],revision=report['revision'],trust_remote_code=False,attn_implementation='eager')
 class Model(nn.Module):
  def __init__(self):
   super().__init__();self.encoder=encoder;self.intent=nn.Linear(encoder.config.hidden_size,3);self.slots=nn.Linear(encoder.config.hidden_size,9)
  def forward(self,input_ids,attention_mask):
   hidden=self.encoder(input_ids=input_ids.long(),attention_mask=attention_mask.long()).last_hidden_state
   return self.intent(hidden[:,0]),self.slots(hidden)
 model=Model();model.load_state_dict(torch.load(a.teacher/'teacher.pt',weights_only=True));model.eval()
 a.output.mkdir(parents=True,exist_ok=True);stem='quick-add-'+a.locale+'-v3';tokenizer.save_pretrained(a.output/'tokenizer')
 sample=tokenizer('后天晚上 8 点打电话' if a.locale=='zh' else 'call the dentist tomorrow at 8 pm',padding='max_length',max_length=128,return_tensors='pt')
 inputs=tuple(sample[n].to(torch.int32) for n in ['input_ids','attention_mask'])
 torch.onnx.export(model,inputs,a.output/(stem+'.onnx'),input_names=['input_ids','attention_mask'],output_names=['intent_logits','slot_logits'],opset_version=13)
 runtime=ort.InferenceSession(str(a.output/(stem+'.onnx')),providers=['CPUExecutionProvider']);actual=runtime.run(None,{n:x.numpy() for n,x in zip(['input_ids','attention_mask'],inputs)})
 with torch.no_grad():expected=model(*inputs)
 errors=[float(np.max(np.abs(x-y.numpy()))) for x,y in zip(actual,expected)]
 if max(errors)>.001:raise ValueError('ONNX export differs from checkpoint')
 import coremltools as ct
 trace=torch.jit.trace(model,inputs,strict=False)
 ml=ct.convert(trace,inputs=[ct.TensorType(name=n,shape=x.shape,dtype=np.int32) for n,x in zip(['input_ids','attention_mask'],inputs)],outputs=[ct.TensorType(name=n) for n in ['intent_logits','slot_logits']],convert_to='neuralnetwork',minimum_deployment_target=ct.target.iOS14)
 ml.short_description='Deadliner quick add '+a.locale+' pretrained intent and slots';ml.save(str(a.output/(stem+'.mlmodel')))
 chars={c for row in load(a.data/'train.jsonl') for c in row['text']}
 wordpiece=tokenizer.get_vocab();vocab=[None]*len(wordpiece)
 for token,index in wordpiece.items():
  vocab[index]=token
  if not token.startswith('['):chars.update(token.replace('##',''))
 characters=['[PAD]','[UNK]']+sorted(chars)
 (a.output/'vocabulary.json').write_text(json.dumps(characters,ensure_ascii=False))
 import unicodedata
 overrides={}
 for code in range(0x10000):
  c=chr(code);normalized=''.join(x for x in unicodedata.normalize('NFD',c.lower()) if unicodedata.category(x)!='Mn')
  if len(normalized)==1 and normalized!=c.lower():overrides[c]=normalized
 codec=dict(locale=a.locale,vocabulary=characters,maxLength=128,intentThreshold=.85,slotThreshold=.9,intentMargin=.25,maxUnknownRatio=.1,wordPiece=dict(vocabulary=vocab,lowercase=True,normalizationOverrides=overrides))
 artifacts={platform:dict(url='https://github.com/DeadlinerStudio/DeadlinerCore/releases/download/quick-add-v3.0.0-preview/'+stem+'.'+extension,sha256=digest(a.output/(stem+'.'+extension)),sizeBytes=(a.output/(stem+'.'+extension)).stat().st_size) for platform,extension in [('android','onnx'),('ios','mlmodel')]}
 manifest=dict(schemaVersion=1,modelId='quick-add-'+a.locale+'-v3.0.0',artifacts=artifacts,codec=codec)
 (a.output/'quick-add-model.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 report.update(modelStem=stem,deployment='exported pretrained candidate; frozen KMP evaluation and backend parity required',onnxCheckpointMaxErrors=errors,tokenizerContract='WordPiece IDs; unicode offsets and token logits decoded in shared KMP')
 (a.output/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(stem,errors,flush=True)
if __name__=='__main__':main()
