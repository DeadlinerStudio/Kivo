"""Auxiliary MASSIVE intent/slot supervision; original labels, no invented proposals."""
import json
from pathlib import Path
from .data import digest, load

def public_rows(path, partition='train'):
 return [r for r in load(path) if r['sourcePartition']==partition and r['annotationAligned'] and 0<len(r['text'])<=128]

def warmstart(model, lookup, path, epochs, seed):
 import numpy as np, torch
 from torch import nn
 rows=public_rows(path)
 intents=sorted({r['sourceIntent'] for r in rows});labels=sorted({s['sourceLabel'] for r in rows for s in r['sourceSpans']})
 ids=np.zeros((len(rows),128),np.int32);mask=ids.copy();slots=np.full_like(ids,-100)
 for i,r in enumerate(rows):
  for j,c in enumerate(r['text']):ids[i,j]=lookup.get(c,1);mask[i,j]=1;slots[i,j]=0
  for s in r['sourceSpans']:
   base=1+2*labels.index(s['sourceLabel']);slots[i,s['start']]=base;slots[i,s['start']+1:s['end']]=base+1
 ids,mask,slots=[torch.from_numpy(x) for x in [ids,mask,slots]];target=torch.tensor([intents.index(r['sourceIntent']) for r in rows])
 original_intent,original_slots=model.intent,model.slots
 model.intent=nn.Linear(original_intent.in_features,len(intents));model.slots=nn.Conv2d(original_slots.in_channels,1+2*len(labels),1)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];model.train()
 for epoch in range(epochs):
  total=[]
  for batch in torch.randperm(len(rows)).split(128):
   il,sl=model(ids[batch],mask[batch]);loss=nn.functional.cross_entropy(il,target[batch])+nn.functional.cross_entropy(sl.reshape(-1,1+2*len(labels)),slots[batch].reshape(-1).long(),ignore_index=-100)
   opt.zero_grad();loss.backward();opt.step();total.append(loss.item())
  history.append(sum(total)/len(total));print(f'Public auxiliary epoch {epoch+1}: loss={history[-1]:.4f}',flush=True)
 model.intent,model.slots=original_intent,original_slots
 return dict(source='MASSIVE-1.1',license='CC-BY-4.0',sourceSha256=digest(path),partition='train only',examples=len(rows),intentLabels=intents,slotLabels=labels,epochs=epochs,loss=history,businessLabelsInvented=False)
