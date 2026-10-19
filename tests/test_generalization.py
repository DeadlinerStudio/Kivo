import copy, unittest, importlib.util
from quickadd_training.generalization_data import authored, perturb, TRAIN_TITLES, EVAL_TITLES
from quickadd_training.data import supervised, validate
from quickadd_training.public_pretrain import public_rows
from quickadd_training.pretrained_teacher import character_logits

class GeneralizationTest(unittest.TestCase):
 def test_title_and_whitespace_offsets_preserved(self):
  row=authored('zh','train',['给爸爸打电话'])[0]
  changed=perturb(row,'zh')[0]
  self.assertIn(' 8 ',changed['text']);self.assertEqual(supervised(changed)['intent'],0)
  title=[s for s in changed['spans'] if s['label']=='TITLE'][0]
  self.assertEqual(changed['text'][title['start']:title['end']],'给爸爸打电话')
  self.assertEqual(row['text'],'明天早上8点给爸爸打电话')
 def test_authoring_families_do_not_overlap(self):
  for language in ['zh','en']:
   self.assertFalse(set(TRAIN_TITLES[language])&set(EVAL_TITLES[language]))
   rows={s:authored(language,s,TRAIN_TITLES[language] if s=='train' else EVAL_TITLES[language]) for s in ['train','test']}
   validate(rows)
   for rs in rows.values():
    for r in rs:supervised(r)
 @unittest.skipUnless(importlib.util.find_spec('numpy'),'optional numpy dependency')
 def test_wordpiece_expands_beginning_into_inside(self):
  import numpy as np
  logits=np.full((1,9),-20.);logits[0,1]=10
  values=character_logits(logits,[(0,4)],'call')
  self.assertEqual(values[:4].argmax(-1).tolist(),[1,2,2,2])
 @unittest.skipUnless(importlib.util.find_spec('numpy'),'optional numpy dependency')
 def test_spaces_do_not_bridge_different_entities(self):
  import numpy as np
  logits=np.full((2,9),-20.);logits[0,1]=10;logits[1,3]=10
  values=character_logits(logits,[(0,4),(5,8)],'call 8pm')
  self.assertEqual(values[4].argmax(),0)
 def test_public_rows_never_use_official_dev(self):
  import tempfile,json
  from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'public.jsonl'
   rows=[dict(sourcePartition='train',annotationAligned=True,text='valid'),dict(sourcePartition='dev',annotationAligned=True,text='reserved'),dict(sourcePartition='train',annotationAligned=False,text='bad offsets')]
   path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
   self.assertEqual([r['text'] for r in public_rows(path)],['valid'])
 def test_native_batching_requires_all_row_ids(self):
  import tempfile,json
  from unittest.mock import patch
  from types import SimpleNamespace
  from quickadd_training.native_adapter import evaluate
  rows=[dict(id=str(i)) for i in range(5)]
  sizes=[]
  def run(command,**kwargs):
   payload=json.loads(__import__('pathlib').Path(kwargs['env']['QUICK_ADD_BENCHMARK_INPUT']).read_text())
   sizes.append(len(payload['rows']))
   return SimpleNamespace(returncode=0,stdout='\n'.join('QUICK_ADD_BENCHMARK '+json.dumps(r) for r in payload['rows']),stderr='')
  with tempfile.TemporaryDirectory() as tmp,patch('quickadd_training.native_adapter.subprocess.run',side_effect=run):
   result=evaluate('/fake',{},rows,tmp,batch_size=2)
  self.assertEqual(sizes,[2,2,1]);self.assertEqual(set(result),{'0','1','2','3','4'})
 @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch dependency')
 def test_inference_tokenization_needs_no_gold_spans(self):
  import torch
  from quickadd_training.pretrained_teacher import tokenize_rows
  class Tokenizer:
   def __call__(self,*args,**kwargs):
    return dict(input_ids=torch.tensor([[2,8,3]]),attention_mask=torch.tensor([[1,1,1]]),offset_mapping=torch.tensor([[[0,0],[0,4],[0,0]]]))
  encoded,offsets,labels=tokenize_rows(Tokenizer(),[dict(text='call')],with_labels=False)
  self.assertEqual(encoded['input_ids'].shape,(1,3));self.assertEqual(labels[0,1].item(),0)
if __name__=='__main__':unittest.main()
