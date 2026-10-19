"""Train-only perturbations and separately authored evaluation; never human gold."""
import argparse, copy, datetime as dt, json, re
from pathlib import Path
from .data import load, validate, digest
from .corpus import BASE

TRAIN_TITLES = {
 'zh': ['打电话','给爸爸打电话','打电话给医生','发邮件给老师','去超市买菜','给朋友回消息','检查快递','取快递','预约牙医','预约体检','办理银行卡','修改密码','上传文件','续费会员','交水费','交电费','接孩子','看牙医','整理行李','给车加油','清洗衣服','给猫喂食','更换电池','检查门窗','预约理发','下载资料','修复登录问题','给客户打电话','去医院取药','打印合同'],
 'en': ['make a phone call','call Dad','call the dentist','email my teacher','shop for groceries','reply to my friend','check the delivery','collect the parcel','book a dentist appointment','schedule a checkup','open a bank account','change my password','upload the files','renew my membership','pay the water bill','pay the electricity bill','pick up the kids','see the dentist','pack my luggage','refuel the car','wash clothes','feed the cat','replace the batteries','check the windows','book a haircut','download the slides','fix the login issue','call the client','collect medicine','print the contract']}
EVAL_TITLES = {
 'zh':['联系房东','问一下维修进度','给外婆报平安','把照片传给姐姐','把资料交给班主任','去菜市场买鱼','给小狗洗澡','取修好的电脑','备份聊天记录','检查烟雾报警器','收拾阳台','给新同事发邀请','预约汽车保养','带身份证去银行','把钥匙还给室友','参加线上培训','整理报税凭证','给花换盆','确认酒店订单','把药装进旅行包','联系售后换货','缴停车费','买生日蛋糕','去图书馆借书','叫醒弟弟','去楼下取外卖','把预算发给经理','做一遍消防演练','更新护照照片','给邻居送包裹'],
 'en':['contact the landlord','ask about the repair','check in with Grandma','send photos to my sister','give the notes to my tutor','buy fish at the market','bathe the puppy','collect my repaired laptop','back up the chat history','check the smoke alarm','tidy the balcony','invite the new colleague','book a car service','take my ID to the bank','return the keys to my roommate','attend the online training','sort the tax receipts','repot the flowers','confirm the hotel booking','pack the medicine','contact support for an exchange','pay for parking','buy a birthday cake','borrow a library book','wake my brother','collect the takeaway','send the budget to my manager','practice the fire drill','update my passport photo','deliver a parcel to my neighbor']}

def insert(row, index, value):
 r=copy.deepcopy(row);r['text']=r['text'][:index]+value+r['text'][index:]
 for s in r['spans']:
  if s['start']>=index:s['start']+=len(value);s['end']+=len(value)
  elif s['end']>index:s['end']+=len(value)
 return r

def perturb(row, locale):
 variants=[]
 # Insertion inside TIME only preserves titles, meaning, and source families.
 for s in row.get('spans',[]):
  if s['label']!='TIME':continue
  text=row['text'];points=[]
  if locale=='zh':
   for i in range(s['start']+1,s['end']):
    if text[i-1].isdigit()!=text[i].isdigit() and (text[i-1].isdigit() or text[i].isdigit()) and not text[i-1].isspace() and not text[i].isspace():points.append(i)
  else:
   points=[m.start()+s['start'] for m in re.finditer(r'(?<=\d)(?=[ap]m)',text[s['start']:s['end']],re.I)]
  if points:
   r=row
   for i in reversed(points):r=insert(r,i,' ')
   variants.append(r)
 # Unicode spaces inside existing clock fields; English token boundaries remain intact.
 for value in ['\u3000','\t']:
  r=copy.deepcopy(row)
  for s in r.get('spans',[]):
   if s['label']=='TIME':
    r['text']=r['text'][:s['start']]+r['text'][s['start']:s['end']].replace(' ',value)+r['text'][s['end']:]
  if r['text']!=row['text']:variants.append(r)
 return variants

def authored(locale, split, titles):
 rows=[];today=dt.date(2026,10,10)
 days=[('明天',1),('后天',2),('大后天',3)] if locale=='zh' else [('tomorrow',1),('the day after tomorrow',2)]
 clocks=[('早上8点','08:00'),('下午3点','15:00'),('晚上8点','20:00'),('晚上 8 点','20:00'),('下午 3 点半','15:30'),('上午九点','09:00')] if locale=='zh' else [('at 8am','08:00'),('at 3pm','15:00'),('at 8 pm','20:00'),('at 3:30pm','15:30'),('at nine a.m.','09:00')]
 for ti,title in enumerate(titles):
  for d,(day,offset) in enumerate(days):
   for ci,(clock,hm) in enumerate(clocks):
    time=day+(' ' if locale=='en' else '')+clock
    layouts=[[(time,'TIME'),(' ' if locale=='en' else '',None),(title,'TITLE')],[(title,'TITLE'),('，' if locale=='zh' else ', ',None),(time,'TIME')]]
    if split=='train':layouts += [[('提醒我' if locale=='zh' else 'remind me to ',None),(title,'TITLE'),(' ' if locale=='en' else '',None),(time,'TIME')]]
    for li,parts in enumerate(layouts):
     text='';spans=[]
     for token,label in parts:
      start=len(text);text+=token
      if label:spans.append(dict(start=start,end=len(text),label=label))
     r=dict(id=f'{locale}-g3-{split}-{len(rows)}',locale=locale,split=split,group=f'{locale}-g3-{split}-title-{ti}',text=text,preferredKind='TASK',now='2026-10-10T12:17:00Z',timezone='Asia/Shanghai' if locale=='zh' else 'UTC',provenance='assistant-authored-generalization-not-human-reviewed',annotationStatus='synthetic',spans=spans,expected=dict(decision='accept',proposal=dict(BASE,title=title,dueAt=(today+dt.timedelta(days=offset)).isoformat()+'T'+hm)))
     rows.append(r)
  for neg in ([f'不要提醒我{title}',f'如果有空再{title}',f'过几天{title}',f'后天晚上{title}，具体几点还没定'] if locale=='zh' else [f"don't remind me to {title}",f'if I have time I will {title}',f'{title} sometime next week',f'{title} tomorrow evening, time undecided']):
   rows.append(dict(id=f'{locale}-g3-{split}-{len(rows)}',locale=locale,split=split,group=f'{locale}-g3-{split}-title-{ti}',text=neg,preferredKind='TASK',now='2026-10-10T12:17:00Z',timezone='Asia/Shanghai' if locale=='zh' else 'UTC',provenance='assistant-authored-generalization-not-human-reviewed',annotationStatus='synthetic',spans=[],trainingIntent='OTHER',expected=dict(decision='defer')))
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('data'));args=ap.parse_args()
 for locale in ['zh','en']:
  original={s:load(args.root/locale/(s+'.jsonl')) for s in ['train','dev']}
  rows=copy.deepcopy(original);rows['test']=authored(locale,'test',EVAL_TITLES[locale])
  rows['train']+=authored(locale,'train',TRAIN_TITLES[locale])
  base=list(rows['train'])
  for row in base:
   for i,r in enumerate(perturb(row,locale)):
    r['id']=row['id']+f'-perturb-{i}';r['augmentationParent']=row['id'];r['provenance']+=';train-only-whitespace-perturbation';rows['train'].append(r)
  # A title can be part of existing dev: exclude full normalized overlaps; keep source group train.
  from unicodedata import normalize
  key=lambda t:normalize('NFKC',t).strip().casefold()
  reserved={key(r['text']) for r in rows['dev']+rows['test']}
  seen=set();rows['train']=[r for r in rows['train'] if key(r['text']) not in reserved and not(key(r['text']) in seen or seen.add(key(r['text'])))]
  validate(rows);out=args.root/(locale+'-g3');out.mkdir(exist_ok=True)
  for split,rs in rows.items():(out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
  evidence=dict(counts={s:len(rs) for s,rs in rows.items()},sha256={s:digest(out/(s+'.jsonl')) for s in rows},independentTitleFamilies=True,humanReviewed=False,realUserData=False,thresholdSelection='dev only; test never used for tuning')
  (out/'provenance.json').write_text(json.dumps(evidence,indent=2));print(locale,evidence['counts'])
if __name__=='__main__':main()
