"""Import official CC BY 4.0 MASSIVE train/dev as review material, never silent gold labels."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import tarfile
from .data import digest

SELECTED={'calendar_set','calendar_query','calendar_remove','alarm_set','alarm_query','alarm_remove','lists_create','lists_query','lists_remove','weather_query','datetime_query','general_greet'}
SOURCE='https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz'


def spans(annotation):
    plain='';labels=[];position=0
    for match in re.finditer(r'\[([a-z_]+)\s*:\s*([^\[\]]+)\]',annotation):
        plain+=annotation[position:match.start()]
        value=match.group(2);start=len(plain);plain+=value
        labels.append(dict(start=start,end=len(plain),sourceLabel=match.group(1),text=value))
        position=match.end()
    return plain+annotation[position:],labels


def align(plain, labels, utterance):
    if plain == utterance:return labels, 'exact'
    left=[(i,c) for i,c in enumerate(plain) if not c.isspace()]
    right=[(i,c) for i,c in enumerate(utterance) if not c.isspace()]
    if [c for _,c in left] != [c for _,c in right]:return [], 'unresolved'
    mapping={i:j for (i,_),(j,_) in zip(left,right)}
    result=[]
    for label in labels:
        indexes=[mapping[i] for i in range(label['start'],label['end']) if i in mapping]
        if not indexes:return [], 'unresolved'
        start,end=min(indexes),max(indexes)+1
        result.append(dict(label,start=start,end=end,text=utterance[start:end]))
    return result, 'whitespace-only'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    counts={};intents={};slots_counts={};skipped={}
    with tarfile.open(args.archive,'r:gz') as archive:
        for member in archive:
            basename=Path(member.name).name
            if not member.isfile() or basename not in ('LICENSE','en-US.jsonl','zh-CN.jsonl'):continue
            stream=archive.extractfile(member)
            if member.size>50*1024*1024:raise ValueError('Unexpected source file size')
            if basename=='LICENSE':
                (args.output/'LICENSE-MASSIVE-CC-BY-4.0.txt').write_bytes(stream.read());continue
            locale=basename.removesuffix('.jsonl');selected=[];all_rows=[];intent_counts=Counter();slot_counts=Counter();skip_test=0
            for line in io.TextIOWrapper(stream,encoding='utf-8'):
                row=json.loads(line)
                if row['partition']=='test':skip_test+=1;continue
                utterance=row['utt'];reconstructed,annotation_spans=spans(row['annot_utt'])
                annotations,alignment=align(reconstructed,annotation_spans,utterance)
                record=dict(source='MASSIVE-1.1',sourceUrl=SOURCE,license='CC-BY-4.0',sourceId=str(row['id']),sourceLocale=locale,
                    parallelGroup='massive:'+str(row['id']),sourcePartition=row['partition'],text=utterance,
                    sourceIntent=row['intent'],sourceAnnotation=row['annot_utt'],sourceSpans=annotations,
                    annotationAligned=alignment!='unresolved',alignmentMethod=alignment,proposalMappingStatus='pending-independent-review',
                    usedInCurrentModels=False)
                # Do not carry worker identifiers; preserve utterance/labels, original split and parallel ID.
                all_rows.append(record);intent_counts[row['intent']]+=1;slot_counts.update(a['sourceLabel'] for a in annotations)
                if row['intent'] in SELECTED:selected.append(record)
            (args.output/(locale+'-train-dev.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows))
            (args.output/(locale+'-review-queue.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
            counts[locale]=dict(trainDev=len(all_rows),selectedForReview=len(selected),aligned=sum(r['annotationAligned'] for r in selected))
            intents[locale]=dict(intent_counts);slots_counts[locale]=dict(slot_counts);skipped[locale]=skip_test
    if set(counts)!={'en-US','zh-CN'} or not (args.output/'LICENSE-MASSIVE-CC-BY-4.0.txt').exists():raise ValueError('Incomplete official archive')
    report=dict(source=SOURCE,archiveSha256=digest(args.archive),license='CC-BY-4.0',counts=counts,officialTestRowsNotImported=skipped,
                intents=intents,slots=slots_counts,usedInCurrentModels=False,sourceOriginalAnnotations=True,deadlinerGoldReviewed=False)
    (args.output/'import-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.output/'NOTICE.md').write_text('''# Attribution

MASSIVE 1.1, Copyright Amazon.com Inc. or its affiliates, CC BY 4.0.
Source: https://github.com/alexa/massive and official dataset URL in import-report.json.
English utterances originated in SLURP, CC BY 4.0: https://github.com/pswietojanski/slurp.
Citations: FitzGerald et al., MASSIVE (2022), https://arxiv.org/abs/2204.08582;
Bastianelli et al., SLURP (2020), https://aclanthology.org/2020.emnlp-main.588/.

Modifications: selected zh-CN/en-US train/dev only; removed worker identifiers; converted original bracket slots to Unicode offsets with exact or validated whitespace-only alignment; selected review intents. No Deadliner proposal gold labels invented. Parallel IDs and original partitions preserved, official test utterances not imported. These samples have NOT trained the current synthetic bootstrap models.
''')
    print(json.dumps(dict(counts=counts,skippedOfficialTest=skipped,usedInCurrentModels=False),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
