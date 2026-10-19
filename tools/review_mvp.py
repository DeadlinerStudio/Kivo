"""Read-only v3 audit: authored probes, production KMP, oracle spans, threshold replay.

Run with the existing training environment and PYTHONPATH=src. No training,
downloads, production source changes, or threshold changes are performed.
Probe results diagnose capabilities; they are not an independent accuracy estimate.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from quickadd_training.corpus import BASE
from quickadd_training.data import digest, load, supervised
from quickadd_training.metrics import accepted, correct, summarize
from quickadd_training.native_adapter import evaluate
from quickadd_training.prepare_adapter import source_digest
from quickadd_training.pretrained_teacher import character_logits

LABELS = ['O', 'B-TITLE', 'I-TITLE', 'B-TIME', 'I-TIME',
          'B-REPEAT', 'I-REPEAT', 'B-NOTE', 'I-NOTE']


def probes(locale):
    rows = []

    def add(category, parts, **fields):
        text, spans = '', []
        preferred = fields.pop('preferredKind', 'TASK')
        decision = fields.pop('decision', 'accept')
        for token, label in parts:
            start = len(text)
            text += token
            if label:
                spans.append(dict(start=start, end=len(text), label=label))
        title = next((text[s['start']:s['end']] for s in spans if s['label'] == 'TITLE'), '')
        note = next((text[s['start']:s['end']] for s in spans if s['label'] == 'NOTE'), '')
        rows.append(dict(id=f'{locale}-review-{len(rows):03}', locale=locale,
                         split='dev', group=f'{locale}-review-{category}', category=category,
                         text=text, preferredKind=preferred,
                         now='2026-10-19T04:00:00Z', timezone='Asia/Shanghai',
                         provenance='review-authored-diagnostic-not-human-gold',
                         annotationStatus='synthetic', spans=spans,
                         expected=dict(decision=decision, proposal=dict(BASE, title=title, note=note, **fields))
                         if decision == 'accept' else dict(decision='defer')))

    if locale == 'zh':
        for title in ['打电话', '交报告书', '买牛奶', '了解社区义工岗位', '报名参加摄影课程', '和王总确认预算']:
            add('familiar-time-new-title', [('明天下午3点', 'TIME'), (title, 'TITLE')], dueAt='2026-10-20T15:00')
            add('plain-title', [(title, 'TITLE')])
        for clock, hm in [('早上6点', '06:00'), ('上午10点', '10:00'), ('上午十一点', '11:00'),
                          ('下午2点', '14:00'), ('下午五点', '17:00'), ('晚上9点半', '21:30'),
                          ('13:45', '13:45'), ('下午三点一刻', '15:15')]:
            add('clock-generalization', [('明天' + clock, 'TIME'), ('打电话', 'TITLE')], dueAt='2026-10-20T' + hm)
        for time, due in [('下周一下午3点', '2026-10-26T15:00'), ('10月25日下午3点', '2026-10-25T15:00'),
                          ('2026年11月1日下午3点', '2026-11-01T15:00'), ('2026-11-01 15:00', '2026-11-01T15:00'),
                          ('三天后下午3点', '2026-10-22T15:00'), ('半小时后', '2026-10-19T12:30'),
                          ('明天下午', '2026-10-20T23:59')]:
            # Vague afternoon has no unique deadline and is a rejection control.
            if time == '明天下午':
                add('rejection-controls', [(time + '打电话', None)], decision='defer')
            else:
                add('date-generalization', [(time, 'TIME'), ('打电话', 'TITLE')], dueAt=due)
        for repeat, fields in [('每天', {}), ('每周一三五', dict(weekdaysMask=21)),
                               ('每个工作日', dict(weekdaysMask=31)), ('每周三次', dict(period='WEEKLY', targetCount=3)),
                               ('每月一次', dict(period='MONTHLY'))]:
            add('habit', [(repeat, 'REPEAT'), ('下午3点', 'TIME'), ('喝水', 'TITLE')],
                kind='HABIT', reminderTime='15:00', **fields)
        add('habit-selected-plain-title', [('喝水', 'TITLE')], kind='HABIT', preferredKind='HABIT')
        add('habit-selected-plain-title', [('每天', 'REPEAT'), ('喝水', 'TITLE')], kind='HABIT', preferredKind='HABIT')
        for note in ['带身份证', '提前十分钟出门', '会议室在三楼']:
            add('note', [('明天下午3点', 'TIME'), ('参加会议', 'TITLE'), ('，备注：', None), (note, 'NOTE')],
                dueAt='2026-10-20T15:00')
        for text in ['不要明天下午3点打电话', '取消明天下午3点打电话的提醒', '如果有空明天下午3点打电话',
                     '明天下午3点打电话然后买牛奶', '明天25点打电话', '2月30日下午3点打电话',
                     '明天下午3点打电话吗', '我昨天已经打过电话了', '明天下午3点打电话或者发邮件']:
            add('rejection-controls', [(text, None)], decision='defer')
        for title in ['联系张鑫', '给龚老师发邮件', '取快递📦', '给Zoë打电话', '购买Type-C转接头']:
            add('vocabulary-and-symbols', [('明天下午3点', 'TIME'), (title, 'TITLE')], dueAt='2026-10-20T15:00')
        for time in ['明天下午 3 点', '明天下午  3  点', '明天下午\t3\t点']:
            add('whitespace', [(time, 'TIME'), ('打电话', 'TITLE')], dueAt='2026-10-20T15:00')
        for gap in ['提醒我', '请提醒我', '麻烦提醒我', '能不能帮我', '记得要']:
            add('request-phrasing', [(gap, None), ('明天下午3点', 'TIME'), ('打电话', 'TITLE')], dueAt='2026-10-20T15:00')
        add('title-time-words', [('明天下午3点', 'TIME'), ('整理明天的演讲稿', 'TITLE')], dueAt='2026-10-20T15:00')
    else:
        for title in ['call Mom', 'submit the report', 'buy milk', 'learn about local volunteer roles',
                      'mail a birthday gift to my cousin', 'confirm the budget with Alex']:
            add('familiar-time-new-title', [('tomorrow at 3pm', 'TIME'), (' ', None), (title, 'TITLE')], dueAt='2026-10-20T15:00')
            add('plain-title', [(title, 'TITLE')])
        for clock, hm in [('at 6am', '06:00'), ('at 10am', '10:00'), ('at eleven a.m.', '11:00'),
                          ('at 2pm', '14:00'), ('at 5pm', '17:00'), ('at 9:30pm', '21:30'),
                          ('at 13:45', '13:45'), ('at quarter past three pm', '15:15')]:
            add('clock-generalization', [('tomorrow ' + clock, 'TIME'), (' ', None), ('call Mom', 'TITLE')], dueAt='2026-10-20T' + hm)
        for time, due in [('next Monday at 3pm', '2026-10-26T15:00'), ('October 25 at 3pm', '2026-10-25T15:00'),
                          ('November 1, 2026 at 3pm', '2026-11-01T15:00'), ('2026-11-01 at 15:00', '2026-11-01T15:00'),
                          ('in three days at 3pm', '2026-10-22T15:00'), ('in 30 minutes', '2026-10-19T12:30')]:
            add('date-generalization', [(time, 'TIME'), (' ', None), ('call Mom', 'TITLE')], dueAt=due)
        for repeat, fields in [('every day', {}), ('every Monday and Wednesday', dict(weekdaysMask=5)),
                               ('every weekday', dict(weekdaysMask=31)), ('three times a week', dict(period='WEEKLY', targetCount=3)),
                               ('once a month', dict(period='MONTHLY'))]:
            add('habit', [(repeat, 'REPEAT'), (' ', None), ('at 3pm', 'TIME'), (' ', None), ('drink water', 'TITLE')],
                kind='HABIT', reminderTime='15:00', **fields)
        add('habit-selected-plain-title', [('drink water', 'TITLE')], kind='HABIT', preferredKind='HABIT')
        add('habit-selected-plain-title', [('every day', 'REPEAT'), (' ', None), ('drink water', 'TITLE')], kind='HABIT', preferredKind='HABIT')
        for note in ['bring my ID', 'leave ten minutes early', 'the meeting room is on floor three']:
            add('note', [('tomorrow at 3pm', 'TIME'), (' ', None), ('attend the meeting', 'TITLE'),
                         (', ', None), (note, 'NOTE')], dueAt='2026-10-20T15:00')
        for text in ["don't call Mom tomorrow at 3pm", 'cancel the reminder to call Mom tomorrow at 3pm',
                     'if I have time call Mom tomorrow at 3pm', 'call Mom tomorrow at 3pm then buy milk',
                     'call Mom tomorrow at 25:00', 'call Mom on February 30 at 3pm',
                     'did I call Mom tomorrow at 3pm?', 'I already called Mom yesterday',
                     'call Mom or email her tomorrow at 3pm', 'call Mom tomorrow afternoon, time undecided']:
            add('rejection-controls', [(text, None)], decision='defer')
        for title in ['call Zoë', 'collect the parcel 📦', 'email support@example.com', 'buy a USB-C adapter', '给妈妈打电话']:
            add('vocabulary-and-symbols', [('tomorrow at 3pm', 'TIME'), (' ', None), (title, 'TITLE')], dueAt='2026-10-20T15:00')
        for time in ['tomorrow at 3 pm', 'tomorrow at  3  pm', 'tomorrow at\t3\tpm']:
            add('whitespace', [(time, 'TIME'), (' ', None), ('call Mom', 'TITLE')], dueAt='2026-10-20T15:00')
        for gap in ['remind me to ', 'please ', 'can you remind me to ', 'I need to ', "don't forget to "]:
            add('request-phrasing', [(gap, None), ('call Mom', 'TITLE'), (' ', None), ('tomorrow at 3pm', 'TIME')], dueAt='2026-10-20T15:00')
        add('title-time-words', [('tomorrow at 3pm', 'TIME'), (' ', None), ("edit tomorrow's presentation", 'TITLE')], dueAt='2026-10-20T15:00')
    return rows


def probabilities(logits):
    p = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def diagnose(row, tokenizer, config):
    """Diagnostic only: production KMP is authoritative for acceptance and correctness."""
    if 'intent' not in row:
        return dict(stage=row.get('encodingIssue', 'encoding'))
    ip = probabilities(np.asarray(row['intent']))
    order = np.argsort(ip)[::-1]
    result = dict(intent=['TASK', 'HABIT', 'OTHER'][order[0]], intentConfidence=float(ip[order[0]]))
    if ip[order[0]] < config['intentThreshold'] or ip[order[0]] - ip[order[1]] < config['intentMargin']:
        return dict(result, stage='intent-confidence')
    if order[0] == 2:
        return dict(result, stage='intent-OTHER')
    offsets = tokenizer(row['text'], padding='max_length', max_length=128, return_offsets_mapping=True)['offset_mapping']
    slots = character_logits(np.asarray(row['slots']).reshape(128, 9), offsets, row['text'])[:len(row['text'])]
    probs = probabilities(slots)
    labels, confidence = probs.argmax(-1), probs.max(-1)
    weakest = int(confidence.argmin())
    result.update(minSlotConfidence=float(confidence[weakest]), weakCharacter=row['text'][weakest],
                  weakLabel=LABELS[labels[weakest]])
    spans, previous, malformed = [[] for _ in range(4)], -1, False
    for char, label in zip(row['text'], labels):
        if label == 0:
            previous = -1
            continue
        group = (int(label) - 1) // 2
        if label % 2:
            spans[group].append(char)
        elif previous != group or not spans[group]:
            malformed = True
        else:
            spans[group][-1] += char
        previous = group
    result.update(rawSpans=dict(zip(['TITLE', 'TIME', 'REPEAT', 'NOTE'], spans)), malformedBIO=malformed)
    result['stage'] = ('slot-confidence' if (confidence < config['slotThreshold']).any()
                       else 'malformed-BIO' if malformed else 'downstream-contract-or-normalization')
    return result


def category_metrics(rows, predictions):
    categories = defaultdict(list)
    for row in rows:
        categories[row.get('category', 'frozen')].append(row)
    return {k: {kk: vv for kk, vv in summarize(v, predictions, 'model').items()
                if kk not in ['failures', 'sources']} for k, v in categories.items()}


def corpus_stats(rows):
    fields = {}
    for label in ['TITLE', 'TIME', 'REPEAT', 'NOTE']:
        values = [r['text'][s['start']:s['end']] for r in rows for s in r.get('spans', []) if s['label'] == label]
        fields[label] = dict(count=len(values), unique=len(set(values)))
    return dict(rows=len(rows), annotationStatus=dict(Counter(r.get('annotationStatus') for r in rows)),
                preferredKind=dict(Counter(r['preferredKind'] for r in rows)), fields=fields)


def run(root, core, output):
    binary = core / 'shared/build/bin/macosArm64/debugTest/test.kexe'
    output.mkdir(parents=True, exist_ok=True)
    audit = dict(scope='Review-authored diagnostics; no independent real-user accuracy claim',
                 adapterBinarySha256=digest(binary), coreQuickAddSourceSha256=source_digest(core),
                 adapterNote='Existing executable, not rebuilt by this audit; token IDs checked by adapter; frozen proposals replayed against archived results',
                 scriptSha256=digest(Path(__file__)), locales={})
    for locale in ['zh', 'en']:
        model_dir = root / 'artifacts' / (locale + '-v3')
        manifest = json.loads((model_dir / 'quick-add-model.json').read_text())
        model_path = model_dir / f'quick-add-{locale}-v3.onnx'
        if digest(model_path) != manifest['artifacts']['android']['sha256']:
            raise ValueError('Model hash differs from manifest')
        tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        runtime = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        loc_out = output / locale
        loc_out.mkdir(exist_ok=True)
        rows = probes(locale)
        probe_path = loc_out / 'probes.jsonl'
        probe_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        vocab = set(manifest['codec']['vocabulary'])
        for row in rows:
            text = row['text']
            if len(text) > 128 or sum(c not in vocab for c in text) / len(text) > manifest['codec']['maxUnknownRatio']:
                row['encodingIssue'] = 'character-vocabulary-or-length'
                continue
            encoded = tokenizer(text, padding='max_length', max_length=128, truncation=False, return_tensors='np')
            if encoded['input_ids'].shape[1] != 128:
                row['encodingIssue'] = 'wordpiece-context'
                continue
            ids = encoded['input_ids'].astype(np.int32)
            mask = encoded['attention_mask'].astype(np.int32)
            il, sl = runtime.run(None, dict(input_ids=ids, attention_mask=mask))
            row.update(inputIds=ids.reshape(-1).tolist(), attentionMask=mask.reshape(-1).tolist(),
                       intent=il.reshape(-1).tolist(), slots=sl.reshape(-1).tolist())
        sets = {'probes': rows}
        frozen_dir = root / 'reports' / (locale + '-v3-final-frozen')
        sets['frozen'] = json.loads((frozen_dir / 'kmp-input.json').read_text())['rows']
        archive = json.loads((frozen_dir / 'predictions.json').read_text())
        result = dict(modelSha256=digest(model_path), manifestSha256=digest(model_dir / 'quick-add-model.json'),
                      probesSha256=digest(probe_path), corpus=corpus_stats(load(root / 'data' / (locale + '-g6-frozen') / 'train.jsonl')),
                      sets={})
        for name, dataset in sets.items():
            predictions = evaluate(binary, manifest, dataset, loc_out / (name + '-native'))
            summary = dict(metrics={k: v for k, v in summarize(dataset, predictions, 'model').items() if k not in ['failures', 'sources']},
                           categories=category_metrics(dataset, predictions))
            if name == 'frozen':
                summary['archivedProposalDifferences'] = sum(
                    predictions[r['id']][mode] != archive[r['id']][mode]
                    for r in dataset for mode in ['rules', 'model', 'hybrid'])
            failures, causes = [], Counter()
            for row in dataset:
                p = predictions[row['id']]['model']
                if row['expected']['decision'] == 'accept' and not correct(row['expected'], p) or row['expected']['decision'] == 'defer' and accepted(p):
                    detail = diagnose(row, tokenizer, manifest['codec'])
                    stage = 'wrong-accepted' if accepted(p) else detail['stage']
                    causes[stage] += 1
                    failures.append(dict(id=row['id'], text=row['text'], expected=row['expected'], actual=p,
                                         diagnosis=detail))
            summary.update(failureStages=dict(causes), failures=failures)
            summary['slotThresholdReplay'] = {}
            for threshold in [.8, .7, .5]:
                changed = deepcopy(manifest)
                changed['codec']['slotThreshold'] = threshold
                replay = evaluate(binary, changed, dataset, loc_out / f'{name}-slot-{threshold}')
                summary['slotThresholdReplay'][str(threshold)] = {
                    k: v for k, v in summarize(dataset, replay, 'model').items() if k not in ['failures', 'sources']}
            # Perfect character spans bypass the learned heads and WordPiece projection.
            # This isolates decoder/normalizer capability, not deployable performance.
            oracle_manifest = deepcopy(manifest)
            oracle_manifest['codec'].pop('wordPiece', None)
            oracle_rows = []
            for row in dataset:
                if row['expected']['decision'] != 'accept':
                    continue
                oracle = {k: v for k, v in row.items() if k not in ['inputIds', 'attentionMask', 'intent', 'slots']}
                labels = supervised(row)
                il = [-20.] * 3
                il[labels['intent']] = 20.
                sl = np.full((128, 9), -20.)
                sl[:, 0] = 20.
                for i, label in enumerate(labels['slots']):
                    sl[i] = -20.
                    sl[i, label] = 20.
                oracle.update(intent=il, slots=sl.reshape(-1).tolist())
                oracle_rows.append(oracle)
            oracle_predictions = evaluate(binary, oracle_manifest, oracle_rows, loc_out / (name + '-oracle'))
            summary['oracleMetrics'] = {k: v for k, v in summarize(oracle_rows, oracle_predictions, 'model').items() if k not in ['failures', 'sources']}
            summary['oracleFailures'] = [dict(id=r['id'], text=r['text'], expected=r['expected'], actual=oracle_predictions[r['id']]['model'])
                                         for r in oracle_rows if not correct(r['expected'], oracle_predictions[r['id']]['model'])]
            result['sets'][name] = summary
            (loc_out / (name + '-predictions.json')).write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
            print(locale, name, summary['metrics'], summary['failureStages'], 'oracle', summary['oracleMetrics']['correct'], flush=True)
        audit['locales'][locale] = result
    (output / 'review.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, default=Path('../DeadlinerCore'))
    parser.add_argument('--output', type=Path, default=Path('reports/mvp-review-2026-10-19'))
    args = parser.parse_args()
    run(Path.cwd(), args.core.resolve(), args.output.resolve())
