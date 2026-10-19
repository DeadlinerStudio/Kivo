"""Bounded candidate combinations, provenance checks and independent ready/needsAI."""
import itertools
import math
import numpy as np

from .corpus import BASE
from .joint_data import ROLES, KINDS
from .normalizer_fidelity import legal_gap, COSMETIC

FIELDS = ('TITLE', 'TIME', 'LOCATION', 'NOTE', 'DAYS', 'GOAL_WINDOW', 'COUNT')
FEATURE_SIZE = 128 + (128 + 64) * len(FIELDS) + 3 + 2 + 1 + 5 * len(FIELDS) + 2


def overlaps(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def candidates(row, raw, beam_size=32):
    kind = KINDS[int(np.argmax(raw['intentProbabilities']))]
    if kind == 'CONFIRM':
        return [draft(row, raw, [], row['preferredKind'], ['request-needs-confirmation'])]
    allowed = FIELDS if kind == 'HABIT' else FIELDS[:4]
    beam = [(0., [])]
    for role in allowed:
        pool = sorted([p for p in raw['candidates'] if p['role'] == role], key=lambda p: p['score'], reverse=True)[:4]
        options = [[]] if role != 'TITLE' else []
        options += [[p] for p in pool]
        if role == 'TIME':
            options += [sorted(pair, key=lambda p: p['start']) for pair in itertools.combinations(pool, 2)
                        if not overlaps(*pair)]
        next_beam = []
        for score, selected in beam:
            for option in options:
                if any(overlaps(a, b) for a in option for b in selected):
                    continue
                next_beam.append((score + sum(p['score'] for p in option), selected + option))
        beam = sorted(next_beam, key=lambda value: value[0], reverse=True)[:beam_size]
        if not beam:
            break
    return [draft(row, raw, selected, kind) for _, selected in beam] or [draft(row, raw, [], kind)]


def coverage(text, selected, kind):
    owners = [None] * len(text)
    for span in selected:
        a, b = span['start'], span['end']
        if not 0 <= a < b <= len(text) or any(owners[j] is not None for j in range(a, b)):
            return ['invalid-overlapping-source'], []
        for j in range(a, b):
            owners[j] = span['role']
    gaps, i = [], 0
    while i < len(text):
        if owners[i] is not None:
            i += 1
            continue
        a = i
        while i < len(text) and owners[i] is None:
            i += 1
        left, right = owners[a - 1] if a else None, owners[i] if i < len(text) else None
        raw = text[a:i]
        clean = ''.join(c for c in raw if c not in COSMETIC and c not in '。.!！、')
        permitted = not clean or legal_gap(raw, left, right)[0]
        if kind == 'HABIT':
            if left is None and right in ['TITLE', 'TIME', 'DAYS', 'GOAL_WINDOW']:
                permitted |= clean in ['创建习惯', '添加习惯', '请创建习惯', '请提醒我', '记得', '提醒我']
            if right == 'LOCATION':
                permitted |= clean in ['在', '地点', '地点在', '要在']
            if right == 'TIME':
                permitted |= clean in ['提醒', '提醒时间', '时间', '在', '每天提醒']
            if right == 'NOTE':
                permitted |= clean in ['备注', '备注是']
            if left == 'TIME' and right is None:
                permitted |= clean == '提醒'
        gaps.append(dict(start=a, end=i, text=raw, permitted=bool(permitted), leftRole=left, rightRole=right))
    return (['source-content-deleted'] if any(not g['permitted'] for g in gaps) else []), gaps


def features(row, raw, selected, kind):
    parts = [np.asarray(raw.get('vector', np.zeros(128)), dtype=np.float32)]
    meta = list(raw['intentProbabilities']) + [float(kind == k) for k in KINDS[:2]] + [float(row['preferredKind'] == 'HABIT')]
    for role in FIELDS:
        spans = [s for s in selected if s['role'] == role]
        vectors = [s.get('vector', np.zeros(128)) for s in spans]
        parts.append(np.mean(vectors, axis=0) if vectors else np.zeros(128, dtype=np.float32))
        parts.append(np.mean([s.get('edge', np.zeros(64)) for s in spans], axis=0) if spans else np.zeros(64, dtype=np.float32))
        meta += [float(bool(spans)), max([s['score'] for s in spans], default=0.) / 10,
                 sum(s['end'] - s['start'] for s in spans) / 96,
                 min([s['start'] for s in spans], default=0) / 96,
                 max([s['end'] for s in spans], default=0) / 96]
    meta += [sum(s['end'] - s['start'] for s in selected) / max(1, len(row['text'])),
             sum(s['score'] for s in selected) / 50]
    vector = np.concatenate(parts + [np.asarray(meta, dtype=np.float32)]).astype(np.float32)
    assert len(vector) == FEATURE_SIZE
    return vector


def draft(row, raw, selected, kind, reasons=None):
    selected = sorted(selected, key=lambda s: s['start'])
    fields = {role: [row['text'][s['start']:s['end']] for s in selected if s['role'] == role] for role in ROLES}
    hard, gaps = coverage(row['text'], selected, kind)
    hard += reasons or []
    if len(fields['TITLE']) != 1:
        hard.append('event-not-established')
    if kind == 'TASK' and not fields['TIME']:
        hard.append('task-time-missing')
    if kind == 'TASK' and row['preferredKind'] == 'HABIT':
        hard.append('explicit-habit-entry-conflict')
    if kind == 'HABIT' and not fields['DAYS'] and not fields['GOAL_WINDOW'] and row['preferredKind'] != 'HABIT':
        hard.append('habit-schedule-not-established')
    note = ('地点：' + fields['LOCATION'][0]) if fields['LOCATION'] else ''
    note += ('；' if note and fields['NOTE'] else '') + (fields['NOTE'][0] if fields['NOTE'] else '')
    title = fields['TITLE'][0] if fields['TITLE'] else row['text'].strip()
    proposal = dict(BASE, kind=kind, title=title, note=note, ready=False, needsAI=True, source='JOINT_PROTOTYPE')
    return dict(proposal=proposal, fields=fields, rawText=row['text'], hardReasons=list(dict.fromkeys(hard)),
                gaps=gaps, spans=[{k: v for k, v in s.items() if k != 'vector'} for s in selected],
                feature=features(row, raw, selected, kind), candidateScore=sum(s['score'] for s in selected))


def quality_features(value, margin=0.):
    p = value['proposal']
    meta = [float(value.get('draftValid', False)), float(p['period'] == 'DAILY'), float(p['period'] == 'WEEKLY')]
    meta += [float(bool(p['weekdaysMask'] & (1 << i))) for i in range(7)]
    meta += [math.log1p(max(0, p['targetCount'])) / 10, float(p['dueAt'] is not None),
             float(p['reminderTime'] is not None), len(value['hardReasons']) / 10, margin / 10]
    return np.concatenate([value['feature'], np.asarray(meta, dtype=np.float32)]).astype(np.float32)


def apply_gate(value, confidence, threshold):
    result = dict(value, proposal=dict(value['proposal']), reasons=list(value['hardReasons']))
    valid = bool(value['draftValid'])
    finite = confidence is not None and math.isfinite(confidence) and 0 <= confidence <= 1
    trusted = valid and finite and threshold is not None and confidence >= threshold
    result['proposal'].update(ready=valid, needsAI=bool(value['rawText'].strip()) and not trusted)
    result.update(confidence=confidence if finite else None, threshold=threshold, localTrusted=bool(trusted))
    if not trusted and valid:
        result['reasons'].append('confidence-unavailable' if not finite else 'no-qualified-threshold' if threshold is None else 'confidence-below-threshold')
    result['state'] = 'TRUSTED' if trusted else 'READY_WITH_AI' if valid else 'PARTIAL'
    return result
