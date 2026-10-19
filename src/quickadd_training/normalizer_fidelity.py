"""Audit copy provenance and position-scoped deletions; never repair missing content."""
import hashlib
import json

COSMETIC = frozenset(' \t\u3000\u00a0，,；;：:')
PREFIX = frozenset(['提醒我', '请提醒我', '麻烦提醒我', '记得', '帮我安排', '请', '麻烦',
                    '请帮我', '帮我', '我要', '打算', '准备'])
LOCATION_CUES = frozenset(['在', '要在', '准备在', '打算在', '需要在', '安排在', '定在',
                           '地点', '地点在', '地点定在', '备注地点'])
EVENT_CUES = frozenset(['要', '准备', '打算', '需要', '记得', '提醒我'])
TIME_CUES = frozenset(['时间', '时间是', '安排到', '安排在', '定在'])
NOTE_CUES = frozenset(['备注', '备注是', '备注为'])
FIELD_ORDER = ['TIME', 'TITLE', 'LOCATION', 'NOTE']


def legal_gap(raw, left, right):
    """A phrase is removable only at a declared edge; sentence tails keep all words."""
    text = ''.join(c for c in raw if c not in COSMETIC)
    if not text:
        return True, 'formatting'
    if right is None:
        return False, 'unexplained-trailing-content'
    if left is None:
        allowed = (right in ['TITLE', 'TIME'] and text in PREFIX)
        allowed |= right == 'TITLE' and text == '请把'
        allowed |= right == 'LOCATION' and text in ['在', '地点', '地点在', '地点定在']
        return bool(allowed), 'request-or-location-prefix' if allowed else 'unexplained-prefix'
    if right == 'LOCATION' and left in ['TITLE', 'TIME'] and text in LOCATION_CUES:
        return True, 'location-introducer'
    if right == 'TITLE' and left in ['TIME', 'LOCATION'] and text in EVENT_CUES:
        return True, 'event-auxiliary'
    if right == 'TIME' and left in ['TITLE', 'LOCATION'] and text in TIME_CUES:
        return True, 'time-introducer'
    if left == right == 'TIME' and text == '的':
        return True, 'date-clock-connector'
    if right == 'NOTE' and left in ['TITLE', 'TIME', 'LOCATION'] and text in NOTE_CUES:
        return True, 'note-introducer'
    if any(c.isdecimal() for c in text):
        return False, 'number-deleted'
    return False, 'unexplained-content'


def policy_sha256():
    policy = dict(cosmetic=sorted(COSMETIC), prefix=sorted(PREFIX), location=sorted(LOCATION_CUES),
                  event=sorted(EVENT_CUES), time=sorted(TIME_CUES), note=sorted(NOTE_CUES), version=1)
    return hashlib.sha256(json.dumps(policy, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audit_copy(text, actions, offsets, controls):
    """Use actual pointer positions, never substring matching or gold field labels."""
    fields, role, used, owners, errors = [], None, set(), [None] * len(text), []
    if not isinstance(actions, list) or not actions or any(type(a) is not int for a in actions):
        return dict(passed=False, errors=['invalid-copy-actions'], gaps=[], unexplainedText=text)
    if controls.get(actions[-1]) != 'EOS':
        errors.append('missing-copy-end')
    for action in actions[:-1]:
        if action in controls:
            name = controls[action]
            if name not in FIELD_ORDER or name in fields:
                errors.append('invalid-field-control')
                role = None
                continue
            fields.append(name)
            role = name
            continue
        if role is None or not 0 <= action < len(offsets):
            errors.append('copy-outside-field-or-context')
            continue
        a, b = offsets[action]
        if not 0 <= a < b <= len(text):
            errors.append('invalid-copy-offset')
            continue
        if action in used or any(owners[i] is not None for i in range(a, b)):
            errors.append('reused-source-content')
            continue
        used.add(action)
        for i in range(a, b):
            owners[i] = role
    if fields != sorted(fields, key=FIELD_ORDER.index) or 'TITLE' not in fields:
        errors.append('invalid-field-order-or-event')
    gaps = []
    index = 0
    while index < len(text):
        if owners[index] is not None:
            index += 1
            continue
        start = index
        while index < len(text) and owners[index] is None:
            index += 1
        left = owners[start - 1] if start else None
        right = owners[index] if index < len(text) else None
        allowed, reason = legal_gap(text[start:index], left, right)
        gaps.append(dict(start=start, end=index, text=text[start:index], leftRole=left,
                         rightRole=right, permitted=allowed, rule=reason))
    unexplained = ''.join(g['text'] for g in gaps if not g['permitted'])
    if unexplained:
        errors.append('source-content-deleted')
    return dict(passed=not errors, errors=list(dict.fromkeys(errors)), gaps=gaps,
                unexplainedText=unexplained, sourceCharacters=len(text), copiedCharacters=sum(v is not None for v in owners))


def guard_normalized(row, prediction, draft, tokenizer):
    """Pure post-validation; learned weights, output string and parsed fields stay unchanged."""
    result = dict(draft, proposal=dict(draft['proposal']), reasons=list(draft['reasons']))
    if prediction['normalized'] in [None, '需要确认']:
        result['fidelity'] = dict(applicable=False, passed=False, reason='already-needs-confirmation')
        return result
    from .normalizer_model import IDS, render
    offsets = tokenizer(row['text'], return_offsets_mapping=True, truncation=False)['offset_mapping']
    actions = prediction.get('actions')
    audit = audit_copy(row['text'], actions, offsets, {value: name for name, value in IDS.items()})
    if not audit['errors']:
        try:
            reconstructed = render(row['text'], actions, offsets)
        except (IndexError, KeyError, TypeError, ValueError):
            reconstructed = None
        if reconstructed != prediction['normalized']:
            audit['passed'] = False
            audit['errors'].append('copy-output-mismatch')
    audit['applicable'] = True
    result['fidelity'] = audit
    if not audit['passed']:
        result['reasons'] = list(dict.fromkeys(result['reasons'] + audit['errors']))
        result['proposal'].update(ready=False, needsAI=True)
        result['needsConfirmation'] = True
    return result
