"""Reference span selection. Dates are deliberately delegated to production KMP."""
from .corpus import BASE


def overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def select(text, prediction, threshold=0.):
    candidates = sorted([dict(s, text=text[s['start']:s['end']])
                         for s in prediction['candidates'] if s['score'] >= threshold],
                        key=lambda s: s['score'], reverse=True)
    selected, rejected = [], []
    limits = dict(TITLE=1, TIME=2, LOCATION=1, NOTE=1, CONNECTOR=12)
    for span in candidates:
        if not span['text'].strip():
            continue
        if any(overlap(span, old) for old in selected):
            rejected.append(span)
            continue
        if sum(old['role'] == span['role'] for old in selected) >= limits[span['role']]:
            rejected.append(span)
            continue
        selected.append(span)
    selected.sort(key=lambda s: s['start'])
    return selected, rejected


def decode(row, prediction, threshold=0., request_threshold=.8):
    """Return a lossless draft even when the supported request cannot be established."""
    text = row['text']
    spans, rejected = select(text, prediction, threshold)
    fields = {role: [s['text'] for s in spans if s['role'] == role]
              for role in ['TITLE', 'TIME', 'LOCATION', 'NOTE', 'CONNECTOR']}
    covered = {i for s in spans for i in range(s['start'], s['end'])}
    residual = ''.join(c for i, c in enumerate(text) if i not in covered and not c.isspace()
                       and c not in '，,。.!！：:、')
    reasons = []
    if prediction['requestProbabilities'][0] < request_threshold:
        reasons.append('request-needs-confirmation')
    if prediction.get('malformedBIO'):
        reasons.append('malformed-BIO')
    if len(fields['TITLE']) != 1:
        reasons.append('event-not-established')
    if residual:
        reasons.append('unexplained-content')
    if row['preferredKind'] != 'TASK':
        reasons.append('habit-outside-prototype')
    # Similar incompatible alternatives produce a reviewable draft, not arbitrary acceptance.
    for alternative in rejected:
        if alternative['role'] == 'CONNECTOR':
            continue
        rivals = [s for s in spans if s['role'] == alternative['role']]
        if rivals and max(s['score'] for s in rivals) - alternative['score'] < .5:
            reasons.append('ambiguous-' + alternative['role'].lower())
            break
    title = fields['TITLE'][0] if fields['TITLE'] else text.strip()
    note = ('地点：' + fields['LOCATION'][0]) if fields['LOCATION'] else ''
    if fields['NOTE']:
        note += ('；' if note else '') + fields['NOTE'][0]
    proposal = dict(BASE, title=title, note=note, ready=False, needsAI=True, source='SPAN_PROTOTYPE')
    return dict(proposal=proposal, spans=spans, fields=fields, reasons=list(dict.fromkeys(reasons)),
                rawText=text, residual=residual, needsConfirmation=True)


def finish(draft, normalized):
    """Consume only the KMP result for MODEL-PREDICTED time text plus a sentinel."""
    result = dict(draft, proposal=dict(draft['proposal']), reasons=list(draft['reasons']))
    if draft['fields']['TIME']:
        if (not normalized or not normalized.get('ready') or normalized.get('needsAI')
                or normalized.get('title') != '片段时间占位' or not normalized.get('dueAt')):
            result['reasons'].append('time-not-normalized')
        else:
            result['proposal']['dueAt'] = normalized['dueAt']
    result['reasons'] = list(dict.fromkeys(result['reasons']))
    complete = not result['reasons']
    result['proposal'].update(ready=complete, needsAI=not complete)
    result['needsConfirmation'] = not complete
    result['state'] = 'COMPLETE' if complete else 'PARTIAL' if any(
        draft['fields'][role] for role in ['TITLE', 'TIME', 'LOCATION', 'NOTE']) else 'CONFIRM'
    return result


def span_metrics(rows, raw, threshold=0.):
    counts = {role: dict(tp=0, predicted=0, gold=0) for role in ['TITLE', 'TIME', 'LOCATION', 'NOTE']}
    for row, prediction in zip(rows, raw):
        selected, _ = select(row['text'], prediction, threshold)
        for role in counts:
            gold = {(s['start'], s['end']) for s in row['spans'] if s['label'] == role}
            actual = {(s['start'], s['end']) for s in selected if s['role'] == role}
            counts[role]['tp'] += len(actual & gold)
            counts[role]['predicted'] += len(actual)
            counts[role]['gold'] += len(gold)
    for value in counts.values():
        value['precision'] = value['tp'] / value['predicted'] if value['predicted'] else None
        value['recall'] = value['tp'] / value['gold'] if value['gold'] else None
        value['f1'] = 2 * value['tp'] / (value['predicted'] + value['gold']) if value['predicted'] + value['gold'] else None
    return counts
