"""Read the independently normalized canonical sentence; no learned span input."""
from .corpus import BASE


def parse_normalized(row, prediction, minimum=0.):
    """The recognizer consumes the model-produced STRING, not source span assignments."""
    text = prediction['normalized']
    fields = {k: [] for k in ['TITLE', 'TIME', 'LOCATION', 'NOTE', 'CONNECTOR']}
    reasons = []
    headers = {'时间': 'TIME', '事项': 'TITLE', '地点': 'LOCATION', '备注': 'NOTE'}
    order = ['TIME', 'TITLE', 'LOCATION', 'NOTE']
    seen = []
    if text in [None, '需要确认']:
        reasons.append('normalizer-needs-confirmation' if text else 'normalization-malformed')
    else:
        for part in text.split('；'):
            header, separator, value = part.partition('：')
            role = headers.get(header)
            if not separator or not role or not value.strip() or role in seen:
                reasons.append('normalization-malformed')
                break
            seen.append(role)
            fields[role] = [value.strip()]
        if seen != sorted(seen, key=order.index):
            reasons.append('normalization-malformed')
    if not fields['TITLE']:
        reasons.append('event-not-established')
    if prediction['minimumProbability'] < minimum:
        reasons.append('normalization-low-confidence')
    if row['preferredKind'] != 'TASK':
        reasons.append('habit-outside-prototype')
    note = ('地点：' + fields['LOCATION'][0]) if fields['LOCATION'] else ''
    if fields['NOTE']:
        note += ('；' if note else '') + fields['NOTE'][0]
    proposal = dict(BASE, title=fields['TITLE'][0] if fields['TITLE'] else row['text'].strip(), note=note,
                    ready=False, needsAI=True, source='NORMALIZER_RULES_PROTOTYPE')
    return dict(proposal=proposal, fields=fields, spans=[], reasons=list(dict.fromkeys(reasons)),
                rawText=row['text'], normalizedText=text, residual='', needsConfirmation=True)
