"""Explain top proposal errors separately from confidence acceptance."""
from collections import Counter

BUSINESS_FIELDS = ('kind', 'title', 'note', 'dueAt', 'period', 'targetCount', 'weekdaysMask', 'reminderTime')


def diagnose(rows, values):
    result = {}
    for kind in ['TASK', 'HABIT']:
        pairs = [(r, v) for r, v in zip(rows, values)
                 if r['expected']['decision'] == 'accept' and r['expected']['proposal']['kind'] == kind]
        counts = {field: sum(r['expected']['proposal'][field] == v['proposal'][field] for r, v in pairs)
                  for field in BUSINESS_FIELDS}
        result[kind] = dict(positiveRows=len(pairs), fieldCorrectCounts=counts,
                            fieldAccuracy={k: n / len(pairs) if pairs else None for k, n in counts.items()},
                            proposalKindCounts=dict(Counter(v['proposal']['kind'] for _, v in pairs)),
                            hardReasons=dict(Counter(reason for _, v in pairs for reason in v['hardReasons'])))
    return dict(scope='All positive top proposals, including invalid and low confidence drafts; not accepted precision',
                kinds=result)
