"""No Python calendar/count parser: normalize only predicted fields in production KMP."""
from pathlib import Path

from .native_adapter import evaluate


class NativeFields:
    def __init__(self, binary, directory):
        self.binary, self.directory = Path(binary), Path(directory)
        self.cache, self.calls = {}, 0

    def normalize(self, rows, drafts):
        keys, missing = [], {}
        for row, draft in zip(rows, drafts):
            f = draft['fields']
            key = (draft['proposal']['kind'], ' '.join(f.get('TIME', [])), ''.join(f.get('DAYS', [])),
                   ''.join(f.get('GOAL_WINDOW', [])), ''.join(f.get('COUNT', [])), row['now'], row['timezone'])
            keys.append(key)
            if key not in self.cache and key not in missing:
                missing[key] = dict(id='fields-' + str(len(missing)), text='模型标题占位', preferredKind=key[0],
                                    now=key[-2], timezone=key[-1],
                                    structuredFields=dict(time=key[1], days=key[2], window=key[3], count=key[4]))
        if missing:
            result = evaluate(self.binary, None, list(missing.values()), self.directory / f'call-{self.calls}')
            self.cache.update({key: result[row['id']].get('structured') for key, row in missing.items()})
            self.calls += 1
        return [self.cache[key] for key in keys]


def attach(draft, normalized):
    value = dict(draft, proposal=dict(draft['proposal']), hardReasons=list(draft['hardReasons']))
    p = value['proposal']
    if not normalized or normalized.get('title') != '模型标题占位' or not normalized.get('ready') or normalized.get('needsAI'):
        value['hardReasons'].append('fields-not-normalized')
    else:
        for field in ['dueAt', 'period', 'targetCount', 'weekdaysMask', 'reminderTime']:
            p[field] = normalized[field]
    value['hardReasons'] = list(dict.fromkeys(value['hardReasons']))
    value['draftValid'] = not value['hardReasons']
    p.update(ready=value['draftValid'], needsAI=True)
    value['reasons'] = list(value['hardReasons'])
    return value
