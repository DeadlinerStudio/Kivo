"""Validate provenance, full proposals, BIO spans and split leakage."""
import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import unicodedata
from pathlib import Path

FIELDS = ('kind', 'title', 'dueAt', 'note', 'period', 'targetCount', 'weekdaysMask', 'reminderTime')
LABELS = ('TITLE', 'TIME', 'REPEAT', 'NOTE')


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(splits):
    seen_ids, seen_text, groups = set(), {}, {}
    for split, rows in splits.items():
        for row in rows:
            if row['id'] in seen_ids:
                raise ValueError('Duplicate id: ' + row['id'])
            seen_ids.add(row['id'])
            if row['split'] != split or not row['provenance'] or not row['group']:
                raise ValueError('Missing provenance/group or inconsistent split')
            key = unicodedata.normalize('NFKC', row['text']).strip().casefold()
            for table, value in ((seen_text, key), (groups, row['group'])):
                if value in table and table[value] != split:
                    raise ValueError('Cross-split leakage: ' + value)
                table[value] = split
            if row['preferredKind'] not in ('TASK', 'HABIT'):
                raise ValueError('Invalid preferred kind')
            moment = datetime.datetime.fromisoformat(row['now'].replace('Z', '+00:00'))
            if moment.tzinfo is None:
                raise ValueError('Reference instant must include UTC offset')
            ZoneInfo(row['timezone'])
            expected = row['expected']
            if expected['decision'] not in ('accept', 'defer'):
                raise ValueError('Unsupported decision')
            if expected['decision'] == 'accept':
                proposal = expected['proposal']
                if set(proposal) != set(FIELDS) or not proposal['title']:
                    raise ValueError('Accept must specify every business field')
                if proposal['kind'] not in ('TASK', 'HABIT'):
                    raise ValueError('Invalid kind')
                if proposal['period'] not in ('DAILY', 'WEEKLY', 'MONTHLY', 'TOTAL'):
                    raise ValueError('Invalid period')
                if type(proposal['targetCount']) is not int or not 1 <= proposal['targetCount'] <= 10000:
                    raise ValueError('Invalid target count')
                if type(proposal['weekdaysMask']) is not int or not 1 <= proposal['weekdaysMask'] <= 127:
                    raise ValueError('Invalid weekday mask')
                if proposal['dueAt'] is not None:
                    datetime.datetime.fromisoformat(proposal['dueAt'])
                if proposal['reminderTime'] is not None:
                    datetime.time.fromisoformat(proposal['reminderTime'])
            occupied = set()
            for span in row.get('spans', []):
                start, end = span['start'], span['end']
                if span['label'] not in LABELS or not 0 <= start < end <= len(row['text']):
                    raise ValueError('Invalid Unicode span')
                indexes = set(range(start, end))
                if occupied & indexes:
                    raise ValueError('Overlapping spans')
                occupied |= indexes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supervised(row):
    if len(row['text']) > 128 or not row['text']:
        raise ValueError('Training text must contain 1..128 Unicode characters')
    spans = row.get('spans')
    if spans is None or (row['expected']['decision'] == 'accept' and not any(s['label'] == 'TITLE' for s in spans)):
        raise ValueError('Training requires reviewed spans; never derive labels from rules')
    if row['expected']['decision'] == 'accept':
        titles = [row['text'][s['start']:s['end']].strip() for s in spans if s['label'] == 'TITLE']
        if titles != [row['expected']['proposal']['title']]:
            raise ValueError('Reviewed TITLE span must agree with expected proposal')
    if row['expected']['decision'] == 'defer' and row.get('trainingIntent') != 'OTHER':
        raise ValueError('Deferred scheduling requests are not OTHER; explicit reviewed trainingIntent required')
    if row['expected']['decision'] == 'defer' and spans:
        raise ValueError('OTHER examples must have empty spans')
    slots = [0] * len(row['text'])
    for span in spans:
        base = 1 + 2 * LABELS.index(span['label'])
        slots[span['start']] = base
        slots[span['start'] + 1:span['end']] = [base + 1] * (span['end'] - span['start'] - 1)
    intent = 2 if row['expected']['decision'] == 'defer' else ('TASK', 'HABIT').index(row['expected']['proposal']['kind'])
    return dict(text=row['text'], intent=intent, slots=slots)
