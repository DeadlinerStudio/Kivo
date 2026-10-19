import copy
import unittest

from quickadd_training.span_data import build, regression, validate
from quickadd_training.span_decode import decode, finish


def perfect_prediction(row):
    return dict(requestProbabilities=[.999, .001], candidates=[
        dict(role=s['label'], start=s['start'], end=s['end'], score=10.) for s in row['spans']])


class CoreSpanTests(unittest.TestCase):
    def test_location_and_modal_are_separate_from_event(self):
        row = regression()[0]
        draft = decode(row, perfect_prediction(row))
        # This tests semantic assembly of extracted spans, including dropping a modeled connector.
        result = finish(draft, dict(ready=True, needsAI=False, title='片段时间占位', dueAt='2026-10-20T16:00'))
        self.assertEqual(result['proposal']['title'], '开会')
        self.assertEqual(result['proposal']['note'], '地点：逸夫楼')
        self.assertTrue(result['proposal']['ready'])

    def test_whitespace_runs_preserve_original_time_span(self):
        row = regression()[6]
        draft = decode(row, perfect_prediction(row))
        self.assertEqual(draft['fields']['TIME'], ['明天下午  4  点'])
        self.assertFalse(draft['residual'])

    def test_location_name_in_event_is_not_moved_to_note(self):
        row = regression()[7]
        draft = decode(row, perfect_prediction(row))
        self.assertEqual(draft['proposal']['title'], '讨论逸夫楼改造方案')
        self.assertEqual(draft['proposal']['note'], '')

    def test_unexplained_content_is_lossless_and_requires_confirmation(self):
        row = regression()[0]
        prediction = perfect_prediction(row)
        prediction['candidates'] = [s for s in prediction['candidates'] if s['role'] != 'LOCATION']
        result = finish(decode(row, prediction), dict(ready=True, needsAI=False, title='片段时间占位', dueAt='2026-10-20T16:00'))
        self.assertFalse(result['proposal']['ready'])
        self.assertIn('unexplained-content', result['reasons'])
        self.assertEqual(result['rawText'], row['text'])
        self.assertEqual(result['residual'], '逸夫楼')

    def test_supported_fields_cannot_override_negative_request(self):
        row = regression()[0]
        prediction = perfect_prediction(row)
        prediction['requestProbabilities'] = [.01, .99]
        result = finish(decode(row, prediction), dict(ready=True, needsAI=False, title='片段时间占位', dueAt='2026-10-20T16:00'))
        self.assertFalse(result['proposal']['ready'])
        self.assertIn('request-needs-confirmation', result['reasons'])

    def test_normalizer_must_consume_all_predicted_time(self):
        row = regression()[0]
        result = finish(decode(row, perfect_prediction(row)), dict(
            ready=True, needsAI=False, title='不明时间片段 片段时间占位', dueAt='2026-10-20T16:00'))
        self.assertFalse(result['proposal']['ready'])
        self.assertIsNone(result['proposal']['dueAt'])
        self.assertIn('time-not-normalized', result['reasons'])

    def test_close_competing_event_boundaries_require_confirmation(self):
        row = regression()[0]
        prediction = perfect_prediction(row)
        title = next(s for s in prediction['candidates'] if s['role'] == 'TITLE')
        prediction['candidates'].append(dict(title, start=title['start'] + 1, score=9.8))
        draft = decode(row, prediction)
        self.assertIn('ambiguous-title', draft['reasons'])

    def test_entity_isolation_detects_different_text_with_same_title(self):
        splits = build(30)
        validate(splits)
        row = copy.deepcopy(splits['train'][0])
        old = row['expected']['proposal']['title']
        source = splits['dev'][0]
        new = source['expected']['proposal']['title']
        start = next(s['start'] for s in row['spans'] if s['label'] == 'TITLE')
        delta = len(new) - len(old)
        row['text'] = row['text'][:start] + new + row['text'][start + len(old):]
        for span in row['spans']:
            if span['label'] == 'TITLE':
                span['end'] += delta
            elif span['start'] >= start + len(old):
                span['start'] += delta
                span['end'] += delta
        row['expected']['proposal']['title'] = new
        splits['train'][0] = row
        with self.assertRaisesRegex(ValueError, 'Entity leakage'):
            validate(splits)

    def test_connectors_cannot_overlap_location_gold(self):
        row = copy.deepcopy(regression()[0])
        location = next(s for s in row['spans'] if s['label'] == 'LOCATION')
        row['spans'].append(dict(location, label='CONNECTOR'))
        with self.assertRaisesRegex(ValueError, 'Overlapping gold spans'):
            validate({'dev': [row]})


if __name__ == '__main__':
    unittest.main()
