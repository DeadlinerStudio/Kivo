import unittest

from quickadd_training.normalizer_rules import parse_normalized
from quickadd_training.span_decode import finish


class NormalizerRulesTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(text='明天下午 4 点要在逸夫楼开会', preferredKind='TASK')

    def parse(self, text):
        return parse_normalized(self.row, dict(normalized=text, minimumProbability=.99))

    def test_canonical_sentence_is_read_without_source_spans(self):
        draft = self.parse('时间：明天下午4点；事项：开会；地点：逸夫楼')
        result = finish(draft, dict(ready=True, needsAI=False, title='片段时间占位', dueAt='2026-10-20T16:00'))
        self.assertTrue(result['proposal']['ready'])
        self.assertEqual(result['proposal']['note'], '地点：逸夫楼')
        self.assertEqual(result['proposal']['title'], '开会')

    def test_duplicate_time_field_cannot_silently_replace_value(self):
        draft = self.parse('时间：明天下午4点；时间：后天下午4点；事项：开会')
        self.assertIn('normalization-malformed', draft['reasons'])

    def test_confirmation_keeps_original_input(self):
        result = finish(self.parse('需要确认'), None)
        self.assertFalse(result['proposal']['ready'])
        self.assertEqual(result['rawText'], self.row['text'])
        self.assertEqual(result['state'], 'CONFIRM')

    def test_missing_or_reordered_fields_require_confirmation(self):
        for text in ['时间：明天；地点：逸夫楼', '事项：开会；时间：明天下午4点', '事项：']:
            with self.subTest(text=text):
                result = finish(self.parse(text), None)
                self.assertFalse(result['proposal']['ready'])

    def test_unrecognized_format_is_not_parsed_as_valid_proposal(self):
        draft = self.parse('开会 地点逸夫楼 明天下午4点')
        self.assertIn('normalization-malformed', draft['reasons'])


if __name__ == '__main__':
    unittest.main()
