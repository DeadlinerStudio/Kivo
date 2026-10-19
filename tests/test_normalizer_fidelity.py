import unittest

from quickadd_training.normalizer_fidelity import audit_copy, legal_gap, guard_normalized
from quickadd_training.normalizer_rules import parse_normalized
from quickadd_training.span_decode import finish


class FidelityTests(unittest.TestCase):
    def audit(self, text, chunks):
        controls = {100: 'TIME', 101: 'TITLE', 102: 'LOCATION', 103: 'NOTE', 104: 'EOS'}
        actions = []
        for role, start, end in chunks:
            actions += [next(k for k, v in controls.items() if v == role)] + list(range(start, end))
        return audit_copy(text, actions + [104], [(i, i + 1) for i in range(len(text))], controls)

    def test_example_keeps_entities_and_allows_location_introducer(self):
        text = '明天下午4点要在逸夫楼开会'
        result = self.audit(text, [('TIME', 0, 6), ('TITLE', 11, 13), ('LOCATION', 8, 11)])
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['gaps'][0]['text'], '要在')

    def test_missing_room_number_title_suffix_or_location_suffix_is_blocked(self):
        for text, end in [('城西阅览室208开会', 5), ('办理自行车年检', 6), ('松林交流中心开会', 2)]:
            with self.subTest(text=text):
                chunks = [('TITLE', 0, end)] if text.startswith('办理') else [
                    ('TITLE', len(text) - 2, len(text)), ('LOCATION', 0, end)]
                result = self.audit(text, chunks)
                self.assertFalse(result['passed'])
                self.assertIn('source-content-deleted', result['errors'])

    def test_deleted_negation_condition_or_tail_cannot_be_treated_as_connector(self):
        for text, chunks in [
            ('不去开会', [('TITLE', 1, 4)]),
            ('如果有空开会', [('TITLE', 4, 6)]),
            ('办理取消预约', [('TITLE', 0, 2)]),
            ('开会需要', [('TITLE', 0, 2)]),
        ]:
            with self.subTest(text=text):
                self.assertFalse(self.audit(text, chunks)['passed'])
        self.assertTrue(self.audit('确认没有录入的成绩', [('TITLE', 0, 9)])['passed'])

    def test_connector_is_scoped_to_source_edge(self):
        self.assertTrue(legal_gap(' 要在，', 'TIME', 'LOCATION')[0])
        self.assertFalse(legal_gap('在', 'TITLE', 'TITLE')[0])
        self.assertFalse(legal_gap('在', 'LOCATION', 'LOCATION')[0])
        self.assertTrue(legal_gap('的', 'TIME', 'TIME')[0])
        self.assertFalse(legal_gap('的', 'TITLE', 'TITLE')[0])
        self.assertFalse(legal_gap('准备', 'TITLE', None)[0])
        self.assertFalse(legal_gap('？', 'TITLE', None)[0])
        self.assertFalse(legal_gap('\n', 'TITLE', None)[0])

    def test_repeated_text_uses_positions_not_substring_presence(self):
        result = self.audit('开会开会', [('TITLE', 0, 2)])
        self.assertFalse(result['passed'])
        self.assertEqual(result['unexplainedText'], '开会')

    def test_bad_copy_trace_cannot_claim_coverage(self):
        for actions in [[101, 0, 0, 104], [101, 0], [101, 9, 104], [100, 0, 104], [101, 104, 104], None]:
            with self.subTest(actions=actions):
                result = audit_copy('会', actions, [(0, 1)], {100: 'TIME', 101: 'TITLE', 104: 'EOS'})
                self.assertFalse(result['passed'])

    def test_unicode_character_offsets_and_spaces(self):
        self.assertTrue(self.audit('开会📝', [('TITLE', 0, 3)])['passed'])
        self.assertTrue(self.audit(' 开会\t', [('TITLE', 1, 3)])['passed'])

    def test_wrapper_rejects_string_trace_mismatch_and_preserves_partial_fields(self):
        from quickadd_training.normalizer_model import IDS

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'offset_mapping': [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]}

        row = dict(text='办理年检', preferredKind='TASK')
        prediction = dict(normalized='事项：办理', minimumProbability=.99,
                          actions=[IDS['TITLE'], 1, 2, IDS['EOS']])
        draft = parse_normalized(row, prediction)
        guarded = guard_normalized(row, prediction, draft, Tokenizer())
        result = finish(guarded, None)
        self.assertFalse(result['proposal']['ready'])
        self.assertEqual(result['rawText'], row['text'])
        self.assertEqual(result['proposal']['title'], '办理')
        self.assertEqual(draft['reasons'], [])
        self.assertIn('source-content-deleted', result['reasons'])
        prediction.update(normalized='事项：办理', actions=[IDS['TITLE'], 1, 2, 3, 4, IDS['EOS']])
        guarded = guard_normalized(row, prediction, parse_normalized(row, prediction), Tokenizer())
        self.assertIn('copy-output-mismatch', guarded['reasons'])


if __name__ == '__main__':
    unittest.main()
