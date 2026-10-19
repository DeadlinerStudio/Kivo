import math
import json
import unittest
import numpy as np

from quickadd_training.corpus import BASE
from quickadd_training.joint_calibration import upper_bound, select, fit
from quickadd_training.joint_decode import apply_gate, candidates, coverage
from quickadd_training.joint_heads import exact


class JointTests(unittest.TestCase):
    def value(self, kind='TASK', valid=True):
        return dict(proposal=dict(BASE, kind=kind, title='跑步', dueAt='2026-10-20T07:00' if kind == 'TASK' else None),
                    draftValid=valid, rawText='明天早上7点跑步', hardReasons=[] if valid else ['source-content-deleted'])

    def test_low_confidence_ready_proposal_can_offer_both_add_and_ai(self):
        result = apply_gate(self.value(), .89, .9)
        self.assertTrue(result['proposal']['ready'])
        self.assertTrue(result['proposal']['needsAI'])
        self.assertFalse(result['localTrusted'])

    def test_threshold_equality_can_hide_ai_but_never_override_hard_failure(self):
        self.assertFalse(apply_gate(self.value(), .9, .9)['proposal']['needsAI'])
        result = apply_gate(self.value(valid=False), 1., .9)
        self.assertFalse(result['proposal']['ready'])
        self.assertTrue(result['proposal']['needsAI'])

    def test_missing_confidence_or_unqualified_threshold_keeps_ready_draft(self):
        for confidence, threshold in [(None, .9), (float('nan'), .9), (.9999, None), (1.1, .9)]:
            result = apply_gate(self.value(), confidence, threshold)
            self.assertTrue(result['proposal']['ready'])
            self.assertTrue(result['proposal']['needsAI'])

    def test_habit_goal_count_is_not_discardable_suffix(self):
        text = '每周跑步三次'
        spans = [dict(role='GOAL_WINDOW', start=0, end=2), dict(role='TITLE', start=2, end=4)]
        self.assertIn('source-content-deleted', coverage(text, spans, 'HABIT')[0])
        spans += [dict(role='COUNT', start=4, end=6)]
        self.assertEqual(coverage(text, spans, 'HABIT')[0], [])

    def test_faithful_but_misplaced_location_is_not_a_correct_proposal(self):
        text = '翠竹社区接待处办理报废登记'
        self.assertEqual(coverage(text, [dict(role='LOCATION', start=0, end=4),
                                         dict(role='TITLE', start=4, end=len(text))], 'TASK')[0], [])
        expected = dict(decision='accept', proposal=dict(BASE, title='办理报废登记', note='地点：翠竹社区接待处'))
        actual = dict(proposal=dict(BASE, title='接待处办理报废登记', note='地点：翠竹社区'))
        self.assertFalse(exact(dict(expected=expected), actual))

    def test_candidate_generation_never_reads_gold_labels(self):
        row = dict(text='每周跑步三次', preferredKind='TASK', expected=dict(decision='defer'))
        raw = dict(intentProbabilities=[.01, .98, .01], candidates=[
            dict(role='GOAL_WINDOW', start=0, end=2, score=5.), dict(role='TITLE', start=2, end=4, score=5.),
            dict(role='COUNT', start=4, end=6, score=5.)])
        before = candidates(row, raw)
        row['expected'] = dict(decision='accept', proposal=dict(BASE, title='任意不相干答案'))
        after = candidates(row, raw)
        self.assertEqual([v['fields'] for v in before], [v['fields'] for v in after])
        self.assertEqual(before[0]['hardReasons'], [])

    def test_explicit_habit_context_does_not_silently_become_task(self):
        row = dict(text='明天跑步', preferredKind='HABIT')
        raw = dict(intentProbabilities=[.99, .005, .005], candidates=[
            dict(role='TIME', start=0, end=2, score=5.), dict(role='TITLE', start=2, end=4, score=5.)])
        self.assertIn('explicit-habit-entry-conflict', candidates(row, raw)[0]['hardReasons'])

    def test_binomial_bound_handles_zero_and_nonzero_errors(self):
        self.assertEqual(upper_bound(0, 0), 1.)
        self.assertAlmostEqual(upper_bound(0, 600), 1 - .05 ** (1 / 600))
        p = upper_bound(1, 10)
        cdf = (1 - p) ** 10 + 10 * p * (1 - p) ** 9
        self.assertAlmostEqual(cdf, .05, places=10)

    def test_small_zero_error_sample_does_not_pass_twelve_threshold_search(self):
        rows, values, probabilities = [], [], []
        for kind in ['TASK', 'HABIT']:
            for _ in range(1000):
                v = self.value(kind)
                rows.append(dict(expected=dict(decision='accept', proposal=v['proposal'])))
                values.append(v); probabilities.append(.9999)
        result = select(rows, values, probabilities)
        self.assertTrue(all(v['threshold'] is None for v in result['choices'].values()))
        self.assertFalse(result['formalGatePassed'])

    def test_wrong_kind_is_a_negative_in_predicted_kind_calibration(self):
        wrong, right = self.value('HABIT'), self.value('HABIT')
        rows = [dict(expected=dict(decision='accept', proposal=self.value('TASK')['proposal'])),
                dict(expected=dict(decision='accept', proposal=right['proposal']))]
        result = fit(rows, [wrong, right], [-2., 2.])
        self.assertEqual(result['HABIT']['samples'], 2)
        self.assertEqual(result['HABIT']['positives'], 1)

    def test_diagnostic_serialization_preserves_positions_without_internal_arrays(self):
        from quickadd_training.joint_evaluate import plain
        value = self.value()
        value.update(feature=np.zeros(4), qualityFeature=np.zeros(4),
                     spans=[dict(role='TITLE', start=0, end=2, score=1., vector=np.zeros(128), edge=np.zeros(64))])
        exported = json.loads(json.dumps(plain(value)))
        self.assertEqual(exported['spans'][0]['end'], 2)
        self.assertNotIn('edge', exported['spans'][0])
        self.assertIn('edge', value['spans'][0])

    def test_type_heads_reuse_one_encoder_and_route_without_gold_at_inference(self):
        import torch
        from transformers import BertConfig
        from quickadd_training.joint_adapter import TypeHeadJointModel
        model = TypeHeadJointModel(BertConfig(vocab_size=32, hidden_size=128, num_hidden_layers=1,
                                            num_attention_heads=2, intermediate_size=256))
        model.eval()
        calls = []
        hook = model.encoder.register_forward_hook(lambda *args: calls.append(True))
        with torch.no_grad():
            model.intent.weight.zero_(); model.intent.bias.copy_(torch.tensor([-5., 5., -5.]))
            _, spans, hidden = model(torch.ones((1, 8), dtype=torch.long), torch.ones((1, 8), dtype=torch.long), torch.tensor([0]))
            expected = model.span_scores(hidden, model.habit_query, model.habit_key, model.habit_start, model.habit_end)
        hook.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(spans, expected)


if __name__ == '__main__':
    unittest.main()
