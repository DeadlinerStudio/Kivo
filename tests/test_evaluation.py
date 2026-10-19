import copy
import unittest
from quickadd_training.data import load, validate, supervised, FIELDS
from quickadd_training.metrics import summarize, streaming
from quickadd_training.gate import reasons


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.row=load('data/dev.jsonl')[0]
        self.proposal=dict(self.row['expected']['proposal'],ready=True,needsAI=False,source='MODEL')

    def prediction(self,proposal):
        return {self.row['id']:dict(hybrid=proposal,hybridMs=1.0)}

    def test_full_proposal_correct(self):
        report=summarize([self.row],self.prediction(self.proposal),'hybrid')
        self.assertEqual(report['correctCoverage'],1)
        self.assertEqual(report['wrongAcceptanceRate'],0)

    def test_every_field_mismatch_is_wrong(self):
        for field in FIELDS:
            with self.subTest(field=field):
                proposal=dict(self.proposal);proposal[field]='different'
                self.assertEqual(summarize([self.row],self.prediction(proposal),'hybrid')['wrongAccepted'],1)

    def test_negative_accept_is_wrong(self):
        row=dict(self.row,expected=dict(decision='defer'))
        self.assertEqual(summarize([row],self.prediction(self.proposal),'hybrid')['negativeWrongAccepted'],1)

    def test_no_accept_has_no_rate(self):
        self.assertIsNone(summarize([self.row],self.prediction(None),'hybrid')['wrongAcceptanceRate'])

    def test_ready_needs_ai_is_deferred(self):
        self.assertEqual(summarize([self.row],self.prediction(dict(self.proposal,needsAI=True)),'hybrid')['accepted'],0)

    def test_group_leak(self):
        other=dict(self.row,id='other',split='test',text='different')
        with self.assertRaises(ValueError):validate(dict(dev=[self.row],test=[other]))

    def test_normalized_text_leak(self):
        other=dict(self.row,id='other',split='test',group='other',text=self.row['text'].replace('3','３'))
        with self.assertRaises(ValueError):validate(dict(dev=[self.row],test=[other]))

    def test_unreviewed_training_rejected(self):
        with self.assertRaises(ValueError):supervised(self.row)

    def test_unicode_slots(self):
        row=dict(self.row,text='跑步🏃',expected=dict(decision='accept',proposal=dict(self.row['expected']['proposal'],title='跑步🏃')),spans=[dict(start=0,end=3,label='TITLE')])
        self.assertEqual(supervised(row)['slots'],[1,2,2])

    def test_overlap_rejected(self):
        row=dict(self.row,spans=[dict(start=0,end=3,label='TITLE'),dict(start=1,end=4,label='TIME')])
        with self.assertRaises(ValueError):validate(dict(dev=[row]))

    def test_streaming_changes(self):
        rows=[dict(self.row,id=str(i),parentId='x',prefixLength=i,isFinal=i==2) for i in (1,2)]
        predictions={'1':dict(hybrid=self.proposal),'2':dict(hybrid=dict(self.proposal,title='other'))}
        self.assertEqual(streaming(rows,predictions,'hybrid')['acceptedProposalChanges'],1)

    def test_dev_cannot_release(self):
        self.assertIn('Independent reviewed test set required',reasons(dict(split='dev'),{}))

    def test_release_gate_positive_and_regression(self):
        report=dict(split='test',datasetSha256='test',modelSha256='model',manifestSha256='manifest',modes=dict(
            rules=dict(wrongAcceptanceRate=.01,correctCoverage=.7),
            hybrid=dict(total=250,eligible=150,wrongAcceptanceRate=.005,correctCoverage=.8)))
        evidence=dict(independentlyReviewed=True,frozenTestSha256='test',modelSha256='model',manifestSha256='manifest',
                      exportParityPassed=True,streamingReviewed=True,devices={p:dict(device='test-device',endToEndPassed=True,samples=100,p95Ms=20) for p in ('android','ios','ohos')})
        self.assertEqual(reasons(report,evidence),[])
        report['modes']['hybrid']['wrongAcceptanceRate']=.02
        self.assertTrue(reasons(report,evidence))

    def test_null_device_timings_fail_closed(self):
        self.assertTrue(reasons({}, dict(devices={'android':dict(device='x',endToEndPassed=True,samples=100,p95Ms=None)})))

    def test_deferred_task_not_other(self):
        row=dict(self.row,expected=dict(decision='defer'),spans=[])
        with self.assertRaises(ValueError):supervised(row)
        row['trainingIntent']='OTHER'
        self.assertEqual(supervised(row)['intent'],2)

    def test_initial_data_valid(self):
        validate({s:load('data/'+s+'.jsonl') for s in ('train','dev','test')})

if __name__=='__main__':unittest.main()
