import unittest
from quickadd_training.import_massive import spans, align

class MassiveAlignmentTests(unittest.TestCase):
    def test_exact_english(self):
        plain,labels=spans('call [person : Mom] at [time : 3pm]')
        result,method=align(plain,labels,'call Mom at 3pm')
        self.assertEqual(method,'exact')
        self.assertEqual(result[0]['text'],'Mom')

    def test_chinese_whitespace_maps_to_actual_unicode_offsets(self):
        plain,labels=spans('[date : 星期五] 早上 [time : 九点] 叫醒我')
        text='星期五早上九点叫醒我'
        result,method=align(plain,labels,text)
        self.assertEqual(method,'whitespace-only')
        for span in result:self.assertEqual(text[span['start']:span['end']],span['text'])
        self.assertEqual(result[1]['start'],5)

    def test_changed_words_are_not_aligned_by_guessing(self):
        plain,labels=spans('[date : 明天] 开会')
        self.assertEqual(align(plain,labels,'后天开会'),([], 'unresolved'))
