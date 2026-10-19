import unittest
from quickadd_training.corpus import build
from quickadd_training.data import validate, supervised

class LocalizedCorpusTests(unittest.TestCase):
    def test_both_languages_have_valid_unleaked_train_spans(self):
        for locale in ('zh','en'):
            with self.subTest(locale=locale):
                splits=build(locale,count=100)
                validate(splits)
                for row in splits['train']:
                    output=supervised(row)
                    self.assertEqual(len(output['slots']),len(row['text']))
                    self.assertEqual(row['locale'],locale)
                    self.assertEqual(row['annotationStatus'],'synthetic')
                self.assertEqual(splits['test'],[])

    def test_generation_is_reproducible(self):
        self.assertEqual(build('en',count=20),build('en',count=20))
