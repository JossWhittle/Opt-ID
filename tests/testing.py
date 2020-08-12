import unittest

from optid.blah import MEANING_OF_LIFE

class MyTestCase(unittest.TestCase):
    def test_meaning_of_life(self):
        self.assertEqual(MEANING_OF_LIFE, 42)

    def test_something_else(self):
        self.assertEqual(True, True)

