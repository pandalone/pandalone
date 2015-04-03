import unittest
import doctest
import pandalone.utils


class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        suite = doctest.DocTestSuite(pandalone.utils)
        unittest.TextTestRunner(verbosity=2).run(suite)
