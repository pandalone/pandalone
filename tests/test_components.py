import doctest
import unittest

import pandalone.components


class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.components, optionflags=doctest.NORMALIZE_WHITESPACE)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))
