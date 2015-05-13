import doctest
import logging
import sys
import unittest

import pandalone.utils


DEFAULT_LOG_LEVEL = logging.INFO


def _init_logging(module_name, loglevel=DEFAULT_LOG_LEVEL):
    logging.basicConfig(level=loglevel)
    logging.getLogger().setLevel(level=loglevel)

    log = logging.getLogger(module_name)
    log.trace = lambda *args, **kws: log.log(0, *args, **kws)

    return log


@unittest.skipIf(sys.version_info < (3, 3), "Doctests are made for py >= 3.3")
class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.utils, optionflags=doctest.NORMALIZE_WHITESPACE)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))
