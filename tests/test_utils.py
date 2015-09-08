#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from __future__ import division, unicode_literals

import doctest
import logging
import os
from pandalone.utils import fullmatch_py2, LoggerWriter
import re
import sys
import unittest

import pandalone.utils as utils


@unittest.skipIf(sys.version_info < (3, 4), "Doctests are made for py >= 3.3")
class Doctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            utils, optionflags=doctest.NORMALIZE_WHITESPACE)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestUtils(unittest.TestCase):

    def _assert_expansion(self, regex, s, template):
        m1 = re.fullmatch(regex, s)
        s1 = m1.expand(template) if m1 else '<NO-MATCH>'
        m2 = fullmatch_py2(regex, s)
        s2 = m2.expand(template) if m2 else '<NO-MATCH>'
        self.assertEqual(s1, s2, "\n  PY-3: '%s' \n  PY-2: '%s' " % (s1, s2))

    @unittest.skipIf(not 'fullmatch' in dir(re), "Platform has no 'fullmatch() to compare with.")
    def test_fullmatch(self):
        self._assert_expansion('.*', 'foo', r'A')
        self._assert_expansion('(.*)', 'foo', r'A_\1')
        self._assert_expansion('(.*)', 'foo', r'A_\g<0>')

        self._assert_expansion('a.*', 'afoo&', r'A')
        self._assert_expansion('a(\w*)', 'afoo&', r'A_\1')
        self._assert_expansion('a(\w*)', 'afoo&', r'A_\g<0>')

    # Not fixed for performance,
    # for full-solution, see:
    #   http://stackoverflow.com/questions/30212413/backport-python-3-4s-regular-expression-fullmatch-to-python-2")
    @unittest.expectedFailure
    def test_fullmatch_hard(self):
        self._assert_expansion(".*?", "Hello", '\g<0>')

    def test_make_unique_filename(self):
        fname = '/dir/_NOT_EXISTS_'
        self.assertEqual(utils.make_unique_filename(fname), fname)

        os.chdir(os.path.dirname(__file__))
        fname = os.path.basename(__file__)
        bname, e = os.path.splitext(fname)
        fname1 = '%s1%s' % (bname, e)
        self.assertEqual(utils.make_unique_filename(fname), fname1)

        fname = os.path.join('..', 'tests', os.path.basename(__file__))
        fname2 = os.path.join('..', 'tests', fname1)
        self.assertEqual(utils.make_unique_filename(fname), fname2)

    @unittest.skipIf(sys.version_info < (3, 4), "TC.assertLog() not exists!")
    def test_LogWriter_smoke(self):
        logname = "foobar"
        log = logging.getLogger(logname)
        level = logging.INFO
        lw = LoggerWriter(log, level)
        with self.assertLogs(logname, level):
            lw.write('Hehe')
        lw.flush()
