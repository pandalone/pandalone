#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

from contextlib import contextmanager
import doctest
from doit.cmd_base import ModuleTaskLoader
from doit.doit_cmd import DoitMain
from doit.reporter import JsonReporter
import pndltasks
import sys
import unittest

import six
import os


class TestDoctest(unittest.TestCase):

    def runTest(self):
        failure_count, test_count = doctest.testmod(pndltasks)
        self.assertEquals(failure_count, 0, (failure_count, test_count))


@contextmanager
def capture(command, *args, **kwargs):
    # Unused
    out, sys.stdout = sys.stdout, six.StringIO()
    err, sys.stderr = sys.stderr, six.StringIO()
    try:
        command(*args, **kwargs)
        sys.stdout.seek(0)
        yield (sys.stdout.getvalue(), sys.stderr.getvalue())
    finally:
        sys.stdout = out
        sys.stderr = err


class CaptureDodo(object):

    """
    Run doit capturing stdout/err in `captured` self.out. 

    :param reporter_opts: show_out
    """

    def run(self, cmdline, pndltasks=pndltasks):
        self.out = '<not_run>'
        outfile = six.StringIO()
        pndltasks.DOIT_CONFIG['reporter'] = JsonReporter(outfile)
        try:
            args = cmdline.split()
            DoitMain(ModuleTaskLoader(pndltasks)).run(args)
        finally:
            self.out = outfile.getvalue()


class TestMakeSamples(unittest.TestCase):

    def test_projects_folder(self):
        self.assertTrue(
            os.path.exists(pndltasks.SAMPLES_FOLDER), pndltasks.SAMPLES_FOLDER)

    def test_no_arg(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 makesam')
        self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)

    def test_target(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 makesam sometarg')
        self.assertIn('sometarg', cdodo.out, cdodo.out)
        self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)
        self.assertNotIn('sometarg.pndl', cdodo.out, cdodo.out)

    def test_sample_target(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 makesam --sample simple_rpw sometarg')
        self.assertIn('sometarg', cdodo.out, cdodo.out)
        self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)
        self.assertNotIn('sometarg.pndl', cdodo.out, cdodo.out)

    def test_multiple_targets(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 makesam t1 t2')
        self.assertIn('Too many', cdodo.out, cdodo.out)

    def test_bad_sample(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 makesam --sample bad_sample')
        self.assertIn('bad_sample', cdodo.out, cdodo.out)
