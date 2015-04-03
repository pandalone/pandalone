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
import dodo
from doit.cmd_base import ModuleTaskLoader
from doit.doit_cmd import DoitMain
from doit.reporter import JsonReporter
import sys
import unittest

import six

import dodo as mydodo


class TestDoctest(unittest.TestCase):

    def runTest(self):
        failure_count, test_count = doctest.testmod(dodo)
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

    def run(self, cmdline, dodo=mydodo):
        self.out = '<not_run>'
        outfile = six.StringIO()
        dodo.DOIT_CONFIG['reporter'] = JsonReporter(outfile)
        try:
            args = cmdline.split()
            DoitMain(ModuleTaskLoader(dodo)).run(args)
        finally:
            self.out = outfile.getvalue()


class TestCreateSamples(unittest.TestCase):

    def test_no_arg(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 createsam')
        self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)

    def test_target(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 createsam sometarg')
        self.assertIn('sometarg', cdodo.out, cdodo.out)
        self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)
        self.assertNotIn('sometarg.pndl', cdodo.out, cdodo.out)

    def test_sample_target(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 createsam --sample simple_rpw sometarg')
        self.assertIn('sometarg', cdodo.out, cdodo.out)
        self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)
        self.assertNotIn('sometarg.pndl', cdodo.out, cdodo.out)

    def test_multiple_targets(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 createsam t1 t2')
        self.assertIn('Too many', cdodo.out, cdodo.out)

    def test_bad_sample(self):
        cdodo = CaptureDodo()
        cdodo.run('-v 2 createsam --sample bad_sample')
        self.assertIn('bad_sample', cdodo.out, cdodo.out)
