#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

from contextlib import contextmanager
import doctest
from doit.reporter import JsonReporter
import os
import sys
import tempfile
import unittest

import six

from pandalone import __main__, pndlcmd
from tests.assertutils import CustomAssertions


@unittest.skipIf(sys.version_info < (3, 3), "Doctests are made for py >= 3.3")
class TestDoctest(unittest.TestCase):

    @unittest.skip('No doctests in `pndl` cmd.')
    def runTest(self):
        failure_count, test_count = doctest.testmod(
            pndlcmd, optionflags=doctest.NORMALIZE_WHITESPACE)
        self.assertGreater(test_count, 0, (failure_count, test_count))
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


@contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)

mydir = os.path.dirname(__file__)  # @UnusedVariable


class CaptureDodo(object):

    """
    Run doit capturing stdout/err in `captured` self.out. 

    :param reporter_opts: show_out
    """

    def run(self, cmdline, pndlcmd=pndlcmd):
        self.out = '<not_run>'
        outfile = six.StringIO()
        pndlcmd.DOIT_CONFIG['reporter'] = JsonReporter(outfile)
        try:
            args = cmdline.split()
            __main__.main(args)
        finally:
            self.out = outfile.getvalue()


def_sample = '%s.pndl' % pndlcmd.opt_sample['default']
_doitdb_files = '.doit.db.dat'


class TestMakeSamples(unittest.TestCase, CustomAssertions):

    def assertGoodSample(self, targetdir, dodo_out, user_msg=None):
        if not user_msg:
            user_msg = dodo_out
        self.assertIn(targetdir, dodo_out, user_msg)
        self.assertFileExists(targetdir, user_msg)
        self.assertFileExists(os.path.join(targetdir, 'dodo.py'), user_msg)
        self.assertFileExists(os.path.join(targetdir, '.gitignore'), user_msg)
        #self.assertFileExists(_doitdb_files, user_msg)

    def test_projects_folder(self):
        self.assertFileExists(pndlcmd.SAMPLES_FOLDER)

    def test_no_arg(self):
        cdodo = CaptureDodo()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                targetdir = def_sample
                cdodo.run('-v 2 makesam')
                self.assertGoodSample(targetdir, cdodo.out)

    def test_no_sample_with_extension(self):
        cdodo = CaptureDodo()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                targetdir = def_sample
                cdodo.run('-v 2 makesam --sample %s' % def_sample)
                self.assertGoodSample(targetdir, cdodo.out)

    def test_target(self):
        cdodo = CaptureDodo()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                targetdir = 'sometarg'
                cdodo.run('-v 2 makesam %s' % targetdir)
                self.assertGoodSample(targetdir, cdodo.out)
                self.assertIn('simple_rpw.pndl', cdodo.out, cdodo.out)
                self.assertNotIn('%s.pndl' % targetdir, cdodo.out, cdodo.out)

    def test_sample_target(self):
        cdodo = CaptureDodo()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                targetdir = 'sometarg'
                cdodo.run('-v 2 makesam --sample %s %s' %
                          (def_sample, targetdir))
                self.assertGoodSample(targetdir, cdodo.out)
                self.assertIn(def_sample, cdodo.out, cdodo.out)
                self.assertNotIn('%s.pndl' % targetdir, cdodo.out, cdodo.out)

    def test_FAILS_multiple_targets(self):
        cdodo = CaptureDodo()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                cdodo.run('-v 2 makesam t1 t2')
                self.assertIn('Too many', cdodo.out, cdodo.out)
                self.assertFileNotExists('t1', cdodo.out)
                self.assertFileNotExists('t2', cdodo.out)
                # TODO: Restore testing of doit-db file
                #self.assertFileExists(_doitdb_files, cdodo.out)

    def test_FAILS_bad_sample(self):
        cdodo = CaptureDodo()
        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(tmpdir):
                cdodo.run('-v 2 makesam --sample bad_sample')
                self.assertIn('bad_sample', cdodo.out, cdodo.out)
                self.assertFileNotExists(def_sample, cdodo.out)
                #self.assertFileExists(_doitdb_files, cdodo.out)
