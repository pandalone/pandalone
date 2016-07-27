#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import doctest
import os
import re
import subprocess
import sys
import unittest

import pandalone

import os.path as osp
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch  # @UnusedImport


mydir = osp.dirname(__file__)
proj_path = osp.normpath(osp.join(mydir, '..'))
readme_path = osp.join(proj_path, 'README.rst')
tutorial_path = osp.join(proj_path, 'doc', 'tutorial.rst')


@unittest.skipIf(sys.version_info < (3, 4), "Doctests are made for py >= 3.3")
class Doctest(unittest.TestCase):

    def test_doctest_README(self):
        failure_count, test_count = doctest.testfile(
            readme_path, module_relative=False,
            optionflags=doctest.NORMALIZE_WHITESPACE)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))

    def test_README_version_opening(self):
        ver = pandalone.__version__
        header_len = 20
        mydir = osp.dirname(__file__)
        with open(readme_path) as fd:
            for i, l in enumerate(fd):
                if ver in l:
                    break
                elif i >= header_len:
                    msg = "Version(%s) not found in README %s header-lines!"
                    raise AssertionError(msg % (ver, header_len))

    def test_README_reldate_opening(self):
        reldate = pandalone.__updated__
        header_len = 20
        mydir = osp.dirname(__file__)
        with open(readme_path) as fd:
            for _, l in zip(range(header_len), fd):
                if reldate in l:
                    break
            else:
                msg = "Version(%s) not found in README first %s header-lines!"
                raise AssertionError(msg % (reldate, header_len))

    def test_README_version_cmdline(self):
        ver = pandalone.__version__
        mydir = osp.dirname(__file__)
        with open(readme_path) as fd:
            ftext = fd.read()
            m = re.search(
                r'pndlcmd --version\s+%s' % ver, ftext, re.MULTILINE | re.IGNORECASE)
            self.assertIsNotNone(m,
                                 "Version(%s) not found in README cmd-line version-check!" %
                                 ver)

    def test_README_as_PyPi_landing_page(self):
        from docutils import core as dcore

        long_desc = subprocess.check_output(
            'python setup.py --long-description'.split(),
            cwd=proj_path)
        self.assertIsNotNone(long_desc, 'Long_desc is null!')

        with patch('sys.exit'):
            dcore.publish_string(long_desc, enable_exit_status=False,
                                 settings_overrides={  # see `docutils.frontend` for more.
                                     'halt_level': 2  # 2=WARN, 1=INFO
                                 })
