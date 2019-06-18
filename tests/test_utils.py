#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import doctest
import logging
import os
from pandalone.utils import LoggerWriter
import re
import sys
import unittest

import ddt

import pandalone.utils as utils


class Doctest(unittest.TestCase):
    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            utils, optionflags=doctest.NORMALIZE_WHITESPACE
        )
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEqual(failure_count, 0, (failure_count, test_count))


class TestUtils(unittest.TestCase):
    def test_make_unique_filename(self):
        fname = "/dir/_NOT_EXISTS_"
        self.assertEqual(utils.make_unique_filename(fname), fname)

        os.chdir(os.path.dirname(__file__))
        fname = os.path.basename(__file__)
        bname, e = os.path.splitext(fname)
        fname1 = "%s1%s" % (bname, e)
        self.assertEqual(utils.make_unique_filename(fname), fname1)

        fname = os.path.join("..", "tests", os.path.basename(__file__))
        fname2 = os.path.join("..", "tests", fname1)
        self.assertEqual(utils.make_unique_filename(fname), fname2)

    def test_LogWriter_smoke(self):
        logname = "foobar"
        log = logging.getLogger(logname)
        level = logging.INFO
        lw = LoggerWriter(log, level)
        with self.assertLogs(logname, level):
            lw.write("Hehe")
        lw.flush()


@unittest.skipIf(os.name != "nt", "Cannot test Windows paths.")
@ddt.ddt
class TPath2Url(unittest.TestCase):
    @ddt.data(
        ("foo", "file:%s/foo"),
        ("foo/", "file:%s/foo/"),
        ("foo/bar", "file:%s/foo/bar"),
        ("foo\\", "file:%s/foo/"),
        ("foo\\bar", "file:%s/foo/bar"),
        ("./foo", "file:%s/foo"),
        ("./foo/", "file:%s/foo/"),
        ("./foo/bar", "file:%s/foo/bar"),
        (".\\foo", "file:%s/foo"),
        (".\\foo\\", "file:%s/foo/"),
        (".\\foo/bar", "file:%s/foo/bar"),
    )
    def test_relative(self, case):
        from urllib.request import pathname2url

        path, url = case
        cwd = pathname2url(os.getcwd())
        self.assertEqual(utils.path2url(path), url % cwd, path)

    @ddt.data(
        ("/", "file:///"),
        ("/foo", "file:///foo"),
        ("/foo/", "file:///foo/"),
        ("/foo/bar", "file:///foo/bar"),
        ("\\", "file:///"),
        ("\\foo", "file:///foo"),
        ("\\foo\\", "file:///foo/"),
        ("\\foo\\bar", "file:///foo/bar"),
    )
    def test_absolute(self, case):
        path, url = case
        self.assertEqual(utils.path2url(path), url, path)

    @ddt.data(
        ("D:\\", "file:///D:/"),
        ("d:\\foo", "file:///d:/foo"),
        ("C:\\foo\\", "file:///C:/foo/"),
        ("c:\\foo\\bar", "file:///c:/foo/bar"),
        ("D:", "file:///D:/"),  # NOTE: destroy Windows per drive cwd!
        ("d:foo", "file:///d:/foo"),
        ("C:foo\\", "file:///C:/foo/"),
        ("c:foo\\bar", "file:///c:/foo/bar"),
    )
    def test_drive(self, case):
        path, url = case
        self.assertEqual(utils.path2url(path), url, path)

    @ddt.data(("d:../foo", "file:///foo"), ("D:../../foo", "file:///foo"))
    def test_corner_cases(self, case):
        path, url = case
        self.assertEqual(utils.path2url(path), url, path)

    @ddt.data(
        ("/%REL%", "file:///RRR"),
        ("/$REL", "file:///RRR"),
        ("/${REL}", "file:///RRR"),
        ("/%REL%/", "file:///RRR/"),
        ("/$REL/", "file:///RRR/"),
        ("/${REL}/", "file:///RRR/"),
        ("%ABS%", "file:///AAA"),
        ("$ABS", "file:///AAA"),
        ("${ABS}", "file:///AAA"),
        ("%ABS%/", "file:///AAA/"),
        ("$ABS/", "file:///AAA/"),
        ("${ABS}/", "file:///AAA/"),
    )
    def test_expansion_vars(self, case):
        path, url = case
        os.environ["REL"] = "RRR"
        os.environ["ABS"] = "/AAA"
        self.assertEqual(utils.path2url(path, 1, 1), url, path)

    @ddt.data(("~/a", "/a"), ("~\\a", "/a"), ("~/a/", "/a/"), ("~\\a\\", "/a/"))
    def test_expansion_user(self, case):
        path, end = case
        url = utils.path2url(path, 0, 1)
        self.assertFalse(url.startswith("~"), (url, path))
        self.assertTrue(url.endswith(end), (url, path))

        url = utils.path2url(path, 0, 0)
        # NO, turned into absolute!
        ##self.assertTrue(url.startswith('~'), (url, path))

    @ddt.data(
        ("http://ser/D:\\", "http://ser/D:/"),
        ("http://ser/d:\\foo", "http://ser/d:/foo"),
        ("http://ser/foo/", "http://ser/foo/"),
    )
    def test_remote(self, case):
        path, url = case
        self.assertEqual(utils.path2url(path), url, path)
