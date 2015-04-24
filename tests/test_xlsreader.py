#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import logging
import tempfile
import doctest
import unittest
import pandalone.xlsreader as xr
import pandas as pd
from tests.test_utils import chdir
import xlrd as xd
# noinspection PyUnresolvedReferences
from six.moves.urllib.parse import urlparse
# noinspection PyUnresolvedReferences
from six.moves.urllib.request import urlopen

DEFAULT_LOG_LEVEL = logging.INFO


def _init_logging(loglevel):
    logging.basicConfig(level=loglevel)
    logging.getLogger().setLevel(level=loglevel)

    log = logging.getLogger(__name__)
    log.trace = lambda *args, **kws: log.log(0, *args, **kws)

    return log


log = _init_logging(DEFAULT_LOG_LEVEL)


def _make_sample_workbook(path, matrix, sheet_name, startrow=0, startcol=0):
    df = pd.DataFrame(matrix)
    writer = pd.ExcelWriter(path)
    df.to_excel(writer, sheet_name, startrow=startrow, startcol=startcol)
    writer.save()


class TestDoctest(unittest.TestCase):

    def runTest(self):
        failure_count, test_count = doctest.testmod(
            xr, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestXlsReader(unittest.TestCase):

    def test_get_no_empty_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5, 6, 7]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            file_path = '/'.join([tmpdir, file_path])

            url = 'file:///%s#Sheet1!A1:C2[1,2]{"1":4,"2":"ciao"}' % file_path

            res = urlparse(url)

            wb = xd.open_workbook(file_contents=urlopen(url).read())

            sheet_name = xr.url_fragment_parser(res.fragment)['xl_sheet_name']
            sheet = wb.sheet_by_name(sheet_name)
            Cell = xr.Cell

            # minimum matrix in the sheet [:]
            args = (sheet, Cell(None, None), Cell(None, None))
            res = [
                [None, 0.0, 1.0, 2.0],
                [0.0, None, None, None],
                [1.0, 5.0, 6.0, 7.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # get single value [D7]
            args = (sheet, Cell(3, 6))
            self.assertEqual(xr.get_range(*args), 0.0)

            # get single value [A1]
            args = (sheet, Cell(0, 0))
            self.assertEqual(xr.get_range(*args), None)

            # get whole column [D]
            args = (sheet, Cell(3, None))
            res = [None, None, None, None, None, None, 0.0, 1.0]
            self.assertEqual(xr.get_range(*args), res)

            # get whole row [6]
            args = (sheet, Cell(None, 5))
            res = [None, None, None, None, 0.0, 1.0, 2.0]
            self.assertEqual(xr.get_range(*args), res)

            # minimum delimited matrix in the sheet [E:]
            args = (sheet, Cell(4, None), Cell(None, None))
            res = [
                [0.0, 1.0, 2.0],
                [None, None, None],
                [5.0, 6.0, 7.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # minimum delimited matrix in the sheet [E7:]
            args = (sheet, Cell(4, 6), Cell(None, None))
            res = [
                [None, None, None],
                [5.0, 6.0, 7.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # delimited matrix in the sheet [D6:F8]
            args = (sheet, Cell(3, 5), Cell(5, 7))
            res = [
                [None, 0.0, 1.0],
                [0.0, None, None],
                [1.0, 5.0, 6.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # minimum delimited matrix in the sheet [:F8]
            args = (sheet, Cell(None, None), Cell(5, 7))
            res = [
                [None, 0.0, 1.0],
                [0.0, None, None],
                [1.0, 5.0, 6.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # minimum delimited matrix in the sheet [7:F8]
            args = (sheet, Cell(None, 6), Cell(5, 7))
            res = [
                [0.0, None, None],
                [1.0, 5.0, 6.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # minimum delimited matrix in the sheet [E:F8]
            args = (sheet, Cell(None, 6), Cell(5, 7))
            res = [
                [0.0, None, None],
                [1.0, 5.0, 6.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # minimum delimited row in the sheet [C6:6]
            args = (sheet, Cell(2, 5), Cell(None, 5))
            res = [None, None, 0.0, 1.0, 2.0]
            self.assertEqual(xr.get_range(*args), res)

            # delimited matrix in the sheet [A1:F8]
            args = (sheet, Cell(0, 0), Cell(5, 7))
            res = [
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, 0.0, 1.0],
                [None, None, None, 0.0, None, None],
                [None, None, None, 1.0, 5.0, 6.0]
            ]
            self.assertEqual(xr.get_range(*args), res)

            # delimited matrix in the sheet [G9:]
            args = (sheet, Cell(6, 8), Cell(None, None))
            res = [[None]]
            self.assertEqual(xr.get_range(*args), res)

            # delimited matrix in the sheet [F9:]
            args = (sheet, Cell(5, 8), Cell(None, None))
            res = [[None]]
            self.assertEqual(xr.get_range(*args), res)

    def test_url_fragment_parser(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5, 6, 7]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            Cell = xr.Cell

            url = 'Sheet1!A1:C2'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=2, row=1))

            url = 'Sheet1!A1:C2[1,2]'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['json_args'], [1, 2])
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=2, row=1))

            url = 'Sheet1!A1:C2{"1":4,"2":"ciao"}'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['json_kwargs'], {'2': 'ciao', '1': 4})
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=2, row=1))

            url = 'Sheet1!A1[1,2]{"1":4,"2":"ciao"}'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['json_kwargs'], {'2': 'ciao', '1': 4})
            self.assertEquals(res['json_args'], [1, 2])
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))

            url = 'Sheet1!A_'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=0, row=None))

            url = 'Sheet1![1,2,3]'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=None, row=None))

            url = ':[1,2,3]'
            res = xr.url_fragment_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=None, row=None))
            self.assertEquals(res['cell_down'], Cell(col=None, row=None))


            self.assertRaises(ValueError, xr.url_fragment_parser, 's![[]')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's![[]{]}')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!{}[]')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!{}[]')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!A')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!A1:!')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!1:2')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!B3:A1')
            self.assertRaises(ValueError, xr.url_fragment_parser, 's!B3:A0')

    def test_fetch_cell(self):
        Cell = xr.Cell

        self.assertEquals(xr.fetch_cell('A1', 'A', '1'), Cell(col=0, row=0))

        self.assertEquals(xr.fetch_cell('A_', 'A', '_'), Cell(col=0, row=None))

        self.assertEquals(xr.fetch_cell('_1', '_','1'), Cell(col=None, row=0))

        self.assertEquals(xr.fetch_cell('__', '_', '_'), Cell(col=None, row=None))

        self.assertRaises(ValueError, xr.fetch_cell, *('_0', '_', '0'))

    def test_col2num(self):
        self.assertEqual(xr.col2num('D'), 3)
        self.assertEqual(xr.col2num('AAA'), 702)
