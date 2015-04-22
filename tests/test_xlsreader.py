#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import logging
import os
import tempfile
import doctest
import unittest

import pandalone.xlsreader as xr
import pandas as pd
import xlrd as xd

DEFAULT_LOG_LEVEL = logging.INFO


def _init_logging(loglevel):
    logging.basicConfig(level=loglevel)
    logging.getLogger().setLevel(level=loglevel)

    log = logging.getLogger(__name__)
    log.trace = lambda *args, **kws: log.log(0, *args, **kws)

    return log


log = _init_logging(DEFAULT_LOG_LEVEL)


def from_my_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


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


class TestGetNoEmptyCells(unittest.TestCase):
    def test_get_no_empty_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = '/'.join([tmpdir, 'sample.xlsx'])
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5, 6, 7]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            url = '%s#%s!A1:C2[1,2]{"1":4,"2":"ciao"}' % (file_path, 'Sheet1')

            sheet = xr.url_parser(url)['xl_sheet']
            Cell = xr.Cell

            # minimum matrix in the sheet [:]
            args = (sheet, Cell(None, None), Cell(None, None))
            res = {
                0: {1: 0.0, 2: 1.0, 3: 2.0},
                1: {0: 0.0},
                2: {0: 1.0, 1: 5.0, 2: 6.0, 3: 7.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # get single value [D7]
            args = (sheet, Cell(3, 6))
            self.assertEqual(xr.get_no_empty_cells(*args), 0.0)

            # get single value [A1]
            args = (sheet, Cell(0, 0))
            self.assertEqual(xr.get_no_empty_cells(*args), None)

            # get whole column [D]
            args = (sheet, Cell(3, None))
            res = {6: 0.0, 7: 1.0}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # get whole row [6]
            args = (sheet, Cell(None, 5))
            res = {4: 0.0, 5: 1.0, 6: 2.0}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [E:]
            args = (sheet, Cell(4, None), Cell(None, None))
            res = {
                0: {0: 0.0, 1: 1.0, 2: 2.0},
                2: {0: 5.0, 1: 6.0, 2: 7.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [E7:]
            args = (sheet, Cell(4, 6), Cell(None, None))
            res = {
                1: {0: 5.0, 1: 6.0, 2: 7.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [D6:F8]
            args = (sheet, Cell(3, 5), Cell(5, 7))
            res = {
                0: {1: 0.0, 2: 1.0},
                1: {0: 0.0},
                2: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [:F8]
            args = (sheet, Cell(None, None), Cell(5, 7))
            res = {
                0: {1: 0.0, 2: 1.0},
                1: {0: 0.0},
                2: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [7:F8]
            args = (sheet, Cell(None, 6), Cell(5, 7))
            res = {
                0: {0: 0.0},
                1: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [E:F8]
            args = (sheet, Cell(None, 6), Cell(5, 7))
            res = {
                0: {0: 0.0},
                1: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited row in the sheet [C6:6]
            args = (sheet, Cell(2, 5), Cell(None, 5))
            res = {2: 0.0, 3: 1.0, 4: 2.0}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [A1:F8]
            args = (sheet, Cell(0, 0), Cell(5, 7))
            res = {
                5: {4: 0.0, 5: 1.0},
                6: {3: 0.0},
                7: {3: 1.0, 4: 5.0, 5: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [G9:]
            args = (sheet, Cell(6, 8), Cell(None, None))
            res = {}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [F9:]
            args = (sheet, Cell(5, 8), Cell(None, None))
            res = {}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

    def test_url_parser(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = '/'.join([tmpdir, 'sample.xlsx'])

            _make_sample_workbook(file_path,
                                  [[None, None, None], [5, 6, 7]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            wb = xd.open_workbook(file_path)

            Cell = xr.Cell

            url = '%s#%s!A1:C2' % (file_path, 'Sheet1')
            res = xr.url_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=2, row=1))

            url = '%s#%s!A1:C2[1,2]' % (file_path, 'Sheet1')
            res = xr.url_parser(url)
            self.assertEquals(res['json_args'], [1, 2])
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=2, row=1))

            url = '%s#%s!A1:C2{"1":4,"2":"ciao"}' % (file_path, 'Sheet1')
            res = xr.url_parser(url)
            self.assertEquals(res['json_kwargs'], {'2': 'ciao', '1': 4})
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))
            self.assertEquals(res['cell_down'], Cell(col=2, row=1))

            url = '%s#%s!A1[1,2]{"1":4,"2":"ciao"}' % (file_path, 'Sheet1')
            res = xr.url_parser(url)
            self.assertEquals(res['json_kwargs'], {'2': 'ciao', '1': 4})
            self.assertEquals(res['json_args'], [1, 2])
            self.assertEquals(res['cell_up'], Cell(col=0, row=0))

            url = '%s#%s!A' % (file_path, 'Sheet1')
            res = xr.url_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=0, row=None))

            url = '%s#%s!:' % (file_path, 'Sheet1')
            res = xr.url_parser(url)
            self.assertEquals(res['cell_up'], Cell(col=None, row=None))
            self.assertEquals(res['cell_down'], Cell(col=None, row=None))

            url = '%s#%s!:' % (file_path, 'Sheet1')
            res = xr.url_parser(url, xl_workbook=wb)
            self.assertEquals(res['xl_workbook'], wb)

            url = '%s#%s#!A1' % (file_path, 'Sheet1')
            self.assertRaises(ValueError, xr.url_parser, url)

            url = '%s#%s!A1::' % (file_path, 'Sheet1')
            self.assertRaises(ValueError, xr.url_parser, url)

            url = '%s#%s!A1:!' % (file_path, 'Sheet1')
            self.assertRaises(ValueError, xr.url_parser, url)

            url = '%s#%s![1,2,3]' % (file_path, 'Sheet1')
            self.assertRaises(ValueError, xr.url_parser, url)

    def test_cell_parser(self):
        Cell = xr.Cell

        self.assertEquals(xr.cell_parser('A1'), Cell(col=0, row=0))

        self.assertEquals(xr.cell_parser('A'), Cell(col=0, row=None))

        self.assertEquals(xr.cell_parser('1'), Cell(col=None, row=0))

        self.assertEquals(xr.cell_parser(''), Cell(col=None, row=None))

        self.assertRaises(ValueError, xr.cell_parser, '@')

        self.assertRaises(ValueError, xr.cell_parser, 'A1A')

        self.assertRaises(ValueError, xr.cell_parser, 'R1C1')

    def test_check_range(self):
        Cell = xr.Cell

        self.assertEquals(xr.check_range(Cell(1, 2), Cell(None, None)), None)
        self.assertRaises(ValueError, xr.check_range, *(Cell(None, None), None))
        self.assertRaises(ValueError, xr.check_range, *(Cell(1, 0), Cell(0, 1)))

    def test_col2num(self):
        self.assertEqual(xr.col2num('D'), 3)
        self.assertEqual(xr.col2num('AAA'), 702)
        self.assertEqual(xr.col2num(''), -1)

        self.assertRaises(TypeError, xr.col2num, 1)

        self.assertRaises(ValueError, xr.col2num, 'a')
        self.assertRaises(ValueError, xr.col2num, '@_')
