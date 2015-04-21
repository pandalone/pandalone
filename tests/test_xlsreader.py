#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import doctest
import logging
import os
import tempfile
import unittest

import pandalone.xlsreader as xr
import pandas as pd
from tests.test_utils import chdir
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
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5, 6, 7]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            sheet = xr.open_xl_sheet(file_path, 'Sheet1')
            Cell = xr.Cell

            # minimum matrix in the sheet [:]
            args = (sheet,  Cell(None, None), Cell(None, None))
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
            args = (sheet,  Cell(4, None), Cell(None, None))
            res = {
                0: {0: 0.0, 1: 1.0, 2: 2.0},
                2: {0: 5.0, 1: 6.0, 2: 7.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [E7:]
            args = (sheet,  Cell(4, 6), Cell(None, None))
            res = {
                1: {0: 5.0, 1: 6.0, 2: 7.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [D6:F8]
            args = (sheet,  Cell(3, 5), Cell(5, 7))
            res = {
                0: {1: 0.0, 2: 1.0},
                1: {0: 0.0},
                2: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [:F8]
            args = (sheet,  Cell(None, None), Cell(5, 7))
            res = {
                0: {1: 0.0, 2: 1.0},
                1: {0: 0.0},
                2: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [7:F8]
            args = (sheet,  Cell(None, 6), Cell(5, 7))
            res = {
                0: {0: 0.0},
                1: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited matrix in the sheet [E:F8]
            args = (sheet,  Cell(None, 6), Cell(5, 7))
            res = {
                0: {0: 0.0},
                1: {0: 1.0, 1: 5.0, 2: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # minimum delimited row in the sheet [C6:6]
            args = (sheet,  Cell(2, 5), Cell(None, 5))
            res = {2: 0.0, 3: 1.0, 4: 2.0}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [A1:F8]
            args = (sheet,  Cell(0, 0), Cell(5, 7))
            res = {
                5: {4: 0.0, 5: 1.0},
                6: {3: 0.0},
                7: {3: 1.0, 4: 5.0, 5: 6.0}
            }
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [G9:]
            args = (sheet,  Cell(6, 8), Cell(None, None))
            res = {}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

            # delimited matrix in the sheet [F9:]
            args = (sheet,  Cell(5, 8), Cell(None, None))
            res = {}
            self.assertEqual(xr.get_no_empty_cells(*args), res)

    def test_open_xl_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = tmpdir + 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5, 6, 7]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            wb = xd.open_workbook(file_path)
            st = wb.sheet_by_name('Sheet1')

            args = (file_path, 'Sheet1')
            self.assertIsInstance(xr.open_xl_sheet(*args), xd.sheet.Sheet)

            args = (wb, 'Sheet1')
            self.assertIsInstance(xr.open_xl_sheet(*args), xd.sheet.Sheet)

            args = (wb, 0)
            self.assertIsInstance(xr.open_xl_sheet(*args), xd.sheet.Sheet)

            args = (st, )
            self.assertEqual(xr.open_xl_sheet(*args), st)

            args = (file_path, 'Sheet1', ':')
            self.assertRaises(TypeError, xr.open_xl_sheet, *args)

            args = (file_path, ('Sheet1', ))
            self.assertRaises(ValueError, xr.open_xl_sheet, *args)

            args = ((file_path,), 'Sheet1')
            self.assertRaises(ValueError, xr.open_xl_sheet, *args)

    def test_cells_parser(self):

        Cell = xr.Cell

        res = [Cell(col=0, row=0), Cell(col=1, row=1)]
        self.assertEqual(xr.cells_parser('A1', 'B2'), res)
        self.assertEqual(xr.cells_parser('a1', 'B2'), res)

        res = [Cell(col=0, row=0), None]
        self.assertEqual(xr.cells_parser('a1'), res)

        res = [Cell(col=1, row=None), None]
        self.assertEqual(xr.cells_parser((1, None)), res)

        res = [Cell(col=1, row=None), Cell(col=2, row=4)]
        self.assertEqual(xr.cells_parser('b', (2, 4)), res)

        res = [Cell(col=0, row=0), Cell(col=3, row=1)]
        self.assertEqual(xr.cells_parser('A1:D2'), res)

        res = [Cell(col=None, row=0), Cell(col=3, row=1)]
        self.assertEqual(xr.cells_parser('1:D2'), res)

        res = [Cell(col=0, row=None), Cell(col=None, row=1)]
        self.assertEqual(xr.cells_parser('A:2'), res)

        args = ('b1', 'a2')
        self.assertRaises(ValueError, xr.cells_parser, *args)

        args = ('b1:', 'a2')
        self.assertRaises(ValueError, xr.cells_parser, *args)

        args = ((1, -3))
        self.assertRaises(ValueError, xr.cells_parser, *args)

        args = ((None, None))
        self.assertRaises(ValueError, xr.cells_parser, *args)

        args = ((1.0, None))
        self.assertRaises(ValueError, xr.cells_parser, *args)

        args = ('b1', 'a2', (None, None))
        self.assertRaises(TypeError, xr.cells_parser, *args)

        self.assertRaises(TypeError, xr.cells_parser)

    def test_check_cell(self):
        self.assertTrue(xr.check_cell((1, 2)))
        self.assertTrue(xr.check_cell((1, None)))
        self.assertFalse(xr.check_cell((1, -1)))
        self.assertFalse(xr.check_cell((None, -3)))
        self.assertFalse(xr.check_cell((1, 1, 2)))
        self.assertFalse(xr.check_cell(('1', '2')))
        self.assertRaises(ValueError, xr.check_cell, 1)

    def test_str2cells_range(self):

        res = [(0, 0), None]
        self.assertEqual(xr.str2cells_range('a1'), res)

        res = [(0, 0), (None, None)]
        self.assertEqual(xr.str2cells_range('a1:'), res)

        res = [(None, None), (1, 1)]
        self.assertEqual(xr.str2cells_range(':b2'), res)

        res = [(0, 0), (1, 1)]
        self.assertEqual(xr.str2cells_range('a1:b2'), res)

        res = [(None, 0), (1, 1)]
        self.assertEqual(xr.str2cells_range('1:b2'), res)

        res = [(0, None), (1, 1)]
        self.assertEqual(xr.str2cells_range('a:b2'), res)

        res = [(None, None), (None, None)]
        self.assertEqual(xr.str2cells_range(':'), res)

        self.assertRaises(TypeError, xr.str2cells_range, 1)

        self.assertRaises(ValueError, xr.str2cells_range, '')
        self.assertRaises(ValueError, xr.str2cells_range, 'b:a')
        self.assertRaises(ValueError, xr.str2cells_range, 'a:b2:')
        self.assertRaises(ValueError, xr.str2cells_range, 'ab2a2')
        self.assertRaises(ValueError, xr.str2cells_range, 'a2a')
        self.assertRaises(ValueError, xr.str2cells_range, '@2')

    def test_col2num(self):
        self.assertEqual(xr.col2num('D'), 3)
        self.assertEqual(xr.col2num('AAA'), 702)
        self.assertEqual(xr.col2num(''), -1)

        self.assertRaises(TypeError, xr.col2num, 1)

        self.assertRaises(ValueError, xr.col2num, 'a')
        self.assertRaises(ValueError, xr.col2num, '@_')
