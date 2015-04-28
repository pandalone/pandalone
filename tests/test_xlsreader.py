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
from datetime import datetime
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

    def test_single_value_get_rect_range(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5.1, 6.1, 7.1]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])

            # get single value [D7]
            args = (sheet, xr.Cell(3, 6))
            self.assertEqual(xr.get_rect_range(*args), 0)

            # get single value [A1]
            args = (sheet, xr.Cell(0, 0))
            self.assertEqual(xr.get_rect_range(*args), None)

            # get single value [H9]
            args = (sheet, xr.Cell(7, 8))
            self.assertEqual(xr.get_rect_range(*args), None)

    def test_vector_get_rect_range(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5.1, 6.1, 7.1]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])

            # single value in the sheet [D7:D7]
            args = (sheet, xr.Cell(3, 6), xr.Cell(3, 6))
            self.assertEqual(xr.get_rect_range(*args), [0])

            # get whole column [D_]
            args = (sheet, xr.Cell(3, None))
            res = [None, None, None, None, None, None, 0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # get whole row [_6]
            args = (sheet, xr.Cell(None, 5))
            res = [None, None, None, None, 0, 1, 2]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [C6:_6]
            args = (sheet, xr.Cell(2, 5), xr.Cell(None, 5))
            res = [None, None, 0, 1, 2]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [_7:_7]
            args = (sheet, xr.Cell(None, 6), xr.Cell(None, 6))
            res = [0]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [A7:_7]
            args = (sheet, xr.Cell(0, 6), xr.Cell(None, 6))
            res = [None, None, None, 0]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited row in the sheet [A7:D7]
            args = (sheet, xr.Cell(0, 6), xr.Cell(3, 6))
            res = [None, None, None, 0]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D_]
            args = (sheet, xr.Cell(3, None), xr.Cell(3, None))
            res = [0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D5:D_]
            args = (sheet, xr.Cell(3, 4), xr.Cell(3, None))
            res = [None, None, 0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D9]
            args = (sheet, xr.Cell(3, None), xr.Cell(3, 8))
            res = [0, 1, None]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited column in the sheet [D3:D9]
            args = (sheet, xr.Cell(3, 2), xr.Cell(3, 8))
            res = [None, None, None, None, 0, 1, None]
            self.assertEqual(xr.get_rect_range(*args), res)

    def test_matrix_get_rect_range(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[None, None, None], [5.1, 6.1, 7.1]],
                                  'Sheet1',
                                  startrow=5, startcol=3)


            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])

            # minimum matrix in the sheet [:]
            args = (sheet, xr.Cell(None, None), xr.Cell(None, None))
            res = [
                [None, 0, 1, 2],
                [0, None, None, None],
                [1, 5.1, 6.1, 7.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E_:__]
            args = (sheet, xr.Cell(4, None), xr.Cell(None, None))
            res = [
                [0, 1, 2],
                [None, None, None],
                [5.1, 6.1, 7.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E7:__]
            args = (sheet, xr.Cell(4, 6), xr.Cell(None, None))
            res = [
                [None, None, None],
                [5.1, 6.1, 7.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [D6:F8]
            args = (sheet, xr.Cell(3, 5), xr.Cell(5, 7))
            res = [
                [None, 0, 1],
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [:F8]
            args = (sheet, xr.Cell(None, None), xr.Cell(5, 7))
            res = [
                [None, 0, 1],
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [7:F8]
            args = (sheet, xr.Cell(None, 6), xr.Cell(5, 7))
            res = [
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E:F8]
            args = (sheet, xr.Cell(None, 6), xr.Cell(5, 7))
            res = [
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [A1:F8]
            args = (sheet, xr.Cell(0, 0), xr.Cell(5, 7))
            res = [
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                [None, None, None, None, 0, 1],
                [None, None, None, 0, None, None],
                [None, None, None, 1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [G9:__]
            args = (sheet, xr.Cell(6, 8), xr.Cell(None, None))
            res = [[None]]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [F9:__]
            args = (sheet, xr.Cell(5, 8), xr.Cell(None, None))
            res = [[None]]
            self.assertEqual(xr.get_rect_range(*args), res)

    def test_basic_parse_xl_ref(self):
        xl_ref = 'Sheet1!A1:C2'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['xl_sheet_name'], 'Sheet1')
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))
        self.assertEquals(res['cell_down'], xr.Cell(col=2, row=1))

        xl_ref = 'Sheet1!A1'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))

        xl_ref = 'Sheet1!a1:c2{"1":4,"2":"ciao"}'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['json'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))
        self.assertEquals(res['cell_down'], xr.Cell(col=2, row=1))

    def test_autocomplete_ref_parse_xl_ref(self):
        xl_ref = 'Sheet1!a1:**'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))
        self.assertEquals(res['cell_down'], xr.Cell(col=None, row=None))

        xl_ref = 'Sheet1!*_:c2'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=None, row='xl_margin'))
        self.assertEquals(res['cell_down'], xr.Cell(col=2, row=1))

        xl_ref = 'Sheet1!A*'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=None))

    def test_short_cuts_parse_xl_ref(self):

        xl_ref = 'Sheet1!a1:'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))
        self.assertEquals(res['cell_down'], xr.Cell(col=None, row=None))

        xl_ref = 'Sheet1!:c2'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=None, row=None))
        self.assertEquals(res['cell_down'], xr.Cell(col=2, row=1))

        xl_ref = 'Sheet1!'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))
        self.assertEquals(res['cell_down'], xr.Cell(col=None, row=None))

        xl_ref = 'Sheet1!{"1": [1, 2], "2": "ciao", "3": {"1": [1, 2, 3]}}'
        res = xr.parse_xl_ref(xl_ref)
        r = {
            "1": [1, 2],
            "2": "ciao",
            "3": {
                "1": [1, 2, 3]
            }
        }
        self.assertEquals(res['cell_up'], xr.Cell(col=0, row=0))
        self.assertEquals(res['cell_down'], xr.Cell(col=None, row=None))
        self.assertEquals(res['json'], r)

    def test_errors_parse_xl_ref(self):
        self.assertRaises(ValueError, xr.parse_xl_ref, 's![[]')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!{}[]')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A1:!')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!1:2')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!B3:A1')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A0:B1')

    def test_fetch_cell_ref(self):
        self.assertEquals(xr.fetch_cell_ref('A1', 'A', '1'),
                          xr.Cell(col=0, row=0))
        self.assertEquals(xr.fetch_cell_ref('A_', 'A', '_'),
                          xr.Cell(col=0, row='xl_margin'))
        self.assertEquals(xr.fetch_cell_ref('A*', 'A', '*'),
                          xr.Cell(col=0, row=None))
        self.assertEquals(xr.fetch_cell_ref('_1', '_','1'),
                          xr.Cell(col='xl_margin', row=0))
        self.assertEquals(xr.fetch_cell_ref('*1', '*','1'),
                          xr.Cell(col=None, row=0))
        self.assertEquals(xr.fetch_cell_ref('__', '_', '_'),
                          xr.Cell(col='xl_margin', row='xl_margin'))
        self.assertEquals(xr.fetch_cell_ref('**', '*', '*'),
                          xr.Cell(col=None, row=None))

        self.assertRaises(ValueError, xr.fetch_cell_ref, *('_0', '_', '0'))
        self.assertRaises(ValueError, xr.fetch_cell_ref, *('@@', '@', '@'))

    def test_col2num(self):
        self.assertEqual(xr.col2num('D'), 3)
        self.assertEqual(xr.col2num('aAa'), 702)

    def test_parse_xl_url(self):
        url = 'file://path/to/file.xls#xl_sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)

        self.assertEquals(res['url_file'], 'file://path/to/file.xls')
        self.assertEquals(res['xl_sheet_name'], 'xl_sheet_name')
        self.assertEquals(res['json'], {"json": "..."})
        self.assertEquals(res['cell_up'], xr.Cell(col=561, row=9))
        self.assertEquals(res['cell_down'], xr.Cell(col=81055, row=19))

        self.assertRaises(ValueError, xr.parse_xl_url, *('#!:{"json":"..."', ))
        url = '#xl_sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)
        self.assertEquals(res['url_file'], '')

    def test_parse_cell(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            xl = [datetime(1900, 8, 2), True, None, u'', 'hi', 1.4, 5.0]
            _make_sample_workbook(file_path,
                                  xl,
                                  'Sheet1')

            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])


            # row vector in the sheet [B2:B_]
            args = (sheet, xr.Cell(1, 1), xr.Cell(1, None), wb.datemode)
            res = [datetime(1900, 8, 2), True, None, None, 'hi', 1.4, 5]
            self.assertEqual(xr.get_rect_range(*args), res)

    def test_comparison_vs_pandas_parse_cell(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            xl = [datetime(1900, 8, 2), True, None, u'', 'hi', 1.4, 5.0]
            _make_sample_workbook(file_path,
                                  xl,
                                  'Sheet1')

            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])

            # row vector in the sheet [B2:B_]
            args = (sheet, xr.Cell(1, 1), xr.Cell(1, None), wb.datemode)

            res = xr.get_rect_range(*args)

            df = pd.read_excel(file_path, 'Sheet1')[0]

            # in pandas None values are converted in float('nan')
            df = df.where(pd.notnull(df), None).values.tolist()

            self.assertEqual(df, res)

    def test_xlwings_vs_get_rect_range(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            _make_sample_workbook(file_path,
                                  [[1, 2, None], [None, 6.1, 7.1]],
                                  'Sheet1',
                                  startrow=5, startcol=3)

            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])

            # load Workbook for --> xlwings
            from xlwings import Workbook, Range
            wb = Workbook('/'.join([tmpdir, file_path]))
            res = {}
            res[0] = Range("Sheet1", "D7").vertical.value
            res[1] = Range("Sheet1", "E6").vertical.value
            res[2] = Range("Sheet1", "E6").horizontal.value
            res[3] = Range("Sheet1", "E6").table.value
            res[4] = Range("Sheet1", "D6:F8").value
            res[5] = Range("Sheet1", "A1:F8").value
            res[6] = Range("Sheet1", "A7:D7").value
            res[7] = Range("Sheet1", "D3:D9").value
            wb.close()

            # minimum delimited column in the sheet [D7:D_]
            args = (sheet, xr.Cell(3, 6), xr.Cell(3, None), False, True)
            self.assertEqual(xr.get_rect_range(*args), res[0])

            # minimum delimited column in the sheet [E6:E_]
            args = (sheet, xr.Cell(4, 5), xr.Cell(4, None), False, True)
            self.assertEqual(xr.get_rect_range(*args), res[1])

            # minimum delimited row in the sheet [E6:_6]
            args = (sheet, xr.Cell(4, 5), xr.Cell(None, 5), False, True)
            self.assertEqual(xr.get_rect_range(*args), res[2])

            # minimum delimited matrix in the sheet [E6:__]
            args = (sheet, xr.Cell(4, 5), xr.Cell(None, None), False, True)
            self.assertEqual(xr.get_rect_range(*args), res[3])

            # delimited matrix in the sheet [D6:F8]
            args = (sheet, xr.Cell(3, 5), xr.Cell(5, 7))
            self.assertEqual(xr.get_rect_range(*args), res[4])

            # delimited matrix in the sheet [A1:F8]
            args = (sheet, xr.Cell(0, 0), xr.Cell(5, 7))
            self.assertEqual(xr.get_rect_range(*args), res[5])

            # delimited row in the sheet [A7:D7]
            args = (sheet, xr.Cell(0, 6), xr.Cell(3, 6))
            self.assertEqual(xr.get_rect_range(*args), res[6])

            # delimited column in the sheet [D3:D9]
            args = (sheet, xr.Cell(3, 2), xr.Cell(3, 8))
            self.assertEqual(xr.get_rect_range(*args), res[7])

            # minimum delimited matrix in the sheet [__:F7]
            args = (sheet, xr.Cell(None, None), xr.Cell(5, 6), False, True)
            res = [[None, 0, 1],
                    [0, 1, 2]]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [_8:G8]
            args = (sheet, xr.Cell(None, 7), xr.Cell(6, 7), False, True)
            res = [6.1, 7.1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D8]
            args = (sheet, xr.Cell(3, None), xr.Cell(3, 7), False, True)
            res = [0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)