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
from tests.test_pndlcmd import chdir
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
            args = (sheet, xr.StartPos(3, 6))
            self.assertEqual(xr.get_rect_range(*args), 0)

            # get single value [A1]
            args = (sheet, xr.StartPos(0, 0))
            self.assertEqual(xr.get_rect_range(*args), None)

            # get single value [H9]
            args = (sheet, xr.StartPos(7, 8))
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
            args = (sheet, xr.StartPos(3, 6), xr.StartPos(3, 6))
            self.assertEqual(xr.get_rect_range(*args), [0])

            # get whole column [D_]
            args = (sheet, xr.StartPos(3, None))
            res = [None, None, None, None, None, None, 0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # get whole row [_6]
            args = (sheet, xr.StartPos(None, 5))
            res = [None, None, None, None, 0, 1, 2]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [C6:_6]
            args = (sheet, xr.StartPos(2, 5), xr.StartPos(None, 5))
            res = [None, None, 0, 1, 2]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [_7:_7]
            args = (sheet, xr.StartPos(None, 6), xr.StartPos(None, 6))
            res = [0]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited row in the sheet [A7:_7]
            args = (sheet, xr.StartPos(0, 6), xr.StartPos(None, 6))
            res = [None, None, None, 0]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited row in the sheet [A7:D7]
            args = (sheet, xr.StartPos(0, 6), xr.StartPos(3, 6))
            res = [None, None, None, 0]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D_]
            args = (sheet, xr.StartPos(3, None), xr.StartPos(3, None))
            res = [0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D5:D_]
            args = (sheet, xr.StartPos(3, 4), xr.StartPos(3, None))
            res = [None, None, 0, 1]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D9]
            args = (sheet, xr.StartPos(3, None), xr.StartPos(3, 8))
            res = [0, 1, None]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited column in the sheet [D3:D9]
            args = (sheet, xr.StartPos(3, 2), xr.StartPos(3, 8))
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
            args = (sheet, xr.StartPos(None, None), xr.StartPos(None, None))
            res = [
                [None, 0, 1, 2],
                [0, None, None, None],
                [1, 5.1, 6.1, 7.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E_:__]
            args = (sheet, xr.StartPos(4, None), xr.StartPos(None, None))
            res = [
                [0, 1, 2],
                [None, None, None],
                [5.1, 6.1, 7.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E7:__]
            args = (sheet, xr.StartPos(4, 6), xr.StartPos(None, None))
            res = [
                [None, None, None],
                [5.1, 6.1, 7.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [D6:F8]
            args = (sheet, xr.StartPos(3, 5), xr.StartPos(5, 7))
            res = [
                [None, 0, 1],
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [:F8]
            args = (sheet, xr.StartPos(None, None), xr.StartPos(5, 7))
            res = [
                [None, 0, 1],
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [7:F8]
            args = (sheet, xr.StartPos(None, 6), xr.StartPos(5, 7))
            res = [
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E:F8]
            args = (sheet, xr.StartPos(None, 6), xr.StartPos(5, 7))
            res = [
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [A1:F8]
            args = (sheet, xr.StartPos(0, 0), xr.StartPos(5, 7))
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
            args = (sheet, xr.StartPos(6, 8), xr.StartPos(None, None))
            res = [[None]]
            self.assertEqual(xr.get_rect_range(*args), res)

            # delimited matrix in the sheet [F9:__]
            args = (sheet, xr.StartPos(5, 8), xr.StartPos(None, None))
            res = [[None]]
            self.assertEqual(xr.get_rect_range(*args), res)

    def test_basic_parse_xl_ref(self):
        xl_ref = 'Sheet1!A1(L):C2(UL)'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['xl_sheet_name'], 'Sheet1')
        self.assertEquals(res['st_cell'].cell, xr.Cell(col=0, row=0))
        self.assertEquals(res['nd_cell'].cell, xr.Cell(col=2, row=1))
        self.assertEquals(res['st_cell'].mov, 'L')
        self.assertEquals(res['nd_cell'].mov, 'UL')

        xl_ref = 'Sheet1!A1'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['st_cell'].cell, xr.Cell(col=0, row=0))
        self.assertEquals(res['nd_cell'], None)

        xl_ref = 'Sheet1!a1(l):c2(ul){"1":4,"2":"ciao"}'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['json'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['st_cell'].cell, xr.Cell(col=0, row=0))
        self.assertEquals(res['nd_cell'].cell, xr.Cell(col=2, row=1))
        self.assertEquals(res['st_cell'].mov, 'L')
        self.assertEquals(res['nd_cell'].mov, 'UL')

    def test_errors_parse_xl_ref(self):
        self.assertRaises(ValueError, xr.parse_xl_ref, 's![[]')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!{}[]')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A1:!')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!1:2')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A0:B1')

    def test_fetch_cell_ref(self):
        self.assertEquals(xr.fetch_cell_ref('A', '1', 'L'),
                          xr.StartPos(xr.Cell(row=0, col=0), 'L'))
        self.assertEquals(xr.fetch_cell_ref('A', '_', 'D'),
                          xr.StartPos(xr.Cell(xr.XL_BOTTOM_ABS, 0), 'D'))
        self.assertEquals(xr.fetch_cell_ref('_', '1', None),
                          xr.StartPos(xr.Cell(0, xr.XL_BOTTOM_ABS), None))
        self.assertEquals(xr.fetch_cell_ref('_', '_', None),
                          xr.StartPos(xr.Cell(xr.XL_BOTTOM_ABS,
                                              xr.XL_BOTTOM_ABS), None))
        self.assertEquals(xr.fetch_cell_ref('A', '^', 'D'),
                          xr.StartPos(xr.Cell(xr.XL_UP_ABS, 0), 'D'))
        self.assertEquals(xr.fetch_cell_ref('^', '1', None),
                          xr.StartPos(xr.Cell(0, xr.XL_UP_ABS), None))
        self.assertEquals(xr.fetch_cell_ref('^', '^', None),
                          xr.StartPos(xr.Cell(xr.XL_UP_ABS,
                                              xr.XL_UP_ABS), None))

        self.assertRaises(ValueError, xr.fetch_cell_ref, *('_0', '_', '0'))
        self.assertRaises(ValueError, xr.fetch_cell_ref, *('@@', '@', '@'))

    def test_col2num(self):
        self.assertEqual(xr.col2num('D'), 3)
        self.assertEqual(xr.col2num('aAa'), 702)

    def test_parse_xl_url(self):
        url = 'file://path/to/file.xls#Sheet1!U10(L):D20(D){"json":"..."}'
        res = xr.parse_xl_url(url)

        self.assertEquals(res['url_file'], 'file://path/to/file.xls')
        self.assertEquals(res['xl_sheet_name'], 'Sheet1')
        self.assertEquals(res['json'], {"json": "..."})
        self.assertEquals(res['st_cell'], xr.StartPos(xr.Cell(9, 20), 'L'))
        self.assertEquals(res['nd_cell'], xr.StartPos(xr.Cell(19, 3), 'D'))

        self.assertRaises(ValueError, xr.parse_xl_url, *('#!:{"json":"..."', ))
        url = '#xl_sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)
        self.assertEquals(res['url_file'], '')

    def test_parse_cell(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            file_path = 'sample.xlsx'
            xl = [datetime(1900, 8, 2), True, None, u'', 'hi', 1.4, 5.0]
            _make_sample_workbook(file_path, xl, 'Sheet1')

            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url = 'file://%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % url
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['xl_sheet_name'])

            indices = xr.get_xl_abs_margins(xr.get_no_empty_cells(sheet))[1]
            # row vector in the sheet [B2:B_]
            args = (sheet, (xr.Cell(1, 1), xr.Cell(7, 1)), indices, wb.datemode)
            res = [datetime(1900, 8, 2), True, None, None, 'hi', 1.4, 5]
            self.assertEqual(xr.get_xl_table(*args), res)

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

            indices = xr.get_xl_abs_margins(xr.get_no_empty_cells(sheet))[1]
            # row vector in the sheet [B2:B_]
            args = (sheet, (xr.Cell(1, 1), xr.Cell(7, 1)), indices, wb.datemode)

            res = xr.get_xl_table(*args)

            df = pd.read_excel(file_path, 'Sheet1')[0]

            # in pandas None values are converted in float('nan')
            df = df.where(pd.notnull(df), None).values.tolist()

            self.assertEqual(df, res)

    def test_xlwings_vs_get_xl_table(self):
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
            datemode = wb.datemode
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



            no_empty, up, dn = xr.get_xl_margins(sheet)
            xl_ma, ind = xr.get_xl_abs_margins(no_empty)

            # minimum delimited column in the sheet [D7:D.(D)]
            st = xr.StartPos(xr.Cell(6, 3), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, xr.CELL_RELATIVE), 'D')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[0])

            # minimum delimited column in the sheet [E6:E.(D)]
            st = xr.StartPos(xr.Cell(5, 4), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, xr.CELL_RELATIVE), 'D')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[1])

            # minimum delimited row in the sheet [E6:.6(R)]
            st = xr.StartPos(xr.Cell(5, 4), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, xr.CELL_RELATIVE), 'R')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[2])

            # minimum delimited matrix in the sheet [E6:..(RD)]
            st = xr.StartPos(xr.Cell(5, 4), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, xr.CELL_RELATIVE), 'RD')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[3])

            st = xr.StartPos(xr.Cell(5, 4), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, xr.CELL_RELATIVE), 'DR')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[3])

            # delimited matrix in the sheet [D6:F8]
            st = xr.StartPos(xr.Cell(7, 5), None)
            nd = xr.StartPos(xr.Cell(5, 3), None)
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[4])

            # delimited matrix in the sheet [A1:F8]
            st = xr.StartPos(xr.Cell(7, 5), None)
            nd = xr.StartPos(xr.Cell(0, 0), None)
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[5])

            # delimited row in the sheet [A7:D7]
            st = xr.StartPos(xr.Cell(6, 0), None)
            nd = xr.StartPos(xr.Cell(6, 3), None)
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[6])

            # delimited column in the sheet [D3:D9]
            st = xr.StartPos(xr.Cell(8, 3), None)
            nd = xr.StartPos(xr.Cell(2, 3), None)
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.get_xl_table(*args), res[7])

            # minimum delimited matrix in the sheet [F7:..(UL)]
            st = xr.StartPos(xr.Cell(6, 5), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, xr.CELL_RELATIVE), 'UL')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [[None, 0, 1],
                   [0, 1, 2]]
            self.assertEqual(xr.get_xl_table(*args), res)

            # minimum delimited matrix in the sheet [F7:F7:LURD]
            st = xr.StartPos(xr.Cell(6, 5), None)
            nd = xr.StartPos(xr.Cell(6, 5), None)
            rng_ext = xr.parse_rng_ext('LURD')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd, rng_ext)
            args = (sheet, rng, ind, datemode)
            res = [[None, 0, 1, 2],
                   [0, 1, 2, None],
                   [1, None, 6.1, 7.1]]
            self.assertEqual(xr.get_xl_table(*args), res)

            # minimum delimited matrix in the sheet [F7:A1(RD)]
            st = xr.StartPos(xr.Cell(6, 5), None)
            nd = xr.StartPos(xr.Cell(0, 0), 'RD')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [[0, 1],
                    [1, 2]]
            self.assertEqual(xr.get_xl_table(*args), res)

            # minimum delimited row in the sheet [_8:G8]
            st = xr.StartPos(xr.Cell(7, 6), None)
            nd = xr.StartPos(xr.Cell(7, xr.CELL_RELATIVE), 'L')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [6.1, 7.1]
            self.assertEqual(xr.get_xl_table(*args), res)

            # minimum delimited column in the sheet [D_:D8]
            st = xr.StartPos(xr.Cell(7, 3), None)
            nd = xr.StartPos(xr.Cell(xr.CELL_RELATIVE, 3), 'U')
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [0, 1]
            self.assertEqual(xr.get_xl_table(*args), res)

            # single value [D8]
            st = xr.StartPos(xr.Cell(7, 3), None)
            nd = None
            rng = xr.get_range(no_empty, up, dn, xl_ma, ind, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [1]
            self.assertEqual(xr.get_xl_table(*args), res)

    def test_open_xl_workbook(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            df = pd.DataFrame()
            file_path = 'sample.xlsx'
            writer = pd.ExcelWriter(file_path)
            df.to_excel(writer, 'Sheet1')
            writer.save()
            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url_parent = 'file://%s#Sheet1!A1' % url
            xl_ref_parent = xr.parse_xl_url(url_parent)
            xr.open_xl_workbook(xl_ref_parent)

            url_child = '#A1:B2'
            xl_ref_child = xr.parse_xl_url(url_child)

            self.assertRaises(ValueError, xr.open_xl_workbook, *(xl_ref_child,))

            xr.open_xl_workbook(xl_ref_child, xl_ref_parent)

            self.assertEquals(xl_ref_child['xl_workbook'],
                              xl_ref_parent['xl_workbook'])

    def test_open_xl_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
            df = pd.DataFrame()
            file_path = 'sample.xlsx'
            writer = pd.ExcelWriter(file_path)
            df.to_excel(writer, 'Sheet1')
            df.to_excel(writer, 'Sheet2')
            writer.save()
            # load sheet for --> get_rect_range
            if tmpdir[0] != '/':
                url = '/'.join(['', tmpdir, file_path])
            else:
                url = '/'.join([tmpdir, file_path])

            url_parent = 'file://%s#Sheet1!A1' % url
            xl_ref_parent = xr.parse_xl_url(url_parent)
            xr.open_xl_workbook(xl_ref_parent)
            xr.open_xl_sheet(xl_ref_parent)

            url_child = '#A1:B2'
            xl_ref_child = xr.parse_xl_url(url_child)

            xr.open_xl_workbook(xl_ref_child, xl_ref_parent)

            self.assertRaises(ValueError, xr.open_xl_sheet, *(xl_ref_child,))

            xr.open_xl_sheet(xl_ref_child, xl_ref_parent)

            self.assertEquals(xl_ref_child['xl_workbook'],
                              xl_ref_parent['xl_workbook'])
            self.assertEquals(xl_ref_child['xl_sheet'],
                              xl_ref_parent['xl_sheet'])

            url_child = '#Sheet2!A1:B2'
            xl_ref_child = xr.parse_xl_url(url_child)
            xr.open_xl_workbook(xl_ref_child, xl_ref_parent)
            xr.open_xl_sheet(xl_ref_child, xl_ref_parent)
            self.assertEquals(xl_ref_child['xl_workbook'],
                              xl_ref_parent['xl_workbook'])
            self.assertNotEquals(xl_ref_child['xl_sheet'],
                                 xl_ref_parent['xl_sheet'])