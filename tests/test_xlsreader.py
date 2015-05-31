#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

from datetime import datetime
import doctest
import os
from pathlib import posixpath
import sys
from tests import _tutils
from tests._tutils import check_xl_installed, xw_Workbook
import unittest

import pandalone.xlsreader as xr
import pandas as pd
from six.moves.urllib.request import urlopen  # @UnresolvedImport
import xlrd as xd
from pandalone.xlsreader import CellPos


log = _tutils._init_logging(__name__)
xl_installed = check_xl_installed()


def _make_xl_margins(sheet):
    full_cells = xr.get_full_cells(sheet)

    up = (0, 0)

    dn = (sheet.nrows - 1, sheet.ncols - 1)

    return full_cells, up, dn


def _make_sample_sheet(path, matrix, sheet_name, startrow=0, startcol=0):
    df = pd.DataFrame(matrix)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name, startrow=startrow, startcol=startcol)


def _make_local_url(fname, fragment=''):
    fpath = os.path.abspath(fname)
    return r'file:///{}#{}'.format(fpath, fragment)


def _read_rect_range(sheet, st_cell, nd_cell=None):
    full_cells = xr.get_full_cells(sheet)
    sheet_margins, indices = xr.get_sheet_margins(full_cells)
    xl_range = xr.resolve_capture_range(full_cells, sheet_margins,
                                        st_cell, nd_cell)  # or CellPos(None, None))
    return xr.read_range_values(sheet, xl_range, indices)


@unittest.skipIf(sys.version_info < (3, 4), "Doctests are made for py >= 3.3")
class TestDoctest(unittest.TestCase):

    def runTest(self):
        failure_count, test_count = doctest.testmod(
            xr, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestXlsReader(unittest.TestCase):

    @unittest.skip('Needs conversion to new logic.')
    def test_single_value_get_rect_range(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            wb_fname = 'sample.xlsx'
            _make_sample_sheet(wb_fname,
                               [[None, None, None], [5.1, 6.1, 7.1]],
                               'Sheet1',
                               startrow=5, startcol=3)

            # load sheet for --> get_rect_range
            fragment = 'Sheet1!A1:C2{"1":4,"2":"ciao"}'
            url = _make_local_url(wb_fname, fragment)
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(
                file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['sheet'])

            # get single value [D7]
            args = (sheet, xr.CellPos(3, 6))
            self.assertEqual(_read_rect_range(*args), 0)

            # get single value [A1]
            args = (sheet, xr.CellPos(0, 0))
            self.assertEqual(_read_rect_range(*args), None)

            # get single value [H9]
            args = (sheet, xr.CellPos(7, 8))
            self.assertEqual(_read_rect_range(*args), None)

    @unittest.skip('Needs conversion to new logic.')
    def test_vector_get_rect_range(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            wb_fname = 'sample.xlsx'
            _make_sample_sheet(wb_fname,
                               [[None, None, None], [5.1, 6.1, 7.1]],
                               'Sheet1',
                               startrow=5, startcol=3)

            # load sheet for --> get_rect_range
            fragment = 'Sheet1!A1:C2{"1":4,"2":"ciao"}'
            url = _make_local_url(wb_fname, fragment)
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(
                file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['sheet'])

            # single value in the sheet [D7:D7]
            args = (sheet, xr.CellPos(3, 6), xr.CellPos(3, 6))
            self.assertEqual(_read_rect_range(*args), [0])

            # get whole column [D_]
            args = (sheet, xr.CellPos(3, None))
            res = [None, None, None, None, None, None, 0, 1]
            self.assertEqual(_read_rect_range(*args), res)

            # get whole row [_6]
            args = (sheet, xr.CellPos(None, 5))
            res = [None, None, None, None, 0, 1, 2]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited row in the sheet [C6:_6]
            args = (sheet, xr.CellPos(2, 5), xr.CellPos(None, 5))
            res = [None, None, 0, 1, 2]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited row in the sheet [_7:_7]
            args = (sheet, xr.CellPos(None, 6), xr.CellPos(None, 6))
            res = [0]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited row in the sheet [A7:_7]
            args = (sheet, xr.CellPos(0, 6), xr.CellPos(None, 6))
            res = [None, None, None, 0]
            self.assertEqual(_read_rect_range(*args), res)

            # delimited row in the sheet [A7:D7]
            args = (sheet, xr.CellPos(0, 6), xr.CellPos(3, 6))
            res = [None, None, None, 0]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D_]
            args = (sheet, xr.CellPos(3, None), xr.CellPos(3, None))
            res = [0, 1]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited column in the sheet [D5:D_]
            args = (sheet, xr.CellPos(3, 4), xr.CellPos(3, None))
            res = [None, None, 0, 1]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited column in the sheet [D_:D9]
            args = (sheet, xr.CellPos(3, None), xr.CellPos(3, 8))
            res = [0, 1, None]
            self.assertEqual(_read_rect_range(*args), res)

            # delimited column in the sheet [D3:D9]
            args = (sheet, xr.CellPos(3, 2), xr.CellPos(3, 8))
            res = [None, None, None, None, 0, 1, None]
            self.assertEqual(_read_rect_range(*args), res)

    @unittest.skip('Needs conversion to new logic.')
    def test_matrix_get_rect_range(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            wb_fname = 'sample.xlsx'
            _make_sample_sheet(wb_fname,
                               [[None, None, None], [5.1, 6.1, 7.1]],
                               'Sheet1',
                               startrow=5, startcol=3)

            # load sheet for --> get_rect_range
            fragment = 'Sheet1!A1:C2{"1":4,"2":"ciao"}'
            url = _make_local_url(wb_fname, fragment)
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(
                file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['sheet'])

            # minimum matrix in the sheet [:]
            args = (sheet, xr.CellPos(None, None), xr.CellPos(None, None))
            res = [
                [None, 0, 1, 2],
                [0, None, None, None],
                [1, 5.1, 6.1, 7.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E_:__]
            args = (sheet, xr.CellPos(4, None), xr.CellPos(None, None))
            res = [
                [0, 1, 2],
                [None, None, None],
                [5.1, 6.1, 7.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E7:__]
            args = (sheet, xr.CellPos(4, 6), xr.CellPos(None, None))
            res = [
                [None, None, None],
                [5.1, 6.1, 7.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # delimited matrix in the sheet [D6:F8]
            args = (sheet, xr.CellPos(3, 5), xr.CellPos(5, 7))
            res = [
                [None, 0, 1],
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited matrix in the sheet [:F8]
            args = (sheet, xr.CellPos(None, None), xr.CellPos(5, 7))
            res = [
                [None, 0, 1],
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited matrix in the sheet [7:F8]
            args = (sheet, xr.CellPos(None, 6), xr.CellPos(5, 7))
            res = [
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # minimum delimited matrix in the sheet [E:F8]
            args = (sheet, xr.CellPos(None, 6), xr.CellPos(5, 7))
            res = [
                [0, None, None],
                [1, 5.1, 6.1]
            ]
            self.assertEqual(_read_rect_range(*args), res)

            # delimited matrix in the sheet [A1:F8]
            args = (sheet, xr.CellPos(0, 0), xr.CellPos(5, 7))
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
            self.assertEqual(_read_rect_range(*args), res)

            # delimited matrix in the sheet [G9:__]
            args = (sheet, xr.CellPos(6, 8), xr.CellPos(None, None))
            res = [[None]]
            self.assertEqual(_read_rect_range(*args), res)

            # delimited matrix in the sheet [F9:__]
            args = (sheet, xr.CellPos(5, 8), xr.CellPos(None, None))
            res = [[None]]
            self.assertEqual(_read_rect_range(*args), res)

    def test_basic_parse_xl_ref(self):
        xl_ref = 'Sheet1!A1(L):C2(UL)'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['sheet'], 'Sheet1')
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
        self.assertEquals(xr.make_CellPos('A', '1', 'L'),
                          xr.CellPos(xr.Cell(row=0, col=0), 'L'))
        self.assertEquals(xr.make_CellPos('A', '_', 'D'),
                          xr.CellPos(xr.Cell('_', 0), 'D'))
        self.assertEquals(xr.make_CellPos('_', '1', None),
                          xr.CellPos(xr.Cell(0, '_'), None))
        self.assertEquals(xr.make_CellPos('_', '_', None),
                          xr.CellPos(xr.Cell('_',
                                             '_'), None))
        self.assertEquals(xr.make_CellPos('A', '^', 'D'),
                          xr.CellPos(xr.Cell('^', 0), 'D'))
        self.assertEquals(xr.make_CellPos('^', '1', None),
                          xr.CellPos(xr.Cell(0, '^'), None))
        self.assertEquals(xr.make_CellPos('^', '^', None),
                          xr.CellPos(xr.Cell('^',
                                             '^'), None))

        self.assertRaises(ValueError, xr.make_CellPos, *('_0', '_', '0'))
        self.assertRaises(ValueError, xr.make_CellPos, *('@@', '@', '@'))

    def test_col2num(self):
        self.assertEqual(xr.col2num('D'), 3)
        self.assertEqual(xr.col2num('aAa'), 702)

    def test_parse_xl_url_Ok(self):
        url = 'file://path/to/file.xlsx#Sheet1!U10(L):D20(D){"json":"..."}'
        res = xr.parse_xl_url(url)

        self.assertEquals(res['url_file'], 'file://path/to/file.xlsx')
        self.assertEquals(res['sheet'], 'Sheet1')
        self.assertEquals(res['json'], {"json": "..."})
        self.assertEquals(res['st_cell'], xr.CellPos(xr.Cell(9, 20), 'L'))
        self.assertEquals(res['nd_cell'], xr.CellPos(xr.Cell(19, 3), 'D'))

    def test_parse_xl_url_Bad(self):
        self.assertRaises(ValueError, xr.parse_xl_url, *('#!:{"json":"..."', ))

    def test_parse_xl_url_Only_fragment(self):
        url = '#sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)
        self.assertEquals(res['url_file'], '')

    def test_parse_cell(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            wb_fname = 'sample.xlsx'
            xl = [datetime(1900, 8, 2), True, None, u'', 'hi', 1.4, 5.0]
            _make_sample_sheet(wb_fname, xl, 'Sheet1')

            # load sheet for --> get_rect_range
            fragment = 'Sheet1!A1:C2{"1":4,"2":"ciao"}'
            url = _make_local_url(wb_fname, fragment)
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(
                file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['sheet'])

            indices = xr.get_sheet_margins(xr.get_full_cells(sheet))[1]
            # row vector in the sheet [B2:B_]
            args = (
                sheet, (xr.Cell(1, 1), xr.Cell(7, 1)), indices, wb.datemode)
            res = [datetime(1900, 8, 2), True, None, None, 'hi', 1.4, 5]
            self.assertEqual(xr.read_range_values(*args), res)

    def test_comparison_vs_pandas_parse_cell(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            wb_fname = 'sample.xlsx'
            wb_fpath = os.path.abspath(wb_fname)
            xl = [datetime(1900, 8, 2), True, None, u'', 'hi', 1.4, 5.0]
            _make_sample_sheet(wb_fname,
                               xl,
                               'Sheet1')

            # load sheet for --> get_rect_range
            url = 'file:///%s#Sheet1!A1:C2{"1":4,"2":"ciao"}' % wb_fpath
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(
                file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['sheet'])

            indices = xr.get_sheet_margins(xr.get_full_cells(sheet))[1]
            # row vector in the sheet [B2:B_]
            args = (
                sheet, (xr.Cell(1, 1), xr.Cell(7, 1)), indices, wb.datemode)

            res = xr.read_range_values(*args)

            df = pd.read_excel(wb_fname, 'Sheet1')[0]

            # in pandas None values are converted in float('nan')
            df = df.where(pd.notnull(df), None).values.tolist()

            self.assertEqual(df, res)

    def test_open_xl_workbook(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            df = pd.DataFrame()
            wb_fname = 'sample.xlsx'
            wb_fpath = os.path.abspath(wb_fname)
            writer = pd.ExcelWriter(wb_fname)
            df.to_excel(writer, 'Sheet1')
            writer.save()
            # load sheet for --> get_rect_range

            url_parent = 'file:///%s#Sheet1!A1' % wb_fpath
            xl_ref_parent = xr.parse_xl_url(url_parent)
            xr.open_xl_workbook(xl_ref_parent)

            url_child = '#A1:B2'
            xl_ref_child = xr.parse_xl_url(url_child)

            self.assertRaises(
                ValueError, xr.open_xl_workbook, *(xl_ref_child,))

            xr.open_xl_workbook(xl_ref_child, xl_ref_parent)

            self.assertEquals(xl_ref_child['xl_workbook'],
                              xl_ref_parent['xl_workbook'])

    def test_open_xl_sheet(self):
        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            df = pd.DataFrame()
            wb_fname = 'sample.xlsx'
            writer = pd.ExcelWriter(wb_fname)
            df.to_excel(writer, 'Sheet1')
            df.to_excel(writer, 'Sheet2')
            writer.save()
            # load sheet for --> get_rect_range
            fragment = 'Sheet1!A1'
            url_parent = _make_local_url(wb_fname, fragment)
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


@unittest.skipIf(not xl_installed, "Cannot test xlwings without MS Excel.")
class TestVsXlwings(unittest.TestCase):

    def test_xlwings_vs_get_xl_table(self):
        import xlwings as xw

        with _tutils.TemporaryDirectory() as tmpdir, _tutils.chdir(tmpdir):
            wb_fname = 'sample.xlsx'
            wb_fpath = os.path.abspath(wb_fname)
            _make_sample_sheet(wb_fname, [
                [1, 2, None],
                [None, 6.1, 7.1]
            ],
                'Sheet1',
                startrow=5, startcol=3)

            fragment = 'Sheet1!A1:C2{"1":4,"2":"ciao"}'
            url = _make_local_url(wb_fname, fragment)
            res = xr.parse_xl_url(url)
            wb = xd.open_workbook(
                file_contents=urlopen(res['url_file']).read())
            sheet = wb.sheet_by_name(res['sheet'])
            datemode = wb.datemode

            # load Workbook for --> xlwings
            with xw_Workbook(wb_fpath) as wb:
                res = {}
                res[0] = xw.Range("Sheet1", "D7").vertical.value
                res[1] = xw.Range("Sheet1", "E6").vertical.value
                res[2] = xw.Range("Sheet1", "E6").horizontal.value
                res[3] = xw.Range("Sheet1", "E6").table.value
                res[4] = xw.Range("Sheet1", "D6:F8").value
                res[5] = xw.Range("Sheet1", "A1:F8").value
                res[6] = xw.Range("Sheet1", "A7:D7").value
                res[7] = xw.Range("Sheet1", "D3:D9").value

            full_cells, up, dn = _make_xl_margins(sheet)
            xl_ma, ind = xr.get_sheet_margins(full_cells)

            # minimum delimited column in the sheet [D7:D.(D)]
            st = xr.CellPos(xr.Cell(6, 3), None)
            nd = xr.CellPos(xr.Cell('.', '.'), 'D')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[0])

            # minimum delimited column in the sheet [E6:E.(D)]
            st = xr.CellPos(xr.Cell(5, 4), None)
            nd = xr.CellPos(xr.Cell('.', '.'), 'D')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[1])

            # minimum delimited row in the sheet [E6:.6(R)]
            st = xr.CellPos(xr.Cell(5, 4), None)
            nd = xr.CellPos(xr.Cell('.', '.'), 'R')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[2])

            # minimum delimited matrix in the sheet [E6:..(RD)]
            st = xr.CellPos(xr.Cell(5, 4), None)
            nd = xr.CellPos(xr.Cell('.', '.'), 'RD')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[3])

            st = xr.CellPos(xr.Cell(5, 4), None)
            nd = xr.CellPos(xr.Cell('.', '.'), 'DR')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[3])

            # delimited matrix in the sheet [D6:F8]
            st = xr.CellPos(xr.Cell(7, 5), None)
            nd = xr.CellPos(xr.Cell(5, 3), None)
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[4])

            # delimited matrix in the sheet [A1:F8]
            st = xr.CellPos(xr.Cell(7, 5), None)
            nd = xr.CellPos(xr.Cell(0, 0), None)
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[5])

            # delimited row in the sheet [A7:D7]
            st = xr.CellPos(xr.Cell(6, 0), None)
            nd = xr.CellPos(xr.Cell(6, 3), None)
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[6])

            # delimited column in the sheet [D3:D9]
            st = xr.CellPos(xr.Cell(8, 3), None)
            nd = xr.CellPos(xr.Cell(2, 3), None)
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            self.assertEqual(xr.read_range_values(*args), res[7])

            # minimum delimited matrix in the sheet [F7:..(UL)]
            st = xr.CellPos(xr.Cell(6, 5), None)
            nd = xr.CellPos(xr.Cell('.', '.'), 'UL')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [[None, 0, 1],
                   [0, 1, 2]]
            self.assertEqual(xr.read_range_values(*args), res)

            # minimum delimited matrix in the sheet [F7:F7:LURD]
            st = xr.CellPos(xr.Cell(6, 5), None)
            nd = xr.CellPos(xr.Cell(6, 5), None)
            rng_exp = xr._parse_range_expansions('LURD')
            rng = xr.resolve_capture_range(
                full_cells, xl_ma, st, nd, rng_exp)
            args = (sheet, rng, ind, datemode)
            res = [[None, 0, 1, 2],
                   [0, 1, 2, None],
                   [1, None, 6.1, 7.1]]
            self.assertEqual(xr.read_range_values(*args), res)

            # minimum delimited matrix in the sheet [F7:A1(RD)]
            st = xr.CellPos(xr.Cell(6, 5), None)
            nd = xr.CellPos(xr.Cell(0, 0), 'RD')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [[0, 1],
                   [1, 2]]
            self.assertEqual(xr.read_range_values(*args), res)

            # minimum delimited row in the sheet [_8:G8]
            st = xr.CellPos(xr.Cell(7, 6), None)
            nd = xr.CellPos(xr.Cell(7, '.'), 'L')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [6.1, 7.1]
            self.assertEqual(xr.read_range_values(*args), res)

            # minimum delimited column in the sheet [D_:D8]
            st = xr.CellPos(xr.Cell(7, 3), None)
            nd = xr.CellPos(xr.Cell('.', 3), 'U')
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [0, 1]
            self.assertEqual(xr.read_range_values(*args), res)

            # single value [D8]
            st = xr.CellPos(xr.Cell(7, 3), None)
            nd = None
            rng = xr.resolve_capture_range(full_cells, xl_ma, st, nd)
            args = (sheet, rng, ind, datemode)
            res = [1]
            self.assertEqual(xr.read_range_values(*args), res)
