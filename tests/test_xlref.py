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
from pandalone.xlref import Edge
import sys
from tests import _tutils
from tests._tutils import check_xl_installed, xw_Workbook
import unittest

import six

from pandalone.xlref import _xlref as xr
from pandalone.xlref import _xlrd as xd
import pandas as pd
import numpy as np
from six.moves.urllib.request import urlopen  # @UnresolvedImport
import xlrd


log = _tutils._init_logging(__name__)
xl_installed = check_xl_installed()


def _make_xl_margins(sheet):
    states_matrix = xd.read_states_matrix(sheet)

    up = (0, 0)

    dn = (sheet.nrows - 1, sheet.ncols - 1)

    return states_matrix, up, dn


def _make_sample_sheet(path, matrix, sheet_name, startrow=0, startcol=0):
    df = pd.DataFrame(matrix)
    with pd.ExcelWriter(path) as w:
        if isinstance(sheet_name, tuple):
            for s in sheet_name:
                df.to_excel(w, s, startrow=startrow, startcol=startcol)
        else:
            df.to_excel(w, sheet_name, startrow=startrow, startcol=startcol)


def _make_local_url(fname, fragment=''):
    fpath = os.path.abspath(fname)
    return r'file://{}#{}'.format(fpath, fragment)


def _read_rect_values(sheet, st_ref, nd_ref=None):
    states_matrix = xd.read_states_matrix(sheet)
    sheet_margins = xr.get_sheet_margins(states_matrix)
    xl_rect = xr.resolve_capture_rect(states_matrix, sheet_margins,
                                      st_ref, nd_ref)  # or Edge(None, None))
    return xr.read_capture_rect_values(sheet, xl_rect, states_matrix)


@unittest.skipIf(sys.version_info < (3, 4), "Doctests are made for py >= 3.3")
class TestDoctest(unittest.TestCase):

    def test_xlref(self):
        failure_count, test_count = doctest.testmod(
            xr, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))

    def test_xlrd(self):
        failure_count, test_count = doctest.testmod(
            xd, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestXlRef(unittest.TestCase):


    def test_parse_xl_ref_Cells_types(self):
        xl_ref = 'b1:C2'
        res = xr.parse_xl_ref(xl_ref)
        st_ref = res['st_ref']
        nd_ref = res['nd_ref']
        self.assertIsInstance(st_ref.cell.row, six.string_types)
        self.assertIsInstance(st_ref.cell.col, six.string_types)
        self.assertIsInstance(nd_ref.cell.row, six.string_types)
        self.assertIsInstance(nd_ref.cell.col, six.string_types)

    def test_parse_xl_ref_Cells_col_row_order(self):
        xl_ref = 'b1:C2'
        res = xr.parse_xl_ref(xl_ref)
        st_ref = res['st_ref']
        nd_ref = res['nd_ref']
        self.assertTrue(st_ref.cell.row.isalnum())
        self.assertTrue(st_ref.cell.col.isalpha())
        self.assertTrue(nd_ref.cell.row.isalnum())
        self.assertTrue(nd_ref.cell.col.isalpha())

    def test_parse_xl_ref_all_upper(self):
        xl_ref = 'b1(uL):C2(Dr):Lur2D'
        res = xr.parse_xl_ref(xl_ref)
        st_ref = res['st_ref']
        nd_ref = res['nd_ref']
        items = [
            st_ref.cell.row, st_ref.cell.col, st_ref.mov,
            nd_ref.cell.row, nd_ref.cell.col, nd_ref.mov,
        ]
        for i in items:
            if i:
                for c in i:
                    if c.isalpha():
                        self.assertTrue(c.isupper(), '%s in %r' % (c, i))

    def test_basic_parse_xl_ref(self):
        xl_ref = 'Sheet1!a1(L):C2(UL)'
        res = xr.parse_xl_ref(xl_ref)
        st_ref = res['st_ref']
        nd_ref = res['nd_ref']
        self.assertEquals(res['sheet'], 'Sheet1')
        self.assertEquals(st_ref.cell, xr.Cell(col='A', row='1'))
        self.assertEquals(nd_ref.cell, xr.Cell(col='C', row='2'))
        self.assertEquals(st_ref.mov, 'L')
        self.assertEquals(nd_ref.mov, 'UL')

        xl_ref = 'Sheet1!A1'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['st_ref'].cell, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_ref'], None)

        xl_ref = 'Sheet1!a1(l):c2(ul){"1":4,"2":"ciao"}'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['json'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['st_ref'].cell, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_ref'].cell, xr.Cell(col='C', row='2'))
        self.assertEquals(res['st_ref'].mov, 'L')
        self.assertEquals(res['nd_ref'].mov, 'UL')

    def test_errors_parse_xl_ref(self):
        self.assertRaises(ValueError, xr.parse_xl_ref, 's![[]')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!{}[]')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A1:!')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!1:2')
        self.assertRaises(ValueError, xr.parse_xl_ref, 's!A0:B1')

    def test_uncooked_Edge_good(self):
        self.assertIsNone(xr._uncooked_Edge(None, None, None))

        self.assertEquals(xr._uncooked_Edge('1', 'A', 'LUR'),
                          xr.Edge(xr.Cell(row='1', col='A'), 'LUR'))
        self.assertEquals(xr._uncooked_Edge('_', '^', 'duL'),
                          xr.Edge(xr.Cell('_', '^'), 'DUL'))
        self.assertEquals(xr._uncooked_Edge('1', '_', None),
                          xr.Edge(xr.Cell('1', '_'), None))
        self.assertEquals(xr._uncooked_Edge('^', '^', None),
                          xr.Edge(xr.Cell('^', '^'), None))

    def test_uncooked_Edge_bad(self):
        self.assertEquals(xr._uncooked_Edge(1, 'A', 'U1'),
                          xr.Edge(xr.Cell(1, 'A'), 'U1'))
        self.assertEquals(xr._uncooked_Edge('1', '%', 'U1'),
                          xr.Edge(xr.Cell('1', '%'), 'U1'))
        self.assertEquals(xr._uncooked_Edge('1', 'A', 'D0L'),
                          xr.Edge(xr.Cell('1', 'A'), 'D0L'))
        self.assertEquals(xr._uncooked_Edge('1', 'A', '@#'),
                          xr.Edge(xr.Cell('1', 'A'), '@#'))

    def test_uncooked_Edge_fail(self):
        self.assertRaises(
            AttributeError, xr._uncooked_Edge, *('1', 1, '0'))
        self.assertRaises(
            AttributeError, xr._uncooked_Edge, *('1', 'A', 23))
#         self.assertRaises(
#             ValueError, xr._uncooked_Edge, *('_0', '_', '0'))
#         self.assertRaises(
#             ValueError, xr._uncooked_Edge, *('@@', '@', '@'))

    def test_col2num(self):
        self.assertEqual(xr._col2num('D'), 3)
        self.assertEqual(xr._col2num('aAa'), 702)

    def test_parse_xl_url_Ok(self):
        url = 'file://path/to/file.xlsx#Sheet1!U10(L):D20(D){"json":"..."}'
        res = xr.parse_xl_url(url)

        self.assertEquals(res['url_file'], 'file://path/to/file.xlsx')
        self.assertEquals(res['sheet'], 'Sheet1')
        self.assertEquals(res['json'], {"json": "..."})
        self.assertEquals(res['st_ref'], xr.Edge(xr.Cell('10', 'U'), 'L'))
        self.assertEquals(res['nd_ref'], xr.Edge(xr.Cell('20', 'D'), 'D'))

    def test_parse_xl_url_Bad(self):
        self.assertRaises(ValueError, xr.parse_xl_url, *('#!:{"json":"..."', ))

    def test_parse_xl_url_Only_fragment(self):
        url = '#sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)
        self.assertEquals(res['url_file'], '')


class TestXlRead(unittest.TestCase):
    def setUp(self):
        from tempfile import mkstemp
        self.tmp = '%s.xlsx' % mkstemp()[1]
        xl = [datetime(1900, 8, 2), True, None, u'', 'hi', 1.4, 5.0]
        _make_sample_sheet(self.tmp, xl, ('Sheet1', 'Sheet2'))

        self.states_matrix = np.array([[0, 1],
                                       [1, 1],
                                       [1, 1],
                                       [1, 0],
                                       [1, 0],
                                       [1, 1],
                                       [1, 1],
                                       [1, 1]], dtype=bool)
        self.sheet = xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1')

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_parse_rect_values(self):
        # row vector in the sheet [B2:B_]
        args = (
            self.sheet, (xr.Cell(1, 1), xr.Cell(7, 1)), self.states_matrix
        )
        res = [datetime(1900, 8, 2), True, None, None, 'hi', 1.4, 5]
        self.assertEqual(xr.read_capture_rect_values(*args), res)

    def test_comparison_vs_pandas_parse_cell(self):

        # row vector in the sheet [B2:B_]
        args = (
            self.sheet, (xr.Cell(1, 1), xr.Cell(7, 1)), self.states_matrix
        )

        res = xr.read_capture_rect_values(*args)

        df = pd.read_excel(self.tmp, 'Sheet1')[0]

        # in pandas None values are converted in float('nan')
        df = df.where(pd.notnull(df), None).values.tolist()

        self.assertEqual(df, res)

    def test_open_xl_workbook(self):

        url_parent = _make_local_url(self.tmp, 'Sheet1!A1')
        xl_ref_parent = xr.parse_xl_url(url_parent)
        xd.open_xlref_workbook(xl_ref_parent)

        url_child = '#A1:B2'
        xl_ref_child = xr.parse_xl_url(url_child)

        self.assertRaises(
            ValueError, xd.open_xlref_workbook, *(xl_ref_child,))

        xd.open_xlref_workbook(xl_ref_child, xl_ref_parent)

        self.assertEquals(xl_ref_child['xl_workbook'],
                          xl_ref_parent['xl_workbook'])

    def test_open_xl_sheet(self):
        url_parent = _make_local_url(self.tmp, 'Sheet1!A1')
        xl_ref_parent = xr.parse_xl_url(url_parent)
        xd.open_xlref_workbook(xl_ref_parent)
        xd.open_sheet(xl_ref_parent)

        url_child = '#A1:B2'
        xl_ref_child = xr.parse_xl_url(url_child)

        xd.open_xlref_workbook(xl_ref_child, xl_ref_parent)

        self.assertRaises(ValueError, xd.open_sheet, *(xl_ref_child,))

        xd.open_sheet(xl_ref_child, xl_ref_parent)

        self.assertEquals(xl_ref_child['xl_workbook'],
                          xl_ref_parent['xl_workbook'])
        self.assertEquals(xl_ref_child['xl_sheet'],
                          xl_ref_parent['xl_sheet'])

        url_child = '#Sheet2!A1:B2'
        xl_ref_child = xr.parse_xl_url(url_child)
        xd.open_xlref_workbook(xl_ref_child, xl_ref_parent)
        xd.open_sheet(xl_ref_child, xl_ref_parent)
        self.assertEquals(xl_ref_child['xl_workbook'],
                          xl_ref_parent['xl_workbook'])
        self.assertNotEquals(xl_ref_child['xl_sheet'],
                             xl_ref_parent['xl_sheet'])

class TestXlRead_rect(unittest.TestCase):
    def setUp(self):
        from tempfile import mkstemp
        self.tmp = '%s.xlsx' % mkstemp()[1]
        xl = [
                 [None, None, None],
                 [5.1, 6.1, 7.1]
             ]

        _make_sample_sheet(self.tmp, xl, 'Sheet1', startrow=5, startcol=3)

        self.sheet = xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1')

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_read_rect_values_Scalar(self):
        sheet = self.sheet

        # get single value [D7]
        args = (sheet, xr.Edge(xr.Cell('7', 'D'), None))
        self.assertEqual(_read_rect_values(*args), [0])

        # get single value [A1]
        args = (sheet, xr.Edge(xr.Cell('1', 'A'), None))
        self.assertEqual(_read_rect_values(*args), [None])

        # get single value [H9]
        args = (sheet, xr.Edge(xr.Cell('9', 'H'), None))
        self.assertEqual(_read_rect_values(*args), [None])

    def test_read_rect_values_Vector(self):
        sheet = self.sheet

        # single value in the sheet [D7:D7]
        args = (sheet,
                xr.Edge(xr.Cell('7', 'D'), None),
                xr.Edge(xr.Cell('7', 'D'), None))
        self.assertEqual(_read_rect_values(*args), [0])

        # get whole column [D_]
        args = (sheet,
                xr.Edge(xr.Cell('1', 'D'), None),
                xr.Edge(xr.Cell('_', 'D'), None))
        res = [None, None, None, None, None, None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res)

        # get whole row [_6]
        args = (sheet,
                xr.Edge(xr.Cell('6', 'A'), None),
                xr.Edge(xr.Cell('6', '_'), None))
        res = [None, None, None, None, 0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited row in the sheet [C6:_6]
        args = (sheet,
                xr.Edge(xr.Cell('6', 'C'), None),
                xr.Edge(xr.Cell('6', '_'), None))
        res = [None, None, 0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited row in the sheet [_7:_7]
        args = (sheet,
                xr.Edge(xr.Cell('7', 'A'), None),
                xr.Edge(xr.Cell('7', '_'), None))
        res = [None, None, None, 0, None, None, None]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited row in the sheet [E6:_6]
        args = (sheet,
                xr.Edge(xr.Cell('1', 'A'), 'RD'),
                xr.Edge(xr.Cell('.', '.'), 'R'))
        res = [0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited row in the sheet [A7:D7]
        args = (sheet,
                xr.Edge(xr.Cell('7', 'A'), None),
                xr.Edge(xr.Cell('7', 'D'), None))
        res = [None, None, None, 0]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited column in the sheet [D_:D_]
        args = (sheet,
                xr.Edge(xr.Cell('^', 'D'), None),
                xr.Edge(xr.Cell('_', 'D'), None))
        res = [None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited column in the sheet [D5:D_]
        args = (sheet,
                xr.Edge(xr.Cell('5', 'D'), None),
                xr.Edge(xr.Cell('_', 'D'), None))
        res = [None, None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited column in the sheet [D_:D9]
        args = (sheet,
                xr.Edge(xr.Cell('^', 'D'), None),
                xr.Edge(xr.Cell('9', 'D'), None))
        res = [None, 0, 1, None]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited column in the sheet [D3:D9]
        args = (sheet,
                xr.Edge(xr.Cell('3', 'D'), None),
                xr.Edge(xr.Cell('9', 'D'), None))
        res = [None, None, None, None, 0, 1, None]
        self.assertEqual(_read_rect_values(*args), res)

    def test_read_rect_values(self):
        sheet = self.sheet

        # minimum matrix in the sheet [:]
        args = (sheet,
                xr.Edge(xr.Cell('^', '^'), None),
                xr.Edge(xr.Cell('_', '_'), None))
        res = [
            [None, 0, 1, 2],
            [0, None, None, None],
            [1, 5.1, 6.1, 7.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [E_:__]
        args = (sheet,
                xr.Edge(xr.Cell('^', 'E'), None),
                xr.Edge(xr.Cell('_', '_'), None))
        res = [
            [0, 1, 2],
            [None, None, None],
            [5.1, 6.1, 7.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [E7:__]
        args = (sheet,
                xr.Edge(xr.Cell('7', 'E'), None),
                xr.Edge(xr.Cell('_', '_'), None))
        res = [
            [None, None, None],
            [5.1, 6.1, 7.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited matrix in the sheet [D6:F8]
        args = (sheet,
                xr.Edge(xr.Cell('6', 'D'), None),
                xr.Edge(xr.Cell('8', 'F'), None))
        res = [
            [None, 0, 1],
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [:F8]
        args = (sheet,
                xr.Edge(xr.Cell('8', 'F'), None),
                xr.Edge(xr.Cell('^', '^'), None))
        res = [
            [None, 0, 1],
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [7:F8]
        args = (sheet,
                xr.Edge(xr.Cell('8', 'F'), None),
                xr.Edge(xr.Cell('7', '^'), None))
        res = [
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [D:F8]
        args = (sheet,
                xr.Edge(xr.Cell('8', 'F'), None),
                xr.Edge(xr.Cell('8', 'D'), 'U'))
        res = [
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited matrix in the sheet [A1:F8]
        args = (sheet,
                xr.Edge(xr.Cell('1', 'A'), None),
                xr.Edge(xr.Cell('8', 'F'), None))
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
        self.assertEqual(_read_rect_values(*args), res)

        # delimited matrix in the sheet [G9:__]
        args = (sheet,
                xr.Edge(xr.Cell('9', 'G'), None),
                xr.Edge(xr.Cell('.', '.'), 'D'))
        self.assertRaises(ValueError, _read_rect_values, *args)

        # delimited matrix in the sheet [F9:__]
        args = (sheet,
                xr.Edge(xr.Cell('9', 'F'), None),
                xr.Edge(xr.Cell('.', '.'), 'R'))
        self.assertRaises(ValueError, _read_rect_values, *args)


@unittest.skipIf(not xl_installed, "Cannot test xlwings without MS Excel.")
class TestVsXlwings(unittest.TestCase):
    def setUp(self):
        from tempfile import mkstemp
        self.tmp = '%s.xlsx' % mkstemp()[1]
        xl = [
            [1, 2, None],
            [None, 6.1, 7.1]
        ]
        _make_sample_sheet(self.tmp, xl, 'Sheet1', startrow=5, startcol=3)



        self.states_matrix = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 0],
             [0, 0, 0, 1, 0, 1, 1]], dtype=bool)

        self.sheet = xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1')
        self.xl_ma = xr.get_sheet_margins(self.states_matrix)

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_xlwings_vs_get_xl_table(self):
        import xlwings as xw
        # load Workbook for --> xlwings
        with xw_Workbook(self.tmp) as wb:
            resTarget = xw.Range("Sheet1", "D7").vertical.value
            res = [None,
                   xw.Range("Sheet1", "E6").vertical.value,
                   xw.Range("Sheet1", "E6").horizontal.value,
                   xw.Range("Sheet1", "E6").table.value,
                   xw.Range("Sheet1", "D6:F8").value,
                   xw.Range("Sheet1", "A1:F8").value,
                   xw.Range("Sheet1", "A7:D7").value,
                   xw.Range("Sheet1", "D3:D9").value,
                   ]

        states_matrix = self.states_matrix
        sheet = self.sheet
        xl_ma = self.xl_ma

        # minimum delimited column in the sheet [D7:D.(D)]
        st = xr.Edge(xr.num2a1_Cell(6, 3), None)
        nd = xr.Edge(xr.num2a1_Cell('.', '.'), 'D')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), resTarget)

        # minimum delimited column in the sheet [E6:E.(D)]
        st = xr.Edge(xr.num2a1_Cell(5, 4), None)
        nd = xr.Edge(xr.num2a1_Cell('.', '.'), 'D')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[1])

        # minimum delimited row in the sheet [E6:.6(R)]
        st = xr.Edge(xr.num2a1_Cell(5, 4), None)
        nd = xr.Edge(xr.num2a1_Cell('.', '.'), 'R')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[2])

        # minimum delimited matrix in the sheet [E6:..(RD)]
        st = xr.Edge(xr.num2a1_Cell(5, 4), None)
        nd = xr.Edge(xr.num2a1_Cell('.', '.'), 'RD')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[3])

        st = xr.Edge(xr.num2a1_Cell(5, 4), None)
        nd = xr.Edge(xr.num2a1_Cell('.', '.'), 'DR')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[3])

        # delimited matrix in the sheet [D6:F8]
        st = xr.Edge(xr.num2a1_Cell(7, 5), None)
        nd = xr.Edge(xr.num2a1_Cell(5, 3), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[4])

        # delimited matrix in the sheet [A1:F8]
        st = xr.Edge(xr.num2a1_Cell(7, 5), None)
        nd = xr.Edge(xr.num2a1_Cell(0, 0), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[5])

        # delimited row in the sheet [A7:D7]
        st = xr.Edge(xr.num2a1_Cell(6, 0), None)
        nd = xr.Edge(xr.num2a1_Cell(6, 3), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[6])

        # delimited column in the sheet [D3:D9]
        st = xr.Edge(xr.num2a1_Cell(8, 3), None)
        nd = xr.Edge(xr.num2a1_Cell(2, 3), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        self.assertEqual(xr.read_capture_rect_values(*args), res[7])

        # minimum delimited matrix in the sheet [F7:..(UL)]
        st = xr.Edge(xr.num2a1_Cell(6, 5), None)
        nd = xr.Edge(xr.num2a1_Cell('.', '.'), 'UL')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        res = [[None, 0, 1],
               [0, 1, 2]]
        self.assertEqual(xr.read_capture_rect_values(*args), res)

        # minimum delimited matrix in the sheet [F7:F7:LURD]
        st = xr.Edge(xr.num2a1_Cell(6, 5), None)
        nd = xr.Edge(xr.num2a1_Cell(6, 5), None)
        rect_exp = 'LURD'
        rng = xr.resolve_capture_rect(
            states_matrix, xl_ma, st, nd, rect_exp)
        args = (sheet, rng, states_matrix)
        res = [[None, 0, 1, 2],
               [0, 1, 2, None],
               [1, None, 6.1, 7.1]]
        self.assertEqual(xr.read_capture_rect_values(*args), res)

        # minimum delimited matrix in the sheet [F7:A1(RD)]
        st = xr.Edge(xr.num2a1_Cell(6, 5), None)
        nd = xr.Edge(xr.num2a1_Cell(0, 0), 'RD')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        res = [[0, 1],
               [1, 2]]
        self.assertEqual(xr.read_capture_rect_values(*args), res)

        # minimum delimited row in the sheet [_8:G8]
        st = xr.Edge(xr.num2a1_Cell(7, 6), None)
        nd = xr.Edge(xr.num2a1_Cell(7, '.'), 'L')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        res = [6.1, 7.1]
        self.assertEqual(xr.read_capture_rect_values(*args), res)

        # minimum delimited column in the sheet [D_:D8]
        st = xr.Edge(xr.num2a1_Cell(7, 3), None)
        nd = xr.Edge(xr.num2a1_Cell('.', 3), 'U')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        res = [0, 1]
        self.assertEqual(xr.read_capture_rect_values(*args), res)

        # single value [D8]
        st = xr.Edge(xr.num2a1_Cell(7, 3), None)
        nd = None
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng, states_matrix)
        res = [1]
        self.assertEqual(xr.read_capture_rect_values(*args), res)
