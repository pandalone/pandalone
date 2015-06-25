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
from pandalone.xlref import _xlrd as xd
from pandalone.xlref import _xlref as xr
import sys
from tests import _tutils
from tests._tutils import check_xl_installed, xw_Workbook
import unittest

from ddt import ddt, data
import six
import xlrd

import numpy as np
import pandas as pd


log = _tutils._init_logging(__name__)
xl_installed = check_xl_installed()


def _make_xl_margins(sheet):
    states_matrix = sheet.read_states_matrix()

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
    return 'file:///{}#{}'.format(fpath, fragment)


def _read_rect_values(sheet, st_edge, nd_edge=None):
    states_matrix = sheet.read_states_matrix()
    special_margins = xr.make_special_margins(*sheet.get_margin_coords())
    xl_rect = xr.resolve_capture_rect(states_matrix, special_margins,
                                      st_edge, nd_edge)  # or Edge(None, None))
    return xr.read_capture_rect(sheet, xl_rect)


@unittest.skipIf(sys.version_info < (3, 4), "Doctests are made for py >= 3.3")
class Doctest(unittest.TestCase):

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


_all_dir_pairs = ['LU', 'LD', 'UL', 'DL']
_all_dir_pairs += _all_dir_pairs[::-1]
_all_dirs = list(['LRUD']) + _all_dir_pairs


@ddt
class Parse(unittest.TestCase):

    def test_parse_xl_ref_Coordss_types(self):
        xl_ref = 'b1:C2'
        res = xr.parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertIsInstance(st_edge.land.row, six.string_types)
        self.assertIsInstance(st_edge.land.col, six.string_types)
        self.assertIsInstance(nd_edge.land.row, six.string_types)
        self.assertIsInstance(nd_edge.land.col, six.string_types)

    def test_parse_xl_ref_Coordss_col_row_order(self):
        xl_ref = 'b1:C2'
        res = xr.parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertTrue(st_edge.land.row.isalnum())
        self.assertTrue(st_edge.land.col.isalpha())
        self.assertTrue(nd_edge.land.row.isalnum())
        self.assertTrue(nd_edge.land.col.isalpha())

    def test_parse_xl_ref_all_upper(self):
        xl_ref = 'b1(uL):C2(Dr):Lur2D'
        res = xr.parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        items = [
            st_edge.land.row, st_edge.land.col, st_edge.mov,
            nd_edge.land.row, nd_edge.land.col, nd_edge.mov,
        ]
        for i in items:
            if i:
                for c in i:
                    if c.isalpha():
                        self.assertTrue(c.isupper(), '%s in %r' % (c, i))

    def test_basic_parse_xl_ref(self):
        xl_ref = 'Sheet1!a1(L):C2(UL)'
        res = xr.parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertEquals(res['sheet'], 'Sheet1')
        self.assertEquals(st_edge.land, xr.Coords(col='A', row='1'))
        self.assertEquals(nd_edge.land, xr.Coords(col='C', row='2'))
        self.assertEquals(st_edge.mov, 'L')
        self.assertEquals(nd_edge.mov, 'UL')

        xl_ref = 'Sheet1!A1'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['st_edge'].land, xr.Coords(col='A', row='1'))
        self.assertEquals(res['nd_edge'], None)

        xl_ref = 'Sheet1!a1(l):c2(ul){"1":4,"2":"ciao"}'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['json'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['st_edge'].land, xr.Coords(col='A', row='1'))
        self.assertEquals(res['nd_edge'].land, xr.Coords(col='C', row='2'))
        self.assertEquals(res['st_edge'].mov, 'L')
        self.assertEquals(res['nd_edge'].mov, 'UL')

    @data('s![[]', 's!{}[]', 's!A', 's!A1:!', 's!1:2', 's!A0:B1', )
    def test_errors_parse_xl_ref(self, case):
        self.assertRaises(ValueError, xr.parse_xl_ref, case)

    def test_uncooked_Edge_good(self):
        self.assertIsNone(xr._uncooked_Edge(None, None, None))

        self.assertEquals(xr._uncooked_Edge('1', 'A', 'LUR'),
                          xr.Edge(xr.Coords(row='1', col='A'), 'LUR'))
        self.assertEquals(xr._uncooked_Edge('_', '^', 'duL'),
                          xr.Edge(xr.Coords('_', '^'), 'DUL'))
        self.assertEquals(xr._uncooked_Edge('1', '_', None),
                          xr.Edge(xr.Coords('1', '_'), None))
        self.assertEquals(xr._uncooked_Edge('^', '^', None),
                          xr.Edge(xr.Coords('^', '^'), None))

    def test_uncooked_Edge_bad(self):
        self.assertEquals(xr._uncooked_Edge(1, 'A', 'U1'),
                          xr.Edge(xr.Coords(1, 'A'), 'U1'))
        self.assertEquals(xr._uncooked_Edge('1', '%', 'U1'),
                          xr.Edge(xr.Coords('1', '%'), 'U1'))
        self.assertEquals(xr._uncooked_Edge('1', 'A', 'D0L'),
                          xr.Edge(xr.Coords('1', 'A'), 'D0L'))
        self.assertEquals(xr._uncooked_Edge('1', 'A', '@#'),
                          xr.Edge(xr.Coords('1', 'A'), '@#'))

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
        self.assertEquals(res['st_edge'], xr.Edge(xr.Coords('10', 'U'), 'L'))
        self.assertEquals(res['nd_edge'], xr.Edge(xr.Coords('20', 'D'), 'D'))

    def test_parse_xl_url_Bad(self):
        self.assertRaises(ValueError, xr.parse_xl_url, *('#!:{"json":"..."', ))

    def test_parse_xl_url_Only_fragment(self):
        url = '#sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)
        self.assertEquals(res['url_file'], '')


@ddt
class Resolve(unittest.TestCase):

    def test_find_states_matrix_margins(self):
        sm = np.array([
            [0, 1, 1, 0]
        ])
        margins = (xr.Coords(0, 1), xr.Coords(0, 2))
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), margins)

        sm = np.asarray([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ])
        margins = (xr.Coords(1, 1), xr.Coords(3, 2))
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), margins)

    def test_find_states_matrix_margins_Single_cell(self):
        sm = np.array([[1], ])
        c = xr.Coords(0, 0)
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), (c, c))

        sm = np.array([[0, 0, 1], ])
        c = xr.Coords(0, 2)
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), (c, c))

        sm = np.array([
            [0, 0],
            [0, 1]
        ])
        c = xr.Coords(1, 1)
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), (c, c))

        sm = np.array([
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        c = xr.Coords(2, 1)
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), (c, c))

    def test_find_states_matrix_margins_Further_empties(self):
        sm = np.asarray([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ])
        margins = (xr.Coords(1, 1), xr.Coords(3, 2))
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), margins)

        sm = np.asarray([
            [1, 0],
            [0, 0],
            [0, 0],
        ])
        margins = (xr.Coords(0, 0), xr.Coords(0, 0))
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), margins)
        sm = np.asarray([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ])
        margins = (xr.Coords(1, 1), xr.Coords(1, 1))
        self.assertEqual(xr.margin_coords_from_states_matrix(sm), margins)

    @data(
        [[]],
        [[0], [0]],
        [[0, 0]],
        [[0, 0], [0, 0]],
    )
    def test_find_states_matrix_margins_EmptySheet(self, states_matrix):
        margins = (xr.Coords(0, 0), xr.Coords(0, 0))
        res = xr.margin_coords_from_states_matrix(np.asarray(states_matrix))
        self.assertEqual(res, margins, states_matrix)

    def make_states_matrix(self):
        states_matrix = np.array([
            # A  B  C  D  E  F
            [0, 0, 0, 0, 0, 0],  # '1'
            [0, 0, 0, 0, 0, 0],  # '2'
            [0, 0, 0, 1, 1, 1],  # '3'
            [0, 0, 1, 0, 0, 1],  # '4'
            [0, 0, 1, 1, 0, 1],  # '5'
        ], dtype=bool)
        margin_coords = xr.margin_coords_from_states_matrix(states_matrix)
        special_margins = xr.make_special_margins(*margin_coords)
        args = (states_matrix, special_margins)

        return args

    def check_target_opposite_state(self, land_state, land_row, land_col,
                                    moves, exp_row, exp_col):
        target_func = xr._target_opposite_state
        self.check_target_func(target_func, land_state, land_row, land_col,
                               moves, exp_row, exp_col)

    def check_target_same_state(self, land_state, land_row, land_col,
                                moves, exp_row, exp_col):
        target_func = xr._target_same_state
        self.check_target_func(target_func, land_state, land_row, land_col,
                               moves, exp_row, exp_col)

    def check_target_func(self, *args):
        (target_func, land_state, land_row, land_col,
         moves, exp_row, exp_col) = args
        states_matrix, margins = self.make_states_matrix()
        argshead = (
            states_matrix, xr.Coords(margins.row['_'], margins.col['_']))

        land_cell = xr.Coords(land_row, land_col)
        args = argshead + (land_state, land_cell, moves)
        res = target_func(*args)
        self.assertEqual(res, xr.Coords(exp_row, exp_col), str(args))

    def check_target_opposite_state_RaisesTargetMissed(self, *args):
        ## args =(land_state, land_row, land_col, moves)
        with self.assertRaisesRegexp(ValueError, "No \w+-target for",
                                     msg=str(args)):
            args += (None, None)
            self.check_target_opposite_state(*args)

    def test_target_opposite_state_Basic(self):
        self.check_target_opposite_state(False, 0, 0, 'DR', 3, 2)
        self.check_target_opposite_state(False, 0, 0, 'RD', 2, 3)

        self.check_target_opposite_state(False, 3, 0, 'UR', 3, 2)
        self.check_target_opposite_state(False, 3, 0, 'RU', 3, 2)
        self.check_target_opposite_state(False, 3, 0, 'DR', 3, 2)
        self.check_target_opposite_state(False, 3, 0, 'RD', 3, 2)

        self.check_target_opposite_state(False, 0, 3, 'DL', 2, 3)
        self.check_target_opposite_state(False, 0, 3, 'LD', 2, 3)
        self.check_target_opposite_state(False, 0, 3, 'DR', 2, 3)
        self.check_target_opposite_state(False, 0, 3, 'RD', 2, 3)

    def test_target_opposite_state_NotMovingFromMatch(self):
        coords = [(2, 3), (3, 2),
                  (2, 4), (2, 5),
                  (3, 5),
                  (4, 2), (4, 3),   (4, 5),
                  ]
        for d in _all_dirs:
            for r, c in coords:
                self.check_target_opposite_state(False, r, c, d, r, c)

    def test_target_opposite_state_NotState(self):
        # FIXME: How is this working!!!
        self.check_target_opposite_state(True, 7, 2, 'U', 7, 2)
        self.check_target_opposite_state(True, 3, 2, 'D', 4, 2)

    def test_target_opposite_state_Beyond_columns(self):
        dirs = ['L', 'LU', 'LD', 'UL', 'DL']
        for d in dirs:
            for row in [2, 3, 4]:
                self.check_target_opposite_state(
                    False, row, 10, d, row, 5)
            if 'D' in d:
                self.check_target_opposite_state(False, 0, 10, d, 2, 5)

    def test_target_opposite_state_Beyond_rows(self):
        dirs = ['U', 'UL', 'UR', 'LU', 'RU']
        for d in dirs:
            for col in [2, 3, 5]:
                self.check_target_opposite_state(
                    False, 10, col, d, 4, col)
            if 'U' in d[0]:
                self.check_target_opposite_state(False, 10, 4, d, 2, 4)
            if 'R' in d:
                self.check_target_opposite_state(False, 10, 0, d, 4, 2)

        self.check_target_opposite_state(False, 10, 4, 'LU', 4, 3)
        self.check_target_opposite_state(False, 10, 4, 'RU', 4, 5)

    def test_target_opposite_state_Beyond_both(self):
        self.check_target_opposite_state(False, 10, 10, 'UL', 4, 5)
        self.check_target_opposite_state(False, 10, 10, 'LU', 4, 5)

    def test_target_opposite_state_InvalidMoves(self):
        bad_dirs = list('UDLR') + ['UR', 'RU', 'UL', 'LU', 'DL', 'LD']
        for d in bad_dirs:
            self.check_target_opposite_state_RaisesTargetMissed(
                False, 0, 0, d)

    def TODO_Check_StateFalse(self):
        pass
#                 >>> states_matrix = np.array([
#         ...     [1, 1, 1],
#         ...     [1, 1, 1],
#         ...     [1, 1, 1],
#         ... ])
#         >>> args = (states_matrix, (2, 2))
#
#         >>> _target_opposite_state(*(args + (True, Coords(0, 2), 'LD')))
#         Coords(row=2, col=2)

    def test_target_same_state_InverseWalking(self):
        self.check_target_same_state(True, 2, 5, 'LD', 4, 3)

        self.check_target_same_state(True, 4, 5, 'LU', 2, 5)

    def check_resolve_capture_rect(self, *args):
        #     st_row, st_col, st_mov,
        #     nd_row, nd_col, nd_mov,
        #     res_st_row, res_st_col, res_nd_row, res_nd_col
        argshead = self.make_states_matrix()

        st_edge = xr.Edge(xr.Coords(*args[0:2]), args[2])
        nd_edge = xr.Edge(xr.Coords(*args[3:5]), args[5])
        res = (xr.Coords(*args[6:8]),
               xr.Coords(*args[8:10]))
        args = argshead + (st_edge, nd_edge)
        self.assertEqual(xr.resolve_capture_rect(*args), res)

    def test_resolve_capture_rect_St_inverseTargetDirs(self):
        self.check_resolve_capture_rect('1', 'A', 'DR', '.', '.', 'DR',
                                        3, 2, 4, 2)
        self.check_resolve_capture_rect('^', '^', 'RD', '.', '.', 'RD',
                                        2, 3, 2, 5)

    @data(
        ('^', '^', 'D', '_', '_', None, 3, 2, 4, 5),
        ('^', '^', 'DR', '_', '_', None, 3, 2, 4, 5),
        ('^', '^', 'R', '_', '_', None, 2, 3, 4, 5),
        ('^', '^', 'RD', '_', '_', None, 2, 3, 4, 5),

        ('4', 'D', 'U', '4', 'E', 'R', 2, 3, 3, 5),
        ('4', 'D', 'UL', '4', 'F', 'RD', 2, 3, 4, 5),
        ('4', 'D', 'L', '4', 'F', 'RD', 3, 2, 4, 5),
        ('4', 'D', 'LU', '4', 'F', 'RD', 3, 2, 4, 5),
    )
    def test_resolve_capture_rect_Target_fromEmpty_st(self, case):
        self.check_resolve_capture_rect(*case)

    @data(
        ('3', 'D', None, '4', 'E', 'R', 2, 3, 3, 5),
        ('3', 'D', None, '4', 'E', 'RD', 2, 3, 3, 5),
        ('3', 'D', None, '4', 'E', 'RU', 2, 3, 3, 5),

        ('3', 'D', None, '4', 'E', 'L', 2, 2, 3, 3),
        ('3', 'D', None, '4', 'E', 'LD', 2, 2, 3, 3),
        ('3', 'D', None, '4', 'E', 'LU', 2, 2, 3, 3),
    )
    def test_resolve_capture_rect_Target_fromEmpty_nd(self, case):
        self.check_resolve_capture_rect(*case)

    def test_resolve_capture_rect_BackwardRelative_singleDir(self):
        self.check_resolve_capture_rect('^', '_', None, '.', '.', 'L',
                                        2, 3, 2, 5)
        self.check_resolve_capture_rect('_', '^', None, '.', '.', 'U',
                                        3, 2, 4, 2)

    @data(
        ('_', '_', None, '.', '.', 'UL', 2, 5, 4, 5),
        ('_', '_', None, '.', '.', 'LU', 2, 5, 4, 5),

        ('^', '_', None, '.', '.', 'LD', 2, 3, 4, 5),
        ('^', '_', None, '.', '.', 'DL', 2, 3, 4, 5),

        ('_', '^', None, '.', '.', 'UR', 3, 2, 4, 3),
        ('_', '^', None, '.', '.', 'RU', 3, 2, 4, 3),
    )
    def test_resolve_capture_rect_BackwardRelative_multipleDirs(self, case):
        self.check_resolve_capture_rect(*case)

    @data(
        ('^', '^', None, '_', '_', None, 2, 2, 4, 5),
        ('_', '_', None, '^', '^', None, 2, 2, 4, 5),
        ('^', '_', None, '_', '^', None, 2, 2, 4, 5),
        ('_', '^', None, '^', '_', None, 2, 2, 4, 5),
    )
    def test_resolve_capture_rect_BackwardMoves_nonRelative(self, case):
        self.check_resolve_capture_rect(*case)


class Read1(unittest.TestCase):

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
        self.sheet = xr.Spreadsheet(
            xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1'))

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_parse_rect_values(self):
        # row vector in the sheet [B2:B_]
        args = (self.sheet, (xr.Coords(1, 1), xr.Coords(7, 1)))
        res = [datetime(1900, 8, 2), True, None, None, 'hi', 1.4, 5]
        self.assertEqual(xr.read_capture_rect(*args), res)

    def test_comparison_vs_pandas_parse_cell(self):

        # row vector in the sheet [B2:B_]
        args = (self.sheet, (xr.Coords(1, 1), xr.Coords(7, 1)))

        res = xr.read_capture_rect(*args)

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


class Read2(unittest.TestCase):  # FIXME: Why another class

    def setUp(self):
        from tempfile import mkstemp
        self.tmp = '%s.xlsx' % mkstemp()[1]
        xl = [
            [None, None, None],
            [5.1, 6.1, 7.1]
        ]

        _make_sample_sheet(self.tmp, xl, 'Sheet1', startrow=5, startcol=3)

        self.sheet = xr.Spreadsheet(
            xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1'))

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_read_rect_values_Scalar(self):
        sheet = self.sheet

        # get single value [D7]
        args = (sheet, xr.Edge(xr.Coords('7', 'D'), None))
        self.assertEqual(_read_rect_values(*args), [0])

        # get single value [A1]
        args = (sheet, xr.Edge(xr.Coords('1', 'A'), None))
        self.assertEqual(_read_rect_values(*args), [None])

        # get single value [H9]
        args = (sheet, xr.Edge(xr.Coords('9', 'H'), None))
        self.assertEqual(_read_rect_values(*args), [None])

    def test_read_rect_values_Vector(self):
        sheet = self.sheet

        # single value in the sheet [D7:D7]
        args = (sheet,
                xr.Edge(xr.Coords('7', 'D'), None),
                xr.Edge(xr.Coords('7', 'D'), None))
        self.assertEqual(_read_rect_values(*args), [0])

        # get whole column [D_]
        args = (sheet,
                xr.Edge(xr.Coords('1', 'D'), None),
                xr.Edge(xr.Coords('_', 'D'), None))
        res = [None, None, None, None, None, None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res)

        # get whole row [_6]
        args = (sheet,
                xr.Edge(xr.Coords('6', 'A'), None),
                xr.Edge(xr.Coords('6', '_'), None))
        res = [None, None, None, None, 0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited row in the sheet [C6:_6]
        args = (sheet,
                xr.Edge(xr.Coords('6', 'C'), None),
                xr.Edge(xr.Coords('6', '_'), None))
        res = [None, None, 0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited row in the sheet [_7:_7]
        args = (sheet,
                xr.Edge(xr.Coords('7', 'A'), None),
                xr.Edge(xr.Coords('7', '_'), None))
        res = [None, None, None, 0, None, None, None]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited row in the sheet [E6:_6]
        args = (sheet,
                xr.Edge(xr.Coords('1', 'A'), 'RD'),
                xr.Edge(xr.Coords('.', '.'), 'R'))
        res = [0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited row in the sheet [A7:D7]
        args = (sheet,
                xr.Edge(xr.Coords('7', 'A'), None),
                xr.Edge(xr.Coords('7', 'D'), None))
        res = [None, None, None, 0]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited column in the sheet [D_:D_]
        args = (sheet,
                xr.Edge(xr.Coords('^', 'D'), None),
                xr.Edge(xr.Coords('_', 'D'), None))
        res = [None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited column in the sheet [D5:D_]
        args = (sheet,
                xr.Edge(xr.Coords('5', 'D'), None),
                xr.Edge(xr.Coords('_', 'D'), None))
        res = [None, None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited column in the sheet [D_:D9]
        args = (sheet,
                xr.Edge(xr.Coords('^', 'D'), None),
                xr.Edge(xr.Coords('9', 'D'), None))
        res = [None, 0, 1, None]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited column in the sheet [D3:D9]
        args = (sheet,
                xr.Edge(xr.Coords('3', 'D'), None),
                xr.Edge(xr.Coords('9', 'D'), None))
        res = [None, None, None, None, 0, 1, None]
        self.assertEqual(_read_rect_values(*args), res)

    def test_read_rect_values(self):
        sheet = self.sheet

        # minimum matrix in the sheet [:]
        args = (sheet,
                xr.Edge(xr.Coords('^', '^'), None),
                xr.Edge(xr.Coords('_', '_'), None))
        res = [
            [None, 0, 1, 2],
            [0, None, None, None],
            [1, 5.1, 6.1, 7.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [E_:__]
        args = (sheet,
                xr.Edge(xr.Coords('^', 'E'), None),
                xr.Edge(xr.Coords('_', '_'), None))
        res = [
            [0, 1, 2],
            [None, None, None],
            [5.1, 6.1, 7.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [E7:__]
        args = (sheet,
                xr.Edge(xr.Coords('7', 'E'), None),
                xr.Edge(xr.Coords('_', '_'), None))
        res = [
            [None, None, None],
            [5.1, 6.1, 7.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited matrix in the sheet [D6:F8]
        args = (sheet,
                xr.Edge(xr.Coords('6', 'D'), None),
                xr.Edge(xr.Coords('8', 'F'), None))
        res = [
            [None, 0, 1],
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [:F8]
        args = (sheet,
                xr.Edge(xr.Coords('8', 'F'), None),
                xr.Edge(xr.Coords('^', '^'), None))
        res = [
            [None, 0, 1],
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [7:F8]
        args = (sheet,
                xr.Edge(xr.Coords('8', 'F'), None),
                xr.Edge(xr.Coords('7', '^'), None))
        res = [
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # minimum delimited matrix in the sheet [D:F8]
        args = (sheet,
                xr.Edge(xr.Coords('8', 'F'), None),
                xr.Edge(xr.Coords('8', 'D'), 'U'))
        res = [
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res)

        # delimited matrix in the sheet [A1:F8]
        args = (sheet,
                xr.Edge(xr.Coords('1', 'A'), None),
                xr.Edge(xr.Coords('8', 'F'), None))
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
                xr.Edge(xr.Coords('9', 'G'), None),
                xr.Edge(xr.Coords('.', '.'), 'D'))
        self.assertRaises(ValueError, _read_rect_values, *args)

        # delimited matrix in the sheet [F9:__]
        args = (sheet,
                xr.Edge(xr.Coords('9', 'F'), None),
                xr.Edge(xr.Coords('.', '.'), 'R'))
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

        self.sheet = xr.Spreadsheet(
            xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1'))

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

        sheet = self.sheet
        states_matrix = sheet.read_states_matrix()
        margin_coords = xr.margin_coords_from_states_matrix(states_matrix)
        xl_ma = xr.make_special_margins(*margin_coords)

        # minimum delimited column in the sheet [D7:D.(D)]
        st = xr.Edge(xr.num2a1_Coords(6, 3), None)
        nd = xr.Edge(xr.num2a1_Coords('.', '.'), 'D')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), resTarget)

        # minimum delimited column in the sheet [E6:E.(D)]
        st = xr.Edge(xr.num2a1_Coords(5, 4), None)
        nd = xr.Edge(xr.num2a1_Coords('.', '.'), 'D')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[1])

        # minimum delimited row in the sheet [E6:.6(R)]
        st = xr.Edge(xr.num2a1_Coords(5, 4), None)
        nd = xr.Edge(xr.num2a1_Coords('.', '.'), 'R')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[2])

        # minimum delimited matrix in the sheet [E6:..(RD)]
        st = xr.Edge(xr.num2a1_Coords(5, 4), None)
        nd = xr.Edge(xr.num2a1_Coords('.', '.'), 'RD')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[3])

        st = xr.Edge(xr.num2a1_Coords(5, 4), None)
        nd = xr.Edge(xr.num2a1_Coords('.', '.'), 'DR')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[3])

        # delimited matrix in the sheet [D6:F8]
        st = xr.Edge(xr.num2a1_Coords(7, 5), None)
        nd = xr.Edge(xr.num2a1_Coords(5, 3), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[4])

        # delimited matrix in the sheet [A1:F8]
        st = xr.Edge(xr.num2a1_Coords(7, 5), None)
        nd = xr.Edge(xr.num2a1_Coords(0, 0), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[5])

        # delimited row in the sheet [A7:D7]
        st = xr.Edge(xr.num2a1_Coords(6, 0), None)
        nd = xr.Edge(xr.num2a1_Coords(6, 3), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[6])

        # delimited column in the sheet [D3:D9]
        st = xr.Edge(xr.num2a1_Coords(8, 3), None)
        nd = xr.Edge(xr.num2a1_Coords(2, 3), None)
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        self.assertEqual(xr.read_capture_rect(*args), res[7])

        # minimum delimited matrix in the sheet [F7:..(UL)]
        st = xr.Edge(xr.num2a1_Coords(6, 5), None)
        nd = xr.Edge(xr.num2a1_Coords('.', '.'), 'UL')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        res = [[None, 0, 1],
               [0, 1, 2]]
        self.assertEqual(xr.read_capture_rect(*args), res)

        # minimum delimited matrix in the sheet [F7:F7:LURD]
        st = xr.Edge(xr.num2a1_Coords(6, 5), None)
        nd = xr.Edge(xr.num2a1_Coords(6, 5), None)
        rect_exp = xr._parse_rect_expansions('LURD')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd, rect_exp)
        args = (sheet, rng)
        res = [[None, 0, 1, 2],
               [0, 1, 2, None],
               [1, None, 6.1, 7.1]]
        self.assertEqual(xr.read_capture_rect(*args), res)

        # minimum delimited matrix in the sheet [F7:A1(RD)]
        st = xr.Edge(xr.num2a1_Coords(6, 5), None)
        nd = xr.Edge(xr.num2a1_Coords(0, 0), 'RD')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        res = [[0, 1],
               [1, 2]]
        self.assertEqual(xr.read_capture_rect(*args), res)

        # minimum delimited row in the sheet [_8:G8]
        st = xr.Edge(xr.num2a1_Coords(7, 6), None)
        nd = xr.Edge(xr.num2a1_Coords(7, '.'), 'L')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        res = [6.1, 7.1]
        self.assertEqual(xr.read_capture_rect(*args), res)

        # minimum delimited column in the sheet [D_:D8]
        st = xr.Edge(xr.num2a1_Coords(7, 3), None)
        nd = xr.Edge(xr.num2a1_Coords('.', 3), 'U')
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        res = [0, 1]
        self.assertEqual(xr.read_capture_rect(*args), res)

        # single value [D8]
        st = xr.Edge(xr.num2a1_Coords(7, 3), None)
        nd = None
        rng = xr.resolve_capture_rect(states_matrix, xl_ma, st, nd)
        args = (sheet, rng)
        res = [1]
        self.assertEqual(xr.read_capture_rect(*args), res)


class Spreadsheet(unittest.TestCase):

    def test_backend_Default(self):
        sheet = xr.Spreadsheet(sheet=None)
        self.assertEqual(sheet._backend, xd)

    def test_backend_Excplicit(self):
        sheet = xr.Spreadsheet(sheet=None, backend='pandalone.xlref._xlref')
        self.assertEqual(sheet._backend, xr)

        sheet = xr.Spreadsheet(sheet=None, backend='pandalone.xlref._xlrd')
        self.assertEqual(sheet._backend, xd)

    def test_get_states_matrix_Caching(self):
        sheet = xr.Spreadsheet(sheet=None)
        obj = object()
        sheet._states_matrix = obj
        self.assertEqual(sheet.read_states_matrix(), obj)

    def test_get_margin_coords_Cached(self):
        sheet = xr.Spreadsheet(sheet=None)
        obj = object()
        sheet._margin_coords = obj
        self.assertEqual(sheet.get_margin_coords(), obj)

    def test_get_margin_coords_Extracted_from_states_matrix(self):
        sheet = xr.Spreadsheet(sheet=None)
        sheet._states_matrix = np.array([
            [0, 1, 1, 0]
        ])
        margins = (xr.Coords(0, 1), xr.Coords(0, 2))
        self.assertEqual(sheet.get_margin_coords(), margins)
        sheet._margin_coords = None
        self.assertEqual(sheet.get_margin_coords(), margins)
        sheet._states_matrix = None
        self.assertEqual(sheet._margin_coords, margins)

        # Modify states_matrix but not cache.
        #
        sheet._states_matrix = np.asarray([
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ])
        self.assertEqual(sheet.get_margin_coords(), margins)
        sheet._margin_coords = None
        margins = (xr.Coords(1, 0), xr.Coords(3, 2))
        self.assertEqual(sheet.get_margin_coords(), margins)
