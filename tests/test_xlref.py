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
from pandalone.xlref._xlrd import XlrdSheet as Sheet
import sys
from tests import _tutils
from tests._tutils import check_xl_installed, xw_Workbook
import unittest

from ddt import ddt, data
import six
import xlrd

import numpy as np
from numpy import testing as npt
import pandas as pd


log = _tutils._init_logging(__name__)
xl_installed = check_xl_installed()
xr.CHECK_CELLTYPE = True

def _make_xl_margins(sheet):
    states_matrix = sheet.get_states_matrix()

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
    states_matrix = sheet.get_states_matrix()
    up, dn = sheet.get_margin_coords()
    st, nd = xr.resolve_capture_rect(states_matrix, up, dn,
                                      st_edge, nd_edge)  # or Edge(None, None))
    return xr.read_capture_rect(sheet, st, nd)


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
_all_dir_single = list('LRUD')
_all_dirs = _all_dir_single + _all_dir_pairs


@ddt
class Parse(unittest.TestCase):

    def test_parse_xl_ref_Cell_types(self):
        xl_ref = 'b1:C2'
        res = xr.parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertIsInstance(st_edge.land.row, six.string_types)
        self.assertIsInstance(st_edge.land.col, six.string_types)
        self.assertIsInstance(nd_edge.land.row, six.string_types)
        self.assertIsInstance(nd_edge.land.col, six.string_types)

    def test_parse_xl_ref_Cell_col_row_order(self):
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
        self.assertEquals(st_edge.land, xr.Cell(col='A', row='1'))
        self.assertEquals(nd_edge.land, xr.Cell(col='C', row='2'))
        self.assertEquals(st_edge.mov, 'L')
        self.assertEquals(nd_edge.mov, 'UL')

        xl_ref = 'Sheet1!A1'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['st_edge'].land, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_edge'], None)

        xl_ref = 'Sheet1!a1(l):c2(ul){"1":4,"2":"ciao"}'
        res = xr.parse_xl_ref(xl_ref)
        self.assertEquals(res['json'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['st_edge'].land, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_edge'].land, xr.Cell(col='C', row='2'))
        self.assertEquals(res['st_edge'].mov, 'L')
        self.assertEquals(res['nd_edge'].mov, 'UL')

    @data('s![[]', 's!{}[]', 's!A', 's!A1:!', 's!1:2', 's!A0:B1', )
    def test_errors_parse_xl_ref(self, case):
        self.assertRaises(ValueError, xr.parse_xl_ref, case)

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
        self.assertEquals(res['st_edge'], xr.Edge(xr.Cell('10', 'U'), 'L'))
        self.assertEquals(res['nd_edge'], xr.Edge(xr.Cell('20', 'D'), 'D'))

    def test_parse_xl_url_Bad(self):
        self.assertRaises(ValueError, xr.parse_xl_url, *('#!:{"json":"..."', ))

    def test_parse_xl_url_Only_fragment(self):
        url = '#sheet_name!UP10:DOWN20{"json":"..."}'
        res = xr.parse_xl_url(url)
        self.assertEquals(res['url_file'], '')


def make_sample_matrix():
    states_matrix = np.array([
        [0, 1, 3],
        [4, 5, 7],
        [8, 9, 11],
        [12, 13, 15],
    ])
    dn = xr.Coords(3, 2)
    args = (states_matrix, dn)

    return args


def make_states_matrix():
    states_matrix = np.array([
        # A  B  C  D  E  F
        [0, 0, 0, 0, 0, 0],  # '1'
        [0, 0, 0, 0, 0, 0],  # '2'
        [0, 0, 0, 1, 1, 1],  # '3'
        [0, 0, 1, 0, 0, 1],  # '4'
        [0, 0, 1, 1, 0, 1],  # '5'
    ], dtype=bool)
    args = (states_matrix, xr.Coords(4, 5))

    return args


@ddt
class StatesVector(unittest.TestCase):

    @data(*_all_dir_single)
    def test_extract_states_vector_1st_element(self, mov):
        args = make_sample_matrix()
        sm = args[0]
        for r in range(sm.shape[0]):
            for c in range(sm.shape[1]):
                nargs = args + (xr.Coords(r, c), mov)
                vect = xr._extract_states_vector(*nargs)[0]

                npt.assert_array_equal(vect[0], sm[r, c], str(args))

    def check_extract_states_vector(self, land_r, land_c, mov, exp_vect):
        args = make_sample_matrix()
        args += (xr.Coords(land_r, land_c), mov)
        vect = xr._extract_states_vector(*args)[0]

        npt.assert_array_equal(vect, exp_vect, str(args))

    @data(
        (1, 1, 'L', [5, 4]),
        (1, 1, 'U', [5, 1]),
        (1, 1, 'R', [5, 7]),
        (1, 1, 'D', [5, 9, 13]),
    )
    def test_extract_states_vector_Center(self, case):
        self.check_extract_states_vector(*case)

    @data(
        (0, 0, 'L', [0]),
        (0, 0, 'U', [0]),
        (0, 0, 'R', [0, 1, 3]),
        (0, 0, 'D', [0, 4, 8, 12]),

        (0, 2, 'L', [3, 1, 0]),
        (0, 2, 'U', [3]),
        (0, 2, 'R', [3]),
        (0, 2, 'D', [3, 7, 11, 15]),

        (3, 2, 'L', [15, 13, 12]),
        (3, 2, 'U', [15, 11, 7, 3]),
        (3, 2, 'R', [15]),
        (3, 2, 'D', [15]),

        (3, 0, 'L', [12]),
        (3, 0, 'U', [12, 8, 4, 0]),
        (3, 0, 'R', [12, 13, 15]),
        (3, 0, 'D', [12]),
    )
    def test_extract_states_vector_Corners(self, case):
        self.check_extract_states_vector(*case)


@ddt
class TargetOpposite(unittest.TestCase):

    def check_target_opposite_full_impl(self, *args):
        (land_row, land_col,
         moves, exp_row, exp_col) = args
        states_matrix, dn = make_states_matrix()
        argshead = (states_matrix, dn)

        land_cell = xr.Coords(land_row, land_col)
        args = argshead + (land_cell, moves)

        if exp_row:
            res = xr._target_opposite(*args)
            self.assertEqual(res, xr.Coords(exp_row, exp_col), str(args))
        else:
            with self.assertRaisesRegexp(ValueError, "No \w+-target for",
                                         msg=str(args)):
                xr._target_opposite(*args)

    def check_target_opposite_state(self, land_row, land_col, moves,
                                    exp_row=None, exp_col=None):
        self.check_target_opposite_full_impl(land_row, land_col,
                                             moves, exp_row, exp_col)

    @data(
        (0, 0, 'DR', 3, 2),
        (0, 0, 'RD', 2, 3),

        (3, 0, 'UR', 3, 2),
        (3, 0, 'RU', 3, 2),
        (3, 0, 'DR', 3, 2),
        (3, 0, 'RD', 3, 2),

        (0, 3, 'DL', 2, 3),
        (0, 3, 'LD', 2, 3),
        (0, 3, 'DR', 2, 3),
        (0, 3, 'RD', 2, 3),
    )
    def test_target_opposite_state_Basic(self, case):
        self.check_target_opposite_state(*case)

    def test_target_opposite_state_NotMovingFromMatch(self):
        coords = [(2, 3), (3, 2),
                  (2, 4), (2, 5),
                  (3, 5),
                  (4, 2), (4, 3),   (4, 5),
                  ]
        for d in _all_dirs:
            for r, c in coords:
                self.check_target_opposite_state(r, c, d, r, c)

    def test_target_opposite_state_Beyond_columns(self):
        dirs = ['L', 'LU', 'LD', 'UL', 'DL']
        for d in dirs:
            for row in [2, 3, 4]:
                self.check_target_opposite_state(row, 10, d, row, 5)
            if 'D' in d:
                self.check_target_opposite_state(0, 10, d, 2, 5)

    @data('U', 'UL', 'UR', 'LU', 'RU')
    def test_target_opposite_state_Beyond_rows1(self, moves):
        for col in [2, 3, 5]:
            self.check_target_opposite_state(10, col, moves, 4, col)
        if 'U' in moves[0]:
            self.check_target_opposite_state(10, 4, moves, 2, 4)
        if 'R' in moves:
            self.check_target_opposite_state(10, 0, moves, 4, 2)

    def test_target_opposite_state_Beyond_rows2(self):
        self.check_target_opposite_state(10, 4, 'LU', 4, 3)
        self.check_target_opposite_state(10, 4, 'RU', 4, 5)

    def test_target_opposite_state_Beyond_both(self):
        self.check_target_opposite_state(10, 10, 'UL', 4, 5)
        self.check_target_opposite_state(10, 10, 'LU', 4, 5)

    @data(*(list('UDLR') + ['UR', 'RU', 'UL', 'LU', 'DL', 'LD']))
    def test_target_opposite_state_InvalidMoves(self, moves):
        self.check_target_opposite_state(0, 0, moves)


@ddt
class TargetSame(unittest.TestCase):

    def check_target_same_full_impl(self, *args):
        (inverse_sm, land_row, land_col,
         moves, exp_row, exp_col) = args
        states_matrix, dn = make_states_matrix()
        if inverse_sm:
            states_matrix = ~states_matrix
        argshead = (states_matrix, dn)

        land_cell = xr.Coords(land_row, land_col)
        args = argshead + (land_cell, moves)

        if exp_row:
            res = xr._target_same(*args)
            self.assertEqual(res, xr.Coords(exp_row, exp_col), str(args))
        else:
            with self.assertRaisesRegexp(ValueError, "No \w+-target for",
                                         msg=str(args)):
                xr._target_same(*args)

    def check_target_same_state(self, inverse_sm, land_row, land_col, moves,
                                exp_row=None, exp_col=None):
        self.check_target_same_full_impl(inverse_sm, land_row, land_col,
                                         moves, exp_row, exp_col)

    @data(
        (True, 0, 0, 'DR', 4, 5),
        (True, 0, 0, 'DR', 4, 5),
        (True, 1, 1, 'DR', 4, 5),

        (False, 2, 3, 'DR', 2, 5),
        (False, 2, 3, 'RD', 2, 5),
    )
    def test_target_same_state_Empty_2moves(self, case):
        state, r, c, mov, rr, rc = case
        self.check_target_same_state(state, r, c, mov, rr, rc)

    @data(
        (False, 2, 5, 'LD', 4, 3),
        (False, 2, 5, 'DL', 4, 3),
    )
    def test_target_same_state_NormalWalking(self, case):
        self.check_target_same_state(*case)

    @data(
        (False, 2, 5, 'L', 2, 3),
        (False, 2, 5, 'U', 2, 5),
        (False, 2, 5, 'LU', 2, 3),
        (False, 2, 5, 'UL', 2, 3),

        (False, 4, 2, 'U', 3, 2),
        (False, 4, 2, 'RU', 3, 3),
        (False, 4, 2, 'UR', 3, 3),


        (False, 4, 5, 'L', 4, 5),
        (False, 4, 5, 'U', 2, 5),
        (False, 4, 5, 'LU', 2, 5),
        (False, 4, 5, 'UL', 2, 5),

        (True, 4, 4, 'U', 3, 4),
        (True, 4, 4, 'UR', 3, 4),
        (True, 4, 4, 'UL', 3, 4),
    )
    def test_target_same_state_InverseWalking(self, case):
        self.check_target_same_state(*case)


@ddt
class Margins(unittest.TestCase):

    def test_find_states_matrix_margins(self):
        sm = np.array([
            [0, 1, 1, 0]
        ])
        margins = (xr.Coords(0, 1), xr.Coords(0, 2))
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), margins)

        sm = np.asarray([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ])
        margins = (xr.Coords(1, 1), xr.Coords(3, 2))
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), margins)

    def test_find_states_matrix_margins_Single_cell(self):
        sm = np.array([
            [1],
        ])
        c = xr.Coords(0, 0)
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), (c, c))

        sm = np.array([
            [0, 0, 1],
        ])
        c = xr.Coords(0, 2)
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), (c, c))

        sm = np.array([
            [0, 0],
            [0, 1]
        ])
        c = xr.Coords(1, 1)
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), (c, c))

        sm = np.array([
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        c = xr.Coords(2, 1)
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), (c, c))

    def test_find_states_matrix_margins_Further_empties(self):
        sm = np.asarray([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ])
        margins = (xr.Coords(1, 1), xr.Coords(3, 2))
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), margins)

        sm = np.asarray([
            [1, 0],
            [0, 0],
            [0, 0],
        ])
        margins = (xr.Coords(0, 0), xr.Coords(0, 0))
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), margins)
        sm = np.asarray([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ])
        margins = (xr.Coords(1, 1), xr.Coords(1, 1))
        self.assertEqual(xr._margin_coords_from_states_matrix(sm), margins)

    @data(
        [[]],
        [[0], [0]],
        [[0, 0]],
        [[0, 0], [0, 0]],
    )
    def test_find_states_matrix_margins_EmptySheet(self, states_matrix):
        margins = (xr.Coords(0, 0), xr.Coords(0, 0))
        res = xr._margin_coords_from_states_matrix(np.asarray(states_matrix))
        self.assertEqual(res, margins, states_matrix)


@ddt
class Expand(unittest.TestCase):

    def make_states_matrix(self):
        states_matrix = np.array([
            # 0  1  2  3  4  5
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 1, 1, 1, 0],  # 1
            [0, 1, 0, 0, 1, 0],  # 2
            [0, 1, 1, 1, 1, 0],  # 3
            [0, 0, 0, 0, 0, 1],  # 4
        ], dtype=bool)
        return states_matrix

    def check_expand_rect(self, rect_in, exp_mov_str, rect_out, states_matrix=None):
        if states_matrix is None:
            states_matrix = self.make_states_matrix()

        exp_mov = xr._parse_expansion_moves(exp_mov_str)
        st = xr.Coords(*rect_in[:2])
        nd = xr.Coords(*rect_in[2:]) if len(rect_in) > 2 else st 
        rect_out = (xr.Coords(*rect_out[:2]), 
            xr.Coords(*rect_out[2:]) if len(rect_out) > 2 else xr.Coords(*rect_out))
        rect_got = xr._expand_rect(states_matrix, st, nd, exp_mov)
        self.assertEqual(rect_got, rect_out)

        exp_mov = xr._parse_expansion_moves(exp_mov_str)
        rect_got = xr._expand_rect(states_matrix, nd, st, exp_mov)
        self.assertEqual(rect_got, rect_out)

    @data(
        ((2, 1), 'U'),
        ((3, 1, 3, 1), 'D'),
        ((2, 1, 3, 1), 'U'),
        ((2, 1, 3, 1), 'D'),

        ((2, 1), 'U1'),
        ((3, 1, 3, 1), 'D1'),
        ((2, 1, 3, 1), 'U1'),
        ((2, 1, 3, 1), 'D1'),

        ((2, 1), 'U?'),
        ((3, 1, 3, 1), 'D?'),
        ((2, 1, 3, 1), 'U?'),
        ((2, 1, 3, 1), 'D?'),

        ((1, 3, 1, 4), 'R'),
        ((1, 2, 1, 3), 'L'),

        ((1, 3, 1, 4), 'R1'),
        ((1, 2, 1, 3), 'L1'),

        ((1, 1, 3, 4), 'LURD'),
    )
    def test_StandStill(self, case):
        case += (case[0], )
        self.check_expand_rect(*case)

    @data(
        ((2, 5), 'R'),
        ((4, 0, 4, 4), 'D'),

        ((4, 5), 'R'),
        ((4, 5), 'D'),

        ((0, 1, 0, 2), 'U'),
        ((1, 0, 3, 0), 'L'),

        ((0, 0), 'U'),
        ((0, 0), 'L'),
    )
    def test_StandStill_beyondMargins(self, case):
        case += (case[0], )
        self.check_expand_rect(*case)

    @data(
        ((3, 1, 3, 5), 'U', (1, 1, 3, 5)),
        ((2, 1, 3, 2), 'RU', (1, 1, 3, 4)),
        ((2, 3, 2, 3), 'LURD', (1, 1, 3, 4)),

        ((1, 1, 3, 2), 'LURD', (1, 1, 3, 4)),
        ((2, 1, 3, 2), 'LURD', (1, 1, 3, 4)),
        ((1, 2, 3, 2), 'LURD', (1, 1, 3, 4)),
        ((1, 1, 3, 2), 'DLRU', (1, 1, 3, 4)),
        ((2, 1, 3, 2), 'DLRU', (1, 1, 3, 4)),
        ((1, 2, 3, 2), 'DLRU', (1, 1, 3, 4)),
    )
    def test_expand_rect(self, case):
        self.check_expand_rect(*case)

    @data(
        ((3, 1, 3, 2), 'U1R1', (2, 1, 3, 3)),
        ((3, 1, 3, 2), 'R1U1', (2, 1, 3, 3)),
        ((3, 1, 3, 2), 'U?R?', (2, 1, 3, 3)),
        ((3, 1, 3, 2), 'R?U?', (2, 1, 3, 3)),

        ((2, 1, 3, 2), 'R1U1', (1, 1, 3, 3)),
        ((2, 1, 3, 2), 'U1R1', (1, 1, 3, 3)),

        ((3, 3, 4, 3), 'L1', (3, 2, 4, 3)),
    )
    def test_Single(self, case):
        self.check_expand_rect(*case)

    @data(
        ((2, 1, 6, 1), 'R', (2, 1, 6, 5)),
    )
    def test_OutOfBounds(self, case):
        self.check_expand_rect(*case)

    def test_spiral(self):
        states_matrix = np.array([
            # 0  1  2  3  4  5
            [0, 1, 0, 0, 0, 0],  # 0
            [0, 0, 1, 0, 1, 0],  # 1
            [0, 0, 1, 1, 0, 0],  # 2
            [0, 1, 0, 0, 1, 0],  # 3
            [0, 0, 0, 0, 0, 1],  # 4
        ], dtype=bool)
        self.check_expand_rect((2, 2, 2, 2), 'LURD', (0, 1, 3, 4),
                               states_matrix=states_matrix)

    def test_spiral_Broken(self):
        states_matrix = np.array([
            # 0  1  2  3  4  5
            [0, 1, 0, 0, 0, 0],  # 0
            [0, 0, 1, 0, 0, 0],  # 1
            [0, 0, 1, 1, 0, 0],  # 2
            [0, 1, 0, 1, 1, 0],  # 3
            [0, 0, 0, 0, 0, 1],  # 4
        ], dtype=bool)
        self.check_expand_rect((2, 2, 2, 2), 'LURD', (0, 1, 3, 4),
                               states_matrix=states_matrix)


@ddt
class Capture(unittest.TestCase):

    def make_states_matrix(self):
        states_matrix = np.array([
            # A  B  C  D  E  F
            [0, 0, 0, 0, 0, 0],  # '1'
            [0, 0, 0, 0, 0, 0],  # '2'
            [0, 0, 0, 1, 1, 1],  # '3'
            [0, 0, 1, 0, 0, 1],  # '4'
            [0, 0, 1, 1, 0, 1],  # '5'
        ], dtype=bool)
        up, dn = xr._margin_coords_from_states_matrix(states_matrix)
        args = (states_matrix, up, dn)

        return args

    def check_resolve_capture_rect(self, *args):
        #     st_row, st_col, st_mov,
        #     nd_row, nd_col, nd_mov,
        #     res_st_row, res_st_col, res_nd_row, res_nd_col
        argshead = self.make_states_matrix()

        st_edge = xr.Edge(xr.Cell(*args[0:2]), args[2], '+')
        nd_edge = xr.Edge(xr.Cell(*args[3:5]), args[5], '+')
        res = (xr.Coords(*args[6:8]),
               xr.Coords(*args[8:10]))
        args = argshead + (st_edge, nd_edge)
        self.assertEqual(xr.resolve_capture_rect(*args), res, str(args))

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
        ('3', 'D', 'R', '_', '_', None, 2, 5, 4, 5),
        ('3', 'D', 'UR', '_', '_', None, 2, 5, 4, 5),
        ('3', 'D', 'RU', '_', '_', None, 2, 5, 4, 5),
        ('3', 'D', 'DR', '_', '_', None, 2, 5, 4, 5),
        ('3', 'D', 'RD', '_', '_', None, 2, 5, 4, 5),

        ('3', 'D', 'L', '_', '_', None, 2, 3, 4, 5),
        ('3', 'D', 'UL', '_', '_', None, 2, 3, 4, 5),
        ('3', 'D', 'LU', '_', '_', None, 2, 3, 4, 5),
        ('3', 'D', 'DL', '_', '_', None, 2, 3, 4, 5),
        ('3', 'D', 'LD', '_', '_', None, 2, 3, 4, 5),

        ('3', 'E', 'R', '_', '_', None, 2, 5, 4, 5),
        ('3', 'E', 'UR', '_', '_', None, 2, 5, 4, 5),
        ('3', 'E', 'RU', '_', '_', None, 2, 5, 4, 5),
        ('3', 'E', 'DR', '_', '_', None, 2, 5, 4, 5),
        ('3', 'E', 'RD', '_', '_', None, 2, 5, 4, 5),

        ('3', 'E', 'L', '_', '_', None, 2, 3, 4, 5),
        ('3', 'E', 'UL', '_', '_', None, 2, 3, 4, 5),
        ('3', 'E', 'LU', '_', '_', None, 2, 3, 4, 5),
        ('3', 'E', 'DL', '_', '_', None, 2, 3, 4, 5),
        ('3', 'E', 'LD', '_', '_', None, 2, 3, 4, 5),

        ('3', 'E', 'L', '_', '_', None, 2, 3, 4, 5),
    )
    def test_resolve_capture_rect_Target_fromFull_st(self, case):
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
                                       [1, 1]
                                       ], dtype=bool)
        self.sheet = Sheet(
            xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1'))

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_parse_rect_values(self):
        # row vector in the sheet [B2:B_]
        args = (self.sheet, xr.Coords(1, 1), xr.Coords(7, 1))
        res = [datetime(1900, 8, 2), True, None, None, 'hi', 1.4, 5]
        self.assertEqual(xr.read_capture_rect(*args), res, str(args))

    def test_comparison_vs_pandas_parse_cell(self):

        # row vector in the sheet [B2:B_]
        args = (self.sheet, xr.Coords(1, 1), xr.Coords(7, 1))

        res = xr.read_capture_rect(*args)

        df = pd.read_excel(self.tmp, 'Sheet1')[0]

        # in pandas None values are converted in float('nan')
        df = df.where(pd.notnull(df), None).values.tolist()

        self.assertEqual(df, res)


class Read2(unittest.TestCase):  # FIXME: Why another class

    def setUp(self):
        from tempfile import mkstemp
        self.tmp = '%s.xlsx' % mkstemp()[1]
        xl = [
            [None, None, None],
            [5.1, 6.1, 7.1]
        ]

        _make_sample_sheet(self.tmp, xl, 'Sheet1', startrow=5, startcol=3)

        self.sheet = Sheet(
            xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1'))

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
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # get whole row [_6]
        args = (sheet,
                xr.Edge(xr.Cell('6', 'A'), None),
                xr.Edge(xr.Cell('6', '_'), None))
        res = [None, None, None, None, 0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # minimum delimited row in the sheet [C6:_6]
        args = (sheet,
                xr.Edge(xr.Cell('6', 'C'), None),
                xr.Edge(xr.Cell('6', '_'), None))
        res = [None, None, 0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # minimum delimited row in the sheet [_7:_7]
        args = (sheet,
                xr.Edge(xr.Cell('7', 'A'), None),
                xr.Edge(xr.Cell('7', '_'), None))
        res = [None, None, None, 0, None, None, None]
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # minimum delimited row in the sheet [E6:_6]
        args = (sheet,
                xr.Edge(xr.Cell('1', 'A'), 'RD'),
                xr.Edge(xr.Cell('.', '.'), 'R'))
        res = [0, 1, 2]
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # delimited row in the sheet [A7:D7]
        args = (sheet,
                xr.Edge(xr.Cell('7', 'A'), None),
                xr.Edge(xr.Cell('7', 'D'), None))
        res = [None, None, None, 0]
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # minimum delimited column in the sheet [D_:D_]
        args = (sheet,
                xr.Edge(xr.Cell('^', 'D'), None),
                xr.Edge(xr.Cell('_', 'D'), None))
        res = [None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res, str(args))

        # minimum delimited column in the sheet [D5:D_]
        args = (sheet,
                xr.Edge(xr.Cell('5', 'D'), None),
                xr.Edge(xr.Cell('_', 'D'), None))
        res = [None, None, 0, 1]
        self.assertEqual(_read_rect_values(*args), res, str(args))

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
                xr.Edge(xr.Cell('8', 'D'), 'U', '+'))
        res = [
            [0, None, None],
            [1, 5.1, 6.1]
        ]
        self.assertEqual(_read_rect_values(*args), res, str(args))

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
        self.assertEqual(_read_rect_values(*args), res, str(args))

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

        self.sheet = Sheet(
            xlrd.open_workbook(self.tmp).sheet_by_name('Sheet1'))

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_xlwings_vs_get_xl_table(self):
        import xlwings as xw
        # load Workbook for --> xlwings
        with xw_Workbook(self.tmp):
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
        states_matrix = sheet.get_states_matrix()
        up, dn = xr._margin_coords_from_states_matrix(states_matrix)

        # minimum delimited column in the sheet [D7:D.(D)]
        st = xr.Edge(xr.coords2Cell(6, 3), None)
        nd = xr.Edge(xr.coords2Cell('.', '.'), 'D')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        args = (sheet, ) + rect
        self.assertEqual(xr.read_capture_rect(sheet, *rect), resTarget)

        # minimum delimited column in the sheet [E6:E.(D)]
        st = xr.Edge(xr.coords2Cell(5, 4), None)
        nd = xr.Edge(xr.coords2Cell('.', '.'), 'D')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[1])

        # minimum delimited row in the sheet [E6:.6(R)]
        st = xr.Edge(xr.coords2Cell(5, 4), None)
        nd = xr.Edge(xr.coords2Cell('.', '.'), 'R')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[2])

        # minimum delimited matrix in the sheet [E6:..(RD)]
        st = xr.Edge(xr.coords2Cell(5, 4), None)
        nd = xr.Edge(xr.coords2Cell('.', '.'), 'RD')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[3])

        st = xr.Edge(xr.coords2Cell(5, 4), None)
        nd = xr.Edge(xr.coords2Cell('.', '.'), 'DR')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[3])

        # delimited matrix in the sheet [D6:F8]
        st = xr.Edge(xr.coords2Cell(7, 5), None)
        nd = xr.Edge(xr.coords2Cell(5, 3), None)
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[4])

        # delimited matrix in the sheet [A1:F8]
        st = xr.Edge(xr.coords2Cell(7, 5), None)
        nd = xr.Edge(xr.coords2Cell(0, 0), None)
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[5])

        # delimited row in the sheet [A7:D7]
        st = xr.Edge(xr.coords2Cell(6, 0), None)
        nd = xr.Edge(xr.coords2Cell(6, 3), None)
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[6])

        # delimited column in the sheet [D3:D9]
        st = xr.Edge(xr.coords2Cell(8, 3), None)
        nd = xr.Edge(xr.coords2Cell(2, 3), None)
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res[7])

        # minimum delimited matrix in the sheet [F7:..(UL)]
        st = xr.Edge(xr.coords2Cell(6, 5), None)
        nd = xr.Edge(xr.coords2Cell('.', '.'), 'UL')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        res = [[None, 0, 1],
               [0, 1, 2]]
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res, str(args))

        # minimum delimited matrix in the sheet [F7:F7:LURD]
        st = xr.Edge(xr.coords2Cell(6, 5), None)
        nd = xr.Edge(xr.coords2Cell(6, 5), None)
        exp_moves = xr._parse_expansion_moves('LURD')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd, exp_moves)
        res = [[None, 0, 1, 2],
               [0, 1, 2, None],
               [1, None, 6.1, 7.1]]
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res, str(args))

        # minimum delimited matrix in the sheet [F7:A1(RD)]
        st = xr.Edge(xr.coords2Cell(6, 5), None)
        nd = xr.Edge(xr.coords2Cell(0, 0), 'RD')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        res = [[0, 1],
               [1, 2]]
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res, str(args))

        # minimum delimited row in the sheet [_8:G8]
        st = xr.Edge(xr.coords2Cell(7, 6), None, '+')
        nd = xr.Edge(xr.coords2Cell(7, '.'), 'L', '+')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        res = [6.1, 7.1]
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res, str(args))

        # minimum delimited column in the sheet [D_:D8]
        st = xr.Edge(xr.coords2Cell(7, 3), None, '+')
        nd = xr.Edge(xr.coords2Cell('.', 3), 'U', '+')
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        res = [0, 1]
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res, str(args))

        # single value [D8]
        st = xr.Edge(xr.coords2Cell(7, 3), None, '+')
        nd = None
        rect = xr.resolve_capture_rect(states_matrix, up, dn, st, nd)
        res = [1]
        self.assertEqual(xr.read_capture_rect(sheet, *rect), res, str(args))


class _Spreadsheet(unittest.TestCase):

    def test_get_states_matrix_Caching(self):
        sheet = xr._Spreadsheet(sheet=None)
        obj = object()
        sheet._states_matrix = obj
        self.assertEqual(sheet.get_states_matrix(), obj)

    def test_get_margin_coords_Cached(self):
        sheet = xr._Spreadsheet(sheet=None)
        obj = object()
        sheet._margin_coords = obj
        self.assertEqual(sheet.get_margin_coords(), obj)

    def test_get_margin_coords_Extracted_from_states_matrix(self):
        sheet = xr._Spreadsheet(sheet=None)
        sheet._states_matrix = np.array([
            [0, 1, 1, 0]
        ])
        margins = (xr.Cell(0, 1), xr.Cell(0, 2))
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
        margins = (xr.Cell(1, 0), xr.Cell(3, 2))
        self.assertEqual(sheet.get_margin_coords(), margins)
