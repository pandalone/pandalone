#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import contextlib
from datetime import datetime
import doctest
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from ddt import ddt, data
from numpy import testing as npt
import six
import xlrd

import numpy as np
from pandalone.xlref import _xlrd as xd
from pandalone.xlref import _xlref as xr
from pandalone.xlref._xlrd import XlrdSheet
import pandas as pd
from tests import _tutils
from tests._tutils import check_xl_installed, xw_Workbook

from ._tutils import assertRaisesRegex, CustomAssertions


log = _tutils._init_logging(__name__)
xl_installed = check_xl_installed()
xr.CHECK_CELLTYPE = True


def _make_xl_margins(sheet):
    states_matrix = sheet.get_states_matrix()

    up = (0, 0)

    dn = (sheet.nrows - 1, sheet.ncols - 1)

    return states_matrix, up, dn


def _write_sample_sheet(path, matrix, sheet_name, **kws):
    df = pd.DataFrame(matrix)
    with pd.ExcelWriter(path) as w:
        if isinstance(sheet_name, tuple):
            for s in sheet_name:
                df.to_excel(w, s, **kws)
        else:
            df.to_excel(w, sheet_name, **kws)


def _make_local_url(fname, fragment=''):
    fpath = os.path.abspath(fname)
    return 'file:///{}#{}'.format(fpath, fragment)


def _read_rect_values(sheet, st_edge, nd_edge, dims, scream=True):
    states_matrix = sheet.get_states_matrix()
    up, dn = sheet.get_margin_coords()
    st, nd = xr.resolve_capture_rect(states_matrix, up, dn,
                                     st_edge, nd_edge)  # or Edge(None, None))
    v = sheet.read_rect(st, nd)
    if dims is not None:
        v = xr._redim(v, dims, scream=scream)

    return v


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
        res = xr._parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertIsInstance(st_edge.land.row, six.string_types)
        self.assertIsInstance(st_edge.land.col, six.string_types)
        self.assertIsInstance(nd_edge.land.row, six.string_types)
        self.assertIsInstance(nd_edge.land.col, six.string_types)

    def test_parse_xl_ref_Cell_col_row_order(self):
        xl_ref = 'b1:C2'
        res = xr._parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertTrue(st_edge.land.row.isalnum())
        self.assertTrue(st_edge.land.col.isalpha())
        self.assertTrue(nd_edge.land.row.isalnum())
        self.assertTrue(nd_edge.land.col.isalpha())

    def test_parse_xl_ref_all_upper(self):
        xl_ref = 'b1(uL):C2(Dr):Lur2D'
        res = xr._parse_xl_ref(xl_ref)
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
        res = xr._parse_xl_ref(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertEquals(res['sheet'], 'Sheet1')
        self.assertEquals(st_edge.land, xr.Cell(col='A', row='1'))
        self.assertEquals(nd_edge.land, xr.Cell(col='C', row='2'))
        self.assertEquals(st_edge.mov, 'L')
        self.assertEquals(nd_edge.mov, 'UL')

        xl_ref = 'Sheet1!A1'
        res = xr._parse_xl_ref(xl_ref)
        self.assertEquals(res['st_edge'].land, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_edge'], None)

        xl_ref = 'Sheet1!a1(l):c2(ul){"1":4,"2":"ciao"}'
        res = xr._parse_xl_ref(xl_ref)
        self.assertEquals(res['json'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['st_edge'].land, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_edge'].land, xr.Cell(col='C', row='2'))
        self.assertEquals(res['st_edge'].mov, 'L')
        self.assertEquals(res['nd_edge'].mov, 'UL')

    @data('s![[]', 's!{}[]', 's!A', 's!A1:!', 's!1:2', 's!A0:B1', )
    def test_errors_parse_xl_ref(self, case):
        self.assertRaises(ValueError, xr._parse_xl_ref, case)

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
            with assertRaisesRegex(self, ValueError, "No \w+-target for",
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
            with assertRaisesRegex(self, ValueError, "No \w+-target for",
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


@ddt
class Redim(unittest.TestCase):

    @data(
        ([],                (0,)),
        ([[[]]],            (1, 1, 0)),
        ([[1, 2, 3]],       (1, 3)),
        ([[[1, 2], [3, 4]]], (1, 2, 2)),
        ([[[[1, 2]], [[3, 4]]]], (1, 2, 1, 2)),
        (3.14,              ()),
        ('ff',              ()),
        (None,              ()),
    )
    def test_shape(self, case):
        self.assertEqual(xr._shape(case[0]), case[1])

    def check_redim_array(self, case):
        arr, dim, exp = case
        res = xr._redim(arr, dim, dim)

        self.assertEqual(res, exp)
#         if isinstance(exp, list):
#             exp = np.asarray(exp, dtype=object)
#             npt.assert_equal(res, exp)

    @data(
        ([1],       1,  [1]),
        ([1],       2,  [[1]]),
        ([13],      3,  [[[13]]]),
        ([1, 2],    1,  [1, 2]),
        ([1, 2],    2,  [[1, 2]]),
        ([1, 2],    3,  [[[1, 2]]]),
    )
    def test_upscale(self, case):
        self.check_redim_array(case)

    @data(
        ([[1, 2]],   1,  [1, 2]),
        ([[[1, 2]]], 2,  [[1, 2]]),
        ([[[1, 2]]], 1,  [1, 2]),
    )
    def test_downscale(self, case):
        self.check_redim_array(case)

    @data(
        ([None],    0,  None),
        ([['str']], 0,  'str'),
        ([[[3.14]]], 0, 3.14),
        ('str',     0,  'str'),
        (None,      0,  None),
        (5.1,       0,  5.1),
    )
    def test_zero(self, case):
        self.check_redim_array(case)

    @data(
        ([],        3,  [[[]]]),
        ([[]],      3,  [[[]]]),
        ([[]],      2,  [[]]),
        ([],        2,  [[]]),
        ([],        1,  []),
        ([[]],      1,  []),
    )
    def test_empty(self, case):
        self.check_redim_array(case)

    @data(
        ([[], []], 1,
         r"Cannot reduce shape\(2, 0\) from 2-->1!"),
        ([[1, 2], [3, 4]], 1,
         r"Cannot reduce shape\(2, 2\) from 2-->1!"),
        ([[[1, 1]], [[2, 2]]], 1,
         r"Cannot reduce shape\(2, 1, 2\) from 3-->1!"),
    )
    def test_cannot_downscale(self, case):
        arr, ndim, err = case
        with assertRaisesRegex(self, ValueError, err, msg=str((arr, ndim))):
            res = xr._redim(arr, ndim, True)
            print(res)

    @data(
        ([1, 2, 3], 0,
         r"Cannot reduce shape\(3,\) from 1-->0!"),
        ([[1, 2], [3, 4]], 0,
         r"Cannot reduce shape\(2, 2\) from 2-->0!"),
        ([[1, 2]], 0,
         r"Cannot reduce shape\(1, 2\) from 2-->0!"),
        ([1, 2], 0,
         r"Cannot reduce shape\(2,\) from 1-->0!"),
        ([], 0,
         r"Cannot reduce shape\(0,\) from 1-->0!"),
        ([[]], 0,
         r"Cannot reduce shape\(1, 0\) from 2-->0!"),
    )
    def test_unreducableZero(self, case):
        arr, ndim, err = case
        with assertRaisesRegex(self, ValueError, err, msg=str((arr, ndim))):
            res = xr._redim(arr, ndim, True)
            print(res)

    @data(
        (dict(),                        (None, False)),
        (dict(dims=None),               (None, False)),
        (dict(dims=[0, 1, 1, 1, 4]),        ((0, 1, 1, 1, 4), False)),
        (dict(dims=[0, 1, 1, 1, 4, True]),  ((0, 1, 1, 1, 4), True)),

        (dict(dims=[0, 1, 1, 1, 4, 5]),     ((0, 1, 1, 1, 4), True)),
    )
    def test_json_extract_dims(self, case):
        json, exp = case
        log = MagicMock()
        res = xr._json_extract_dims(json, log)
        self.assertEqual(res, exp)
        self.assertEqual(len(log.mock_calls), 0, log.mock_calls)

    @data(
        (dict(dims=[0, 1, 1, 1, 4, 5, 6]),     ((0, 1, 1, 1, 4), True), 1),
        (dict(dims=[0, 1, 1, 1, 4, 5, 6, 7]), ((0, 1, 1, 1, 4), True), 1),
    )
    def test_json_extract_dims_ExtraArgs(self, case):
        json, exp, mock_calls = case
        log = MagicMock()
        res = xr._json_extract_dims(json, log)
        self.assertEqual(res, exp)
        self.assertEqual(len(log.mock_calls), mock_calls, log.mock_calls)


@ddt
class ReadRect(unittest.TestCase):

    def setUp(self):
        arr = np.array([
            # A     B       C      D
            [None, None,   None,  None],  # 1
            [None, None,   None,  None],  # 2
            [None, np.NaN, 5.1,   7.1],   # 3
            [None, None,   43,    'str'],  # 4
            [None, -1,     None,  None],  # 5
        ])
        self.sheet = ArraySheet(arr)

    def check_read_capture_rect(self, case):
        sheet = self.sheet

        e1 = case[0]
        e1 = xr.Edge(xr.Cell(*e1[:2]), *e1[2:])
        e2 = case[1]
        e2 = e2 and xr.Edge(xr.Cell(*e2[:2]), *e2[2:])
        args = (sheet, e1, e2, case[2])
        res = _read_rect_values(*args)
        exp = case[3]
        if isinstance(exp, list):
            exp = np.asarray(exp, dtype=object)
        else:
            npt.assert_array_equal(res, exp)

    @data(
        # Edge-1            Edge2       Dims    Result
        (('1', 'A'),        None,       None,   None),
        (('4', 'D'),        None,       None,   'str'),
        (('1', 'A', 'DR'),  None,       None,   np.NaN),
        (('^', '^'),        None,       None,   np.NaN),
        (('_', '^'),        None,       None,   -1),
        (('_', 'C', 'U'),   None,       None,   43),
    )
    def test_Scalar(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1            Edge2       Dims    Result
        (('1', 'A'),        None,       0,      None),
        (('4', 'D'),        None,       1,      ['str']),
        (('1', 'A', 'DR'),  None,       2,      [[np.NaN]]),
        (('^', '^'),        None,       3,      [[[np.NaN]]]),
        (('_', '^'),        None,       0,      -1),
        (('_', 'C', 'U'),   None,       1,      [43]),
    )
    def test_Scalar_withDims(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1         Edge2          Dims    Result
        (('1', 'A'),    ('1', 'A'),     None,   [None]),
        (('4', 'D'),    ('4', 'D'),     None,   ['str']),
        (('1', 'A', 'DR'), ('.', '.'),  None,   [np.NaN]),
        (('^', '^'),    ('^', '^'),     None,   [np.NaN]),
        (('_', '^'),    ('_', '^'),     None,   [-1]),
        (list('_CU'),   list('^CD+'),   None,   [43]),
    )
    def test_Cell(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1         Edge2          Dims    Result
        (('1', 'A'),    ('1', 'A'),     2,      [[None]]),
        (('4', 'D'),    ('4', 'D'),     3,      [[['str']]]),
        (('1', 'A', 'DR'), ('.', '.'),  0,      np.NaN),
        (('^', '^'),    ('^', '^'),     1,      [np.NaN]),
        (('_', '^'),    ('_', '^'),     2,      [[-1]]),
        (list('_CU'),   list('^CD+'),   3,      [[[43]]]),
    )
    def test_Cell_withDims(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1         Edge2          Dims    Result
        (('1', 'A'),    ('1', '_'),     None,   [None] * 4),
        (('3', 'A'),    list('.BR+'),   None,   [None, np.NaN, 5.1, 7.1]),
        (list('^_L'),   list('^AR'),    None,   [np.NaN, 5.1, 7.1]),
        (('_', '_', 'UL'), list('..L'), None,   [43, 'str']),
        (('_', '^'),    ('_', '_'),     None,   [-1, None, None]),
        (list('^CR'),   list('^DR'),    None,   [5.1, 7.1]),
    )
    def test_Row(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1         Edge2          Dims    Result
        (('1', 'A'),    ('1', '_'),     1,      [None] * 4),
        (('3', 'A'),    list('.BR+'),   2,      [[None, np.NaN,  5.1, 7.1]]),
        (list('^_L'),   list('^AR'),    3,      [[[np.NaN, 5.1, 7.1]]]),
        (('_', '_', 'UL'), list('..L'), 2,      [[43, 'str']]),
        (('_', '^'),    ('_', '_'),     1,      [-1, None, None]),
        (list('^CR'),   list('^DR'),    2,      [[5.1, 7.1]]),
    )
    def test_Row_withDims(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1         Edge2          Dims    Result
        (('1', 'A'),    ('_', 'A'),     None,   [[None] * 5]),
        (('^', '^'),    list('_^'),     None,   [[5.1,  None, -1]]),
        (('1', 'B'),    ('_', 'A', 'RU'), None,
         [[None, None, np.NaN,  None, -1]]),
        (('1', '_', 'LD'), list('..D'), None,   [[7.1,  'str']]),
        (('_', 'C'), list('1CD+'),      None,   [[np.NaN, 43, None]]),
        (('_', '_'), ('1', '_', 'D'),   None,   [[7.1,  'str', None]]),
    )
    def test_Col(self, case):
        self.check_read_capture_rect(case)

    @data(
        # Edge-1         Edge2          Dims    Result
        (('1', 'A'),    ('_', 'A'),     1,      [None] * 5),
        (('^', '^'),    list('_^'),     2,      [[5.1,  None, -1]]),
        (('1', 'B'),    ('_', 'A', 'RU'), 3,
         [[[None, None, np.NaN,  None, -1]]]),
        (('1', '_', 'LD'), list('..D'), 2,      [[7.1,  'str']]),
        (('_', 'C'), list('1CD+'),      1,      [np.NaN, 43, None]),
        (('_', '_'), ('1', '_', 'D'),   2,      [[7.1,  'str', None]]),
    )
    def test_Col_withDims(self, case):
        self.check_read_capture_rect(case)


_xlref.read(
    'A1:..(D):{"pipe":[{"func": "redim", "kws":{"col": [null, 1]}}, "numpy" ], "opts":{"show_help": 1}}', sheets)


@ddt
class VsPandas(unittest.TestCase, CustomAssertions):

    @contextlib.contextmanager
    def sample_xl_file(self, matrix, **df_write_kws):
        try:
            tmp_file = '%s.xlsx' % tempfile.mktemp()
            _write_sample_sheet(tmp_file, matrix, 'Sheet1', **df_write_kws)

            yield tmp_file
        finally:
            try:
                os.unlink(tmp_file)
            except:
                log.warning("Failed deleting %s!", tmp_file, exc_info=1)

    dt = datetime(1900, 8, 2)
    m1 = np.array([
        # A     B       C      D       E
        [1,    True,   None, False,  None],   # 1
        [5,    True,   dt,    u'',    3.14],  # 2
        [7,    False,  5.1,   7.1,    ''],    # 3
        [9,    True,   43,    'str',  dt],    # 4
    ])

    def test_pandas_can_write_multicolumn(self):
        df = pd.DataFrame([1, 2])
        df.columns = pd.MultiIndex.from_arrays([list('A'), list('a')])
        err = "Writing as Excel with a MultiIndex is not yet implemented."
        msg = ("\n\nTIP: Pandas-%s probably saves DFs with MultiIndex columns now. \n"
               "     Update _xlref._to_df() accordingly!")
        with assertRaisesRegex(self, NotImplementedError, err,
                               msg=msg % pd.__version__):
            try:
                tmp_file = '%s.xlsx' % tempfile.mktemp()
                df.to_excel(tmp_file)
            finally:
                try:
                    os.unlink(tmp_file)
                except:
                    pass

    def check_vs_read_df(self, table, st, nd, write_df_kws={}, parse_df_kws={}):
        with self.sample_xl_file(table, **write_df_kws) as xl_file:
            pd_df = pd.read_excel(xl_file, 'Sheet1')
            xlrd_wb = xlrd.open_workbook(xl_file)
            self.sheet = XlrdSheet(xlrd_wb.sheet_by_name('Sheet1'))
            xlref_res = xr.read_capture_rect(self.sheet, st, nd)
            xlref_df = xr._to_df(xlref_res, **parse_df_kws)

            msg = '\n---\n%s\n--\n%s\n-\n%s' % (xlref_res, xlref_df, pd_df)
            self.assertTrue(xlref_df.equals(pd_df), msg=msg)

    def test_vs_read_df(self):
        self.check_vs_read_df(self.m1.tolist(),
                              xr.Coords(0, 0), xr.Coords(4, 5),
                              parse_df_kws=dict(header=0))

    @unittest.expectedFailure  # Dims mismatch.
    @data(
        *range(m1.shape[0])
    )
    def test_vs_read_df_Row(self, i):
        self.check_vs_read_df(self.m1[i, :].tolist(),
                              xr.Coords(1, 1), xr.Coords(1, 5),
                              parse_df_kws=dict(header=None))

    @data(
        *range(m1.shape[1])
    )
    def test_vs_read_df_Col(self, i):
        self.check_vs_read_df(self.m1[:, i].tolist(),
                              xr.Coords(1, 1), xr.Coords(4, 1),
                              parse_df_kws=dict(header=None))


@unittest.skipIf(not xl_installed, "Cannot test xlwings without MS Excel.")
class VsXlwings(unittest.TestCase):

    def setUp(self):
        self.tmp = '%s.xlsx' % tempfile.mktemp()[1]
        xl = [
            [1, 2, None],
            [None, 6.1, 7.1]
        ]
        _write_sample_sheet(self.tmp, xl, 'Sheet1', startrow=5, startcol=3)

        xlrd_wb = xlrd.open_workbook(self.tmp)
        self.sheet = XlrdSheet(xlrd_wb.sheet_by_name('Sheet1'))

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    def test_xlwings_vs_get_xl_table(self):
        import xlwings as xw  # @UnresolvedImport
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
        rect = xr.resolve_capture_rect(
            states_matrix, up, dn, st, nd, exp_moves)
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
        sheet = xr._Spreadsheet()
        obj = object()
        sheet._states_matrix = obj
        self.assertEqual(sheet.get_states_matrix(), obj)

    def test_get_margin_coords_Cached(self):
        sheet = xr._Spreadsheet()
        obj = object()
        sheet._margin_coords = obj
        self.assertEqual(sheet.get_margin_coords(), obj)

    def test_get_margin_coords_Extracted_from_states_matrix(self):
        sheet = xr._Spreadsheet()
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


class ArraySheet(xr._Spreadsheet):

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def _read_states_matrix(self):
        return ~np.equal(self._arr, None)

    def read_rect(self, st, nd):
        if nd is None:
            return self._arr[st]
        rect = np.array([st, nd]) + [[0, 0], [1, 1]]
        return self._arr[slice(*rect[:, 0]), slice(*rect[:, 1])].tolist()
