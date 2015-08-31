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

from ddt import ddt, data
from future import utils as fututis  # @UnresolvedImport
from future.backports import ChainMap
from numpy import testing as npt
from past.builtins import basestring
import xlrd

import itertools as itt
import numpy as np
from pandalone.xlref import _xlrd as xd
from pandalone.xlref import _xlref as xr
from pandalone.xlref._xlrd import XlrdSheet
import pandas as pd
from tests import _tutils
from tests._tutils import (check_xl_installed, xw_Workbook)

from ._tutils import assertRaisesRegex, CustomAssertions


try:
    from unittest.mock import MagicMock, patch, sentinel
except ImportError:
    from mock import MagicMock, patch, sentinel


log = _tutils._init_logging(__name__)
xl_installed = check_xl_installed()
xr.CHECK_CELLTYPE = True


def _write_sample_sheet(path, matrix, sheet_name, **kwds):
    df = pd.DataFrame(matrix)
    with pd.ExcelWriter(path) as w:
        if isinstance(sheet_name, tuple):
            for s in sheet_name:
                df.to_excel(w, s, **kwds)
        else:
            df.to_excel(w, sheet_name, **kwds)


def _make_local_url(fname, fragment=''):
    fpath = os.path.abspath(fname)
    return 'file:///{}#{}'.format(fpath, fragment)


def _read_rect_values(sheet, st_edge, nd_edge, dims):
    states_matrix = sheet.get_states_matrix()
    margin_coords = sheet.get_margin_coords()
    st, nd = xr.resolve_capture_rect(states_matrix, margin_coords,
                                     st_edge, nd_edge)  # or Edge(None, None))
    v = sheet.read_rect(st, nd)
    if dims is not None:
        v = xr._redim(v, dims)

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
        res = xr._parse_xlref_fragment(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertIsInstance(st_edge.land.row, basestring)
        self.assertIsInstance(st_edge.land.col, basestring)
        self.assertIsInstance(nd_edge.land.row, basestring)
        self.assertIsInstance(nd_edge.land.col, basestring)

    def test_parse_xl_ref_Cell_col_row_order(self):
        xl_ref = 'b1:C2'
        res = xr._parse_xlref_fragment(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertTrue(st_edge.land.row.isalnum())
        self.assertTrue(st_edge.land.col.isalpha())
        self.assertTrue(nd_edge.land.row.isalnum())
        self.assertTrue(nd_edge.land.col.isalpha())

    def test_parse_xl_ref_all_upper(self):
        xl_ref = 'b1(uL):C2(Dr):Lur2D'
        res = xr._parse_xlref_fragment(xl_ref)
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
        res = xr._parse_xlref_fragment(xl_ref)
        st_edge = res['st_edge']
        nd_edge = res['nd_edge']
        self.assertEquals(res['sh_name'], 'Sheet1')
        self.assertEquals(st_edge.land, xr.Cell(col='A', row='1'))
        self.assertEquals(nd_edge.land, xr.Cell(col='C', row='2'))
        self.assertEquals(st_edge.mov, 'L')
        self.assertEquals(nd_edge.mov, 'UL')

        xl_ref = 'Sheet1!A1'
        res = xr._parse_xlref_fragment(xl_ref)
        self.assertEquals(res['st_edge'].land, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_edge'], None)

        xl_ref = 'Sheet1!a1(l):c2(ul):{"1":4,"2":"ciao"}'
        res = xr._parse_xlref_fragment(xl_ref)
        self.assertEquals(res['js_filt'], {'2': 'ciao', '1': 4})
        self.assertEquals(res['st_edge'].land, xr.Cell(col='A', row='1'))
        self.assertEquals(res['nd_edge'].land, xr.Cell(col='C', row='2'))
        self.assertEquals(res['st_edge'].mov, 'L')
        self.assertEquals(res['nd_edge'].mov, 'UL')

    @data(*list(itt.product(
        ['A1:B1', 'A1:B1:LURD',
         'A1:B1:"as"', 'A1:B1:["func"]', 'A1:B1:{"opts": {}}', 'A1:B1:{"opts": {"a": 1}}',
         'A1:B1:LURD:"as"', 'A1:B1:LURD:["func"]', 'A1:B1:LURD:{"opts": {}}', 'A1:B1:LURD:{"opts": {"a": 1}}',
         'A1(U):B1:LURD:"as"', 'A1(U):B1:LURD:["func"]', 'A1(U):B1:LURD:{"opts": {}}', 'A1(U):B1:LURD:{"opts": {"a": 1}}',
         'A1:B1(D):LURD:"as"', 'A1:B1(D):LURD:["func"]', 'A1:B1(D):LURD:{"opts": {}}', 'A1:B1(D):LURD:{"opts": {"a": 1}}',
         'A1(U):B1(D):LURD:"as"', 'A1(U):B1(D):LURD:["func"]', 'A1(U):B1(D):LURD:{"opts": {}}', 'A1(U):B1(D):LURD:{"opts": {"a": 1}}',
         'sh!A1(U):B1(D):LURD:"as"', '0!A1(U):B1(D):LURD:["func"]', 'sh!A1(U):B1(D):LURD:{"opts": {}}', 'sh!A1(U):B1(D):LURD:{"opts": {"a": 1}}',
         ],
        [':', ':"as"', ':["func"]', ':{"opts": {}}', ':{"opts": {"a": 1}}',
         'sheet!:', 'sheet!:"as"', 'sheet!:["func"]', 'sheet!:{"opts": {}}', 'sheet!:{"opts": {"a": 1}}',
         ]
    )))
    def test_shortcut_vs_regular_fieldsCount(self, case):
        regular, shortcut = case
        res1 = xr._parse_xlref_fragment(regular)
        res2 = xr._parse_xlref_fragment(shortcut)
        self.assertEquals(len(res1), len(res2), (res1, res2))

    @data('s![[]', 's!{}[]', 's!A', 's!A1:!', 's!1:2', 's!A0:B1', )
    def test_errors_parse_xl_ref(self, case):
        self.assertRaises(ValueError, xr._parse_xlref_fragment, case)

    @data(':{"opts": "..."}', ':{"opts": [2]}',
          'A1:b1:{"opts": "..."}', 'A1:B1:{"opts": [4]}')
    def test_parse_xl_ref_BadOpts(self, xlref):
        err_msg = 'must be a json-object\(dictionary\)'
        assertRaisesRegex(self, ValueError, err_msg,
                          xr._parse_xlref_fragment, xlref)

    def test_uncooked_Edge_good(self):
        self.assertIsNone(xr.Edge_uncooked(None, None, None))

        self.assertEquals(xr.Edge_uncooked('1', 'A', 'LUR'),
                          xr.Edge(xr.Cell(row='1', col='A'), 'LUR'))
        self.assertEquals(xr.Edge_uncooked('_', '^', 'duL'),
                          xr.Edge(xr.Cell('_', '^'), 'DUL'))
        self.assertEquals(xr.Edge_uncooked('1', '_', None),
                          xr.Edge(xr.Cell('1', '_'), None))
        self.assertEquals(xr.Edge_uncooked('^', '^', None),
                          xr.Edge(xr.Cell('^', '^'), None))

    def test_uncooked_Edge_bad(self):
        self.assertEquals(xr.Edge_uncooked(1, 'A', 'U1'),
                          xr.Edge(xr.Cell(1, 'A'), 'U1'))
        self.assertEquals(xr.Edge_uncooked('1', '%', 'U1'),
                          xr.Edge(xr.Cell('1', '%'), 'U1'))
        self.assertEquals(xr.Edge_uncooked('1', 'A', 'D0L'),
                          xr.Edge(xr.Cell('1', 'A'), 'D0L'))
        self.assertEquals(xr.Edge_uncooked('1', 'A', '@#'),
                          xr.Edge(xr.Cell('1', 'A'), '@#'))

    def test_uncooked_Edge_fail(self):
        self.assertRaises(
            AttributeError, xr.Edge_uncooked, *('1', 1, '0'))
        self.assertRaises(
            AttributeError, xr.Edge_uncooked, *('1', 'A', 23))
#         self.assertRaises(
#             ValueError, xr.Edge_uncooked, *('_0', '_', '0'))
#         self.assertRaises(
#             ValueError, xr.Edge_uncooked, *('@@', '@', '@'))

    def test_col2num(self):
        self.assertEqual(xr._col2num('D'), 3)
        self.assertEqual(xr._col2num('aAa'), 702)

    def test_parse_xl_url_Ok(self):
        url = 'file://path/to/file.xlsx#Sheet1!U10(L):D20(D):{"json":"..."}'
        res = xr.parse_xlref(url)

        self.assertEquals(res['url_file'], 'file://path/to/file.xlsx')
        self.assertEquals(res['sh_name'], 'Sheet1')
        self.assertEquals(res['js_filt'], {"json": "..."})
        self.assertEquals(res['st_edge'], xr.Edge(xr.Cell('10', 'U'), 'L'))
        self.assertEquals(res['nd_edge'], xr.Edge(xr.Cell('20', 'D'), 'D'))

    def test_parse_xl_url_Bad(self):
        self.assertRaises(ValueError, xr.parse_xlref, *('#!:{"json":"..."', ))

    def test_parse_xl_url_Only_fragment(self):
        url = '#sheet_name!UP10:DOWN20:{"json":"..."}'
        res = xr.parse_xlref(url)
        self.assertEquals(res['url_file'], None)

    def test_parse_xl_url_No_fragment(self):
        url = 'sdadsggfds'
        err_text = "No fragment-part"
        with assertRaisesRegex(self, ValueError, err_text):
            xr.parse_xlref(url)


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
        margin_coords = xr._margin_coords_from_states_matrix(states_matrix)
        args = (states_matrix, margin_coords)

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
class TRedim(unittest.TestCase):

    def check_redim(self, case):
        arr, dim, exp = case
        res = xr._redim(arr, dim)

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
        self.check_redim(case)

    @data(
        ([[1, 2]],   1,  [1, 2]),
        ([[[1, 2]]], 2,  [[1, 2]]),
        ([[[1, 2]]], 1,  [1, 2]),
        ([[[1], [2]]], 2,  [[1], [2]]),
        ([[[[1]], [[2]]]], 2,  [[1], [2]]),
        ([[[[1]], [[2]]]], 3,  [[[1]], [[2]]]),
        ([[[1], [2]]], 1,  [1, 2]),
        ([[[1], [2]]], 0,  [1, 2]),
    )
    def test_downscale(self, case):
        self.check_redim(case)

    @data(
        ([None],    0,  None),
        ([['str']], 0,  'str'),
        ([[[3.14]]], 0, 3.14),
        ('str',     0,  'str'),
        (None,      0,  None),
        (5.1,       0,  5.1),
    )
    def test_zero(self, case):
        self.check_redim(case)

    @data(
        ([],        3,  [[[]]]),
        ([[]],      3,  [[[]]]),
        ([[]],      2,  [[]]),
        ([],        2,  [[]]),
        ([],        1,  []),
        ([[]],      1,  []),
    )
    def test_empty(self, case):
        self.check_redim(case)

    @data(
        ([[], []], 1,               []),
        ([[1, 2], [3, 4]], 1,       [1, 2, 3, 4]),
        ([[[1, 1]], [[2, 2]]], 1,   [1, 1, 2, 2]),

        ([[], []], 0,               []),
        ([[1, 2], [3, 4]], 0,       [1, 2, 3, 4]),
        ([[[1, 1]], [[2, 2]]], 0,   [1, 1, 2, 2]),
    )
    def test_flatten(self, case):
        self.check_redim(case)

    @data(
        ([1, 2, 3], 0,          [1, 2, 3]),
        ([[1, 2], [3, 4]], 0,   [1, 2, 3, 4]),
        ([[1, 2]], 0,           [1, 2]),
        ([1, 2], 0,             [1, 2]),
        ([], 0,                 []),
        ([[]], 0,               []),
    )
    def test_unreducableZero(self, case):
        self.check_redim(case)


@ddt
class TCallSpec(unittest.TestCase):

    @data(
        ('func',                ('func', [], {})),
        ('',                    ('', [], {})),

        (['f', [], {}],         ('f', [], {})),
        (['f', None, None],     ('f', [], {})),
        (['f', [1], {2: 2}],    ('f', [1], {2: 2})),
        (['f', [2], {3: 3}],    ('f', [2], {3: 3})),
        (['f', {}, []],         ('f', [], {})),
        (['f', {2: 3}, [1]],    ('f', [1], {2: 3})),
        (['f', []],             ('f', [], {})),
        (['f', [1, 2]],         ('f', [1, 2], {})),
        (['f', {}],             ('f', [], {})),
        (['f', {1: 1, 2: 2}],   ('f', [], {1: 1, 2: 2})),

        ({'func': 'f', 'args': [], 'kwds': {}},     ('f', [], {})),
        ({'func': 'f', 'args': None, 'kwds': None}, ('f', [], {})),
        ({'func': 'f', 'args': [1], 'kwds': {1: 2}}, ('f', [1], {1: 2})),
        ({'func': 'f', 'args': [], },               ('f', [], {})),
        ({'func': 'f', 'args': [1, 2], },            ('f', [1, 2], {})),
        ({'func': 'f', 'kwds': {}},                 ('f', [], {})),
        ({'func': 'f', 'kwds': {2: 3, 3: 4}},         ('f', [], {2: 3, 3: 4})),
    )
    def test_OK(self, case):
        call_desc, exp = case
        cspec, opts = xr._parse_call_spec(call_desc)
        self.assertEqual(cspec, exp)
        self.assertIsNone(opts)

    _bad_struct = "One of str, list or dict expected"
    _func_not_str = "Expected a `string` for func"
    _func_missing = ("missing 1 required positional argument: 'func'"
                     if fututis.PY3 else
                     'takes at least 1 argument')
    _cannot_decide = "Cannot decide `args`/`kwds`"
    _more_args = ("takes from 1 to 3 positional arguments"
                  if fututis.PY3 else
                  'takes at most 3 arguments')
    _more_kwds = "unexpected keyword argument"
    _args_not_list = "Expected a `list`"
    _kwds_not_dict = "Expected a `dict`"

    @data(
        (1,                     _bad_struct),
        (True,                  _bad_struct),
        (None,                  _bad_struct),
        ([],                    _func_missing),
    )
    def test_Fail_base(self, case):
        call_desc, err = case
        with assertRaisesRegex(self, ValueError, err):
            xr._parse_call_spec(call_desc)

    @data(
        ([1, [], {}],           _func_not_str),
        ([[], 'f', {}],         _cannot_decide),
        ([[], {}, 'f'],         _cannot_decide),

        (['f', [], []],         _cannot_decide),
        (['f', [], {}, []],     _more_args),  # 5
        (['f', {}, {}],         _cannot_decide),
        (['f', [], {}, {}],     _more_args),

        (['f', {}, 33],         _cannot_decide),
        (['f', [], 33],         _cannot_decide),

        (['f', [], {}, 33],     _more_args),  # 10
        (['f', [], {}, 33],     _more_args),

    )
    def test_Fail_List(self, case):
        call_desc, err = case
        with assertRaisesRegex(self, ValueError, err):
            xr._parse_call_spec(call_desc)

    @data(
        ({'args': [], 'kwds': {}},                      _func_missing),
        ({'args': []},                                  _func_missing),
        ({'kwds': {}},                                  _func_missing),

        ({'gg': 1, 'args': [], 'kwds': {}},              _more_kwds),
        ({'func': 'f', 'args': [], 'y': 5},              _more_kwds),
        ({'func': 'f', 'kwds': {}, 'y': 5},              _more_kwds),

        ({'func': None, 'args': [], 'kwds': {}},        _func_not_str),
        ({'func': 1, 'args': [], 'kwds': {}},           _func_not_str),
        ({'func': True, 'args': [1], 'kwds': {}},       _func_not_str),
        ({'func': [], 'args': [1], 'kwds': {}},         _func_not_str),

        ({'func': 'f', 'args': 1, 'kwds': {}},          _args_not_list),
        ({'func': 'f', 'args': True, 'kwds': {}},       _args_not_list),
        ({'func': 'f', 'args': {}, 'kwds': {}},         _args_not_list),

        ({'func': 'f', 'args': [], 'kwds': 1},          _kwds_not_dict),
        ({'func': 'f', 'args': [], 'kwds': True},       _kwds_not_dict),
        ({'func': 'f', 'args': [], 'kwds': []},         _kwds_not_dict),
    )
    def test_Fail_Object(self, case):
        call_desc, err = case
        with assertRaisesRegex(self, ValueError, err):
            xr._parse_call_spec(call_desc)

    @data(
        ({'opts': {1: 2}},                  None, {1: 2}),
        ({'func': 'f', 'kwds': {'opts': {1: 2}}}, ('f', [], {}), {1: 2}),
        ({'func': 'f', 'kwds': {'a': 'b', 'opts': {1: 2}}},
         ('f', [], {'a': 'b'}), {1: 2}),
        (['f', {'a': 2, 'opts': {1: 2}}],   ('f', [], {'a': 2}), {1: 2}),
        (['f', {'a': 2, 'opts': {1: 2}}, [5, 6]],
         ('f', [5, 6], {'a': 2}), {1: 2}),  # 5
        ({'opts': {1: 2}, 'func': 'f', 'args': [1, 2]},
         ('f', [1, 2], {}), {1: 2}),
        ({'opts': {1: 2}, 'func': 'f', 'args': [1, 2], 'kwds': {'a': 2}},
         ('f', [1, 2], {'a': 2}), {1: 2}),
        ({'opts': {1: 2, 11: 22}, 'func': 'merge',
          'kwds': {'a': 2, 'opts': {1: 1, 3: 4}}},
         ('merge', [], {'a': 2}), {1: 1, 11: 22, 3: 4}),
    )
    def test_opts(self, case):
        inp, call_spec, opts = case
        res = xr._parse_call_spec(inp)
        self.assertEqual(res[0], call_spec)
        self.assertEqual(res[1], opts)


@ddt
class TReadRect(unittest.TestCase):

    def setUp(self):
        arr = np.array([
            # A     B       C      D
            [None, None,   None,  None],  # 1
            [None, None,   None,  None],  # 2
            [None, np.NaN, 5.1,   7.1],   # 3
            [None, None,   43,    'str'],  # 4
            [None, -1,     None,  None],  # 5
        ])
        self.sheet = xr.ArraySheet(arr)

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


@ddt
class TRecursive(unittest.TestCase):

    @data(
        1,
        [1, 2],
        'str',
        [],
        [1],
        [[1, 2], [3, 4]],
        [[], [1, 'a', 'b'], [11, list('abc')]],
    )
    def test_dontExpand_nonDicts(self, vals):
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', side_effect=lambda x: x)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso).values
        self.assertEqual(res, vals)

    @data(
        'str',
        (1, 'str', []),
        (1, 'str', [[1, 2]]),
        [[], set([1, 'a', 'b']), [11, list('abc')]],
    )
    def test_expandNestedStrings(self, vals):
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', return_value=sentinel.BINGO)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso).values
        self.assertIn(sentinel.BINGO.name, str(res))
        self.assertNotIn("'", str(res))

    @data(
        {1: 'str'},
        [{2: 'str'}],
        [{3: 'str'}],
        [{4: ['str', 'a', {4: [list('ab')]}]}],
    )
    def test_expandDicts_nonStrKeys(self, vals):
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', return_value=sentinel.BINGO)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso).values
        self.assertIn(sentinel.BINGO.name, str(res))
        self.assertNotIn("'", str(res))

    @data(
        {'key': 'str'},
        [{'key': 'str'}],
        [{'key': ['str', 'a', {'key': [list('ab')]}]}],
        [{'key1': ['str', 'foo', {'key2': ['abc', 'bar'], 'k3':'123'}]}],
    )
    def test_expandDicts_preservingKeys(self, vals):
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', return_value=sentinel.BINGO)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso).values
        # print(res)
        self.assertIn(sentinel.BINGO.name, str(res))
        self.assertIn('key', str(res))

        # Mask all str-keys, and check
        #    no other strings left.
        #
        res = str(res)
        for k in ['key', 'key1', 'key2', 'k3']:
            res = res.replace("'%s'" % k, 'OFF')
        self.assertNotIn("'", str(res))

    @data(
        (0, ['bang', 'str', 'foo', 'abc', 'bar', '123'], []),
        (1, ['bang', 'str', 'foo', 'abc', 'bar', '123'], []),
        (2, ['bang', 'str', 'foo', 'abc', 'bar', '123'], []),
        (3, ['str', 'foo', 'abc', 'bar', '123'], ['bang']),
        (4, ['abc', 'bar', '123'], ['bang', 'str', 'foo']),
        (5, ['abc', 'bar'], ['bang', 'str', 'foo', '123']),
        (6, [], ['bang', 'str', 'foo', 'abc', 'bar', '123']),
    )
    def test_expandDicts_depth(self, case):
        depth, exists, missing = case
        vals = [{
            'key1': ['str', 'foo', {'key2': ['abc', 'bar'], 'k3':'123'}],
            'key11': 'bang'}]
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', return_value=sentinel.BINGO)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso, depth=depth).values
        print(res)
        if missing:
            self.assertIn(sentinel.BINGO.name, str(res))
        self.assertIn("key", str(res))
        for v in exists:
            self.assertIn(v, str(res))
        for v in missing:
            self.assertNotIn(v, str(res))

    @data(
        pd.DataFrame({'key1': list('abc'), 'key2': list('def')}),
        [pd.DataFrame({'key1': list('abc'), 'key2': list('def')})],
    )
    def test_expandDFs(self, vals):
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', return_value=sentinel.BINGO)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso).values
        print(res)
        self.assertIn(sentinel.BINGO.name, str(res))
        self.assertIn("key", str(res))

    @data(
        ([{'key1': ['str', 'foo', {'key2': ['abc', 'bar'], 'k3':'123'}]}],
         ('key1', None),
         ['abc', 'bar', '123'], ['str', 'foo']),

        ([{'key1': ['str', 'foo', {'key2': ['abc', 'bar'], 'k3':'123'}]}],
         (None, ['key2', 'k3']),
         ['abc', 'bar', '123'], ['str', 'foo']),

        ([{'key1': ['str', 'foo', {'key2': ['abc', 'bar'], 'k3':'123'}]}],
         (['key1', 'key2'], None),
         ['123'], ['str', 'foo', 'abc', 'bar']),

        ([{'key1': ['str', 'foo', {'key2': ['abc', 'bar'], 'k3':'123'}]}],
         (None, ['key2', 'k3']),
         ['abc', 'bar', '123'], ['str', 'foo']),
    )
    def test_expandDicts_IncExcFilters(self, case):
        vals, incexc, exist, missing = case
        ranger = xr.Ranger(None)
        ranger.lasso = MagicMock(name='lasso()', return_value=sentinel.BINGO)
        lasso = xr.Lasso(values=vals)
        res = xr.Ranger.recursive_filter(ranger, lasso, *incexc).values
        print(res)
        self.assertIn(sentinel.BINGO.name, str(res))
        for k in ['key1', 'key2', 'k3']:
            self.assertIn(k, str(res))
        for v in exist:
            self.assertIn(v, str(res))
        for v in missing:
            self.assertNotIn(v, str(res))


@ddt
class TLasso(unittest.TestCase):

    def m1(self):
        dt = datetime(1900, 8, 2)
        return np.array([
            # A     B       C      D       E
            [1,    True,   None, False,  None],   # 1
            [5,    True,   dt,    '',    3.14],  # 2
            [7,    False,  5.1,   7.1,    ''],    # 3
            [9,    True,   43,    'str', dt],    # 4
        ])

    def test_read_Colon(self):
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        res = xr.lasso('#:', sf)
        npt.assert_array_equal(res, self.m1().tolist())

    def test_read_ColonWithJson(self):
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        res = xr.lasso('''#:
            [
                "pipe", [
                    ["redim", {"col": [2, 1]}], 
                    "numpy"
                ], {"opts":
                    {"verbose": true}
                }
            ]''',
                       sf)
        self.assertIsInstance(res, np.ndarray)
        npt.assert_array_equal(res, self.m1())

    def test_read_A1(self):
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        res = xr.lasso('''#A1:..(D):
            [
                "pipe", [
                    ["redim", {"col": [2, 1]}], 
                    "numpy"
                ], {"opts":
                    {"verbose": true}
                }
            ]''',
                       sf)
        self.assertIsInstance(res, np.ndarray)
        npt.assert_array_equal(res, [[1, 5, 7, 9]])

    def test_read_RC(self):
        m1 = self.m1()
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        res = xr.lasso('#R1C1:..(D):["pipe", [["redim", {"col": [2,1]}]]]',
                       sf)
        self.assertIsInstance(res, list)
        npt.assert_array_equal(res, m1[:, 0].reshape((1, -1)))

    def test_read_RC_negative(self):
        m1 = self.m1()
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        res = xr.lasso('#R-1C-2:..(U):["pipe", [["redim", {"col": 1}]]]',
                       sf)
        npt.assert_array_equal(res, m1[:, -2].astype('<U5'))

    def test_read_asLasso(self):
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        res = xr.lasso('''#A1:..(D)''', sf, return_lasso=True)
        self.assertIsInstance(res, xr.Lasso)

    def test_Ranger_intermediateLaso(self):
        sf = xr.SheetsFactory()
        sf.add_sheet(xr.ArraySheet(self.m1()))
        ranger = xr.make_default_Ranger(sheets_factory=sf)
        ranger.lasso(
            '''#A1(DR):__(UL+):RULD:["pipe", [["redim"], ["numpy"]]]''')
        self.assertEqual(ranger.intermediate_lasso[0], 'numpy',
                         ranger.intermediate_lasso)

        ranger = xr.make_default_Ranger(sheets_factory=sf)
        self.assertRaises(ValueError, ranger.lasso,
                          '''#A1(DR):__(UL+):RULD:["pipe", [["redim"], ["dab_func"]]]''')
        self.assertEqual(ranger.intermediate_lasso[0], 'dab_func',
                         ranger.intermediate_lasso, )

    @data(
        ('#R5C4:..(UL):%s',      [[None, 0, 1], [0, 1, 2]]),
        ('#R5C4:R5C4:LURD:%s',   [
            [None, 0,    1,   2],
            [0,    1,    2,   None],
            [1,    None, 6.1, 7.1]
        ]),
        ('#R5C_(LU):A1(RD):%s',      [[0, 1], [1, 2]]),
        ('#__(LU+):^^(RD):%s',       [[0, 1], [1, 2], [None, 6.1]]),
        ('#R_C5:R6C.(L+):%s',        [6.1, 7.1]),  # 5
        ('#R^C3(U+):..(D+):%s',      [0, 1]),
        ('#D6:%s',                   6.1),
    )
    def test_read_xlwings_dims(self, case):
        xlref, res = case
        table = np.array([
            # A(1)  B(2)   C(3)  D(4)  E(5)
            [None, None,  None, None, None],  # 1
            [None, None,  None, None, None],  # 2
            [None, None,  None, None, None],  # 3
            [None, None,  0.,   1.,   2.],    # 4
            [None, 0.,    1.,   2.,   None],  # 5
            [None, 1.,    None, 6.1,  7.1]    # 6
        ])
        sheetsFact = xr.SheetsFactory()
        sheetsFact.add_sheet(xr.ArraySheet(table), 'wb', 'sheet1')

        dims = xr.xlwings_dims_call_spec()
        self.assertEqual(xr.lasso(xlref % dims, sheetsFact), res)


@ddt
class VsPandas(unittest.TestCase, CustomAssertions):

    @contextlib.contextmanager
    def sample_xl_file(self, matrix, **df_write_kwds):
        try:
            tmp_file = '%s.xlsx' % tempfile.mktemp()
            _write_sample_sheet(tmp_file, matrix, 'Sheet1', **df_write_kwds)

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
        [9,    True,   43,    b'str', dt],    # 4
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

    def check_vs_read_df(self, table, st, nd, write_df_kwds={}, parse_df_kwds={}):
        with self.sample_xl_file(table, **write_df_kwds) as xl_file:
            pd_df = pd.read_excel(xl_file, 'Sheet1')
            pd_df = pd_df.iloc[
                slice(st[0], nd[0] + 1), slice(st[1], nd[1] + 1)]
            xlrd_wb = xlrd.open_workbook(xl_file)
            self.sheet = XlrdSheet(xlrd_wb.sheet_by_name('Sheet1'))
            xlref_res = self.sheet.read_rect(st, nd)
            lasso = xr.Lasso(
                st=st, nd=nd, values=xlref_res, opts=ChainMap())

            lasso1 = xr.redim_filter(None, lasso, row=[2, True])

            df_filter = xr.get_default_filters()['df']['func']
            lasso2 = df_filter(None, lasso1, **parse_df_kwds)

            xlref_df = lasso2.values

            msg = '\n---\n%s\n--\n%s\n-\n%s' % (xlref_res, xlref_df, pd_df)
            self.assertTrue(xlref_df.equals(pd_df), msg=msg)

    def test_vs_read_df(self):
        self.check_vs_read_df(self.m1.tolist(),
                              xr.Coords(0, 0), xr.Coords(4, 5),
                              parse_df_kwds=dict(header=0))


@unittest.skipIf(not xl_installed, "Cannot test xlwings without MS Excel.")
@ddt
class VsXlwings(unittest.TestCase):

    def setUp(self):
        self.tmp = '%s.xlsx' % tempfile.mktemp()
        table = [
            [1, 2, None],
            [None, 6.1, 7.1]
        ]
        _write_sample_sheet(self.tmp, table, 'Sheet1', startrow=5, startcol=3)

        xlrd_wb = xlrd.open_workbook(self.tmp)
        self.sheet = XlrdSheet(xlrd_wb.sheet_by_name('Sheet1'))
        self.sheetsFact = xr.SheetsFactory()
        self.sheetsFact.add_sheet(self.sheet, 'wb', 'sheet1')

    def tearDown(self):
        del self.sheet
        os.remove(self.tmp)

    @data(
        ('#R7C4:..(D):%s', lambda xw: xw.Range("Sheet1", "D7").vertical.value),
        ('#R6C5:..(D):%s', lambda xw: xw.Range("Sheet1", "E6").vertical.value),
        ('#R6C5:..(R):%s', lambda xw: xw.Range("Sheet1",
                                               "E6").horizontal.value),
        ('#R6C5:..(RD):%s', lambda xw: xw.Range("Sheet1", "E6").table.value),
        ('#R6C5:..(DR):%s', lambda xw: xw.Range("Sheet1", "E6").table.value),
        ('#R8C6:R6R4:%s', lambda xw: xw.Range("Sheet1", "D6:F8").value),
        ('#R8C6:R1C1:%s', lambda xw: xw.Range("Sheet1", "A1:F8").value),
        ('#R7C1:R7C4:%s', lambda xw: xw.Range("Sheet1", "A7:D7").value),
        ('#R9C4:R3C4:%s', lambda xw: xw.Range("Sheet1", "D3:D9").value),
    )
    def test_vs_xlwings(self, case):
        xlref, res = case
        # load Workbook for --> xlwings
        with xw_Workbook(self.tmp) as xw:
            res = res(xw)

        dims = xr.xlwings_dims_call_spec()
        self.assertEqual(xr.lasso(xlref % dims, self.sheetsFact), res)


class TSheet(unittest.TestCase):

    class MySheet(xr.ABCSheet):

        def open_sibling_sheet(self, sheet_id):
            raise NotImplementedError()

        def get_sheet_ids(self):
            raise NotImplementedError()

        def read_rect(self, st, nd):
            return [[]]

        def _read_states_matrix(self):
            return [[]]

    def test_get_states_matrix_Caching(self):
        sheet = self.MySheet()
        obj = object()
        sheet._states_matrix = obj
        self.assertEqual(sheet.get_states_matrix(), obj)

    def test_get_margin_coords_Cached(self):
        sheet = self.MySheet()
        obj = object()
        sheet._margin_coords = obj
        self.assertEqual(sheet.get_margin_coords(), obj)

    def test_get_margin_coords_Extracted_from_states_matrix(self):
        sheet = self.MySheet()
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


@ddt
class TSheetFactory(unittest.TestCase):

    @data(
        (('wb1', ['sh1', 0]), None, None,     [('wb1', 'sh1'), ('wb1', 0),
                                               ('wb1', None)]),
        (('wb1', [None, 0]), None, None,      [('wb1', None), ('wb1', 0)]),
        (('wb1', ['sh1', 0]), 'wb2', None,    [('wb1', 'sh1'), ('wb1', 0),
                                               ('wb1', None), ('wb2', 'sh1'),
                                               ('wb2', 0), ('wb2', None)]),
        (('wb1', ['sh1', 0]), 'wb2', 'sh2',   [('wb1', 'sh1'), ('wb1', 0),
                                               ('wb1', 'sh2'), ('wb2', 'sh1'),
                                               ('wb2', 'sh2'), ('wb2', 0)]),
        (('wb1', ['sh1', 0]), None, 'sh2',   [('wb1', 'sh1'), ('wb1', 0),
                                              ('wb1', 'sh2')]),
    )
    def test_derive_keys(self, case):
        sh_ids, wb_id, sheet_ids, exp = case
        sf = xr.SheetsFactory()
        sheet = MagicMock()
        sheet.get_sheet_ids.return_value = sh_ids
        keys = sf._derive_sheet_keys(sheet, wb_id, sheet_ids)
        self.assertEqual(sorted(keys, key=str),
                         sorted(exp, key=str))

    cache_keys = [
        ([('w1', ['s1'])],                                      1),
        ([('w1', ['s1', None])],                                2),
        ([('w1', ['s1', 0, 1])],                                2),
        ([('w1', ['s1', 0, 1, None])],                          3),
        ([('wb', ['s1']), ('w1', [0])],                         2),
        ([('w1', ['s1', 0, None]), ('w2', ['s1', 0, None])],    4),

    ]

    @data(
        *cache_keys
    )
    def test_cache_sheet(self, case):
        extra_ids, open_calls = case
        k1 = ('wb', 'sh')
        k2 = ('wb',  0)
        sheet = MagicMock()
        sheet.get_sheet_ids.return_value = ('wb', ['sh', 0])
        sf = xr.SheetsFactory()
        for wb_id, sh_ids in extra_ids:
            for sh_id in sh_ids:
                sf.add_sheet(sheet, wb_id, sh_id)

        self.assertIs(sf._cache_get(k1), sheet)
        self.assertIs(sf._cache_get(k2), sheet)
        for wb_id, sh_ids in extra_ids:
            for sh_id in sh_ids:
                self.assertIs(sf._cache_get((wb_id, sh_id)), sheet)
        self.assertIsNone(sf._cache_get(('no', 'key')))

    @data(
        *cache_keys
    )
    def test_close_sheet(self, case):
        extra_ids, open_calls = case
        # Try closings by all keys.
        #
        extra_ids = extra_ids + [('wb', ['sh', 0])]
        for wb_id1, sh_ids1 in extra_ids:
            for sh_id1 in sh_ids1:

                sheet = MagicMock()
                sheet.get_sheet_ids.return_value = ('wb', ['sh', 0])

                # Populate cache
                sf = xr.SheetsFactory()
                for wb_id, sh_ids in extra_ids:
                    for sh_id in sh_ids:
                        sf.add_sheet(sheet, wb_id, sh_id)

                # Close the key
                sf._close_sheet((wb_id1, sh_id1))
                sheet._close.assert_called_once_with()

                # Expect to be empty.
                for k, sh_dict in sf._cached_sheets.items():
                    self.assertIsNot(sh_dict, k)

                # Check cache returns no value.
                for wb_id, sh_ids in extra_ids:
                    for sh_id in sh_ids:
                        self.assertIsNone(sf._cache_get((wb_id, sh_id)))

    @data(
        *cache_keys
    )
    def test_fetch_sheet_prePopulated(self, case):
        extra_ids, open_calls = case
        k1 = ('wb', 'sh')
        k2 = ('wb',  0)
        sheet = MagicMock()
        sheet.get_sheet_ids.return_value = ('wb', ['sh', 0])

        sf = xr.SheetsFactory()
        sf._open_sheet = MagicMock(side_effect=AssertionError("OPENED!"))
        for wb_id, sh_ids in extra_ids:
            for sh_id in sh_ids:
                sf.add_sheet(sheet, wb_id, sh_id)

        extra_ids = extra_ids + [('wb', ['sh', 0])]
        for wb_id, sh_ids in extra_ids:
            for sh_id in sh_ids:
                self.assertIs(sf.fetch_sheet(wb_id, sh_id), sheet)
                self.assertIs(sf.fetch_sheet(None, None), sheet)
                self.assertIs(sf.fetch_sheet(None, sh_id), sheet)
                self.assertIs(sf.fetch_sheet(*k1), sheet)
                self.assertIs(sf.fetch_sheet(*k2), sheet)

    @data(
        *cache_keys
    )
    def test_fetch_sheet_andOpen(self, case):
        k1 = ('wb', 'sh')
        k2 = ('wb',  0)
        extra_ids, open_calls = case
        sheet = MagicMock(name='sheet')
        sheet.get_sheet_ids.return_value = ('wb', ['sh', 0])

        sf = xr.SheetsFactory()
        sf._open_sheet = MagicMock(name='open_sheet', return_value=sheet)

        extra_ids = extra_ids + [('wb', ['sh', 0])]
        for wb_id, sh_ids in extra_ids:
            for sh_id in sh_ids:
                self.assertIs(sf.fetch_sheet(wb_id, sh_id), sheet)
                self.assertIs(sf.fetch_sheet(None, None), sheet)
                self.assertIs(sf.fetch_sheet(None, sh_id), sheet)
                self.assertIs(sf.fetch_sheet(*k1), sheet)
                self.assertIs(sf.fetch_sheet(*k2), sheet)

        self.assertEqual(sf._open_sheet.call_count, open_calls,
                         sf._open_sheet.mock_calls)
