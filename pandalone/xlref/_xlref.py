#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The user-facing implementation of *xlref*.

Prefer accessing the public members from the parent module.
"""
from collections import namedtuple
import json
import logging
import re
from string import ascii_uppercase
from types import ModuleType

import six

import itertools as itt
import numpy as np
import pandas as pd
from six.moves.urllib.parse import urldefrag  # @UnresolvedImport

from . import _xlrd


log = logging.getLogger(__name__)


Cell = namedtuple('Cell', ['row', 'col'])
"""
A pair of 1-based strings, denoting the "A1" coordinates of a cell.

The "num" coords (numeric, 0-based) are specified using numpy-arrays
(:class:`Coords`).
"""


Coords = namedtuple('Coords', ['row', 'col'])
"""
A pair of 0-based integers denoting the "num" coordinates of a cell.

The "A1" coords (1-based coordinates) are specified using numpy-arrays
(:class:`Cell`).
"""
#     return np.array([row, cell], dtype=np.int16)

Edge = namedtuple('Edge', ['land', 'mov'])
"""
An :term:`Edge` might be "cooked" or "uncooked" depending on its `land`:

- An *uncooked* edge contains *A1* :class:`Cell`.
- An *cooked* edge contains a *resolved* :class:`Coords`.

Use None for missing moves.
"""

_special_coord_symbols = {'^', '_', '.'}

_primitive_dir_vectors = {
    'L': Coords(0, -1),
    'U': Coords(-1, 0),
    'R': Coords(0, 1),
    'D': Coords(1, 0)
}


def coords2Cell(row, col):
    """Make *A1* :class:`Cell` from *resolved* coords, with rudimentary error-checking.

    Examples::

        >>> coords2Cell(row=0, col=0)
        Cell(row='1', col='A')
        >>> coords2Cell(row=0, col=26)
        Cell(row='1', col='AA')

        >>> coords2Cell(row=10, col='.')
        Cell(row='11', col='.')

        >>> coords2Cell(row=-3, col=-2)
        Traceback (most recent call last):
        AssertionError: negative row!


    """
    if row not in _special_coord_symbols:
        assert row >= 0, 'negative row!'
        row = str(row + 1)
    if col not in _special_coord_symbols:
        assert col >= 0, 'negative col!'
        col = _xlrd.colname(col)
    return Cell(row=row, col=col)


def _uncooked_Edge(row, col, mov):
    """
    Make a new `Edge` from any non-values supplied, as is capitalized, or nothing.

    :param str, None col:    ie ``A``
    :param str, None row:    ie ``1``
    :param str, None mov:    ie ``RU1D?``

    :return:    a `Edge` if any non-None
    :rtype:     Edge, None


    Examples::

        >>> tr = _uncooked_Edge('1', 'a', 'Rul')
        >>> tr
        Edge(land=Cell(row='1', col='A'), mov='RUL')


    No error checking performed::

        >>> _uncooked_Edge('Any', 'foo', 'BaR')
        Edge(land=Cell(row='Any', col='FOO'), mov='BAR')

        >>> print(_uncooked_Edge(None, None, None))
        None


    except were coincidental::

        >>> _uncooked_Edge(row=0, col=123, mov='BAR')
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'

        >>> _uncooked_Edge(row=0, col='A', mov=123)
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'
    """

    if col == row == mov is None:
        return None

    return Edge(land=Cell(col=col and col.upper(), row=row), mov=mov and mov.upper())

_re_xl_ref_parser = re.compile(
    r"""
    ^\s*(?:(?P<sheet>[^!]+)?!)?                          # xl sheet name
    (?:                                                  # first cell
        (?P<st_col>[A-Z]+|_|\^)                          # first col
        (?P<st_row>[123456789]\d*|_|\^)                  # first row
        (?:\(
            (?P<st_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves from st cell
            \)
        )?
    )
    (?::                                                 # second cell [opt]
        (?P<nd_col>[A-Z]+|_|\^|\.)                       # second col
        (?P<nd_row>[123456789]\d*|_|\^|\.)               # second row
        (?:\(
            (?P<nd_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves from nd cell
            \)
        )?
        (?::
            (?P<rect_exp>[LURD?123456789]+)              # rect expansion [opt]
        )?
    )?
    \s*
    (?::?
        (?P<json>\{.*\})?                                # any json object [opt]
    )\s*$""",
    re.IGNORECASE | re.X)

_re_rect_exp_splitter = re.compile('([LURD]\d+)', re.IGNORECASE)

# TODO: Make rect_expansions `?` work different from numbers.
_re_rect_expansion_parser = re.compile(
    r"""
    ^(?P<moves>[LURD]+)                                  # primitive moves
    (?P<times>\?|\d+)?                                   # repetition times
    $""",
    re.IGNORECASE | re.X)


def _repeat_moves(moves, times=None):
    """
    Returns an iterator that repeats `moves` x `times`, or infinite if unspecified.

    Used when parsing primitive :term:`directions`.

   :param str moves: the moves to repeat
   :param str times: N of repetitions. If `None` it means infinite repetitions.
   :return:    An iterator of the moves
   :rtype:     iterator

    Examples::

         >>> list(_repeat_moves('LUR', '3'))
         ['LUR', 'LUR', 'LUR']
         >>> list(_repeat_moves('ABC', '0'))
         []
         >>> _repeat_moves('ABC')  ## infinite repetitions
         repeat('ABC')
     """
    args = (moves,)
    if times is not None:
        args += (int(times), )
    return itt.repeat(*args)


def _parse_rect_expansions(rect_exp):
    """
    Parse rect-expansion into a list of dir-letters iterables.

    :param rect_exp:
        A string with a sequence of primitive moves:
        es. L1U1R1D1
    :type xl_ref: str

    :return:
        A list of primitive-dir chains.
    :rtype: list


    Examples::

        >>> res = _parse_rect_expansions('LURD?')
        >>> res
        [repeat('LUR'), repeat('D', 1)]

        # infinite generator
        >>> [next(res[0]) for i in range(10)]
        ['LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR']

        >>> list(res[1])
        ['D']

        >>> _parse_rect_expansions('1LURD')
        Traceback (most recent call last):
        ValueError: Invalid rect-expansion(1LURD) due to:
                'NoneType' object has no attribute 'groupdict'

    """
    try:
        res = _re_rect_exp_splitter.split(rect_exp.replace('?', '1'))

        return [_repeat_moves(**_re_rect_expansion_parser.match(v).groupdict())
                for v in res
                if v != '']

    except Exception as ex:
        msg = 'Invalid rect-expansion({}) due to: {}'
        raise ValueError(msg.format(rect_exp, ex))


def parse_xl_ref(xl_ref):
    """
    Parses a :term:`xl-ref` and splits it in its "ingredients".

    :param str xl_ref:
        a string with the following format:
        <sheet>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):
        <rect_exp>{<json>}
        i.e.::

            sheet!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}

    :return:
        dictionary containing the following parameters::

        - sheet: str
        - st_edge: (Edge, None) the 1st-ref, uncooked, with raw cell
        - nd_edge: (Edge, None) the 2nd-ref, uncooked, with raw cell
        - rect_exp: (str) as found on the xl-ref
        - json: parsed

    :rtype: dict


    Examples::

        >>> res = parse_xl_ref('Sheet1!A1(DR):Z20(UL):L1U2R1D1:{"json":"..."}')
        >>> sorted(res.items())
        [('json', {'json': '...'}),
         ('nd_edge', Edge(land=Cell(row='20', col='Z'), mov='UL')),
         ('rect_exp', [repeat('L', 1), repeat('U', 2), repeat('R', 1), repeat('D', 1)]),
         ('sheet', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='DR'))]

        >>> parse_xl_ref('A1(DR)Z20(UL)')
        Traceback (most recent call last):
        ValueError: Invalid xl-ref(A1(DR)Z20(UL)) due to: not an `xl-ref` syntax.
    """

    try:
        m = _re_xl_ref_parser.match(xl_ref)
        if not m:
            raise ValueError('not an `xl-ref` syntax.')
        gs = m.groupdict()

        # Replace coords of 1st and 2nd cells
        #     with "uncooked" edge.
        #
        p = gs.pop
        gs['st_edge'] = _uncooked_Edge(p('st_row'), p('st_col'), p('st_mov'))
        gs['nd_edge'] = _uncooked_Edge(p('nd_row'), p('nd_col'), p('nd_mov'))

        js = gs['json']
        gs['json'] = json.loads(js) if js else None

        rect_exp = gs['rect_exp']
        gs['rect_exp'] = _parse_rect_expansions(rect_exp) if rect_exp else None

        return gs

    except Exception as ex:
        msg = "Invalid xl-ref(%s) due to: %s"
        log.debug(msg, xl_ref, ex, exc_info=1)
        raise ValueError(msg % (xl_ref, ex))


def parse_xl_url(url, base_url=None, backend=None):
    """
    Parses a :term:`xl-url`.

    :param str url:
        a string with the following format::

            <url_file>#<sheet>!<1st_edge>:<2nd_edge>:<expand><json>

        Exxample::

            file:///path/to/file.xls#sheet_name!UP10:DN20:LDL1{"dim":2}
    :param XlUrl base_url:
    :param module backend: one of :mod:`_xlrd` or mod:`_xlwings`

    :return:
        dictionary containing the following parameters::

        - url_file
        - sheet
        - st_col
        - st_row
        - st_mov
        - nd_col
        - nd_row
        - nd_mov
        - json

    :rtype: dict


    Examples::

        >>> url = 'file:///sample.xlsx#Sheet1!A1(UL):.^(DR):LU?:{"2": "ciao"}'
        >>> res = parse_xl_url(url)
        >>> sorted(res.items())
        [('json', {'2': 'ciao'}),
         ('nd_edge', Edge(land=Cell(row='^', col='.'), mov='DR')),
         ('rect_exp', [repeat('L'), repeat('U', 1)]),
         ('sheet', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='UL')),
         ('url_file', 'file:///sample.xlsx')]
    """

    try:
        url_file, frag = urldefrag(url)
        res = parse_xl_ref(frag)
        res['url_file'] = url_file

        return res

    except Exception as ex:
        raise ValueError("Invalid xl-url({}) due to: {}".format(url, ex))


def _margin_coords_from_states_matrix(states_matrix):
    """
    Returns top-left/bottom-down margins of full cells from a :term:`state` matrix.

    May be used by :meth:`Spreadsheet.get_margin_coords()` if a backend
    does not report the sheet-margins internally.

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`Spreadsheet.read_states_matrix()` to derrive it.
    :return:    the 2 coords of the top-left & bottom-right full cells
    :rtype:     (Coords, Coords)

    Examples::

        >>> states_matrix = np.asarray([
        ...    [0, 0, 0],
        ...    [0, 1, 0],
        ...    [0, 1, 1],
        ...    [0, 0, 1],
        ... ])
        >>> margins = _margin_coords_from_states_matrix(states_matrix)
        >>> margins
        (Coords(row=1, col=1), Coords(row=3, col=2))


    Note that the botom-left cell is not the same as `states_matrix` matrix size::

        >>> states_matrix = np.asarray([
        ...    [0, 0, 0, 0],
        ...    [0, 1, 0, 0],
        ...    [0, 1, 1, 0],
        ...    [0, 0, 1, 0],
        ...    [0, 0, 0, 0],
        ... ])
        >>> _margin_coords_from_states_matrix(states_matrix) == margins
        True

    """
    if not states_matrix.any():
        c = Coords(0, 0)
        return c, c
    indices = np.array(np.where(states_matrix), dtype=np.int16).T

    # return indices.min(0), indices.max(0)
    return Coords(*indices.min(0)), Coords(*indices.max(0))


def _row2num(coord):
    """
    Resolves special coords or converts Excel 1-based rows to zero-based, reporting invalids.

    :param str, int coord:     excel-row coordinate or one of ``^_.``
    :return:    excel row number, >= 0
    :rtype:     int

    Examples::

        >>> row = _row2num('1')
        >>> row
        0
        >>> row == _row2num(1)
        True
        >>> _row2num('-1')
        -2

    Fails ugly::

        >>> _row2num('.')
        Traceback (most recent call last):
        ValueError: invalid literal for int() with base 10: '.'
    """
    return int(coord) - 1


def _col2num(coord):
    """
    Resolves special coords or converts Excel A1 columns to a zero-based, reporting invalids.

    :param str coord:          excel-column coordinate or one of ``^_.``
    :return:    excel column number, >= 0
    :rtype:     int

    Examples::

        >>> col = _col2num('D')
        >>> col
        3
        >>> _col2num('d') == col
        True
        >>> _col2num('AaZ')
        727

    Fails ugly::

        >>> _col2num('12')
        Traceback (most recent call last):
        ValueError: substring not found

        >>> _col2num(1)
        Traceback (most recent call last):
        TypeError: 'int' object is not iterable
    """

    rcoord = 0
    for c in coord:
        rcoord = rcoord * 26 + ascii_uppercase.rindex(c.upper()) + 1

    rcoord -= 1

    return rcoord


def _resolve_coord(cname, cfunc, coord, up_coord, dn_coord, base_coord=None):
    """
    Translates special coords or converts Excel string 1-based rows/cols to zero-based, reporting invalids.

    :param str        cname:  the coord-name, one of 'row', 'column'
    :param function   cfunc:  the function to convert coord ``str --> int``
    :param int, str   coord:  the "A1" coord to translate
    :param int up_coord:
            the resolved *top* or *left* margin coordinate
    :param int up_coord:
            the resolved *bottom* or *right* margin coordinate
    :param int, None base_coord:  the resolved basis for dependent coord, if any

    :return: the resolved coord or `None` if it were not a special coord.


    Row examples::

        >>> cname = 'row'

        >>> r0 = _resolve_coord(cname, _row2num, '1', 1, 10)
        >>> r0
        0
        >>> r0 == _resolve_coord(cname, _row2num, 1, 1, 10)
        True
        >>> _resolve_coord(cname, _row2num, '^', 1, 10)
        1
        >>> _resolve_coord(cname, _row2num, '_', 1, 10)
        10
        >>> _resolve_coord(cname, _row2num, '.', 1, 10, 13)
        13


    But notice when base-cell missing::

        >>> _resolve_coord(cname, _row2num, '.', 0, 10, base_coord=None)
        Traceback (most recent call last):
        ValueError: invalid row('.') due to: '.'

    Other ROW error-checks::

        >>> _resolve_coord(cname, _row2num, '0', 0, 10)
        Traceback (most recent call last):
        ValueError: invalid row('0') due to: resolved to negative(-1)!

        >>> _resolve_coord(cname, _row2num, 'a', 0, 10)
        Traceback (most recent call last):
        ValueError: invalid row('a') due to: invalid literal for int() with base 10: 'a'

        >>> _resolve_coord(cname, _row2num, None, 0, 10)
        Traceback (most recent call last):
        ValueError: invalid row(None) due to:
                int() argument must be a string,
                a bytes-like object or a number, not 'NoneType'


    Column examples::

        >>> cname = 'column'

        >>> _resolve_coord(cname, _col2num, 'A', 1, 10)
        0
        >>> _resolve_coord(cname, _col2num, 'DADA', 1, 10)
        71084
        >>> _resolve_coord(cname, _col2num, '.', 1, 10, 13)
        13

    And COLUMN error-checks::

        >>> _resolve_coord(cname, _col2num, None, 0, 10)
        Traceback (most recent call last):
        ValueError: invalid column(None) due to: 'NoneType' object is not iterable

        >>> _resolve_coord(cname, _col2num, '4', 0, 10)
        Traceback (most recent call last):
        ValueError: invalid column('4') due to: substring not found

        >>> _resolve_coord(cname, _col2num, 4, 0, 10)
        Traceback (most recent call last):
        ValueError: invalid column(4) due to: 'int' object is not iterable


    """
    try:
        if coord in _special_coord_symbols:
            special_dict = {
                '^': up_coord,
                '_': dn_coord
            }
            if base_coord is not None:
                special_dict['.'] = base_coord
            rcoord = special_dict[coord]
        else:
            rcoord = cfunc(coord)

        if rcoord < 0:
            msg = 'resolved to negative(%s)!'
            raise ValueError(msg % rcoord)

        return rcoord
    except Exception as ex:
        msg = 'invalid {}({!r}) due to: {}'
        six.raise_from(ValueError(msg.format(cname, coord, ex)), ex)


def _resolve_cell(cell, up_coords, dn_coords, base_cords=None):
    """
    Translates any special coords to absolute ones.

    To get the margin_coords, use one of:

    * :meth:`Spreadsheet.get_margin_coords()`
    * :func:`_margin_coords_from_states_matrix()`

    :param Cell cell:
            The "A1" cell to translate its coords.
    :param Coords up_coords:
            the top-left resolved coords with full-cells
    :param Coords dn_coords:
            the bottom-right resolved coords with full-cells
    :param Coords base_cords:
                A resolved cell to base dependent coords (``.``).
    :return: the resolved cell-coords
    :rtype:  Coords


    Examples::

        >>> up = Coords(1, 2)
        >>> dn = Coords(10, 6)
        >>> base = Coords(40, 50)

        >>> _resolve_cell(Cell(col='B', row=5), up, dn)
        Coords(row=4, col=1)

        >>> _resolve_cell(Cell('^', '^'), up, dn)
        Coords(row=1, col=2)

        >>> _resolve_cell(Cell('_', '_'), up, dn)
        Coords(row=10, col=6)

        >>> base == _resolve_cell(Cell('.', '.'), up, dn, base)
        True

        >>> _resolve_cell(Cell('1', '5'), up, dn)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='1', col='5')) due to:
                invalid col('5') due to: substring not found

        >>> _resolve_cell(Cell('A', 'B'), up, dn)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='A', col='B')) due to:
                invalid row('A') due to: invalid literal for int() with base 10: 'A'

    But notice when base-cell missing::

        >>> _resolve_cell(Cell('1', '.'), up, dn)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='1', col='.')) due to: invalid col('.') due to: '.'

    """
    assert isinstance(cell, Cell), cell
    assert isinstance(up_coords, Coords), up_coords
    assert isinstance(dn_coords, Coords), dn_coords
    try:
        if base_cords is None:
            base_row = base_col = None
        else:
            base_row, base_col = base_cords
        row = _resolve_coord('row', _row2num, cell.row,
                             up_coords[0], dn_coords[0], base_row)
        col = _resolve_coord('col', _col2num, cell.col,
                             up_coords[1], dn_coords[1], base_col)

        return Coords(row, col)
    except Exception as ex:
        msg = "invalid cell(%s) due to: %s\n  margins(%s)\n  base_cords(%s)"
        log.debug(msg, cell, ex, (up_coords, dn_coords), base_cords)
        six.raise_from(ValueError("invalid cell(%s) due to: %s" % (cell, ex)),
                       ex)


def _target_opposite_state(states_matrix, up_coords, dn_coords,
                           state, land, moves):
    """

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`Spreadsheet.read_states_matrix()` to derrive it.
    :param Coords up_coords:
            the resolved-coords for the top-left of full-cells
    :param Coords dn_coords:
            the bottom-right for the top-left of full-cells
    :param bool state:
            the state of the landing-cell, or `False` if beyond limits
    :param land:
            the landing-cell
    :param moves:
    :return: the resolved num-coords
    :rtype: Coords


    Examples::

        >>> states_matrix = np.array([
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 1],
        ...     [0, 0, 1, 0, 0, 1],
        ...     [0, 0, 1, 1, 1, 1]
        ... ])
        >>> args = (states_matrix, Coords(2, 2), Coords(4, 5))

        >>> _target_opposite_state(*(args + (False, Coords(0, 0), 'DR')))
        Coords(row=3, col=2)

        >>> _target_opposite_state(*(args + (False, Coords(0, 0), 'RD')))
        Coords(row=2, col=3)

    It fails if a non-empty target-cell cannot be found, or
    it ends-up beyond bounds::

        >>> _target_opposite_state(*(args + (False, Coords(0, 0), 'D')))
        Traceback (most recent call last):
        ValueError: No full-target for landing-Coords(row=0, col=0) with movement(D)!

        >>> _target_opposite_state(*(args + (False, Coords(0, 0), 'UR')))
        Traceback (most recent call last):
        ValueError: No full-target for landing-Coords(row=0, col=0) with movement(UR)!


    But notice that the landing-cell maybe outside of bounds::

        >>> _target_opposite_state(*(args + (False, Coords(3, 10), 'L')))
        Coords(row=3, col=5)

    """
    assert isinstance(up_coords, Coords), up_coords
    assert isinstance(dn_coords, Coords), dn_coords
    assert isinstance(land, Coords), land

    target, last_move = _target_opposite_state_impl(
        states_matrix, up_coords, dn_coords, state, land, moves)

    if state and (land != target).any():
        target -= last_move

    # return target
    return Coords(target[0], target[1])


def _target_opposite_state_impl(states_matrix, up_coords, dn_coords,
                                state, land, moves):
    up_coords = Coords(0, 0)  # FIXME: up-margin on target-oposite.
    c0 = np.array(land)  # .copy()
    mv1 = _primitive_dir_vectors[moves[0]]
    mv2 = _primitive_dir_vectors[moves[1]] if len(moves) > 1 else None

    if not state:
        if land[0] > dn_coords[0] and 'U' in moves:
            c0[0] = dn_coords[0]
        if land[1] > dn_coords[1] and 'L' in moves:
            c0[1] = dn_coords[1]
        if land[0] < up_coords[0] and 'D' in moves:
            c0[0] = up_coords[0]
        if land[1] < up_coords[1] and 'R' in moves:
            c0[1] = up_coords[1]

    while (up_coords <= c0).all() and (c0 <= dn_coords).all():
        c1 = c0.copy()
        # Why rescan each time when searching-same?
        while (up_coords <= c1).all():
            try:
                if states_matrix[c1[0], c1[1]] != state:
                    return c1, mv1
            except IndexError:
                if state:
                    return c1, mv1
                break
            c1 += mv1

        if mv2 is None:
            break

        c0 += mv2

    if state:
        return c0, mv2

    raise ValueError(
        'No {}-target for landing-{} with movement({})!'.format(
            'empty' if state else 'full', land, moves))


def _target_same_state(states_matrix, up_coords, dn_coords, state, land, moves):
    """

    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`Spreadsheet.read_states_matrix()` to derrive it.
    :param Coords up_coords:
            the resolved-coords for the top-left of full-cells
    :param Coords dn_coords:
            the bottom-right for the top-left of full-cells
    :param bool state:
            the state of the landing-cell, or `False`
            if beyond limits
    :param Coords land:
            the landing-cell
    :param moves:
    :return: the identified cell
    :rtype: Coords


    Examples::

        >>> states_matrix = np.array([
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 1],
        ...     [0, 0, 1, 0, 0, 1],
        ...     [0, 0, 1, 1, 1, 1]
        ... ])
        >>> args = (states_matrix, Coords(2, 2), Coords(4, 5))

        >>> _target_same_state(*(args + (True, Coords(4, 5), 'U')))
        Coords(row=2, col=5)

        >>> _target_same_state(*(args + (True, Coords(4, 5), 'L')))
        Coords(row=4, col=2)

        >>> _target_same_state(*(args + (True, Coords(4, 5), 'UL', )))
        Coords(row=2, col=2)

        >>> _target_same_state(*(args + (True, Coords(2, 2), 'DR')))
        Coords(row=2, col=2)


    It fails if a non-empty target-cell cannot be found, or
    it ends-up beyond bounds::

        >>> _target_same_state(*(args + (False, Coords(2, 2), 'UL')))
        Traceback (most recent call last):
        ValueError: No full-target for landing-Coords(row=2, col=2) with movement(U)!


    But notice that the landing-cell maybe outside of bounds::

        >>> _target_same_state(*(args + (False, Coords(10, 3), 'U')))
        Coords(row=4, col=3)

    And this is the *negative* (??)::

        >>> _target_same_state(*(args + (True, Coords(2, 5), 'DL')))
        Coords(row=4, col=3)

    """

    c1 = land
    for mv in moves:
        c = _target_opposite_state(states_matrix, up_coords, dn_coords,
                                   state, land, mv)
        dis = _primitive_dir_vectors[mv]
        c1 = [i if not k == 0 else j for i, j, k in zip(c, c1, dis)]
    # return c1
    return Coords(*c1)


def _expand_rect(states_matrix, state, xl_rect, exp_mov):
    """
    Applies the :term:`expansion-moves` based on the `states_matrix`.

    :param state:
    :param xl_rect:
    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`Spreadsheet.read_states_matrix()` to derrive it.
    :param exp_mov:
    :return:


    Examples::

        >>> states_matrix = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 1, 1, 1],
        ...     [0, 0, 0, 1, 0, 0, 1],
        ...     [0, 0, 0, 1, 1, 1, 1]
        ... ])

        >>> rng = (Coords(6, 3), Coords(6, 3))
        >>> exp_mov = [_repeat_moves('U')]
        >>> _expand_rect(states_matrix, True, rng, exp_mov)
        [Coords(row=6, col=3), Coords(row=6, col=3)]

        >>> rng = (Coords(6, 3), Coords(7, 3))
        >>> exp_mov = [_repeat_moves('R')]
        >>> _expand_rect(states_matrix, True, rng, exp_mov)
        [Coords(row=6, col=3), Coords(row=7, col=6)]

        >>> rng = (Coords(6, 3), Coords(10, 3))
        >>> exp_mov = [_repeat_moves('R')]
        >>> _expand_rect(states_matrix, True, rng, exp_mov)
        [Coords(row=6, col=3), Coords(row=10, col=6)]

        >>> rng = (Coords(6, 5), Coords(6, 5))
        >>> exp_mov = [_repeat_moves('LURD')]
        >>> _expand_rect(states_matrix, True, rng, exp_mov)
        [Coords(row=5, col=3), Coords(row=7, col=6)]

    """
    assert isinstance(xl_rect[0], Coords), xl_rect
    assert isinstance(xl_rect[1], Coords), xl_rect
    _m = {
        'L': (0, 1),
        'U': (0, 1),
        'R': (1, 0),
        'D': (1, 0),
    }
    xl_rect = [np.array(v) for v in xl_rect]
    for moves in exp_mov:
        for directions in moves:
            flag = True
            for d in directions:
                mv = _primitive_dir_vectors[d]
                i, j = _m[d]
                st, nd = (xl_rect[i], xl_rect[j])
                st = st + mv
                nd = [p2 if k == 0 else p1 for p1, p2, k in zip(st, nd, mv)]
                if i == 1:
                    v = states_matrix[nd[0]:st[0] + 1, nd[1]:st[1] + 1]
                else:
                    v = states_matrix[st[0]:nd[0] + 1, st[1]:nd[1] + 1]
                if (not v.size and state) or (v != state).all():
                    continue
                xl_rect[i] = st
                flag = False

            if flag:
                break

    # return xl_rect
    return [Coords(*v) for v in xl_rect]


def resolve_capture_rect(states_matrix, up_coords, dn_coords, st_edge,
                         nd_edge=None, rect_exp=None):
    """
    Performs :term:`targeting` and applies :term:`expansions` but does not extract values.

    To get the margin_coords, use one of:

    * :meth:`Spreadsheet.get_margin_coords()`
    * :func:`_margin_coords_from_states_matrix()`

    Feed its results into :func:`read_capture_values()`.

    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`Spreadsheet.read_states_matrix()` to derrive it.
    :param Coords up_coords:
            the top-left coords with full-cells
    :param Coords dn_coords:
            the bottom-right coords with full-cells
    :param Edge st_edge: "uncooked" as matched by regex
    :param Edge nd_edge: "uncooked" as matched by regex
    :param list or none rect_exp:
            the result of :func:`_parse_rect_expansions()`

    :return:    a ``(Coords, Coords)`` with the 1st and 2nd :term:`capture-cell`
                ordered from top-left --> bottom-right.
    :rtype: tuple

    Examples::

        >>> states_matrix = np.array([
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 1],
        ...     [0, 0, 1, 0, 0, 1],
        ...     [0, 0, 1, 1, 1, 1]
        ... ], dtype=bool)
        >>> up, dn = _margin_coords_from_states_matrix(states_matrix)

        >>> st_edge = Edge(Cell('1', 'A'), 'DR')
        >>> nd_edge = Edge(Cell('.', '.'), 'DR')
        >>> resolve_capture_rect(states_matrix, up, dn, st_edge, nd_edge)
        (Coords(row=3, col=2), Coords(row=4, col=2))

    Walking backwards::

        >>> st_edge = Edge(Cell('_', '_'), None)
        >>> nd_edge = Edge(Cell('.', '.'), 'UL')
        >>> rect = resolve_capture_rect(states_matrix, up, dn, st_edge, nd_edge)
        >>> rect
        (Coords(row=2, col=2), Coords(row=4, col=5))

        >>> st_edge = Edge(Cell('^', '_'), None)
        >>> nd_edge = Edge(Cell('_', '^'), None)
        >>> rect == resolve_capture_rect(states_matrix, up, dn, st_edge, nd_edge)
        True

    """
    assert isinstance(up_coords, Coords), up_coords
    assert isinstance(dn_coords, Coords), dn_coords

    st = _resolve_cell(st_edge.land, up_coords, dn_coords)
    try:
        state = states_matrix[st]
    except IndexError:
        state = False

    if st_edge.mov is not None:
        st = _target_opposite_state(
            states_matrix, up_coords, dn_coords, state, st, st_edge.mov)
        state = not state

    if nd_edge is None:
        capt_rect = (st, st)
    else:
        nd = _resolve_cell(nd_edge.land, up_coords, dn_coords, st)

        if nd_edge.mov is not None:
            mov = nd_edge.mov

            try:
                nd_state = states_matrix[nd]
            except IndexError:
                nd_state = False

            if state == nd_state:
                nd = _target_same_state(states_matrix, up_coords, dn_coords,
                                        state, nd, mov)
            else:
                nd = _target_opposite_state(states_matrix, up_coords, dn_coords,
                                            not state, nd, mov)

        # Order rect-cells.
        #
        c = np.array([st, nd], dtype=np.int16)
        #capt_rect = c.min(0), c.max(0)
        capt_rect = (Coords(*c.min(0).tolist()), Coords(*c.max(0).tolist()))

    if rect_exp:
        capt_rect = _expand_rect(states_matrix, state, capt_rect, rect_exp)

    return capt_rect


def read_capture_rect(sheet, xl_rect):
    """
    Extracts :term:`capture-rect` values from excel-sheet and apply :term:`filters`.

    :param sheet:
            anything supporting the :func:`read_rect(states_matrix, xl_rect)`
            such as the the :class:`Spreadsheet` which can hide away
            the backend-module .
    :param tuple xl_rect:  tuple (num_cell, num_cell) with the edge targets of
                           the capture-rect
    :return: the rect values TODO: pre-processed

    .. testsetup::
        >>> import os, tempfile, xlrd, pandas as pd

        >>> df = pd.DataFrame([
        ... # Cols: 0       1    2
        ...        [None, None, None],
        ...        [5.1,  6.1,  7.1]
        ... ])
        >>> tmp = ''.join([tempfile.mkstemp()[1], '.xlsx'])
        >>> writer = pd.ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

    Examples::

        >>> sheet = Spreadsheet(xlrd.open_workbook(tmp).sheet_by_name('Sheet1'))
        >>> sheet.read_states_matrix()
        array([[False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False],
           [False, False, False, False,  True,  True,  True],
           [False, False, False,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True]], dtype=bool)

        # minimum matrix in the sheet
        >>> read_capture_rect(sheet, (Coords(5, 3), Coords(7, 6)))
        [[None,  0,    1,    2],
         [0,    None, None, None],
         [1,     5.1,  6.1,  7.1]]

        # single-value
        >>> read_capture_rect(sheet, (Coords(6, 3), Coords(6, 3)))
        [0]

        # column
        >>> read_capture_rect(sheet, (Coords(0, 3), Coords(7, 3)))
        [None, None, None, None, None, None, 0, 1]

        # row
        >>> read_capture_rect(sheet, (Coords(5, 0), Coords(5, 6)))
        [None, None, None, None, 0, 1, 2]

        # row beyond sheet-limits
        >>> read_capture_rect(sheet, (Coords(5, 0), Coords(5, 10)))
        [None, None, None, None, 0, 1, 2, None, None, None, None]

    .. testcleanup::
        >>> os.remove(tmp)
    """

    st_target, nd_target = xl_rect
    table = sheet.read_rect(xl_rect)

    # column
    if nd_target[1] == st_target[1]:
        table = [v[0] for v in table]

    # row
    if nd_target[0] == st_target[0]:
        table = table[0]

    if isinstance(table, list):
        return table
    else:
        return [table]


def _get_value_dim(value):
    try:
        if isinstance(value, list):
            return 1 + _get_value_dim(value[0])
    except IndexError:
        return 1
    return 0


def _redim_value(value, n):
    if n > 0:
        return [_redim_value(value, n - 1)]
    elif n < 0:
        if len(value) > 1:
            raise Exception
        return _redim_value(value[0], n + 1)
    return value


def _redim_captured_values(value, dim_min, dim_max=None):
    """
    Reshapes the output value of :func:`read_capture_rect()`.

    :param value: matrix or vector or value
    :type value: list of lists, list, value

    :param dim_min: minimum dimension or 'auto'
    :type dim_min: int, None

    :param dim_max: maximum dimension
    :type dim_max: int, None

    :return: reshaped value
    :rtype: list of lists, list, value


    Examples::

        >>> _redim_captured_values([1, 2], 2)
        [[1, 2]]

        >>> _redim_captured_values([[1, 2]], 1)
        [[1, 2]]

        >>> _redim_captured_values([[1, 2]], 1, 1)
        [1, 2]

        >>> _redim_captured_values([], 2)
        [[]]

        >>> _redim_captured_values([[1, 2]], 0, 0)
        Traceback (most recent call last):
        ValueError: Cannot reduce Captured-values dimension(2) to (0, 0)!

    """
    val_dim = _get_value_dim(value)
    try:
        if val_dim < dim_min:
            return _redim_value(value, dim_min - val_dim)
        elif dim_max is not None and val_dim > dim_max:
            return _redim_value(value, dim_max - val_dim)
        return value
    except:
        # TODO: Make redimming use np-arrays.
        msg = 'Cannot reduce Captured-values dimension({}) to ({}, {})!'
        raise ValueError(msg.format(val_dim, dim_min, dim_max))


def _type_df_with_numeric_conversion(df, args, kws):
    df = pd.DataFrame(args, kws)
    return df.convert_objects(convert_numeric=True)

_default_filters = {
    None: {'fun': lambda x: x},  # TODO: Actually _redim_captured_values().
    'df': {'fun': pd.DataFrame},
    'df_num': {'fun': _type_df_with_numeric_conversion},
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}


def _process_captured_values(value, func=None, args=(), kws=None, filters=None,
                             available_filters=_default_filters):
    """
    Processes the output value of :func:`read_capture_rect()` function.

    FIXME: Actually use _process_captured_values()!

    :param value: matrix or vector or a scalar-value
    :type value: list of lists, list, value

    :param str, None type:
            The 1st-filter to apply, if missing, applies the mapping found in
            the ``None --> <filter`` entry of the `available_filters` dict.
    :param dict, None kws:  keyword arguments for the filter function
    :param sequence, None args:
            arguments for the type-function
    :param list filters:
            A list of 3-tuples ``(filter_callable, *args, **kws)``
            to further process rect-values.
    :param dict available_filters:
            Entries of ``<fun_names> --> <callables>`` for pre-configured
            filters available to post-process rect-values.
            The callable for `None` key will be always called
            to the original values to ensure correct dimensionality
    :return: processed rect-values
    :rtype: given type, or list of lists, list, value


    Examples::

        >>> value = [[1, 2], [3, 4], [5, 6]]
        >>> res = _process_captured_values(value, func='dict')
        >>> sorted(res.items())
        [(1, 2),
         (3, 4),
         (5, 6)]

        >>> value = [[1, 9], [8, 10], [5, 11]]
        >>> _process_captured_values(value,
        ...     filters=[{'func':'sorted', 'kws':{'reverse': True}}])
        [[8, 10],
         [5, 11],
         [1, 9]]
    """
    if not kws:
        kws = {}
    val = available_filters[func]['fun'](value, *args, **kws)
    if filters:
        for v in filters:
            val = _process_captured_values(val, **v)
    return val


class Spreadsheet(object):
    """
    A delegating to backends excel-worksheets wrapper that is utilized by this module.

    :param np.array _states_matrix:
            The :term:`states-matrix` cached, so recreate object
            to refresh it.
    :param dict _margin_coords:
            limits used by :func:`_resolve_cell`, cached, so recreate object
            to refresh it.

    Resource management is outside of the scope of this class,
    and must happen in the backend workbook/sheet instance.

    *xlrd* examples::

        >>> import xlrd                                       #  doctest: +SKIP
        >>> with xlrd.open_workbook(self.tmp) as wb:          #  doctest: +SKIP
        ...     sheet = xlref.xlrdSheet(wb.sheet_by_name('Sheet1'))
        ...     ## Do whatever

    *win32* examples::

        >>> with dsgdsdsfsd as wb:          #  doctest: +SKIP
        ...     sheet = xlref.win32Sheet(wb.sheet['Sheet1'])
        TODO
    """
    _states_matrix = None
    _margin_coords = None

    def __init__(self, sheet, backend=_xlrd):
        if not isinstance(backend, ModuleType):
            import importlib
            backend = importlib.import_module(backend)
        self._backend = backend
        self._sheet = sheet

    def read_states_matrix(self):
        """
        Deduce the :term:`states-matrix` of the wrapped sheet.

        :return:   A 2D-array with `False` wherever cell are blank or empty.
        :rtype:     ndarray
        """
        if self._states_matrix is None:
            self._states_matrix = self._backend.read_states_matrix(self._sheet)
        return self._states_matrix

    def read_rect(self, xl_rect):
        return self._backend.read_rect(self._sheet, self.read_states_matrix(),
                                       xl_rect)

    def _read_margin_coords(self):
        """
        Override if possible to read (any of the) limits directly from the sheet.

        :return:    the 2 coords of the top-left & bottom-right full cells;
                    anyone coords can be None.
                    By default returns ``(None, None)``.
        :rtype:     (Coords, Coords)

        """
        return None, None  # pragma: no cover

    def get_margin_coords(self):
        """
        Extract (and cache) margins either internally or from :func:`_margin_coords_from_states_matrix()`.

        :return:    the resolved top-left and bottom-right :class:`Coords`
        :rtype:     tuple


        Examples::

            >>> sheet = Spreadsheet(sheet=None)
            >>> sheet._states_matrix = np.asarray([       ## Mock states_matrix.
            ...    [0, 0, 0, 0],
            ...    [1, 1, 0, 0],
            ...    [0, 1, 1, 0],
            ...    [0, 0, 1, 0],
            ... ])
            >>> sheet.get_margin_coords()
            (Coords(row=1, col=0), Coords(row=3, col=2))

        """
        if not self._margin_coords:
            up, dn = self._read_margin_coords()
            if up is None or dn is None:
                sm = self.read_states_matrix()
                up1, dn1 = _margin_coords_from_states_matrix(sm)
                up = up or up1
                dn = dn or dn1
            self._margin_coords = up, dn

        return self._margin_coords
