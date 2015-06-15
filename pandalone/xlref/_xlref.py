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
from . import _xlrd

import itertools as itt
import numpy as np
import pandas as pd
from six.moves.urllib.parse import urldefrag  # @UnresolvedImport


log = logging.getLogger(__name__)


_special_coords = {'^', '_', '.'}

_primitive_dir = {
    'L': np.array([0, -1]),
    'U': np.array([-1, 0]),
    'R': np.array([0, 1]),
    'D': np.array([1, 0])
}

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
    ^(?P<moves>[LURD]+)                                    # primitive moves
    (?P<times>\?|\d+)?                                   # repetition times
    $""",
    re.IGNORECASE | re.X)


Cell = namedtuple('Cell', ['row', 'col'])
"""Its coords might be "A1" (strings, 1-based) or "num" (0-based)."""

Edge = namedtuple('Edge', ['cell', 'mov'])
"""
An :term:`Edge` might be "cooked" or "uncooked" depending on its `Cell`.

- An *uncooked* edge contains *A1* :class:`Cell`.
- An *cooked* edge contains a *num* :class:`Cell`.
"""


def num2a1_Cell(row, col):
    """Make *A1* :class:``Cell`` from *num* or special coords, with rudimentary error-checking.

    Examples::

        >>> num2a1_Cell(row=0, col=0)
        Cell(row='1', col='A')
        >>> num2a1_Cell(row=0, col=26)
        Cell(row='1', col='AA')

        >>> num2a1_Cell(row=10, col='.')
        Cell(row='11', col='.')

        >>> num2a1_Cell(row=-3, col=-2)
        Traceback (most recent call last):
        AssertionError: negative row!


    """
    if row not in _special_coords:
        assert row >= 0, 'negative row!'
        row = str(row + 1)
    if col not in _special_coords:
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
        Edge(cell=Cell(row='1', col='A'), mov='RUL')


    No error checking performed::

        >>> _uncooked_Edge('Any', 'foo', 'BaR')
        Edge(cell=Cell(row='Any', col='FOO'), mov='BAR')

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

    return Edge(cell=Cell(col=col and col.upper(), row=row), mov=mov and mov.upper())


def _repeat_moves(moves, times=None):
    """
    Returns an iterator that repeats `moves` x `times`, or infinite if unspecified.

    Used when parsing :term:`primitive-directions`.

   :param str moves: the moves to repeat
   :param str times: N of repetitions. If `None` it means infinite repetitions.
   :return:    An iterator of the moves
   :rtype:     iterator

    Examples::

         >>> list(_repeat_moves('LuR', '3'))
         ['LUR', 'LUR', 'LUR']
         >>> list(_repeat_moves('ABC', '0'))
         []
         >>> _repeat_moves('ABC')  ## infinite repetitions
         repeat('ABC')
     """
    args = (moves.upper(),)
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
        - st_ref: (Edge, None) the 1st-ref, uncooked, with raw cell
        - nd_ref: (Edge, None) the 2nd-ref, uncooked, with raw cell
        - rect_exp: (str) as found on the xl-ref
        - json: parsed

    :rtype: dict


    Examples::

        >>> res = parse_xl_ref('Sheet1!A1(DR):Z20(UL):L1U2R1D1:{"json":"..."}')
        >>> sorted(res.items())
        [('json', {'json': '...'}),
         ('nd_ref', Edge(cell=Cell(row='20', col='Z'), mov='UL')),
         ('rect_exp', 'L1U2R1D1'),
         ('sheet', 'Sheet1'),
         ('st_ref', Edge(cell=Cell(row='1', col='A'), mov='DR'))]

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
        gs['st_ref'] = _uncooked_Edge(
            p('st_row'), p('st_col'), p('st_mov'))
        gs['nd_ref'] = _uncooked_Edge(
            p('nd_row'), p('nd_col'), p('nd_mov'))

        js = gs['json']
        gs['json'] = json.loads(js) if js else None

        return gs

    except Exception as ex:
        msg = "Invalid xl-ref(%s) due to: %s"
        log.debug(msg, xl_ref, ex, exc_info=1)
        raise ValueError(msg % (xl_ref, ex))


def parse_xl_url(url):
    """
    Parses the contents of an :term:`xl-url`.

    :param str url:
        a string with the following format::

            <url_file>#<sheet>!<1st_ref>:<2nd_ref>:<expand><json>

        Exxample::

            file:///path/to/file.xls#sheet_name!UP10:DN20:LDL1{"dim":2}

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
         ('nd_ref', Edge(cell=Cell(row='^', col='.'), mov='DR')),
         ('rect_exp', 'LU?'),
         ('sheet', 'Sheet1'),
         ('st_ref', Edge(cell=Cell(row='1', col='A'), mov='UL')),
         ('url_file', 'file:///sample.xlsx')]
    """

    try:
        res = {}

        res['url_file'], frag = urldefrag(url)  # parse excel url

        res.update(parse_xl_ref(frag))  # resolve excel reference

        return res

    except Exception as ex:
        raise ValueError("Invalid xl-url({}) due to: {}".format(url, ex))


def get_sheet_margins(states_matrix):
    """
    Returns top-left and bottom-down margins and all full-incdices from a :term:`state` matrix.

    Cache its return-value to use it in other functions here needing it.

    :param ndarray states_matrix:
            An array with `False` wherever cell are blank or empty.
            Use :func:`read_states_matrix()` to derrive it.
    :return:  a `Cell` with zero-based top-left/bottom-right margins for rows/cols,
    :rtype: tuple

    Examples::

        >>> states_matrix = [
        ...    [0, 0, 0],
        ...    [0, 1, 0],
        ...    [0, 1, 1],
        ...    [0, 0, 1],
        ... ]
        >>> margins = get_sheet_margins(states_matrix)
        >>> margins                                         # doctest: +SKIP
        Cell(row={'_': 3, '^': 1}, col={'_': 2, '^': 1})


    Note that the botom-left cell is not the same as `states_matrix` matrix size::

        >>> states_matrix = [
        ...    [0, 0, 0, 0],
        ...    [0, 1, 0, 0],
        ...    [0, 1, 1, 0],
        ...    [0, 0, 1, 0],
        ...    [0, 0, 0, 0],
        ... ]
        >>> margins_2 = get_sheet_margins(states_matrix)
        >>> margins_2 == margins
        True

    """
    indices = np.asarray(np.where(states_matrix)).T
    up_r, up_c = indices.min(0)
    dn_r, dn_c = indices.max(0)
    sheet_margins = Cell(
        row={'^': up_r, '_': dn_r},
        col={'^': up_c, '_': dn_c})
    return sheet_margins


def _build_special_coords(cord_bounds, base_coord):
    """Make a stacked dict of margins and base-coord, used for resolving all specials coords. """
    try:
        from collections import ChainMap
        return ChainMap(cord_bounds, {'.': base_coord})
    except ImportError:
        # TODO: FIX hack when ChainMap backported to py2.
        c = {'.': base_coord}
        c.update(cord_bounds)

        return c


def _resolve_coord(cname, cfunc, coord, cbounds, bcoord=None):
    """
    Translates special coords or converts Excel string 1-based rows/cols to zero-based, reporting invalids.

    :param str        cname:  the coord-name, one of 'row', 'column'
    :param function   cfunc:  the function to convert coord ``str --> int``
    :param int, str   coord:  the coord to translate
    :param dict     cbounds:  the coord part of :func:`get_sheet_margins()`
    :param int, None bcoord:  the basis for dependent coord, if any

    :return: the resolved coord or `None` if it were not a special coord.


    Row examples::

        >>> cbounds = {'_': 10, '^':1}
        >>> cname = 'row'

        >>> r0 = _resolve_coord(cname, _row2num, '1', cbounds)
        >>> r0
        0
        >>> r0 == _resolve_coord(cname, _row2num, 1, cbounds)
        True
        >>> _resolve_coord(cname, _row2num, '_', cbounds)
        10
        >>> _resolve_coord(cname, _row2num, '^', cbounds)
        1
        >>> _resolve_coord(cname, _row2num, '.', cbounds, 13)
        13


    But notice when base-cell missing::

        >>> _resolve_coord(cname, _row2num, '.', cbounds, bcoord=None)
        Traceback (most recent call last):
        ValueError: invalid row('.') due to: '.'

    Other ROW error-checks::

        >>> _resolve_coord(cname, _row2num, '0', cbounds)
        Traceback (most recent call last):
        ValueError: invalid row('0') due to: resolved to negative(-1)!

        >>> _resolve_coord(cname, _row2num, 'a', cbounds)
        Traceback (most recent call last):
        ValueError: invalid row('a') due to: invalid literal for int() with base 10: 'a'

        >>> _resolve_coord(cname, _row2num, None, cbounds)
        Traceback (most recent call last):
        ValueError: invalid row(None) due to: int() argument must be a string,
        a bytes-like object or a number, not 'NoneType'


    Column examples::

        >>> cname = 'column'

        >>> _resolve_coord(cname, _col2num, 'A', cbounds)
        0
        >>> _resolve_coord(cname, _col2num, 'DADA', cbounds)
        71084
        >>> _resolve_coord(cname, _col2num, '.', cbounds, 13)
        13

    And COLUMN error-checks::

        >>> _resolve_coord(cname, _col2num, None, cbounds)
        Traceback (most recent call last):
        ValueError: invalid column(None) due to: 'NoneType' object is not iterable

        >>> _resolve_coord(cname, _col2num, '4', cbounds)
        Traceback (most recent call last):
        ValueError: invalid column('4') due to: substring not found

        >>> _resolve_coord(cname, _col2num, 4, cbounds)
        Traceback (most recent call last):
        ValueError: invalid column(4) due to: 'int' object is not iterable


    """
    try:
        if coord in _special_coords:
            if bcoord:
                cbounds = _build_special_coords(cbounds, bcoord)
            rcoord = cbounds[coord]
        else:
            rcoord = cfunc(coord)

        if rcoord < 0:
            msg = 'resolved to negative(%s)!'
            raise ValueError(msg % rcoord)

        return rcoord
    except Exception as ex:
        msg = 'invalid {}({!r}) due to: {}'
        raise ValueError(msg.format(cname, coord, ex))


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


def _resolve_cell(cell, margins, bcell=None):
    """
    Translates any special coords to absolute ones.

    :param Cell cell:     The raw cell to translate its coords.
    :param Cell bcell:    A resolved cell to base any dependent coords (``.``).
    :param Cell margins:  see :func:`get_sheet_margins()`
    :rtype: Cell


    Examples::

        >>> margins = Cell(
        ...     row={'^':1, '_':10},
        ...     col={'^':2, '_':6})

        >>> _resolve_cell(Cell(col='A', row=5), margins)
        Cell(row=4, col=0)

        >>> _resolve_cell(Cell('^', '^'), margins)
        Cell(row=1, col=2)

        >>> _resolve_cell(Cell('_', '_'), margins)
        Cell(row=10, col=6)

        >>> _resolve_cell(Cell('1', '5'), margins)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='1', col='5')) due to:
                invalid col('5') due to: substring not found

        >>> _resolve_cell(Cell('A', 'B'), margins)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='A', col='B')) due to:
                invalid row('A') due to: invalid literal for int() with base 10: 'A'

        >>> _resolve_cell(Cell('1', '.'), margins)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='1', col='.')) due to: invalid col('.') due to: '.'

    """
    try:
        row = _resolve_coord('row', _row2num, cell.row, margins.row,
                             bcell and bcell.row)
        col = _resolve_coord('col', _col2num, cell.col, margins.col,
                             bcell and bcell.col)

        return Cell(row=row, col=col)
    except Exception as ex:
        msg = "invalid cell(%s) due to: %s\n  margins(%s)\n  bcell(%s)"
        log.debug(msg, cell, ex, margins, bcell)
        raise ValueError("invalid cell(%s) due to: %s" % (cell, ex))


def _target_opposite_state(state, cell, states_matrix, dn, moves, last=False):
    """

    :param bool state:      the starting-state
    :param cell:
    :param ndarray states_matrix:
            An array with `False` wherever cell are blank or empty.
            Use :func:`read_states_matrix()` to derrive it.
    :param sheet:
    :param moves:
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

        >>> args = (False, Cell(1, 1), states_matrix, (7, 6))
        >>> _target_opposite_state(*(args + ('DR', )))
        Cell(row=6, col=3)

        >>> _target_opposite_state(*(args + ('RD', )))
        Cell(row=5, col=4)

        >>> _target_opposite_state(*(args + ('D', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(D)

        >>> _target_opposite_state(*(args + ('U', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(U)

        >>> _target_opposite_state(*(args + ('R', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(R)

        >>> _target_opposite_state(*(args + ('L', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(L)

        >>> _target_opposite_state(*(args + ('LU', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(LU)

        >>> args = (True, Cell(6, 3), states_matrix, (7, 6))
        >>> _target_opposite_state(*(args + ('D', )))
        Cell(row=7, col=3)

        >>> args = (True, Cell(10, 3), states_matrix, (7, 6))
        >>> _target_opposite_state(*(args + ('U', )))
        Cell(row=10, col=3)

        >>> args = (False, Cell(10, 10), states_matrix, (7, 6))
        >>> _target_opposite_state(*(args + ('UL', )))
        Cell(row=7, col=6)

        >>> states_matrix = np.array([
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ... ])
        >>> args = (True, Cell(0, 2), states_matrix, (2, 2))
        >>> _target_opposite_state(*(args + ('LD', )))
        Cell(row=2, col=2)
    """
    up = (0, 0)
    mv = _primitive_dir[moves[0]]  # first move
    c0 = np.array(cell)

    if not state:
        if not c0[0] <= dn[0] and 'U' in moves:
            c0[0] = dn[0]
        if not c0[1] <= dn[1] and 'L' in moves:
            c0[1] = dn[1]
    else:
        last = True

    flag = False
    while True:
        c1 = c0
        while (up <= c1).all():
            try:
                if states_matrix[c1[0], c1[1]] != state:
                    if last and flag:
                        c1 = c1 - mv
                    return Cell(*(c1[0], c1[1]))
            except IndexError:
                if state:
                    if last and flag:
                        c1 = c1 - mv
                    return Cell(*(c1[0], c1[1]))
                break
            c1 = c1 + mv
            flag = True

        try:
            c0 = c0 + _primitive_dir[moves[1]]  # second move
        except IndexError:
            break

        if not ((up <= c0).all() and (c0 <= dn).all()):
            if state:
                if last:
                    c0 = c0 - _primitive_dir[moves[1]]
                return Cell(*(c0[0], c0[1]))
            break

    raise ValueError('Invalid {} with movement({})'.format(cell, moves))


def _target_same_state(state, cell, states_matrix, dn, moves):
    """

    :param bool state:      the starting-state
    :param cell:
    :param ndarray states_matrix:
            An array with `False` wherever cell are blank or empty.
            Use :func:`read_states_matrix()` to derrive it.
    :param sheet:
    :param moves:
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
        >>> args = (True, Cell(7, 6), states_matrix, (7, 6))
        >>> _target_same_state(*(args + ('UL', )))
        Cell(row=5, col=3)

        >>> _target_same_state(*(args + ('U', )))
        Cell(row=5, col=6)

        >>> _target_same_state(*(args + ('L', )))
        Cell(row=7, col=3)

        >>> args = (True, Cell(5, 3), states_matrix, (7, 6))
        >>> _target_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> args = (False, Cell(5, 3), states_matrix, (7, 6))
        >>> _target_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> _target_same_state(*(args + ('UL', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=5, col=3) with movement(U)

        >>> args = (True, Cell(5, 6), states_matrix, (7, 6))
        >>> _target_same_state(*(args + ('DL', )))
        Cell(row=7, col=4)

    """

    c1 = list(cell)

    for mv in moves:
        c = _target_opposite_state(state, cell, states_matrix, dn, mv, True)
        dis = _primitive_dir[mv]
        c1 = [i if not k == 0 else j for i, j, k in zip(c, c1, dis)]
    return Cell(*c1)


def _expand_rect(state, xl_rect, states_matrix, rect_exp):
    """

    :param state:
    :param xl_rect:
    :param ndarray states_matrix:
            An array with `False` wherever cell are blank or empty.
            Use :func:`read_states_matrix()` to derrive it.
    :param rect_exp:
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

        >>> rng = (Cell(row=6, col=3), Cell(row=6, col=3))
        >>> rect_exp = [_repeat_moves('U')]
        >>> _expand_rect(True, rng, states_matrix, rect_exp)
        [Cell(row=6, col=3), Cell(row=6, col=3)]

        >>> rng = (Cell(row=6, col=3), Cell(row=7, col=3))
        >>> rect_exp = [_repeat_moves('R')]
        >>> _expand_rect(True, rng, states_matrix, rect_exp)
        [Cell(row=6, col=3), Cell(row=7, col=6)]

        >>> rng = (Cell(row=6, col=3), Cell(row=10, col=3))
        >>> rect_exp = [_repeat_moves('R')]
        >>> _expand_rect(True, rng, states_matrix, rect_exp)
        [Cell(row=6, col=3), Cell(row=10, col=6)]

        >>> rng = (Cell(row=6, col=5), Cell(row=6, col=5))
        >>> rect_exp = [_repeat_moves('LURD')]
        >>> _expand_rect(True, rng, states_matrix, rect_exp)
        [Cell(row=5, col=3), Cell(row=7, col=6)]

    """
    _m = {
        'L': (0, 1),
        'U': (0, 1),
        'R': (1, 0),
        'D': (1, 0)
    }
    xl_rect = [np.array(v) for v in xl_rect]
    for moves in rect_exp:
        for directions in moves:
            flag = True
            for d in directions:
                mv = _primitive_dir[d]
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

    return [Cell(*v) for v in xl_rect]


def resolve_capture_rect(states_matrix, sheet_margins, st_ref,
                         nd_ref=None, rect_exp=None):
    """
    Performs :term:`targeting` and applies :term:`expansions` but does not extract values.

    Feed the results into :func:`read_capture_values()`.

    :param ndarray states_matrix:
            An array with `False` wherever cell are blank or empty.
            Use :func:`read_states_matrix()` to derrive it.
    :param Edge st_ref:  "uncooked" as matched by regex
    :param Edge nd_ref:  "uncooked" as matched by regex
    :param rect_exp:
    :return: a ``(Cell, Cell)`` with the 1st and 2nd :term:`capture-cell`
    :rtype: tuple

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
        >>> sheet_margins = Cell(row={'_': 7, '^': 5}, col={'_': 6, '^': 3})

        >>> st_ref = Edge(Cell('1', 'A'), 'DR')
        >>> nd_ref = Edge(Cell('.', '.'), 'DR')
        >>> resolve_capture_rect(states_matrix, sheet_margins, st_ref, nd_ref)
        (Cell(row=6, col=3), Cell(row=7, col=3))

        >>> nd_ref = Edge(Cell('8', 'G'), 'UL')
        >>> resolve_capture_rect(states_matrix, sheet_margins, st_ref, nd_ref)
        (Cell(row=5, col=3), Cell(row=6, col=3))
    """

    dn = (sheet_margins[0]['_'], sheet_margins[1]['_'])

    st = _resolve_cell(st_ref.cell, sheet_margins)
    try:
        state = states_matrix[st]
    except IndexError:
        state = False

    if st_ref.mov is not None:
        st = _target_opposite_state(state, st, states_matrix, dn, st_ref.mov)
        state = not state

    if nd_ref is None:
        nd = Cell(*st)
    else:
        nd = _resolve_cell(nd_ref.cell, sheet_margins, st)

        if nd_ref.mov is not None:
            mov = nd_ref.mov
            if state == states_matrix[nd]:
                nd = _target_same_state(state, nd, states_matrix, dn, mov)
            else:
                nd = _target_opposite_state(
                    not state, nd, states_matrix, dn, mov)

        c = np.array([st, nd])

        st, nd = (Cell(*list(c.min(0))), Cell(*list(c.max(0))))

    if rect_exp is None:
        return (st, nd)
    else:
        rect_exp = _parse_rect_expansions(rect_exp)
        return _expand_rect(state, (st, nd), states_matrix, rect_exp)


def read_capture_rect_values(sheet, xl_rect, states_matrix):
    """
    Extracts :term:`capture-rect` values from excel-sheet and apply :term:`filters`.

    :param sheet:
    :param tuple xl_rect:  tuple (num_cell, num_cell) with the edge targets of
                           the capture-rect
    :param ndarray states_matrix:
            An array with `False` wherever cell are blank or empty.
            Use :func:`read_states_matrix()` to derrive it.
    :return:

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
        >>> sheet = xlrd.open_workbook(tmp).sheet_by_name('Sheet1')

    Examples::

        >>> stm = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 1, 1, 1],
        ...     [0, 0, 0, 1, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 1, 1]], dtype=bool)

        # minimum matrix in the sheet
        >>> read_capture_rect_values(sheet, (Cell(5, 3), Cell(7, 6)), stm)
        [[None,  0,    1,    2],
         [0,    None, None, None],
         [1,     5.1,  6.1,  7.1]]

        # get single value
        >>> read_capture_rect_values(sheet, (Cell(6, 3), Cell(6, 3)), stm)
        [0]

        # get column vector
        >>> read_capture_rect_values(sheet, (Cell(0, 3), Cell(7, 3)), stm)
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> read_capture_rect_values(sheet, (Cell(5, 0), Cell(5, 6)), stm)
        [None, None, None, None, 0, 1, 2]

        # get row vector
        >>> read_capture_rect_values(sheet, (Cell(5, 0), Cell(5, 10)), stm)
        [None, None, None, None, 0, 1, 2, None, None, None, None]

    .. testcleanup::
        >>> os.remove(tmp)
    """

    st_target = xl_rect[0]
    nd_target = xl_rect[1]

    table = _xlrd.read_rect(sheet, states_matrix, st_target, nd_target)
    # vector
    if nd_target.col == st_target.col:
        table = [v[0] for v in table]

    # vector
    if nd_target.row == st_target.row:
        table = table[0]

    if isinstance(table, list):
        return table
    else:
        return [table]


def _get_value_dim(value):
    """ FIXME: _get_value_dim() UNUSED? """
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
    Reshapes the output value of :func:`read_capture_rect_values()`.

    :param value: matrix or vector or value
    :type value: list of lists, list, value

    :param dim_min: minimum dimension
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


def _process_captured_values(value, type=None, args=(), kws=None, filters=None,
                             available_filters=_default_filters):
    """
    Processes the output value of :func:`read_capture_rect_values()` function.

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
        >>> res = _process_captured_values(value, type='dict')
        >>> sorted(res.items())
        [(1, 2),
         (3, 4),
         (5, 6)]

        >>> value = [[1, 9], [8, 10], [5, 11]]
        >>> _process_captured_values(value,
        ...     filters=[{'type':'sorted', 'kws':{'reverse': True}}])
        [[8, 10],
         [5, 11],
         [1, 9]]
    """
    if not kws:
        kws = {}
    val = available_filters[type]['fun'](value, *args, **kws)
    if filters:
        for v in filters:
            val = _process_captured_values(val, **v)
    return val
