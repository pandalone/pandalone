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
from __future__ import unicode_literals

from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict, defaultdict
import inspect
import json
import logging
import re
from string import ascii_uppercase
import textwrap

from future import utils as fututils  # @UnresolvedImport
from future.backports import ChainMap  # @UnresolvedImport
from future.moves.urllib.parse import urldefrag  # @UnresolvedImport
from future.utils import with_metaclass
from past.builtins import basestring
from toolz import dicttoolz as dtz

import itertools as itt
import numpy as np


log = logging.getLogger(__name__)

try:
    from xlrd import colname as xl_colname
    # TODO: Try different backends providing `colname` function.
except ImportError:
    log.warning(
        'One of `xlrd`, `...` libraries is needed, will crash later!')


CHECK_CELLTYPE = False
"""When `True`, most coord-functions accept any 2-tuples."""

OPTS = 'opts'
"""The key for specifying options within :term:`filters`."""

Cell = namedtuple('Cell', ['row', 'col'])
"""
A pair of 1-based strings, denoting the "A1" coordinates of a cell.

The "num" coords (numeric, 0-based) are specified using numpy-arrays
(:class:`Coords`).
"""


Coords = namedtuple('Coords', ['row', 'col'])
"""
A pair of 0-based integers denoting the "num" coordinates of a cell.

The "A1" coords (1-based coordinates) are specified using :class:`Cell`.
"""
#     return np.array([row, cell], dtype=np.int16)


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
        col = xl_colname(col)
    return Cell(row=row, col=col)

_Edge = namedtuple('Edge', ['land', 'mov', 'mod'])
"""
:param Cell land:
:param str mov: use None for missing moves.
:param str mod: one of (`+`, `-` or `None`)

An :term:`Edge` might be "cooked" or "uncooked" depending on its `land`:

- An *uncooked* edge contains *A1* :class:`Cell`.
- An *cooked* edge contains a *resolved* :class:`Coords`.

"""


def Edge(land, mov=None, mod=None):
    return _Edge(land, mov, mod)


def _uncooked_Edge(row, col, mov, mod=None):
    """
    Make a new `Edge` from any non-values supplied, as is capitalized, or nothing.

    :param str, None col:    ie ``A``
    :param str, None row:    ie ``1``
    :param str, None mov:    ie ``RU``
    :param str, None mod:    ie ``+``

    :return:    a `Edge` if any non-None
    :rtype:     Edge, None


    Examples::

        >>> tr = _uncooked_Edge('1', 'a', 'Rul', '-')
        >>> tr
        Edge(land=Cell(row='1', col='A'), mov='RUL', mod='-')


    No error checking performed::

        >>> _uncooked_Edge('Any', 'foo', 'BaR', '+_&%')
        Edge(land=Cell(row='Any', col='FOO'), mov='BAR', mod='+_&%')

        >>> print(_uncooked_Edge(None, None, None, None))
        None


    except were coincidental::

        >>> _uncooked_Edge(row=0, col=123, mov='BAR', mod=None)
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'

        >>> _uncooked_Edge(row=0, col='A', mov=123, mod=None)
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'
    """

    if col == row == mov == mod is None:
        return None

    return Edge(land=Cell(col=col and col.upper(), row=row),
                mov=mov and mov.upper(), mod=mod)

_special_coord_symbols = {'^', '_', '.'}

_primitive_dir_vectors = {
    'L': Coords(0, -1),
    'U': Coords(-1, 0),
    'R': Coords(0, 1),
    'D': Coords(1, 0)
}

_re_xl_ref_parser = re.compile(
    r"""
    ^\s*(?:(?P<sheet>[^!]+)?!)?                          # xl sheet name
    (?:                                                  # 1st-edge
        (?:
            (?:
            (?P<st_col>[A-Z]+|[_^])                      # col
            (?P<st_row>[123456789]\d*|[_^])              # row
            ) | (?:
            R(?P<st_row2>-?[123456789]\d*|[_^])
            C(?P<st_col2>-?[123456789]\d*|[_^])
            )
        )
        (?:\(
            (?P<st_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves
            (?P<st_mod>[+-])?                            # move modifiers
            \)
        )?
    )
    (?::                                                 # 2nd-edge [opt]
        (?:
            (?:
            (?P<nd_col>[A-Z]+|[_^.])                     # col
            (?P<nd_row>[123456789]\d*|[_^.])             # row
            ) | (?:
            R(?P<nd_row2>-?[123456789]\d*|[_^.])
            C(?P<nd_col2>-?[123456789]\d*|[_^.])
            )
        )
        (?:\(
            (?P<nd_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves
            (?P<nd_mod>[+-])?                            # move-modifiers
            \)
        )?
        (?::
            (?P<exp_moves>[LURD?123456789]+)              # expansion moves [opt]
        )?
    )?
    \s*
    (?::?\s*
        (?P<json>[[{"].*)?                                #  [opt] json
    )$
    """,
    re.IGNORECASE | re.X | re.DOTALL)
"""
This regex produces the following capture-groups:

- sheet: i.e. ``Sheet 1``
- st_col: i.e. ``UPT``
- st_row: i.e. ``8``
- st_mov: i.e. ``LU``
- st_mod: i.e. ``-``
- nd_col: i.e. ``_``
- nd_row: i.e. ``.``
- nd_mov: i.e. ``D``
- nd_mod: i.e. ``+``
- exp_moves: i.e. ``LDL1``
"""

_re_exp_moves_splitter = re.compile('([LURD]\d+)', re.IGNORECASE)

# TODO: Make exp_moves `?` work different from numbers.
_re_exp_moves_parser = re.compile(
    r"""
    ^(?P<moves>[LURD]+)                                  # primitive moves
    (?P<times>\?|\d+)?                                   # repetition times
    $""",
    re.IGNORECASE | re.X)


def _repeat_moves(moves, times=None):
    """
    Returns an iterator that repeats `moves` x `times`, or infinite if unspecified.

    Used when parsing primitive :term:`directions`.

   :param str moves: the moves to repeat ie ``RU1D?``
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


def _parse_expansion_moves(exp_moves):
    """
    Parse rect-expansion into a list of dir-letters iterables.

    :param exp_moves:
        A string with a sequence of primitive moves:
        es. L1U1R1D1
    :type xl_ref: str

    :return:
        A list of primitive-dir chains.
    :rtype: list


    Examples::

        >>> res = _parse_expansion_moves('LURD?')
        >>> res
        [repeat('LUR'), repeat('D', 1)]

        # infinite generator
        >>> [next(res[0]) for i in range(10)]
        ['LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR']

        >>> list(res[1])
        ['D']

        >>> _parse_expansion_moves('1LURD')
        Traceback (most recent call last):
        ValueError: Invalid rect-expansion(1LURD) due to:
                'NoneType' object has no attribute 'groupdict'

    """
    try:
        res = _re_exp_moves_splitter.split(exp_moves.replace('?', '1'))

        return [_repeat_moves(**_re_exp_moves_parser.match(v).groupdict())
                for v in res
                if v != '']

    except Exception as ex:
        msg = 'Invalid rect-expansion({}) due to: {}'
        raise ValueError(msg.format(exp_moves, ex))


def _parse_xl_ref(xl_ref):
    """
    Parses a :term:`xl-ref` and splits it in its "ingredients".

    :param str xl_ref:
            a string with the following format::

                <sheet>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):<exp_moves>{<json>}

            i.e.::

                sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}

    :return:
        dictionary containing the following parameters:

        - sheet: (str, int, None) i.e. ``sheet_name``
        - st_edge: (Edge, None) the 1st-ref, uncooked, with raw cell
          i.e. ``Edge(land=Cell(row='8', col='UPT'), mov='LU', mod='-')``
        - nd_edge: (Edge, None) the 2nd-ref, uncooked, with raw cell
          i.e. ``Edge(land=Cell(row='_', col='.'), mov='D', mod='+')``
        - exp_moves: (sequence, None), as i.e. ``LDL1`` parsed by 
          :func:`_parse_expansion_moves()`
        - json: dict i.e. ``{"dims: 1}``

    :rtype: dict


    Examples::

        >>> res = _parse_xl_ref('Sheet1!A1(DR+):Z20(UL):L1U2R1D1:{"json":"..."}')
        >>> sorted(res.items())
        [('exp_moves', [repeat('L', 1), repeat('U', 2), repeat('R', 1), repeat('D', 1)]),
         ('json', {'json': '...'}),
         ('nd_edge', Edge(land=Cell(row='20', col='Z'), mov='UL', mod=None)),
         ('sheet', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='DR', mod='+'))]
        >>> _parse_xl_ref('A1(DR)Z20(UL)')
        Traceback (most recent call last):
        ValueError: Not an `xl-ref` syntax.
    """

    m = _re_xl_ref_parser.match(xl_ref)
    if not m:
        raise ValueError('Not an `xl-ref` syntax.')
    gs = m.groupdict()

    # Replace coords of 1st and 2nd cells
    #     with "uncooked" edge.
    #
    p = gs.pop
    r, c = p('st_row'), p('st_col')
    r2, c2 = p('st_row2'), p('st_col2')
    if r2 is not None:
        r, c = r2, c2
    gs['st_edge'] = _uncooked_Edge(r, c,
                                   p('st_mov'), p('st_mod'))
    r, c = p('nd_row'), p('nd_col')
    r2, c2 = p('nd_row2'), p('nd_col2')
    if r2 is not None:
        r, c = r2, c2
    gs['nd_edge'] = _uncooked_Edge(r, c,
                                   p('nd_mov'), p('nd_mod'))

    exp_moves = gs['exp_moves']
    gs['exp_moves'] = _parse_expansion_moves(
        exp_moves) if exp_moves else None

    js = gs['json']
    if js:
        try:
            gs['json'] = json.loads(js) if js else None
        except ValueError as ex:
            raise ValueError('%s, json:\n %s' % (ex, js))

    return gs


def parse_xl_url(url):
    """
    Parses a :term:`xl-url`.

    :param str url:
        a string with the following format::

            <url_file>#<sheet>!<1st_edge>:<2nd_edge>:<expand><json>

        i.e.::

            file:///path/to/file.xls#sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}

    :return:
        the dictionary returned by :func:`_parse_xl_ref()`  augmented 
        with the following:

        - url_file: i.e. ``path/to/file.xls``

    :rtype: dict


    Examples::

        >>> url = 'file:///sample.xlsx#Sheet1!A1(UL):.^(DR):LU?:{"2": "ciao"}'
        >>> res = parse_xl_url(url)
        >>> sorted(res.items())
        [('exp_moves', [repeat('L'), repeat('U', 1)]),
         ('json', {'2': 'ciao'}),
         ('nd_edge', Edge(land=Cell(row='^', col='.'), mov='DR', mod=None)),
         ('sheet', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='UL', mod=None)),
         ('url_file', 'file:///sample.xlsx')]
    """

    try:
        url_file, frag = urldefrag(url)
        if not frag:
            frag = url_file
            url_file = None
        res = _parse_xl_ref(frag)
        res['url_file'] = url_file

        return res

    except Exception as ex:
        msg = "Invalid xl-ref(%s) due to: %s"
        log.debug(msg, url, ex, exc_info=1)
        raise ValueError(msg % (url, ex))


def _margin_coords_from_states_matrix(states_matrix):
    """
    Returns top-left/bottom-down margins of full cells from a :term:`state` matrix.

    May be used by :meth:`ABCSheet.get_margin_coords()` if a backend
    does not report the sheet-margins internally.

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
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

    Negatives (from bottom) are preserved::

        >>> _row2num('-1')
        -1

    Fails ugly::

        >>> _row2num('.')
        Traceback (most recent call last):
        ValueError: invalid literal for int() with base 10: '.'
    """
    rcoord = int(coord)
    if rcoord == 0:
        msg = 'Uncooked-coord cannot be zero!'
        raise ValueError(msg.format(coord))
    if rcoord > 0:
        rcoord -= 1

    return rcoord


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
        >>> _col2num('10')
        9
        >>> _col2num(9)
        8

    Negatives (from left-end) are preserved::

        >>> _col2num('AaZ')
        727

    Fails ugly::

        >>> _col2num('%$')
        Traceback (most recent call last):
        ValueError: substring not found

        >>> _col2num([])
        Traceback (most recent call last):
        TypeError: int() argument must be a string, a bytes-like object or 
                    a number, not 'list'

    """
    try:
        rcoord = int(coord)
    except ValueError:
        rcoord = 0
        for c in coord:
            rcoord = rcoord * 26 + ascii_uppercase.rindex(c.upper()) + 1

        rcoord -= 1
    else:
        if rcoord == 0:
            msg = 'Uncooked-coord cannot be zero!'
            raise ValueError(msg.format(coord))
        elif rcoord > 0:
            rcoord -= 1

    return rcoord


def _resolve_coord(cname, cfunc, coord, up_coord, dn_coord, base_coord=None):
    """
    Translates special coords or converts Excel string 1-based rows/cols to zero-based, reporting invalids.

    :param str cname:  
            the coord-name, one of 'row', 'column'
    :param function cfunc:  
            the function to convert coord ``str --> int``
    :param int, str coord:  
            the "A1" coord to translate
    :param int up_coord:
            the resolved *top* or *left* margin zero-based coordinate
    :param int dn_coord:
            the resolved *bottom* or *right* margin zero-based coordinate 
    :param int, None base_coord:  
            the resolved basis for dependent coord, if any

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
        >>> _resolve_coord(cname, _row2num, '-3', 0, 10)
        8

    But notice when base-cell missing::

        >>> _resolve_coord(cname, _row2num, '.', 0, 10, base_coord=None)
        Traceback (most recent call last):
        ValueError: invalid row('.') due to: '.'

    Other ROW error-checks::

        >>> _resolve_coord(cname, _row2num, '0', 0, 10)
        Traceback (most recent call last):
        ValueError: invalid row('0') due to: Uncooked-coord cannot be zero!

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
        >>> _resolve_coord(cname, _col2num, '-4', 0, 10)
        7

    And COLUMN error-checks::

        >>> _resolve_coord(cname, _col2num, None, 0, 10)
        Traceback (most recent call last):
        ValueError: invalid column(None) due to: int() argument must be a string, 
                    a bytes-like object or a number, not 'NoneType'

        >>> _resolve_coord(cname, _col2num, 0, 0, 10)
        Traceback (most recent call last):
        ValueError: invalid column(0) due to: Uncooked-coord cannot be zero!

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

        # Resolve negatives as from the end.
        if rcoord < 0:
            rcoord = dn_coord + rcoord + 1

        return rcoord
    except Exception as ex:
        msg = 'invalid {}({!r}) due to: {}'
        fututils.raise_from(ValueError(msg.format(cname, coord, ex)), ex)


def _resolve_cell(cell, up_coords, dn_coords, base_cords=None):
    """
    Translates any special coords to absolute ones.

    To get the margin_coords, use one of:

    * :meth:`ABCSheet.get_margin_coords()`
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

        >>> _resolve_cell(Cell('-1', '-2'), up, dn)
        Coords(row=10, col=5)

        >>> _resolve_cell(Cell('A', 'B'), up, dn)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='A', col='B')) due to:
                invalid row('A') due to: invalid literal for int() with base 10: 'A'

    But notice when base-cell missing::

        >>> _resolve_cell(Cell('1', '.'), up, dn)
        Traceback (most recent call last):
        ValueError: invalid cell(Cell(row='1', col='.')) due to: invalid col('.') due to: '.'

    """
    assert not CHECK_CELLTYPE or isinstance(cell, Cell), cell
    assert not CHECK_CELLTYPE or isinstance(up_coords, Coords), up_coords
    assert not CHECK_CELLTYPE or isinstance(dn_coords, Coords), dn_coords
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
        msg = "invalid cell(%s) due to: %s"
        fututils.raise_from(ValueError(msg % (cell, ex)), ex)


_mov_vector_slices = {
    # VECTO_SLICE        REVERSE  COORD_INDEX
    'L': (1, -1, lambda r, c: (r, slice(None, c + 1))),
    'U': (0, -1, lambda r, c: (slice(None, r + 1), c)),
    'R': (1, 1, lambda r, c: (r, slice(c, None))),
    'D': (0, 1, lambda r, c: (slice(r, None), c)),
}


def _extract_states_vector(states_matrix, dn_coords, land, mov,
                           mov_slices=_mov_vector_slices):
    """Extract a slice from the states-matrix by starting from `land` and following `mov`."""
    coord_indx, is_reverse, slice_func = mov_slices[mov]
    vect_slice = slice_func(*land)
    states_vect = states_matrix[vect_slice]
    states_vect = states_vect[::is_reverse]

    return states_vect, coord_indx, is_reverse


def _target_opposite(states_matrix, dn_coords, land, moves,
                     primitive_dir_vectors=_primitive_dir_vectors):
    """
    Follow moves from `land` and stop on the 1st full-cell.

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param Coords dn_coords:
            the bottom-right for the top-left of full-cells
    :param Coords land:
            the landing-cell
    :param str moves: MUST not be empty
    :return: the identified target-cell's coordinates
    :rtype: Coords


    Examples::

        >>> states_matrix = np.array([
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 1],
        ...     [0, 0, 1, 0, 0, 1],
        ...     [0, 0, 1, 1, 1, 1]
        ... ])
        >>> args = (states_matrix, Coords(4, 5))

        >>> _target_opposite(*(args + (Coords(0, 0), 'DR')))
        Coords(row=3, col=2)

        >>> _target_opposite(*(args + (Coords(0, 0), 'RD')))
        Coords(row=2, col=3)

    It fails if a non-empty target-cell cannot be found, or
    it ends-up beyond bounds::

        >>> _target_opposite(*(args + (Coords(0, 0), 'D')))
        Traceback (most recent call last):
        ValueError: No opposite-target for landing-Coords(row=0, col=0) with movement(D)!

        >>> _target_opposite(*(args + (Coords(0, 0), 'UR')))
        Traceback (most recent call last):
        ValueError: No opposite-target for landing-Coords(row=0, col=0) with movement(UR)!


    But notice that the landing-cell maybe outside of bounds::

        >>> _target_opposite(*(args + (Coords(3, 10), 'L')))
        Coords(row=3, col=5)

    """
    assert not CHECK_CELLTYPE or isinstance(dn_coords, Coords), dn_coords
    assert not CHECK_CELLTYPE or isinstance(land, Coords), land

    up_coords = np.array([0, 0])
    target = np.array(land)

    if land[0] > dn_coords[0] and 'U' in moves:
        target[0] = dn_coords[0]
    if land[1] > dn_coords[1] and 'L' in moves:
        target[1] = dn_coords[1]

#     if states_matrix[target].all():
#         return Coords(*target)

    imoves = iter(moves)
    mov1 = next(imoves)
    mov2 = next(imoves, None)
    dv2 = mov2 and primitive_dir_vectors[mov2]

    # Limit negative coords, since they are valid indices.
    while (up_coords <= target).all():
        try:
            states_vect, coord_indx, is_reverse = _extract_states_vector(
                states_matrix, dn_coords, target, mov1)
        except IndexError:
            break
        else:
            if states_vect.any():
                indices = states_vect.nonzero()[0]
                target[coord_indx] += is_reverse * indices.min()

                return Coords(*target)

            if not dv2:
                break

            target += dv2

    msg = 'No opposite-target for landing-{} with movement({})!'
    raise ValueError(msg.format(land, moves))


def _target_same_vector(states_matrix, dn_coords, land, mov):
    """
    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param Coords dn_coords:
            the bottom-right for the top-left of full-cells
    :param Coords land:
            The landing-cell, which MUST be full!
    """
    states_vect, coord_indx, is_reverse = _extract_states_vector(
        states_matrix, dn_coords, land, mov)
    if states_vect.all():
        same_len = len(states_vect) - 1
    else:
        indices = np.diff(states_vect).nonzero()[0]
        same_len = indices.min()
    target_coord = land[coord_indx] + is_reverse * same_len

    return target_coord, coord_indx


def _target_same(states_matrix, dn_coords, land, moves):
    """
    Scan term:`exterior` row and column on specified `moves` and stop on the last full-cell.

    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param Coords dn_coords:
            the bottom-right for the top-left of full-cells
    :param Coords land:
            the landing-cell which MUST be within bounds
    :param moves: which MUST not be empty
    :return: the identified target-cell's coordinates
    :rtype: Coords


    Examples::

        >>> states_matrix = np.array([
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 1],
        ...     [0, 0, 1, 0, 0, 1],
        ...     [0, 0, 1, 1, 1, 1]
        ... ])
        >>> args = (states_matrix, Coords(4, 5))

        >>> _target_same(*(args + (Coords(4, 5), 'U')))
        Coords(row=2, col=5)

        >>> _target_same(*(args + (Coords(4, 5), 'L')))
        Coords(row=4, col=2)

        >>> _target_same(*(args + (Coords(4, 5), 'UL', )))
        Coords(row=2, col=2)


    It fails if landing is empty or beyond bounds::

        >>> _target_same(*(args + (Coords(2, 2), 'DR')))
        Traceback (most recent call last):
        ValueError: No same-target for landing-Coords(row=2, col=2) with movement(DR)!

        >>> _target_same(*(args + (Coords(10, 3), 'U')))
        Traceback (most recent call last):
        ValueError: No same-target for landing-Coords(row=10, col=3) with movement(U)!

    """
    assert not CHECK_CELLTYPE or isinstance(dn_coords, Coords), dn_coords
    assert not CHECK_CELLTYPE or isinstance(land, Coords), land

    target = np.asarray(land)
    if (target <= dn_coords).all() and states_matrix[land]:
        for mov in moves:
            coord, indx = _target_same_vector(states_matrix, dn_coords,
                                              np.asarray(land), mov)
            target[indx] = coord

        return Coords(*target)

    msg = 'No same-target for landing-{} with movement({})!'
    raise ValueError(msg.format(land, moves))


def _sort_rect(r1, r2):
    """
    Sorts rect-vertices in a 2D-array (with vertices in rows).

    Example::

        >>> _sort_rect((5, 3), (4, 6))
        array([[4, 3],
               [5, 6]])
    """
    rect = np.array([r1, r2], dtype=int)
    rect.sort(0)
    return rect


def _expand_rect(states_matrix, r1, r2, exp_moves):
    """
    Applies the :term:`expansion-moves` based on the `states_matrix`.

    :param state:
    :param Coords r1:
              any vertice of the rect to expand
    :param Coords r2:
              any vertice of the rect to expand
    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param exp_moves:
    :return: a sorted rect top-left/bottom-right


    Examples::

        >>> states_matrix = np.array([
        ...     #0  1  2  3  4  5
        ...     [0, 0, 0, 0, 0, 0], #0
        ...     [0, 0, 1, 1, 1, 0], #1
        ...     [0, 1, 0, 0, 1, 0], #2
        ...     [0, 1, 1, 1, 1, 0], #3
        ...     [0, 0, 0, 0, 0, 1], #4
        ... ], dtype=bool)

        >>> r1, r2 = (Coords(2, 1), Coords(2, 1))
        >>> exp_moves = [_repeat_moves('U')]
        >>> _expand_rect(states_matrix, r1, r2, exp_moves)
        (Coords(row=2, col=1), Coords(row=2, col=1))

        >>> r1, r2 = (Coords(3, 1), Coords(2, 1))
        >>> exp_moves = [_repeat_moves('R')]
        >>> _expand_rect(states_matrix, r1, r2, exp_moves)
        (Coords(row=2, col=1), Coords(row=3, col=4))

        >>> r1, r2 = (Coords(2, 1), Coords(6, 1))
        >>> exp_moves = [_repeat_moves('R')]
        >>> _expand_rect(states_matrix, r1, r2, exp_moves)
        (Coords(row=2, col=1), Coords(row=6, col=5))

        >>> r1, r2 = (Coords(2, 3), Coords(2, 3))
        >>> exp_moves = [_repeat_moves('LURD')]
        >>> _expand_rect(states_matrix, r1, r2, exp_moves)
        (Coords(row=1, col=1), Coords(row=3, col=4))

    """
    assert not CHECK_CELLTYPE or isinstance(r1, Coords), r1
    assert not CHECK_CELLTYPE or isinstance(r2, Coords), r2

    nd_offsets = np.array([0, 1, 0, 1])
    coord_offsets = {
        'L': np.array([0,  0, -1, 0]),
        'R': np.array([0,  0,  0, 1]),
        'U': np.array([-1, 0,  0, 0]),
        'D': np.array([0,  1,  0, 0]),
    }
    coord_indices = {
        'L': [0, 1, 2, 2],
        'R': [0, 1, 3, 3],
        'U': [0, 0, 2, 3],
        'D': [1, 1, 2, 3],
    }

    # Sort rect's vertices top-left/bottom-right.
    #
    rect = _sort_rect(r1, r2)
    # ``[r1, r2, c1, c2]`` to use slices, below
    rect = rect.T.flatten()
    for dirs_repeated in exp_moves:
        for dirs in dirs_repeated:
            orig_rect = rect
            for d in dirs:
                exp_rect = rect + coord_offsets[d]
                exp_vect_i = exp_rect[coord_indices[d]] + nd_offsets
                exp_vect_v = states_matrix[slice(*exp_vect_i[:2]),
                                           slice(*exp_vect_i[2:])]
                if exp_vect_v.any():
                    rect = exp_rect
            if (rect == orig_rect).all():
                break

    return Coords(*rect[[0, 2]]), Coords(*rect[[1, 3]])


def resolve_capture_rect(states_matrix, up_coords, dn_coords,
                         st_edge, nd_edge=None, exp_moves=None):
    """
    Performs :term:`targeting`, :term:`capturing` and :term:`expansions` based on the :term:`states-matrix`.

    To get the margin_coords, use one of:

    * :meth:`ABCSheet.get_margin_coords()`
    * :func:`_margin_coords_from_states_matrix()`

    Its results can be fed into :func:`read_capture_values()`.

    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param Coords up_coords:
            the top-left coords with full-cells
    :param Coords dn_coords:
            the bottom-right coords with full-cells
    :param Edge st_edge: "uncooked" as matched by regex
    :param Edge nd_edge: "uncooked" as matched by regex
    :param list or none exp_moves:
            the result of :func:`_parse_expansion_moves()`

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

    Using dependenent coordinates for the 2nd edge::

        >>> st_edge = Edge(Cell('_', '_'), None)
        >>> nd_edge = Edge(Cell('.', '.'), 'UL')
        >>> rect = resolve_capture_rect(states_matrix, up, dn, st_edge, nd_edge)
        >>> rect
        (Coords(row=2, col=2), Coords(row=4, col=5))

    Using sheet's margins::

        >>> st_edge = Edge(Cell('^', '_'), None)
        >>> nd_edge = Edge(Cell('_', '^'), None)
        >>> rect == resolve_capture_rect(states_matrix, up, dn, st_edge, nd_edge)
        True

    Walking backwards::

        >>> st_edge = Edge(Cell('^', '_'), 'L')          # Landing is full, so 'L' ignored.
        >>> nd_edge = Edge(Cell('_', '_'), 'L', '+')    # '+' or would also stop.
        >>> rect == resolve_capture_rect(states_matrix, up, dn, st_edge, nd_edge)
        True

    """
    assert not CHECK_CELLTYPE or isinstance(up_coords, Coords), up_coords
    assert not CHECK_CELLTYPE or isinstance(dn_coords, Coords), dn_coords

    st = _resolve_cell(st_edge.land, up_coords, dn_coords)
    try:
        st_state = states_matrix[st]
    except IndexError:
        st_state = False

    if st_edge.mov is not None:
        if st_state:
            if st_edge.mod == '+':
                st = _target_same(states_matrix, dn_coords, st, st_edge.mov)
        else:
            st = _target_opposite(states_matrix, dn_coords, st, st_edge.mov)

    if nd_edge is None:
        nd = None
    else:
        nd = _resolve_cell(nd_edge.land, up_coords, dn_coords, st)

        if nd_edge.mov is not None:
            try:
                nd_state = states_matrix[nd]
            except IndexError:
                nd_state = False

            mov = nd_edge.mov
            if nd_state:
                if (nd_edge.mod == '+' or
                        nd_edge.land == Cell('.', '.') and nd_edge.mod != '-'):
                    nd = _target_same(states_matrix, dn_coords, nd, mov)
            else:
                nd = _target_opposite(states_matrix, dn_coords, nd, mov)

    if exp_moves:
        st, nd = _expand_rect(states_matrix, st, nd or st, exp_moves)
    else:
        if nd is not None:
            rect = _sort_rect(st, nd)
            st, nd = tuple(Coords(*c) for c in rect)

    return st, nd


def _classify_rect_shape(st, nd):
    """
    Identifies rect from its edge-coordinates (row, col, 2d-table)..

    :param Coords st:
            the top-left edge of capture-rect, inclusive
    :param Coords or None nd:
            the bottom-right edge of capture-rect, inclusive
    :return: 
            in int based on the input like that:

            - 0: only `st` given 
            - 1: `st` and `nd` point the same cell 
            - 2: row 
            - 3: col 
            - 4: 2d-table 

    Examples::

        >>> _classify_rect_shape((1,1), None)
        0
        >>> _classify_rect_shape((2,2), (2,2))
        1
        >>> _classify_rect_shape((2,2), (2,20))
        2
        >>> _classify_rect_shape((2,2), (20,2))
        3
        >>> _classify_rect_shape((2,2), (20,20))
        4
    """
    if nd is None:
        return 0
    rows = nd[0] - st[0]
    cols = nd[1] - st[1]
    return 1 + bool(cols) + 2 * bool(rows)


def _decide_ndim_by_rect_shape(shape_idx, ndims_list):
    return ndims_list[shape_idx]


def _updim(values, new_ndim):
    """
    Append trivial dimensions to the left.

    :param values:      The scalar ot 2D-results of :meth:`Sheet.read_rect()`
    :param int new_dim: The new dimension the result should have
    """
    new_shape = (1,) * (new_ndim - values.ndim) + values.shape
    return values.reshape(new_shape)


def _downdim(values, new_ndim):
    """
    Squeeze it, and then flatten it, before inflating it.

    :param values:       The scalar ot 2D-results of :meth:`Sheet.read_rect()`
    :param int new_dim: The new dimension the result should have
    """
    trivial_indxs = [i for i, d in enumerate(values.shape) if d == 1]
    offset = values.ndim - new_ndim
    trivial_ndims = len(trivial_indxs)
    if offset > trivial_ndims:
        values = values.flatten()
    elif offset == trivial_ndims:
        values = values.squeeze()
    else:
        for _, indx in zip(range(offset), trivial_indxs):
            values = values.squeeze(indx)

    return values


def _redim(values, new_ndim):
    """
    Reshapes the :term:`capture-rect` values of :func:`read_capture_rect()`.

    :param values:      The scalar ot 2D-results of :meth:`Sheet.read_rect()`
    :type values: (nested) list, *
    :param new_ndim: 
    :type int, (int, bool) or None new_ndim: 

    :return: reshaped values
    :rtype: list of lists, list, *


    Examples::

        >>> _redim([1, 2], 2)
        [[1, 2]]

        >>> _redim([[1, 2]], 1)
        [1, 2]

        >>> _redim([], 2)
        [[]]

        >>> _redim([[3.14]], 0)
        3.14

        >>> _redim([[11, 22]], 0)
        [11, 22]

        >>> arr = [[[11], [22]]]
        >>> arr == _redim(arr, None)
        True

        >>> _redim([[11, 22]], 0)
        [11, 22]
    """
    if new_ndim is None:
        return values

    values = np.asarray(values)
    try:
        new_ndim, transpose = new_ndim
        if transpose:
            values = values.T
    except:
        pass
    if new_ndim is not None:
        if values.ndim < new_ndim:
            values = _updim(values, new_ndim)
        elif values.ndim > new_ndim:
            values = _downdim(values, new_ndim)

    return values.tolist()

Lash = namedtuple('Lash', ('xl_ref', 'st', 'nd', 'values', OPTS))
"""
The `xl-ref` fields, the resolved :term:`capture-rect` and the read values along with `opts` passed between filters.

:param ChainMap opts:
        stacked dictionaries with options from previoues invocations 
"""

CallSpec = namedtuple('CallSpec', ('func', 'args', 'kwds'))
"""The :term:`call-specifier` for holding the parsed json-values."""


def _parse_call_spec(call_spec_values):
    """
    Parse :term:`call-specifier` from json-parsed values.

    :param call_spec_values:
        This is a *non-null* structure specifying some function call 
        in the `filter` part, which it can be either:

        - string: ``"func_name"`` 
        - list:   ``["func_name", ["arg1", "arg2"], {"k1": "v1"}]``
          where the last 2 parts are optional and can be given in any order;
        - object: ``{"func": "func_name", "args": ["arg1"], "kwds": {"k":"v"}}`` 
          where the `args` and `kwds` are optional.

    :return: 
        the 3-tuple ``func, args=(), kwds={}`` with the defaults as shown 
        when missing. 
    """
    def boolarr(l):
        return np.fromiter(l, dtype=bool)

    def parse_list(func, item1=None, item2=None):
        items = (item1, item2)
        isargs = boolarr(isinstance(a, list) for a in items)
        iskwds = boolarr(isinstance(a, dict) for a in items)
        isnull = boolarr(a is None for a in items)

        if isargs.all() or iskwds.all() or not (
                isargs ^ iskwds ^ isnull).all():
            msg = "Cannot decide `args`/`kwds` for call_spec(%s)!"
            raise ValueError(msg.format(call_spec_values))
        args, kwds = None, None
        if isargs.any():
            args = items[isargs.nonzero()[0][0]]
        if iskwds.any():
            kwds = items[iskwds.nonzero()[0][0]]
        return func, args, kwds

    def parse_object(func, args=None, kwds=None):
        return func, args, kwds

    opts = None
    try:
        if isinstance(call_spec_values, basestring):
            func, args, kwds = call_spec_values, None, None
        elif isinstance(call_spec_values, list):
            func, args, kwds = parse_list(*call_spec_values)
        elif isinstance(call_spec_values, dict):
            # Parse a lone OPTS.
            #
            opts = call_spec_values.pop(OPTS, None)
            if not call_spec_values:
                return None, opts
            func, args, kwds = parse_object(**call_spec_values)
        else:
            msg = "One of str, list or dict expected for call-spec(%s)!"
            raise ValueError(msg.format(call_spec_values))
    except ValueError:
        raise
    except Exception as ex:
        msg = "Cannot parse call-spec({}) due to: {}"
        raise ValueError(msg.format(call_spec_values, ex))

    if not isinstance(func, basestring):
        msg = "Expected a `string` for func({}) for call-spec({})!"
        raise ValueError(msg.format(func, call_spec_values))
    if args is None:
        args = []
    elif not isinstance(args, list):
        msg = "Expected a `list` for args({}) for call-spec({})!"
        raise ValueError(msg.format(args, call_spec_values))
    if kwds is None:
        kwds = {}
    elif not isinstance(kwds, dict):
        msg = "Expected a `dict` for kwds({}) for call-spec({})!"
        raise ValueError(msg.format(kwds, call_spec_values))

    # Extract any OPTS kewd from func-kwds,
    #     and merge it with base opts.
    if opts:
        opts.update(kwds.pop(OPTS, {}))
    else:
        opts = kwds.pop(OPTS, None)

    return CallSpec(func, args, kwds), opts


def _build_call_help(name, func, desc):
    sig = inspect.formatargspec(*inspect.getfullargspec(func))
    desc = textwrap.indent(textwrap.dedent(desc), '    ')
    return '\n\nFilter: %s%s:\n%s' % (name, sig, desc)


def _make_call(lash, func_name, args, kwds):
    opts = lash.opts
    lax = opts['lax']
    verbose = opts['verbose']
    available_funcs = opts['available_funcs']
    func_desc = ''
    try:
        func_rec = available_funcs[func_name]
        func, func_desc = func_rec
        lash = func(lash, *args, **kwds)
        assert isinstance(lash, Lash), (func_name, lash)
    except Exception as ex:
        if func_desc and verbose:
            func_desc = _build_call_help(func_name, func, func_desc)
        msg = "Error in call-specifier(%s, %s, %s): %s%s"
        if lax:
            log.warning(msg, func_name, args, kwds, ex, func_desc, exc_info=1)
        else:
            raise ValueError(msg % (func_name, args, kwds, ex, func_desc))
    return lash


###############
# FILTER-DEFS
###############

def _redim_filter(lash,
                  scalar=None, cell=None, row=None, col=None, table=None):
    """
    Reshape and/or transpose captured values, depending on rect's shape.

    Each dimension might be a single int or None, or a pair [dim, transpose].  
    """
    ndims_list = (scalar, cell, row, col, table)
    shape_idx = _classify_rect_shape(lash.st, lash.nd)
    new_ndim = _decide_ndim_by_rect_shape(shape_idx, ndims_list)
    values = lash.values
    if new_ndim is not None:
        lash = lash._replace(values=_redim(values, new_ndim))

    return lash


def _pipe_filter(lash, *pipe):
    """
    Apply all call-specifiers one after another on the captured values, and updates options.

    :param list pipe: the call-specifiers
    """
    orig_opts = lash.opts.copy()
    for call_spec_values in pipe:
        call_spec, opts = _parse_call_spec(call_spec_values)
        if opts:
            lash.maps.append(opts)
        lash = _make_call(lash, *call_spec)
    lash = lash._replace(opts=orig_opts)

    return lash

_default_filters = {
    'pipe': [
        _pipe_filter,
        _pipe_filter.__doc__,
    ],
    'redim': [
        _redim_filter,
        _redim_filter.__doc__,
    ],
    'numpy': [
        lambda lash, *
        args, **kwds: lash._replace(
            values=np.array(lash.values, *args, **kwds)),
        np.array.__doc__,
    ],
    'dict': [
        lambda lash, *
        args, **kwds: lash._replace(
            values=dict(lash.values, *args, **kwds)),
        dict.__doc__,
    ],
    'odict': [
        lambda lash, *
        args, **kwds: lash._replace(
            values=OrderedDict(lash.values, *args, **kwds)),
        OrderedDict.__doc__,
    ],
    'sorted': [
        lambda lash, *
        args, **kwds: lash._replace(
            values=sorted(lash.values, *args, **kwds)),
        sorted.__doc__,
    ],
}

try:
    import pandas as pd
    from pandas.io import parsers, excel as pdexcel

    def _df_filter(lash, *args, **kwds):
        values = lash.values
        header = kwds.get('header', 'infer')
        if header == 'infer':
            header = kwds['header'] = 0 if kwds.get('names') is None else None
        if header is not None:
            values[header] = pdexcel._trim_excel_header(values[header])
        parser = parsers.TextParser(values, **kwds)  # , convert_float=True,
        lash = lash._replace(values=parser.read())

        return lash

    _default_filters['df'] = [
        _df_filter,
        parsers.TextParser.__doc__,
    ]
    _default_filters['series'] = [
        lambda lash, *args, **kwds: pd.Series(OrderedDict(lash.values),
                                              *args, **kwds),
        ("Makes a pd.Series from a 2 columns.\n" + pd.Series.__doc__),
    ]


except ImportError:
    pass

_default_opts = {
    'lax': False,
    'verbose': False,
    'available_funcs': _default_filters,
    'read': {'on_demand': False, },
}


class SheetFactory(object):
    """"
    Creates and caches sheets from backends.

    :ivar dict _cached_sheets: 
            A cache of all _Spreadsheets accessed so far, 
            keyed by multiple keys generated by :meth:`_derive_sheet_keys`.
    :ivar ABCSheet _current_sheet:
            The last used sheet, used when unspecified by the xlref.
    """

    def __init__(self, current_sheet=None):
        self._current_sheet = current_sheet
        self._cached_sheets = {}

    def _cache_get(self, key):
        wb, sh = key
        if wb in self._cached_sheets:
            shs = self._cached_sheets[wb]
            return shs.get(sh, None)

    def _cache_put(self, key, sheet):
        wb, sh = key
        if wb in self._cached_sheets:
            sh_dict = self._cached_sheets[wb]
        else:
            sh_dict = self._cached_sheets[wb] = {}
        sh_dict[sh] = sheet

    def _build_sheet_key(self, wb, sh):
        assert wb is not None, (wb, sh)
        return (wb, sh)

    def _derive_sheet_keys(self, sheet,  new_wb_id=None, new_sheet_id=None):
        wb_id, sh_ids = sheet.get_sheet_ids()
        assert wb_id is not None, (wb_id, sh_ids)

        key_pairs = itt.product([wb_id, new_wb_id], sh_ids + [new_sheet_id])
        keys = list(set(self._build_sheet_key(*p)
                        for p in key_pairs
                        if p[0] is not None))
        assert keys, (sheet,  new_wb_id, new_sheet_id)

        return keys

    def _close_sheet(self, key):
        sheet = self._cache_get(key)
        if sheet:
            sheet._close()
            for sh_dict in self._cached_sheets.values():
                for sh_id, sh in list(sh_dict.items()):
                    if sh is sheet:
                        del sh_dict[sh_id]
            if self._current_sheet is sheet:
                self._current_sheet = None

    def close_all(self):
        for sh_dict in self._cached_sheets.values():
            for sh in sh_dict.values():
                sh._close_all()
        self._cached_sheets = {}
        self._current_sheet = None

    def _open_sheet(self, wb_id, sheet_id, opts):
        from .import _xlrd
        return _xlrd.open_sheet(wb_id, sheet_id, opts)

    def add_sheet(self, sheet, wb_id=None, sheet_id=None, no_current=False):
        assert sheet, (sheet, wb_id, sheet_id)

        keys = self._derive_sheet_keys(sheet, wb_id, sheet_id)
        for k in keys:
            old_sheet = self._cache_get(k)
            if old_sheet and old_sheet is not sheet:
                self._close_sheet(k)
            self._cache_put(k, sheet)
        if not no_current:
            self._current_sheet = sheet

    def fetch_sheet(self, wb_id, sheet_id, read_opts={}):
        csheet = self._current_sheet
        if wb_id is None:
            if not csheet:
                msg = "No current-sheet exists yet!. Specify a Workbook."
                raise ValueError(msg)

            if sheet_id is None:
                return csheet

            wb_id, c_sh_ids = csheet.get_sheet_ids()
            assert wb_id is not None, (csheet, c_sh_ids)

            key = self._build_sheet_key(wb_id, sheet_id)
            sheet = self._cache_get(key)

            if not sheet:
                sheet = csheet.sibling_sheet(sheet_id)
                assert sheet, (wb_id, sheet_id, read_opts)
                self.add_sheet(sheet, wb_id, sheet_id)
        else:
            key = self._build_sheet_key(wb_id, sheet_id)
            sheet = self._cache_get(key)
            if not sheet:
                sheet = self._open_sheet(wb_id, sheet_id, read_opts)
                assert sheet, (wb_id, sheet_id, read_opts)
                self.add_sheet(sheet, wb_id, sheet_id)

        return sheet

def read(xlref, sheets_fact=None,
         available_funcs=_default_filters, default_opts=_default_opts):
    """
    Parses xlref, resolves rect and fetches values from spreadsheet, and filters them,
    """
    opts = ChainMap()
    opts.maps.append(default_opts)

    fields = parse_xl_url(xlref)

    call_spec = None
    js = fields.get('json', None)
    if json:
        call_spec, user_opts = _parse_call_spec(js)
        if user_opts:
            opts.maps.append(user_opts)

    if not sheets_fact:
        sheets_fact = SheetFactory()
    wb, sh = fields['url_file'],fields['sheet']
    try:
        sheet = sheets_fact.fetch_sheet(wb, sh)
    except Exception as ex:
        msg = "Loading sheet(%s:%s) failed due to: %s"
        raise ValueError(msg % (wb, sh, ex))
        
    capture_args = (sheet.get_states_matrix(),) + sheet.get_margin_coords()
    fields_to_keep = ['st_edge', 'nd_edge', 'exp_moves']
    kwds = dtz.keyfilter(lambda k: k in fields_to_keep, fields)
    st, nd = resolve_capture_rect(*capture_args, **kwds)
    log.info("Resolved to [%s, %s] <-- %s", st, nd, xlref)

    values = sheet.read_rect(st, nd)

    lash = Lash(fields, st, nd, values, opts)

    if call_spec:
        lash = _make_call(lash, *call_spec)

    return lash


class ABCSheet(with_metaclass(ABCMeta, object)):
    """
    A delegating to backend factory and sheet-wrapper with utility methods.

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

    def _close(self):
        """ Override it to release resources for this sheet."""

    def _close_all(self):
        """ Override it to release resources this and all sibling sheets."""

    @abstractmethod
    def get_sheet_ids(self):
        """
        :return: a 2-tuple of its wb-name and a sheet-ids of this sheet i.e. name & indx
        :rtype: ([str or None, [str or int or None])
        """

    @abstractmethod
    def sibling_sheet(self, sheet_id):
        """Return a sibling sheet by the given index or name"""

    @abstractmethod
    def _read_states_matrix(self):
        """
        Read the :term:`states-matrix` of the wrapped sheet.

        :return:   A 2D-array with `False` wherever cell are blank or empty.
        :rtype:     ndarray
        """

    def get_states_matrix(self):
        """
        Read and cache the :term:`states-matrix` of the wrapped sheet.

        :return:   A 2D-array with `False` wherever cell are blank or empty.
        :rtype:     ndarray
        """
        if self._states_matrix is None:
            self._states_matrix = self._read_states_matrix()
        return self._states_matrix

    @abstractmethod
    def read_rect(self, st, nd):
        """
        Fecth the actual values  from the backend Excel-sheet.

        :param Coords st:
                the top-left edge, inclusive
        :param Coords, None nd:
                the bottom-right edge, inclusive(!); when `None`,
                must return a scalar value.
        :return: 
                a 1D or 2D-list with the values fenced by the rect,
                which might be empty if beyond limits.
        :rtype: list
        """

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

        """
        if not self._margin_coords:
            up, dn = self._read_margin_coords()
            if up is None or dn is None:
                sm = self.get_states_matrix()
                up1, dn1 = _margin_coords_from_states_matrix(sm)
                up = up or up1
                dn = dn or dn1
            self._margin_coords = up, dn

        return self._margin_coords
