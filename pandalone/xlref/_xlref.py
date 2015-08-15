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

from abc import abstractmethod, ABCMeta
from collections import namedtuple, Iterable
import json
import logging
import re
from string import ascii_uppercase

import six

import itertools as itt
import numpy as np
from six.moves.urllib.parse import urldefrag  # @UnresolvedImport


log = logging.getLogger(__name__)

try:
    from xlrd import colname as xl_colname
    # TODO: Try different backends providing `colname` function.
except ImportError:
    log.warning(
        'One of `xlrd`, `...` libraries is needed, failures might occure later!')


CHECK_CELLTYPE = False
"""When `True`, most coord-functions accept any 2-tuples."""

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
        (?P<st_col>[A-Z]+|_|\^)                          # col
        (?P<st_row>[123456789]\d*|_|\^)                  # row
        (?:\(
            (?P<st_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves
            (?P<st_mod>[+-])?                            # move modifiers
            \)
        )?
    )
    (?::                                                 # 2nd-edge [opt]
        (?P<nd_col>[A-Z]+|_|\^|\.)                       # col
        (?P<nd_row>[123456789]\d*|_|\^|\.)               # row
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
    (?::?
        (?P<json>\{.*\})?                                # any json object [opt]
    )\s*$""",
    re.IGNORECASE | re.X)

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


def parse_xl_ref(xl_ref):
    """
    Parses a :term:`xl-ref` and splits it in its "ingredients".

    :param str xl_ref:
        a string with the following format:
        <sheet>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):
        <exp_moves>{<json>}
        i.e.::

            sheet!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}

    :return:
        dictionary containing the following parameters::

        - sheet: str
        - st_edge: (Edge, None) the 1st-ref, uncooked, with raw cell
        - nd_edge: (Edge, None) the 2nd-ref, uncooked, with raw cell
        - exp_moves: (str) as found on the xl-ref
        - json: parsed

    :rtype: dict


    Examples::

        >>> res = parse_xl_ref('Sheet1!A1(DR+):Z20(UL):L1U2R1D1:{"json":"..."}')
        >>> sorted(res.items())
        [('exp_moves', [repeat('L', 1), repeat('U', 2), repeat('R', 1), repeat('D', 1)]),
         ('json', {'json': '...'}),
         ('nd_edge', Edge(land=Cell(row='20', col='Z'), mov='UL', mod=None)),
         ('sheet', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='DR', mod='+'))]
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
        gs['st_edge'] = _uncooked_Edge(p('st_row'), p('st_col'),
                                       p('st_mov'), p('st_mod'))
        gs['nd_edge'] = _uncooked_Edge(p('nd_row'), p('nd_col'),
                                       p('nd_mov'), p('nd_mod'))

        js = gs['json']
        gs['json'] = json.loads(js) if js else None

        exp_moves = gs['exp_moves']
        gs['exp_moves'] = _parse_expansion_moves(
            exp_moves) if exp_moves else None

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

            file:///path/to/file.xls#sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}
    :param XlUrl base_url:
    :param module backend: one of :mod:`_xlrd` or mod:`_xlwings`

    :return:
        dictionary containing the following parameters:

        - url_file: i.e. ``foo.xls``
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
        - json: i.e. ``{"dims: 1}``

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
        res = parse_xl_ref(frag)
        res['url_file'] = url_file

        return res

    except Exception as ex:
        raise ValueError("Invalid xl-url({}) due to: {}".format(url, ex))


def _margin_coords_from_states_matrix(states_matrix):
    """
    Returns top-left/bottom-down margins of full cells from a :term:`state` matrix.

    May be used by :meth:`_Spreadsheet.get_margin_coords()` if a backend
    does not report the sheet-margins internally.

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`_Spreadsheet.get_states_matrix()` to derrive it.
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

    * :meth:`_Spreadsheet.get_margin_coords()`
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
        six.raise_from(ValueError("invalid cell(%s) due to: %s" % (cell, ex)),
                       ex)


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
            Use :meth:`_Spreadsheet.get_states_matrix()` to derrive it.
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
            Use :meth:`_Spreadsheet.get_states_matrix()` to derrive it.
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
            Use :meth:`_Spreadsheet.get_states_matrix()` to derrive it.
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


def _expand_rect(states_matrix, r1, r2, exp_mov):
    """
    Applies the :term:`expansion-moves` based on the `states_matrix`.

    :param state:
    :param Coords r1:
              any vertice of the rect to expand
    :param Coords r2:
              any vertice of the rect to expand
    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`_Spreadsheet.get_states_matrix()` to derrive it.
    :param exp_mov:
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
        >>> exp_mov = [_repeat_moves('U')]
        >>> _expand_rect(states_matrix, r1, r2, exp_mov)
        (Coords(row=2, col=1), Coords(row=2, col=1))

        >>> r1, r2 = (Coords(3, 1), Coords(2, 1))
        >>> exp_mov = [_repeat_moves('R')]
        >>> _expand_rect(states_matrix, r1, r2, exp_mov)
        (Coords(row=2, col=1), Coords(row=3, col=4))

        >>> r1, r2 = (Coords(2, 1), Coords(6, 1))
        >>> exp_mov = [_repeat_moves('R')]
        >>> _expand_rect(states_matrix, r1, r2, exp_mov)
        (Coords(row=2, col=1), Coords(row=6, col=5))

        >>> r1, r2 = (Coords(2, 3), Coords(2, 3))
        >>> exp_mov = [_repeat_moves('LURD')]
        >>> _expand_rect(states_matrix, r1, r2, exp_mov)
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
    rect = rect.T.flatten()  # ``[r1, r2, c1, c2]`` to use slices, below
    for moves in exp_mov:
        for directions in moves:
            foundFull = False
            for d in directions:
                exp_rect = rect + coord_offsets[d]
                exp_vect_i = exp_rect[coord_indices[d]] + nd_offsets
                exp_vect_v = states_matrix[
                    slice(*exp_vect_i[:2]), slice(*exp_vect_i[2:])]
                if exp_vect_v.any():
                    rect = exp_rect
                    foundFull = True
            if not foundFull:
                break

    return Coords(*rect[[0, 2]]), Coords(*rect[[1, 3]])


def resolve_capture_rect(states_matrix, up_coords, dn_coords, st_edge,
                         nd_edge=None, exp_moves=None):
    """
    Performs :term:`targeting`, :term:`capturing` and :term:`expansions` based on the :term:`states-matrix`.

    To get the margin_coords, use one of:

    * :meth:`_Spreadsheet.get_margin_coords()`
    * :func:`_margin_coords_from_states_matrix()`

    Its results can be fed into :func:`read_capture_values()`.

    :param Coords states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`_Spreadsheet.get_states_matrix()` to derrive it.
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


def read_capture_rect(sheet, st, nd, dims=None):
    """
    Extracts and process :term:`capture-rect` values from excel-sheet by applying :term:`filters`.

    :param sheet:
            anything supporting the :func:`read_rect(states_matrix, rect)`
            such as the the :class:`_Spreadsheet` which can hide away
            the backend-module .
    :param Coords st:
            the the top-left edge of capture-rect, inclusive
    :param Coords or None nd:
            the the bottom-right edge of capture-rect, inclusive
    :param int dims:    
            the non-negative minimum num of dimensions for the results. 
            The num of dims is derived from the shape of rect-edges according 
            to the following matrix:

             ========  ====  =====  =====  =====
             `dims`:   None      0      1      2
             ========  ====  =====  =====  =====
             1 coord:     0      0      1      2
                cell:     1      0      1      2
                 row:     1  raise      1      2
                 col:     2  raise      1      2
               table:     2  raise  raise      2
             ========  ====  =====  =====  =====


    :return: 
            The rect values appropriately dimensioned ,or 
            a scalar-value if `nd` is `None`.

    Examples::

        import numpy as np
        from pandalone import xref

        ## A mockup sheet.
        #
        class ArraySheet(xref._Spreadsheet):
            def __init__(self, arr):
                self._arr = np.asarray(arr)

            def get_states_matrix(self):
                return ~np.equal(self._arr, None)

            def read_rect(self, st, nd):
                if nd is None:
                    return self._arr[st]
                rect = np.array([st, nd]) + [[0, 0], [1, 1]]
                return self._arr[slice(*rect[:, 0]), slice(*rect[:, 1])]



    """
    assert not CHECK_CELLTYPE or isinstance(st, Coords), st
    assert not CHECK_CELLTYPE or nd is None or isinstance(nd, Coords), nd
    assert dims is None or dims >= 0, dims

    res = sheet.read_rect(st, nd)
    res = np.asarray(res)

    if dims is None:
        if nd is None:
            dims = 0
        else:
            # Calculate dims implied by CaptureRect,
            #    preserving cells as 1D & cols as 2D.
            #
            rect = np.array([st, nd])
            coord_dims = (rect[0] - rect[1]).astype(bool)
            rect_dims = coord_dims.sum()
            col_dims = 2 * coord_dims[0]
            dims = max(1, rect_dims, col_dims)

    if res.ndim != dims:
        res = _redim_array(res, dims)

    return res


def _redim_array(arr, dims):
    """
    :param array arr:     what to redim
    :param int dims:      the new dimensions

    Examples::

        >>> _redim_array(np.array([[1, 2]]), 1)
        array([1, 2])

        >>> _redim_array(np.array([1, 2]), 2)
        array([[1, 2]])

        >>> _redim_array(np.array([]), 3)
        array([], shape=(1, 1, 0), dtype=float64)

        >>> _redim_array(np.array([[1, 2], [3, 4]]), 1)
        Traceback (most recent call last):
        ValueError: Cannot reduce dimensions of (2, 2) from 2-->1!

        >>> _redim_array(np.array([[3.14]]), 0)
        3.14

        >>> repr(_redim_array(np.array([[]]), 0))
        Traceback (most recent call last):
        ValueError: Cannot reduce dimensions of (1, 0) from 2-->0!

        >>> _redim_array(np.array([[1, 2]]), 0)
        Traceback (most recent call last):
        ValueError: Cannot reduce dimensions of (1, 2) from 2-->0!

    """
    def raise_cannot_reduce(a_shape, dims):
        msg = "Cannot reduce dimensions of {} from {}-->{}!"
        raise ValueError(msg.format(a_shape, len(a_shape), dims))
    a_shape = arr.shape
    if dims == 0:
        if arr.size == 1:
            return arr.item()
        else:
            raise_cannot_reduce(a_shape, dims)

    if dims < arr.ndim:
        arr = arr.squeeze()

    if dims == arr.ndim:
        return arr
    elif dims < arr.ndim:
        raise_cannot_reduce(a_shape, dims)
    else:
        # Append trivial dimensions o the left.
        new_shape = (1,) * (dims - arr.ndim) + arr.shape

        return arr.reshape(new_shape)


_default_filters = {
    None: {'fun': lambda x: x},  # TODO: Actually _redim_captured_values().
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}
try:
    import pandas as pd

    def _type_df_with_numeric_conversion(df, args, kws):
        df = pd.DataFrame(args, kws)
        return df.convert_objects(convert_numeric=True)
    _default_filters['df'] = {'fun': pd.DataFrame}
    _default_filters['df_num'] = {'fun': _type_df_with_numeric_conversion}
except ImportError:
    pass


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


class _Spreadsheet(object):
    """
    An abstract  delegating to backends excel-worksheets wrapper that is utilized by this module.

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
    __metaclass__ = ABCMeta

    _states_matrix = None
    _margin_coords = None

    @abstractmethod
    def _read_states_matrix(self):
        pass

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


        Examples::

            >>> sheet = _Spreadsheet()
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
                sm = self.get_states_matrix()
                up1, dn1 = _margin_coords_from_states_matrix(sm)
                up = up or up1
                dn = dn or dn1
            self._margin_coords = up, dn

        return self._margin_coords
