#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The algorithmic part of *xlasso*.

Prefer accessing the public members from the parent module.
"""

from __future__ import unicode_literals

from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict
from copy import deepcopy
import inspect
import logging
from string import ascii_uppercase
import textwrap

from future.backports import ChainMap
from future.builtins import str
from future.utils import iteritems, with_metaclass
from past.builtins import basestring
from toolz import dicttoolz as dtz

import itertools as itt
import numpy as np
from pandalone.utils import as_list

from pandalone.xlasso import _parse
from pandalone.xlasso._parse import Cell, Coords, _Edge_to_str


log = logging.getLogger(__name__)

try:
    from xlrd import colname as xl_colname
    # TODO: Try different backends providing `colname` function.
except ImportError:
    log.warning(
        'One of `xlrd`, `...` libraries is needed, will crash later!')


CHECK_CELLTYPE = False
"""When `True`, most coord-functions accept any 2-tuples."""

"""The key for specifying options within :term:`filters`."""

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
        col = xl_colname(col)
    return Cell(row=row, col=col)


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
        ValueError: Cannot resolve `relative-row` without `base-coord`!

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
        if isinstance(ex, KeyError) and ex.args == ('.',):
            msg = "Cannot resolve `relative-{}` without `base-coord`!"
            raise ValueError(msg.format(cname))
        msg = 'invalid {}({!r}) due to: {}'
        # fututils.raise_from(ValueError(msg.format(cname, coord, ex)), ex) see
        # GH 141
        raise ValueError(msg.format(cname, coord, ex))


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
        ValueError: invalid cell(Cell(row='1', col='.')) due to: 
        Cannot resolve `relative-col` without `base-coord`!

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
        # fututils.raise_from(ValueError(msg % (cell, ex)), ex) see GH 141
        raise ValueError(msg % (cell, ex))


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
                     edge_name='', primitive_dir_vectors=_primitive_dir_vectors):
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
        ValueError: No opposite-target found while moving(D) from landing-Coords(row=0, col=0)!

        >>> _target_opposite(*(args + (Coords(0, 0), 'UR')))
        Traceback (most recent call last):
        ValueError: No opposite-target found while moving(UR) from landing-Coords(row=0, col=0)!


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

    msg = 'No opposite-target found while moving({}) from {}landing-{}!'
    raise ValueError(msg.format(moves, edge_name, land))


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


def _target_same(states_matrix, dn_coords, land, moves, edge_name=''):
    """
    Scan term:`exterior` row and column on specified `moves` and stop on the last full-cell.

    :param np.ndarray states_matrix:
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
        ValueError: No same-target found while moving(DR) from landing-Coords(row=2, col=2)!

        >>> _target_same(*(args + (Coords(10, 3), 'U')))
        Traceback (most recent call last):
        ValueError: No same-target found while moving(U) from landing-Coords(row=10, col=3)!

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
    msg = 'No same-target found while moving({}) from {}landing-{}!'
    raise ValueError(msg.format(moves, edge_name, land))


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
    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param exp_moves:
            Just the parsed string, and not `None`.
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
        >>> _expand_rect(states_matrix, r1, r2, 'U')
        (Coords(row=2, col=1), Coords(row=2, col=1))

        >>> r1, r2 = (Coords(3, 1), Coords(2, 1))
        >>> _expand_rect(states_matrix, r1, r2, 'R')
        (Coords(row=2, col=1), Coords(row=3, col=4))

        >>> r1, r2 = (Coords(2, 1), Coords(6, 1))
        >>> _expand_rect(states_matrix, r1, r2, 'r')
        (Coords(row=2, col=1), Coords(row=6, col=5))

        >>> r1, r2 = (Coords(2, 3), Coords(2, 3))
        >>> _expand_rect(states_matrix, r1, r2, 'LURD')
        (Coords(row=1, col=1), Coords(row=3, col=4))

    """
    assert not CHECK_CELLTYPE or isinstance(r1, Coords), r1
    assert not CHECK_CELLTYPE or isinstance(r2, Coords), r2

    exp_moves = _parse.parse_expansion_moves(exp_moves)

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


def resolve_capture_rect(states_matrix, up_dn_margins,
                         st_edge, nd_edge=None, exp_moves=None,
                         base_coords=None):
    """
    Performs :term:`targeting`, :term:`capturing` and :term:`expansions` based on the :term:`states-matrix`.

    To get the margin_coords, use one of:

    * :meth:`ABCSheet.get_margin_coords()`
    * :func:`_margin_coords_from_states_matrix()`

    Its results can be fed into :func:`read_capture_values()`.

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :param (Coords, Coords) up_dn_margins:
            the top-left/bottom-right coords with full-cells
    :param Edge st_edge: "uncooked" as matched by regex
    :param Edge nd_edge: "uncooked" as matched by regex
    :param list or none exp_moves:
            Just the parsed string, and not `None`.
    :param Coords base_coords:
            The base for a :term:`dependent` :term;`1st` edge.

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
        >>> resolve_capture_rect(states_matrix, (up, dn), st_edge, nd_edge)
        (Coords(row=3, col=2), Coords(row=4, col=2))

    Using dependenent coordinates for the 2nd edge::

        >>> st_edge = Edge(Cell('_', '_'), None)
        >>> nd_edge = Edge(Cell('.', '.'), 'UL')
        >>> rect = resolve_capture_rect(states_matrix, (up, dn), st_edge, nd_edge)
        >>> rect
        (Coords(row=2, col=2), Coords(row=4, col=5))

    Using sheet's margins::

        >>> st_edge = Edge(Cell('^', '_'), None)
        >>> nd_edge = Edge(Cell('_', '^'), None)
        >>> rect == resolve_capture_rect(states_matrix, (up, dn), st_edge, nd_edge)
        True

    Walking backwards::

        >>> st_edge = Edge(Cell('^', '_'), 'L')          # Landing is full, so 'L' ignored.
        >>> nd_edge = Edge(Cell('_', '_'), 'L', '+')    # '+' or would also stop.
        >>> rect == resolve_capture_rect(states_matrix, (up, dn), st_edge, nd_edge)
        True

    """
    up_margin, dn_margin = up_dn_margins
    assert not CHECK_CELLTYPE or isinstance(up_margin, Coords), up_margin
    assert not CHECK_CELLTYPE or isinstance(dn_margin, Coords), dn_margin

    st = _resolve_cell(st_edge.land, up_margin, dn_margin, base_coords)
    try:
        st_state = states_matrix[st]
    except IndexError:
        st_state = False

    if st_edge.mov is not None:
        if st_state:
            if st_edge.mod == '+':
                st = _target_same(states_matrix, dn_margin, st, st_edge.mov,
                                  '1st-')
        else:
            st = _target_opposite(states_matrix, dn_margin, st, st_edge.mov,
                                  '1st-')

    if nd_edge is None:
        nd = None
    else:
        nd = _resolve_cell(nd_edge.land, up_margin, dn_margin, st)

        if nd_edge.mov is not None:
            try:
                nd_state = states_matrix[nd]
            except IndexError:
                nd_state = False

            mov = nd_edge.mov
            if nd_state:
                if (nd_edge.mod == '+' or
                        nd_edge.land == Cell('.', '.') and nd_edge.mod != '-'):
                    nd = _target_same(
                        states_matrix, dn_margin, nd, mov, '2nd-')
            else:
                nd = _target_opposite(
                    states_matrix, dn_margin, nd, mov, '2nd-')

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


class SheetsFactory(object):
    """
    A caching-store of :class:`ABCSheet` instances, serving them based on (workbook, sheet) IDs, optionally creating them from backends.

    :ivar dict _cached_sheets: 
            A cache of all _Spreadsheets accessed so far, 
            keyed by multiple keys generated by :meth:`_derive_sheet_keys`.
    :ivar ABCSheet _current_sheet:
            The last used sheet, used when unspecified by the xlref.

    - To avoid opening non-trivial workbooks, use the :meth:`add_sheet()` 
      to pre-populate this cache with them.

    - The last sheet added becomes the *current-sheet*, and will be 
      served when :term:`xl-ref` does not specify any workbook and sheet.

      .. Tip::
          For the simplest API usage, try this::

              >>> sf = SheetsFactory()
              >>> sf.add_sheet(some_sheet)              # doctest: +SKIP
              >>> lasso('A1:C3(U)', sf)                 # doctest: +SKIP

    - The *current-sheet* is served only when wokbook-id is `None`, that is,
      the id-pair ``('foo.xlsx', None)`` does not hit it, so those ids 
      are send to the cache as they are.

    - To add another backend, modify the opening-sheets logic (ie clipboard), 
      override :meth:`_open_sheet()`.

    - It is a resource-manager for contained sheets, wo it can be used wth 
      a `with` statement.

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

    def _derive_sheet_keys(self, sheet,  wb_ids=None, sh_ids=None):
        """
        Retuns the product of user-specified and sheet-internal keys.

        :param wb_ids:
                a single or a sequence of extra workbook-ids (ie: file, url)
        :param sh_ids:
                a single or sequence of extra sheet-ids (ie: name, index, None)
        """
        wb_id, sh_ids2 = sheet.get_sheet_ids()
        assert wb_id is not None, (wb_id, sh_ids2)
        wb_ids = [wb_id] + as_list(wb_ids)
        sh_ids = sh_ids2 + as_list(sh_ids)

        key_pairs = itt.product(wb_ids, sh_ids)
        keys = list(set(self._build_sheet_key(*p)
                        for p in key_pairs
                        if p[0] is not None))
        assert keys, (keys, sheet,  wb_ids, sh_ids)

        return keys

    def _close_sheet(self, key):
        sheet = self._cache_get(key)
        if sheet:
            sheet._close()
            for sh_dict in self._cached_sheets.values():
                for sh_id, sh in list(iteritems(sh_dict)):
                    if sh is sheet:
                        del sh_dict[sh_id]
            if self._current_sheet is sheet:
                self._current_sheet = None

    def close(self):
        """Closes all contained sheets and empties cache."""
        for sh_dict in self._cached_sheets.values():
            for sh in sh_dict.values():
                sh._close_all()
        self._cached_sheets = {}
        self._current_sheet = None

    def add_sheet(self, sheet, wb_ids=None, sh_ids=None,
                  no_current=False):
        """
        Updates cache and (optionally) `_current_sheet`.

        :param wb_ids:
                a single or sequence of extra workbook-ids (ie: file, url)
        :param sh_ids:
                a single or sequence of extra sheet-ids (ie: name, index, None)
        """
        assert sheet, (sheet, wb_ids, sh_ids)
        keys = self._derive_sheet_keys(sheet, wb_ids, sh_ids)
        for k in keys:
            old_sheet = self._cache_get(k)
            if old_sheet and old_sheet is not sheet:
                self._close_sheet(k)
            self._cache_put(k, sheet)
        if not no_current:
            self._current_sheet = sheet

    def fetch_sheet(self, wb_id, sheet_id, opts={}):
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
                sheet = csheet.open_sibling_sheet(sheet_id, opts)
                assert sheet, (wb_id, sheet_id, opts)
                self.add_sheet(sheet, wb_id, sheet_id)
        else:
            key = self._build_sheet_key(wb_id, sheet_id)
            sheet = self._cache_get(key)
            if not sheet:
                sheet = self._open_sheet(wb_id, sheet_id, opts)
                assert sheet, (wb_id, sheet_id, opts)
                self.add_sheet(sheet, wb_id, sheet_id)

        return sheet

    def _open_sheet(self, wb_id, sheet_id, opts):
        """OVERRIDE THIS to change backend."""
        from . import _xlrd
        return _xlrd.open_sheet(wb_id, sheet_id, opts)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()


def _build_call_help(name, func, desc):
    sig = func and inspect.formatargspec(*inspect.getfullargspec(func))
    desc = textwrap.indent(textwrap.dedent(desc), '    ')
    return '\n\nFilter: %s%s:\n%s' % (name, sig, desc)

Lasso = namedtuple('Lasso',
                   ('xl_ref', 'url_file', 'sh_name',
                    'st_edge', 'nd_edge', 'exp_moves',
                    'call_spec',
                    'sheet', 'st', 'nd', 'values', 'base_cell',
                    'opts'))
"""
All the fields used by the algorithm, populated stage-by-stage by :class:`Ranger`.

:param str xl_ref:
        The full url, populated on parsing.
:param str sh_name:
        Parsed sheet name (or index, but still as string), populated on parsing.
:param Edge st_edge:
        The 1st edge, populated on parsing.
:param Edge nd_edge:
        The 2nd edge, populated on parsing.
:param Coords st:
        The top-left targeted coords of the :term:`capture-rect`, 
        populated on :term:`capturing`.`
:param Coords nd:
        The bottom-right targeted coords of the :term:`capture-rect`, 
        populated on :term:`capturing`
:param ABCSheet sheet:
        The fetched from factory or ranger's current sheet, populated 
        after :term:`capturing` before reading.
:param values:
        The excel's table-values captured by the :term:`lasso`, 
        populated after reading updated while applying :term:`filters`. 
:param dict or ChainMap opts:
        - Before `parsing`, they are just any 'opts' dict found in the 
          :term:`filters`. 
        - After *parsing, a 2-map ChainMap with :attr:`Ranger.base_opts` and
          options extracted from *filters* on top.
"""


Lasso.__new__.__defaults__ = (None,) * len(Lasso._fields)
"""Make :class:`Lasso` construct with all missing fields as `None`."""


def _Lasso_to_edges_str(lasso):
    st = _Edge_to_str(lasso.st_edge) if lasso.st_edge else ''
    nd = _Edge_to_str(lasso.nd_edge) if lasso.nd_edge else ''
    s = st if st and not nd else '%s:%s' % (st, nd)
    exp = ':%s' % lasso.exp_moves.upper() if lasso.exp_moves else ''
    return s + exp


class Ranger(object):
    """
    The director-class that performs all stages required for "throwing the lasso" around rect-values.

    Use it when you need to have total control of the procedure and 
    configuration parameters, since no defaults are assumed.

    The :meth:`do_lasso()` does the job.

    :ivar SheetsFactory sheets_factory:
            Factory of sheets from where to parse rect-values; does not 
            close it in the end.
            Maybe `None`, but :meth:`do_lasso()` will scream unless invoked 
            with a `context_lasso` arg containing a concrete :class:`ABCSheet`.
    :ivar dict base_opts: 
            The :term:`opts` that are deep-copied and used as the defaults 
            for every :meth:`do_lasso()`, whether invoked directly or 
            recursively by :meth:`recursive_filter()`.
            If unspecified, no opts are used, but this attr is set to an 
            empty dict.
            See :func:`get_default_opts()`.
    :ivar dict or None available_filters: 
            No filters exist if unspecified. 
            See :func:`get_default_filters()`.
    :ivar Lasso intermediate_lasso:
            A ``('stage', Lasso)`` pair with the last :class:`Lasso` instance 
            produced during the last execution of the :meth:`do_lasso()`.
            Used for inspecting/debuging.
    :ivar _context_lasso_fields:
            The name of the fields taken from `context_lasso` arg of 
            :meth:`do_lasso()`, when the parsed ones are `None`.
            Needed for recursive invocations, see :meth:`recursive_filter`.
    """

    _context_lasso_fields = ['sheet', 'st', 'nd', 'base_cell']

    def __init__(self, sheets_factory,
                 base_opts=None, available_filters=None):
        self.sheets_factory = sheets_factory
        if base_opts is None:
            base_opts = {}
        self.base_opts = base_opts
        self.available_filters = available_filters
        self.intermediate_lasso = None

    def _relasso(self, lasso, stage, **kwds):
        """Replace lasso-values and updated :attr:`intermediate_lasso`."""
        lasso = lasso._replace(**kwds)
        self.intermediate_lasso = (stage, lasso)

        return lasso

    def _make_call(self, lasso, func_name, args, kwds):
        def parse_avail_func_rec(func, desc=None):
            if not desc:
                desc = func.__doc__
            return func, desc

        # Just to update intermediate_lasso.
        lasso = self._relasso(lasso, func_name)

        lax = lasso.opts.get('lax', False)
        verbose = lasso.opts.get('verbose', False)
        func, func_desc = '', ''
        try:
            func_rec = self.available_filters[func_name]
            func, func_desc = parse_avail_func_rec(**func_rec)
            lasso = func(self, lasso, *args, **kwds)
            assert isinstance(lasso, Lasso), (func_name, lasso)
        except Exception as ex:
            if verbose:
                func_desc = _build_call_help(func_name, func, func_desc)
            msg = "While invoking(%s, %s, %s): %s%s"
            help_msg = func_desc if verbose else ''
            if lax:
                log.warning(
                    msg, func_name, args, kwds, ex, help_msg, exc_info=1)
            else:
                raise ValueError(msg % (func_name, args, kwds, ex, help_msg))

        return lasso

    def pipe_filter(self, lasso, *pipe):
        """
        Apply all call-specifiers one after another on the captured values.

        :param list pipe: the call-specifiers
        """

        for call_spec_values in pipe:
            call_spec = _parse.parse_call_spec(call_spec_values)
            lasso = self._make_call(lasso, *call_spec)

        return lasso

    def recursive_filter(self, lasso, include=None, exclude=None, depth=-1):
        """
        Recursively expand any :term:`xl-ref` strings found by treating values as mappings (dicts, df, series) and/or nested lists.

        - The `include`/`exclude` filter args work only for dict-like objects
          with ``items()`` or ``iteritems()`` and indexing methods, 
          i.e. Mappings, series and dataframes.

          - If no filter arg specified, expands for all keys. 
          - If only `include` specified, rejects all keys not explicitly 
            contained in this filter arg.
          - If only `exclude` specified, expands all keys not explicitly 
            contained in this filter arg.
          - When both `include`/`exclude` exist, only those explicitely 
            included are accepted, unless also excluded.

        - Lower the :mod:`logging` level to see other than syntax-errors on
          recursion reported on :data:`log`.

        :param list or str include:
                Items to include in the recursive-search.
                See descritpion above.
        :param list or str exclude:
                Items to include in the recursive-search.
                See descritpion above.
        :param int or None depth:
                How deep to dive into nested structures for parsing xl-refs.
                If `< 0`, no limit. If 0, stops completely.
        """
        include = include and as_list(include)
        exclude = exclude and as_list(exclude)

        def verbose(msg):
            if lasso.opts.get('verbose', False):
                msg = '%s \n    @Lasso: %s' % (msg, lasso)
            return msg

        def is_included(key):
            ok = not include or key in include
            ok &= not exclude or key not in exclude
            return ok

        def new_base_cell(base_cell, cdepth, i):
            if base_cell:
                if cdepth == 0:
                    base_cell = base_cell._replace(row=i)
                elif cdepth == 1:
                    base_cell = base_cell._replace(col=i)
            return base_cell

        def dive_list(vals, base_cell, cdepth):
            if isinstance(vals, basestring):
                context = lasso._asdict()
                context['base_cell'] = base_cell
                try:
                    rlasso = self.do_lasso(vals, **context)
                    vals = rlasso and rlasso.values
                except SyntaxError as ex:
                    msg = "Skipped recursive-lassoing value(%s) due to: %s"
                    log.debug(msg, vals, ex)
                except Exception as ex:
                    msg = "Lassoing  %s stopped due to: \n  %s" % (vals, ex)
                    raise ValueError(verbose(msg))
            elif isinstance(vals, list):
                for i, v in enumerate(vals):
                    nbc = new_base_cell(base_cell, cdepth, i)
                    vals[i] = dive_indexed(v, nbc, cdepth + 1)

            return vals

        def dive_indexed(vals, base_cell, cdepth):
            if cdepth != depth:
                dived = False
                try:
                    items = iteritems(vals)
                except:
                    pass  # Just to avoid chained ex.
                else:
                    for k, v in items:
                        if is_included(k):
                            # No base_cell possible with Indexed.
                            vals[k] = dive_indexed(v, None, cdepth + 1)
                    dived = True
                if not dived:
                    vals = dive_list(vals, base_cell, cdepth)

            return vals

        values = dive_indexed(lasso.values, lasso.st, 0)

        return lasso._replace(values=values)

    def _make_init_Lasso(self, **context_kwds):
        """Creates the lasso to be used for each new :meth:`do_lasso()` invocation."""
        def is_context_field(field):
            return field in self._context_lasso_fields

        context_fields = dtz.keyfilter(is_context_field, context_kwds)
        context_fields['opts'] = ChainMap(deepcopy(self.base_opts))
        init_lasso = Lasso(**context_fields)

        return init_lasso

    def _parse_and_merge_with_context(self, xlref, init_lasso):
        """
        Merges xl-ref parsed-parsed_fields with `init_lasso`, reporting any errors.

        :param Lasso init_lasso: 
                Default values to be overridden by non-nulls.
                Note that ``init_lasso.opts`` must be a `ChainMap`,
                as returned by :math:`_make_init_Lasso()`. 

        :return: a Lasso with any non `None` parsed-fields updated
        """
        assert isinstance(init_lasso.opts, ChainMap), init_lasso

        try:
            parsed_fields = _parse.parse_xlref(xlref)
            parsed_opts = parsed_fields.pop('opts', None)
            if parsed_opts:
                init_lasso.opts.maps.insert(0, parsed_opts)
            filled_fields = dtz.valfilter(lambda v: v is not None,
                                          parsed_fields)
            init_lasso = init_lasso._replace(**filled_fields)
        except SyntaxError:
            raise
        except Exception as ex:
            msg = "Parsing xl-ref(%r) failed due to: %s"
            log.debug(msg, xlref, ex, exc_info=1)
            # raise fututils.raise_from(ValueError(msg % (xlref, ex)), ex) see GH
            # 141
            raise ValueError(msg % (xlref, ex))

        return init_lasso

    def _fetch_sheet_from_lasso(self, sheet, url_file, sh_name, opts):
        if sheet and url_file is None:
            if sh_name is not None:
                sheet = sheet.open_sibling_sheet(sh_name, opts)
                if sheet and self.sheets_factory:
                    self.sheets_factory.add_sheet(sheet, sh_ids=sh_name)
            return sheet

    def _open_sheet(self, lasso):
        try:
            sheet = self._fetch_sheet_from_lasso(lasso.sheet,
                                                 lasso.url_file, lasso.sh_name,
                                                 lasso.opts)
            if not sheet:
                if not self.sheets_factory:
                    msg = "The xl-ref(%r) specifies 'url-file` part but Ranger has no sheet-factory!"
                    raise Exception(msg % lasso.xl_ref)
                sheet = self.sheets_factory.fetch_sheet(
                    lasso.url_file, lasso.sh_name,
                    lasso.opts)  # Maybe context had a Sheet already.
        except Exception as ex:
            msg = "Loading sheet([%s]%s) failed due to: %s"
            raise ValueError(msg % (lasso.url_file, lasso.sh_name, ex))
        return sheet

    def _resolve_capture_rect(self, lasso, sheet):
        try:
            st, nd = resolve_capture_rect(sheet.get_states_matrix(), sheet.get_margin_coords(),
                                          lasso.st_edge, lasso.nd_edge, lasso.exp_moves)
        except Exception as ex:
            msg = "Resolving capture-rect(%r) failed due to: %s"
            raise ValueError(msg % (_Lasso_to_edges_str(lasso), ex))
        return st, nd

    def do_lasso(self, xlref, **context_kwds):
        """
        The director-method that does all the job of hrowing a :term:`lasso`
        around spreadsheet's rect-regions according to :term:`xl-ref`.

        :param str xlref:
            a string with the :term:`xl-ref` format::

                <url_file>#<sheet>!<1st_edge>:<2nd_edge>:<expand><js_filt>

            i.e.::

                file:///path/to/file.xls#sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}

        :param Lasso context_kwds: 
                Default :class:`Lasso` fields in case parsed ones are `None` 
                Only those in :attr:`_context_lasso_fields` are taken 
                into account.
                Utilized  by :meth:`recursive_filter()`.
        :return: 
                The final :class:`Lasso` with captured & filtered values.
        :rtype: Lasso
        """
        self.intermediate_lasso = None

        lasso = self._make_init_Lasso(**context_kwds)
        lasso = self._relasso(lasso, 'context')

        lasso = self._parse_and_merge_with_context(xlref, lasso)
        lasso = self._relasso(lasso, 'parse')

        sheet = self._open_sheet(lasso)
        lasso = self._relasso(lasso, 'open', sheet=sheet)

        st, nd = self._resolve_capture_rect(lasso, sheet)
        lasso = self._relasso(lasso, 'resolve_capture_rect',
                              st=st, nd=nd, base_cell=lasso.base_cell)

        values = sheet.read_rect(st, nd)
        lasso = self._relasso(lasso, 'read_rect', values=values)

        if lasso.call_spec:
            try:
                # relasso() internally
                lasso = self._make_call(lasso, *lasso.call_spec)
            except Exception as ex:
                msg = "Filtering xl-ref(%r) failed due to: %s"
                raise ValueError(msg % (lasso.xl_ref, ex))

        return lasso

###############
# FILTER-DEFS
###############


def xlwings_dims_call_spec():
    """A list :term:`call-spec` for :meth:`_redim_filter` :term:`filter` that imitates results of *xlwings* library."""
    return '["redim", [0, 1, 1, 1, 2]]'


def redim_filter(ranger, lasso,
                 scalar=None, cell=None, row=None, col=None, table=None):
    """
    Reshape and/or transpose captured values, depending on rect's shape.

    Each dimension might be a single int or None, or a pair [dim, transpose]. 
    """
    ndims_list = (scalar, cell, row, col, table)
    shape_idx = _classify_rect_shape(lasso.st, lasso.nd)
    new_ndim = _decide_ndim_by_rect_shape(shape_idx, ndims_list)
    values = lasso.values
    if new_ndim is not None:
        lasso = lasso._replace(values=_redim(values, new_ndim))

    return lasso


def get_default_filters(overrides=None):
    """
   The default available :term:`filters` used by :func:`lasso()` when constructing its internal :class:`Ranger`.

    :param dict or None overrides:
            Any items to update the default ones.

    :return: 
            a dict-of-dicts with 2 items: 

            - *func*: a function with args: ``(Ranger, Lasso, *args, **kwds)``
            - *desc*:  help-text replaced by ``func.__doc__`` if missing.

    :rtype: 
            dict
    """
    filters = {
        'pipe': {
            'func': Ranger.pipe_filter,
        },
        'recurse': {
            'func': Ranger.recursive_filter,
        },
        'redim': {
            'func': redim_filter,
        },
        'numpy': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=np.array(lasso.values, *args, **kwds)),
            'desc': np.array.__doc__,
        },
        'dict': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=dict(lasso.values, *args, **kwds)),
            'desc': dict.__doc__,
        },
        'odict': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=OrderedDict(lasso.values, *args, **kwds)),
            'desc': OrderedDict.__doc__,
        },
        'sorted': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=sorted(lasso.values, *args, **kwds)),
            'desc': sorted.__doc__,
        },
    }

    try:
        import pandas as pd
        from pandas.io import parsers, excel as pdexcel

        def _df_filter(ranger, lasso, *args, **kwds):
            values = lasso.values
            header = kwds.get('header', 'infer')
            if header == 'infer':
                header = kwds['header'] = 0 if kwds.get(
                    'names') is None else None
            if header is not None:
                values[header] = pdexcel._trim_excel_header(values[header])
            # , convert_float=True,
            parser = parsers.TextParser(values, **kwds)
            lasso = lasso._replace(values=parser.read())

            return lasso

        filters.update({
            'df': {
                'func': _df_filter,
                'desc': parsers.TextParser.__doc__,
            },
            'series': {
                'func': lambda ranger, lasso, *args, **kwds: pd.Series(OrderedDict(lasso.values),
                                                                       *args, **kwds),
                'desc': ("Converts a 2-columns list-of-lists into pd.Series.\n" +
                         pd.Series.__doc__),
            }
        })
    except ImportError as ex:
        msg = "The 'df' and 'series' filters were notinstalled, due to: %s"
        log.info(msg, ex)

    if overrides:
        filters.update(overrides)

    return filters


def get_default_opts(overrides=None):
    """
    Default :term:`opts` used by :func:`lasso()` when constructing its internal :class:`Ranger`.

    :param dict or None overrides:
            Any items to update the default ones.
    """
    opts = {
        'lax': False,
        'verbose': False,
        'read': {'on_demand': True, },
    }

    if overrides:
        opts.update(overrides)

    return opts


def make_default_Ranger(sheets_factory=None,
                        base_opts=None,
                        available_filters=None):
    """
    Makes a defaulted :class:`Ranger`.

    :param sheets_factory:
            Factory of sheets from where to parse rect-values; if unspecified, 
            a new :class:`SheetsFactory` is created.
            Remember to invoke its :meth:`SheetsFactory.close()` to clear
            resources from any opened sheets. 
    :param dict or None base_opts: 
            Default opts to affect the lassoing, to be merged with defaults; 
            uses :func:`get_default_opts()`.

            Read the code to be sure what are the available choices :-(. 
    :param dict or None available_filters: 
            The :term:`filters` available to xlrefs, to be merged with defaults;.
            Uses :func:`get_default_filters()` if unspecified.

    """
    return Ranger(sheets_factory or SheetsFactory(),
                  base_opts or get_default_opts(),
                  available_filters or get_default_filters())


def lasso(xlref,
          sheets_factory=None,
          base_opts=None,
          available_filters=None,
          return_lasso=False,
          **context_kwds):
    """
    High-level function to :term:`lasso` around spreadsheet's rect-regions 
    according to :term:`xl-ref` strings by using internally a :class:`Ranger` .

    :param str xlref:
        a string with the :term:`xl-ref` format::

            <url_file>#<sheet>!<1st_edge>:<2nd_edge>:<expand><js_filt>

        i.e.::

            file:///path/to/file.xls#sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}

    :param sheets_factory:
            Factory of sheets from where to parse rect-values; if unspecified, 
            the new :class:`SheetsFactory` created is closed afterwards.
            Delegated to :func:`make_default_Ranger()`, so items override
            default ones; use a new :class:`Ranger` if that is not desired.
    :ivar dict or None base_opts: 
            Opts affecting the lassoing procedure that are deep-copied and used
            as the base-opts for every :meth:`Ranger.do_lasso()`, whether invoked 
            directly or recursively by :meth:`Ranger.recursive_filter()`. 
            Read the code to be sure what are the available choices. 
            Delegated to :func:`make_default_Ranger()`, so items override
            default ones; use a new :class:`Ranger` if that is not desired.
    :param dict or None available_filters: 
            Delegated to :func:`make_default_Ranger()`, so items override
            default ones; use a new :class:`Ranger` if that is not desired.
    :param bool return_lasso:
            If `True`, values are contained in the returned Lasso instance,
            along with all other artifacts of the :term:`lassoing` procedure.

            For more debugging help, create a :class:`Range` yourself and 
            inspect the :attr:`Ranger.intermediate_lasso`.
    :param Lasso context_kwds: 
            Default :class:`Lasso` fields in case parsed ones are `None`
            (i.e. you can specify the sheet like that).
            Only those in :attr:`Ranger._context_lasso_fields` are taken 
            into account.

    :return: 
            Either the captured & filtered values or the final :class:`Lasso`,
            depending on the `return_lassos` arg.
    """
    factory_is_mine = not sheets_factory
    if base_opts is None:
        base_opts = get_default_opts()
    if available_filters is None:
        available_filters = get_default_filters()

    try:
        ranger = make_default_Ranger(sheets_factory=sheets_factory,
                                     base_opts=base_opts,
                                     available_filters=available_filters)
        lasso = ranger.do_lasso(xlref, **context_kwds)
    finally:
        if factory_is_mine:
            ranger.sheets_factory.close()

    return lasso if return_lasso else lasso.values


class ABCSheet(with_metaclass(ABCMeta, object)):
    """
    A delegating to backend factory and sheet-wrapper with utility methods.

    :param np.ndarray _states_matrix:
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
        ...     sheet = xlasso.xlrdSheet(wb.sheet_by_name('Sheet1'))
        ...     ## Do whatever

    *win32* examples::

        >>> with dsgdsdsfsd as wb:          #  doctest: +SKIP
        ...     sheet = xlasso.win32Sheet(wb.sheet['Sheet1'])
        TODO: Win32 Sheet example
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
    def open_sibling_sheet(self, sheet_id, opts=None):
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
        Fecth the actual values from the backend Excel-sheet.

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

    def __repr__(self):
        return '%s%s' % (type(self), self.get_sheet_ids())


class ArraySheet(ABCSheet):
    """A sample :class:`ABCSheet` made out of 2D-list or numpy-arrays, for facilitating tests."""

    def __init__(self, arr, ids=('wb', ['sh', 0])):
        self._arr = np.asarray(arr)
        self._ids = ids

    def open_sibling_sheet(self, sheet_id):
        raise NotImplementedError()

    def get_sheet_ids(self):
        return self._ids

    def _read_states_matrix(self):
        return ~np.equal(self._arr, None)

    def read_rect(self, st, nd):
        if nd is None:
            return self._arr[st]
        rect = np.array([st, nd]) + [[0, 0], [1, 1]]
        return self._arr[slice(*rect[:, 0]), slice(*rect[:, 1])].tolist()

    def __repr__(self):
        return 'ArraySheet%s \n%s' % (self.get_sheet_ids(), self._arr)