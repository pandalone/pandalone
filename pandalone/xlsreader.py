#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
A mini-language to capture rectangular-ranges from Excel-sheets by scanning empty/full cells.

.. default-role:: term


Introduction
============

The purpose of this library is to use of simple traversal operations 
to extract rectangular `regions` from excel-sheets when their exact positions 
is not known beforehand.

The `capturing` depends only on the full/empty `state` of the cells, 
and not on their values . Use another library (i.e. "pandas") to examine 
the values of the `capture-region' afterwards.


Excel-ref Syntax
----------------
::

    <1st-cell>[:[<2nd-cell>][:<expansions>]][<filters>]
    :


Annotated example
-----------------
::

    target-moves───┐
    cell-coords──────┐ │
                    ┌┤┌┴─┐
                    A1(RD):..(RD):L1DR{"type": "df", "kws": {"header": false}}
                    └─┬──┘ └─┬──┘ └┬─┘└───────────────┬─────────────────────┘
    1st-cell-pos──────┘      │     │                  │
    2nd-cell-pos─────────────┘     │                  │
    range-expansions───────────────┘                  │
    filters───────────────────────────────────────────┘

which means:

    Capture any rectangular range from the 1st `full-cell` beyond ``A1`` 
    while *moving* Right and Down, till the 1st `exterior` `empty-cell`;
    then try to `expand` the `capture-range` once to the Left,
    and then Down and Right untill a full-empty line/row is met, repectively.


.. seealso:: Example spreadsheet: :download:`xls_ref.xlsx`

Example
-------
::

      A B C D E
    1  ┌───────┐   
    2  │  X X  │ 
    3  │X X    │ 
    3  │X      │ 
    4  │  X   X│
       └───────┴─► Β2:E4   ^^.__    A1(RD):__:L1   ^^(DR)..(DR):U1DR


      A B C D E
    1  ┌─────┐   
    2  │  X X│ 
    3  │X X  │ 
    3  │X    │ 
    4  │  X  │X
       └─────┴───► Β2:D_   A1(RD):..(RD):L1D   C_:^^   A^(DR):C_:U


      A B C D E
    1    ┌───┐   
    2    │X X│ 
    3   X│X  │
         └───┴───► ^^(RD):..(RD)   _^(R):^.(DR) 
    3   X      
    4     X   X


      A B C D E
    1  ┌───┐   
    2  │  X│X  
    3  │X X│   
    3  │X  │   
    4  │  X│  X
       └───┴─────► Β2:C4   A1(RD):^_   C_:^^   A^(DR):C_:U   ^^(RD):..(D):D


Definitions
===========

.. glossary::

    excel-url
    xl-url
        Any url with its fragment abiding to the `excel-ref` syntax.
        Its file-part should resolve to an excel-file. 

    excel-ref
    xl-ref
        The syntax for `capturing` ranges from excel-sheets, 
        specified within the fragment-part of a `xl-url`.

    cell-pos
    cell-position
        A pair of row/col cell `coordinates` optionally followed by 
        a parenthesized `target-moves`.
        It actually specifies 2 cells, `start-cell` and `target-cell`.

    coord
    coordinate
        Either a *cell-column* (in letters) or *cell-row* (number);
        each one might be `absolute` or `dependent`, independently.

    coords
    coordinates
        The column/row pair of a `start-cell`.

    absolute-coordinate
    absolute
        Any cell row/col identified with column-characters, row-numbers, or
        the following special-characters:

        - ``^``          The top/Left full-cell `coordinate`.
        - ``_``          The bottom/right full-cell `coordinate`.

    dependent-coordinate
    dependent
        Any `2nd-cell` `coordinate` identified with a dot(``.``),
        which means that:

            *"2nd-start-cell coordinate = 1st target-cell coordinate"*

        The `2nd-cell` might contain a "mix" of `absolute` and *dependent* 
        coordinates.

    directions
    primitive-directions
        The 4 *primitive-directions* that are denoted with one of the letters
        ``LURD``.

    target-moves
        Specify the traversing order while `targeting` from `start-cell`, 
        based on `primitive-directions` pairs; so ``DR`` means:  
        
            *"Start going right, column-by-column, travesring columns 
            from top to bottom."*
        
        The pairs ``UD`` and ``LR`` (and their inverse) are invalid.

    targeting
        The search for the `target-cell` starts from the `start-cell`,
        follows the specified `target-moves`, and ends when a `state-change` 
        is detected on an `exterior` column or row.
        column or row, according to the enacted `termination-rule`.

    exterior
    exterior-column
    exterior-row
        The column and the row of the `start-cell`; Any `state-change` on 
        them, triggers the `termination-rule`.


    start-cell
    start
        The cell identified by the `coordinates` of the `cell-pos` alone.

    target-cell
    target
        The cell identified after applying `target-moves` on the `start-cell`.
        Failure to identify a target-cell raises an error.

    1st-cell
    1st-start-cell
    1st-target-cell
        The`capturing` STARTS from the `target` of *this* `cell-pos`.
        It supports `absolute` coordinates only.

    2nd-cell
    2nd-start-cell
    2nd-target-cell
        The `capturing` STOPS at the `target` of this `cell-pos`.
        It supports both `absolute` coordinates, and `dependent` ones from the 
        `1st-target-cell`.

    capture-range
    range
        The sheet's rectangular area bounded by the `1st-target-cell` and 
        the `2nd-target-cell`.

    capturing
    capture-moves
        The reading of the `capture-range` by traversing from 
        the `1st-target-cell` to the `2nd-target-cell`.

    state
    full-cell
    empty-cell
        A cell is *full* when it is not *empty* or *blank* 
        (in Excel's parlance).

    state-change
        Whether we are traversing from an `empty-cell` to a `full-cell`, and
        vice-versa, while `targeting`.

    termination-rule
        The condition to stop `targeting` while traversing an `exterior` 
        column/row and detecting a `state-change`.
        The are 2 rules: `search-same` and `search-opposite`.
        
        .. seealso:: 
            Check `Target-termination rules`_ for when each rule 
            applies.

    search-same
        The `target-cell` is the LAST cell with the SAME `state` as
        the `start-cell`, while `targeting` from it.

    search-opposite
        The `target-cell` is the FIRST cell with OPPOSITE `state` from 
        the `start-cell`, while `targeting` from it.

    expand
    expansions
    range-expansions
        Due to `state-change` on the 'exterior' cells the `capture-range` 
        might be smaller that a wider contigious but "convex" rectangular area.
        
        The * expansions* attempt to remedy this by providing for expanding on 
        arbitrary `directions` accompanied by a multiplicity for each one.
        If multiplicity is unspecified, infinite assumed, so it exampndas 
        until an empty/full row/column is met.

    filter
    filters
    filter-function
        Predefined functions to apply for transforming the `capture-range`
        specified as nested *json* dictionaries.


Details
=======

Target-moves
-------------

There are 12 `target-moves` named with a *single* or a *pair* of
letters denoting the 4 primitive directions, ``LURD``::

            U
     UL◄───┐▲┌───►UR
    LU     │││     RU
     ▲     │││     ▲
     │     │││     │
     └─────┼│┼─────┘
    L◄──────X──────►R
     ┌─────┼│┼─────┐
     │     │││     │
     ▼     │││     ▼
    LD     │││     RD
     DL◄───┘▼└───►DR
            D

    - The 'X' at the center points the starting cell.


So a ``RD`` move means *"traverse cells first by rows then by columns"*, 
or more lengthy description would be:

    *"Start moving *right* till 1st state change, and then 
    move *down* to the next row, and start traversing right again."*


Target-cells
------------

Using these moves we can identify a `target-cell` in relation to 
the `start-cell`. For instance, given this xl-sheet below, there are
multiple ways to identify (or target) the non-empty values ``X``, below::

      A B C D E F
    1
    2
    3     X        ──────► C3    A1(RD)   _^(L)      F3(L)
    4         X    ──────► E4    A4(R)    _4(L)      D1(DR)
    5   X          ──────► B5    A1(DR)   A_(UR)     _5(L)
    6           X  ──────► F6    __       _^(D)      A_(R)

    - The 'X' signifies non-empty cells.


So we can target cells with "absolute coordinates", the usual ``A1`` notation,
augmented with the following special characters:

  - undesrcore(``_``) for bottom/right, and
  - accent(``^``) for top/left

columns/rows of the sheet with non-empty values.

When no ``LURD`` moves are specified, the target-cell coinceds with the starting one.

.. Seealso:: `Target-termination rules`_ section


Ranges
------

To specify a complete `capture-range` we need to identify a 2nd cell.
The 2nd target-cell may be specified:

  - either with `absolute` coordinates, as above, or
  - with `dependent` coords, using the dot(``.``) to refer to the 1st cell.


In the above example-sheet, here are some ways to specify ranges::

      A  B C D E  F
    1

    2
          ┌─────┐
       ┌──┼─┐   │
    3  │  │X│   │
       │┌─┼─┼───┼┐
    4  ││ │ │  X││
       ││ └─┼───┴┼───► C3:E4   A1(RD):..(RD)   _^(L):..(DR)   _4(L):A1(RD)
    5  ││X  │    │
       │└───┼────┴───► B4:E5   A_(UR):..(RU)   _5(L):1_(UR)    E1(D):A.(DR)
    6  │    │     X
       └────┴────────► Β3:C6   A1(RD):^_       ^^:C_           C_:^^


.. Warning::
   Of course, the above ranges WILL FAIL since the `target-moves`
   will stop immediately due to ``X`` values being surrounded by empty-cells.

   But the above diagram was to just convey the general idea.
   To make it work, all the in-between cells of the peripheral row and columns 
   should have been also non-empty.

.. Note::
    The `capture-moves` from `1st-cell` to `2nd-target-cell` are independent from 
    the implied `target-moves` in the case of `dependent` coords.

    More specifically, the `capturing` will always fetch the same values 
    regardless of "row-first" or "column-first" order; this is not the case 
    with `targeting` (``LURD``) moves.

    For instance, to capture ``B4:E5`` in the above sheet we may use 
    ``_5(L):E.(U)``.
    In that case the target cells are ``B5`` and ``E4`` and the `target-moves`
    to reach the 2nd one are ``UR`` which are different from the ``U`` 
    specified on the 2nd cell.


.. Seealso:: `Target-termination rules`_ section


Target-termination rules
------------------------

- For the 1st target-cell:
  Target-cell is identified using `search-opposite` rule.

  .. Note:: It might be useful to allow the user to reverse this behavior 
      (ie by the use of the ``-`` char).

- For the 2nd target cell:

  - If the `state` of the `2nd-start-cell` == `1st-target-cell`:
    - Use `search-same` to identify target.

  - Otherwise:
    - Use `search-opposite` to identify target.



Expansions
----------

Captured-ranges ("values") may be limited due to empty-cells in the 1st 
row/column traversed.  To overcome this, the xl-ref may specify `expansions` 
directions using a 3rd ``:``-section like that::

    _5(L):1_(UR):RDL?U?

This particular case means:

     *"Try expanding Right and Down repeatedly and then try once Left and Up."*

Expansion happens on a row-by-row or column-by-column basis, and terminates 
when a full empty(or non-empty) line is met.

Example-refs are given below for capturing the 2 marked tables::

      A  B C D E F  G
    1
       ┌───────────┐
       │┌─────────┐│
    2  ││  1 X X  ││
       ││         ││
    3  ││X X   X X││
       ││         ││
    4  ││X X X 2 X││
       ││         ││
    5  ││X   X X X││
       └┼─────────┼┴──► A1(RD):..(RD):DRL?
    6   │X        │
        └─────────┴───► A1(RD):..(RD):L?DR       A_(UR):^^(RD)
    7               X

    - The 'X' signify non-empty cells.
    - The '1' and '2' signify the identified target-cells.



.. default-role:: obj

"""
from collections import namedtuple
import datetime
from distutils.version import LooseVersion
import json
import logging
import re
from string import ascii_uppercase

from xlrd import (xldate, XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_TEXT,
                  XL_CELL_BLANK, XL_CELL_ERROR, XL_CELL_BOOLEAN, XL_CELL_NUMBER,
                  open_workbook)
import xlrd

import itertools as itt
import numpy as np
import pandas as pd
from six.moves.urllib.parse import urldefrag  # @UnresolvedImport
from six.moves.urllib.request import urlopen  # @UnresolvedImport


# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    xlrd_0_9_3 = True
else:
    xlrd_0_9_3 = False

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
        (?P<st_row>\d+|_|\^)                             # first row
        (?:\(
            (?P<st_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves from st cell
            \)
        )?
    )
    (?::                                                 # second cell [opt]
        (?P<nd_col>[A-Z]+|_|\^|\.)                       # second col
        (?P<nd_row>\d+|_|\^|\.)                          # second row
        (?:\(
            (?P<nd_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves from nd cell
            \)
        )?
        (?::
            (?P<rng_exp>[LURD?\d]+)                      # range expansion [opt]
        )?
    )?
    \s*
    (?P<json>\{.*\})?                                    # any json object [opt]
    \s*$""", re.IGNORECASE | re.X)

_re_rng_exp_splitter = re.compile('([LURD]\d+)', re.IGNORECASE)

# TODO: Drop `?` from range_expansions, use numbers only.
_re_range_expansion_parser = re.compile(
    r"""
    ^(?P<moves>[LURD]+)                                    # primitive moves
    (?P<times>\?|\d+)?                                   # repetition times
    $""", re.IGNORECASE | re.X)


CellPos = namedtuple('CellPos', ['cell', 'mov'])


Cell = namedtuple('Cell', ['row', 'col'])


def row2num(coord):
    """
    Converts the Excel `str` row to a zero-based `int`, reporting invalids.

    :param str, int coord:  excel-row coordinate or one of ``^_.``

    :return:    excel row number, >= 0
    :rtype:     int


    Examples::

        >>> row2num('1')
        0

        >>> row2num('10') == row2num(10)
        True

        ## "Special" cells are also valid.
        >>> row2num('_'), row2num('^')
        ('_', '^')

        >>> row2num('0')
        Traceback (most recent call last):
        ValueError: Invalid row('0')!

        >>> row2num('a')
        Traceback (most recent call last):
        ValueError: Invalid row('a')!

        >>> row2num(None)
        Traceback (most recent call last):
        ValueError: Invalid row(None)!

    """
    if coord in _special_coords:
        return coord

    try:
        row = int(coord) - 1
        if row < 0:
            raise
        return row
    except Exception:
        raise ValueError('Invalid row({!r})!'.format(coord))


def col2num(coord):
    """
    Converts the Excel `str` column to a zero-based `int`, reporting invalids.

    :param str coord:     excel-column coordinate or one of ``^_.``

    :return:    excel column number, >= 0
    :rtype:     int


    Examples::

        >>> col2num('D')
        3

        >>> col2num('d')
        3

        >>> col2num('AaZ')
        727

        ## "Special" cells are also valid.
        >>> col2num('_'), col2num('^')
        ('_', '^')

        >>> col2num(None)
        Traceback (most recent call last):
        ValueError: Invalid column(None)!

        >>> col2num('4')
        Traceback (most recent call last):
        ValueError: Invalid column('4')!

        >>> col2num(4)
        Traceback (most recent call last):
        ValueError: Invalid column(4)!

    """

    if coord in _special_coords:
        return coord

    try:
        num = 0
        for c in coord:
            num = num * 26 + ascii_uppercase.rindex(c.upper()) + 1

        return num - 1
    except Exception:
        raise ValueError('Invalid column({!r})!'.format(coord))


def make_CellPos(cell_col, cell_row, cell_mov):
    """
    Fetch a cell reference string.

    :param cell_col:    column reference
    :type cell_col: str, None

    :param cell_row:    row reference
    :type cell_row: str, None

    :param cell_mov:    target-moves
    :type cell_mov: str, None

    :return:
        a cell-start
    :rtype: CellPos


    Examples::
        >>> make_CellPos('A', '1', 'R')
        CellPos(cell=Cell(row=0, col=0), mov='R')

        >>> make_CellPos('^', '^', 'R').cell
        Cell(row='^', col='^')

        >>> make_CellPos('_', '_', 'L').cell
        Cell(row='_', col='_')

        >>> make_CellPos('.', '.', 'D').cell
        Cell(row='.', col='.')

        >>> make_CellPos(None, None, None)

        >>> make_CellPos('1', '.', None)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='1', row='.') due to: Invalid column('1')!

        >>> make_CellPos('A', 'B', None)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='A', row='B') due to: Invalid row('B')!

        >>> make_CellPos('A', '1', 12)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='A', row='1') due to: 
            'int' object has no attribute 'upper'
    """

    try:
        if cell_col == cell_row == cell_mov is None:
            return None
        else:
            row = row2num(cell_row)
            col = col2num(cell_col)
            mov = cell_mov.upper() if cell_mov else None

            return CellPos(cell=Cell(col=col, row=row), mov=mov)
    except Exception as ex:
        msg = 'Invalid cell(col={!r}, row={!r}) due to: {}'
        raise ValueError(msg.format(cell_col, cell_row, ex))


def _repeat_moves(moves, times=None):
    """
    Examples::

         >>> list(_repeat_moves('ABC', '3'))
         ['ABC', 'ABC', 'ABC']

         >>> list(_repeat_moves('ABC', '0'))
         []

         >>> _repeat_moves('ABC')  ## infinite repetitions
         repeat('ABC')

     """
    args = (moves,)
    if times is not None:
        args += (int(times), )
    return itt.repeat(*args)


def _parse_range_expansions(rng_exp):
    """
    Parse range-expansion into a list of dir-letters iterables.

    :param rng_exp:
        A string with a sequence of primitive moves:
        es. L1U1R1D1
    :type xl_ref: str

    :return:
        A list of primitive-dir chains.
    :rtype: list


    Examples::

        >>> res = _parse_range_expansions('LURD?')
        >>> res
        [repeat('LUR'), repeat('D', 1)]

        # infinite generator
        >>> [next(res[0]) for i in range(10)]
        ['LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR']

        >>> list(res[1])
        ['D']

        >>> _parse_range_expansions('1LURD')
        Traceback (most recent call last):
        ValueError: Invalid range-expansion(1LURD) due to: 
                'NoneType' object has no attribute 'groupdict'

    """
    try:
        res = _re_rng_exp_splitter.split(rng_exp.replace('?', '1'))

        return [_repeat_moves(**_re_range_expansion_parser.match(v).groupdict())
                for v in res
                if v != '']

    except Exception as ex:
        msg = 'Invalid range-expansion({}) due to: {}'
        raise ValueError(msg.format(rng_exp, ex))


def parse_xl_ref(xl_ref):
    """
    Parses a :term:`excel-ref` and splits it in its "ingredients".

    :param xl_ref:
        a string with the following format:
        <sheet>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):
        <rng_exp>{<json>}
        es. sheet!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}
    :type xl_ref: str

    :return:
        dictionary containing the following parameters::

        - sheet
        - st_cell
        - nd_cell
        - rng_exp
        - json

    :rtype: dict


    Examples::
        >>> from itertools import chain
        >>> xl_ref = 'Sheet1!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}'
        >>> res = parse_xl_ref(xl_ref)

        >>> res['sheet']
        'Sheet1'

        >>> res['st_cell']
        CellPos(cell=Cell(row=0, col=0), mov='DR')

        >>> res['nd_cell']
        CellPos(cell=Cell(row=19, col=25), mov='UL')

        >>> list(chain(*res['rng_exp']))
        ['L', 'U', 'U', 'R', 'D']

        >>> res['json'] == {'json': '...'}
        True
    """

    try:
        r = _re_xl_ref_parser.match(xl_ref).groupdict()

        # resolve json
        r['json'] = json.loads(r['json']) if r['json'] else None

        # resolve range expansions
        r['rng_exp'] = _parse_range_expansions(
            r['rng_exp']) if r['rng_exp'] else None

        p = r.pop

        # fetch 1st cell
        r['st_cell'] = make_CellPos(p('st_col'), p('st_row'), p('st_mov'))

        # fetch 2nd cell
        r['nd_cell'] = make_CellPos(p('nd_col'), p('nd_row'), p('nd_mov'))

        return r

    except Exception as ex:
        log.debug("Invalid excel-ref(%s) due to: %s", xl_ref, ex, exc_info=1)
        raise ValueError("Invalid excel-ref({})!".format(xl_ref))


def parse_xl_url(url):
    """
    Parses the contents of an excel url.

    :param str url:
        a string with the following format::

            <url_file>#<sheet>!<1st_cell>:<2nd_cell>:<expand><json>

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

        >>> url = 'file:///sample.xlsx#Sheet1!A1{"2": "ciao"}'
        >>> res = parse_xl_url(url)
        >>> sorted(res.items())
        [('json', {'2': 'ciao'}),
         ('nd_cell', None),
         ('rng_exp', None),
         ('sheet', 'Sheet1'),
         ('st_cell', CellPos(cell=Cell(row=0, col=0), mov=None)),
         ('url_file', 'file:///sample.xlsx')]

    """

    try:
        res = {}

        res['url_file'], frag = urldefrag(url)  # parse excel url

        res.update(parse_xl_ref(frag))  # resolve excel reference

        return res

    except Exception as ex:
        raise ValueError("Invalid excel-url({}) due to: {}".format(url, ex))


# noinspection PyProtectedMember
def get_full_cells(sheet):
    """
    Returns a boolean ndarray with `False` wherever cell are blank or empty.
    """
    types = np.array(sheet._cell_types)
    return (types != xlrd.XL_CELL_EMPTY) & (types != xlrd.XL_CELL_BLANK)


def get_sheet_margins(full_cells):
    """ 
    Returns upper and lower absolute positions.

    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :return:  a 2-tuple with margins and indixes for full-cells


    Examples::

        >>> full_cells = [
        ...    [0, 0, 0],
        ...    [0, 1, 0],
        ...    [0, 1, 1],
        ...    [0, 0, 1],
        ... ]
        >>> sheet_margins, indices = get_sheet_margins(full_cells)

        #>>> sorted(sheet_margins.items()) ## FIXME: Nested DICT??
        [('col', {'^': 1, '_': 2}), 
         ('row', {'^': 1, '_': 3})]

        >>> indices
         [[1, 1], [2, 1], [2, 2], [3, 2]]

        >>> full_cells = [
        ...    [0, 0, 0, 0],
        ...    [0, 1, 0, 0],
        ...    [0, 1, 1, 0],
        ...    [0, 0, 1, 0],
        ...    [0, 0, 0, 0],
        ... ]
        >>> sheet_margins_2, _ = get_sheet_margins(full_cells)
        >>> sheet_margins_2 == sheet_margins
        True

    """
    indices = np.array(np.where(full_cells)).T  # XXX: Loads all sheet here?!?
    up_r, up_c = indices.min(0)
    dn_r, dn_c = indices.max(0)
    sheet_margins = {
        'col': {
            '^': up_c,
            '_': dn_c
        },
        'row': {
            '^': up_r,
            '_': dn_r
        }
    }
    return sheet_margins, indices.tolist()


def _get_abs_coord(coord, coord_margins, pcoord=None):
    """
    Translates any special or dependent coord to absolute ones.

    :param int, str coord:    the coord to translate
    :param int, None pcoord:  the basis for dependent coord, if any


    No other checks performed::

        >>> margins = {}
        >>> _get_abs_coord('_', margins)
        '_'

        >>> _get_abs_coord('$', margins)
        '$'
    """
    if pcoord:
        try:
            from collections import ChainMap
            coord_margins = ChainMap(coord_margins, {'.': pcoord})
        except ImportError:
            # TODO: FIX hack when ChainMap backported to py2.
            c = {'.': pcoord}
            c.update(coord_margins)
            coord_margins = c

    return coord_margins.get(coord, coord)


def _make_start_Cell(cell, sheet_margins, pcell=None):
    """
    Makes a Cell by translating any special coords to absolute ones.

    :param Cell cell:    The cell to translate its coords.
    :param Cell pcell:   The cell to base any dependent coords (``.``).


    Examples::

        >>> _make_start_Cell(Cell(3, 1), {'row':{}, 'col':{}})
        Cell(row=3, col=1)

    """
    row = _get_abs_coord(
        cell.row, sheet_margins['row'], pcell and pcell.row)
    col = _get_abs_coord(
        cell.col, sheet_margins['col'], pcell and pcell.col)

    return Cell(row=row, col=col)


def _search_opposite_state(state, cell, full_cells, up, dn, moves, last=False):
    """

    :param bool state:      the starting-state
    :param cell:
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :param sheet:
    :param directions:
    :return:


    Examples::
        >>> full_cells = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 1, 1, 1],
        ...     [0, 0, 0, 1, 0, 0, 1],
        ...     [0, 0, 0, 1, 1, 1, 1]
        ... ])
        >>> args = (False, Cell(1, 1), full_cells, (0, 0), (7, 6))
        >>> _search_opposite_state(*(args + ('DR', )))
        Cell(row=6, col=3)

        >>> _search_opposite_state(*(args + ('RD', )))
        Cell(row=5, col=4)

        >>> _search_opposite_state(*(args + ('D', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(D)

        >>> _search_opposite_state(*(args + ('U', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(U)

        >>> _search_opposite_state(*(args + ('R', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(R)

        >>> _search_opposite_state(*(args + ('L', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(L)

        >>> _search_opposite_state(*(args + ('LU', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(LU)

        >>> args = (True, Cell(6, 3), full_cells, (0, 0), (7, 6))
        >>> _search_opposite_state(*(args + ('D', )))
        Cell(row=8, col=3)

        >>> args = (True, Cell(10, 3), full_cells, (0, 0), (7, 6))
        >>> _search_opposite_state(*(args + ('U', )))
        Cell(row=10, col=3)

        >>> args = (False, Cell(10, 10), full_cells, (0, 0), (7, 6))
        >>> _search_opposite_state(*(args + ('UL', )))
        Cell(row=7, col=6)

        >>> full_cells = np.array([
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ... ])
        >>> args = (True, Cell(0, 2), full_cells, (0, 0), (2, 2))
        >>> _search_opposite_state(*(args + ('LD', )))
        Cell(row=3, col=2)
    """
    mv = _primitive_dir[moves[0]]  # first move

    c0 = np.array(cell)

    flag = False

    if not state:
        if not c0[0] <= dn[0] and 'U' in moves:
            c0[0] = dn[0]
        if not c0[1] <= dn[1] and 'L' in moves:
            c0[1] = dn[1]

    while True:
        c1 = c0
        while (up <= c1).all():
            try:
                if full_cells[c1[0], c1[1]] != state:
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


def _search_same_state(state, cell, full_cells, up, dn, moves):
    """

    :param bool state:      the starting-state
    :param cell:
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :param sheet:
    :param directions:
    :return:


    Examples::
        >>> full_cells = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 1, 1, 1],
        ...     [0, 0, 0, 1, 0, 0, 1],
        ...     [0, 0, 0, 1, 1, 1, 1]
        ... ])
        >>> args = (True, Cell(7, 6), full_cells, (0, 0), (7, 6))
        >>> _search_same_state(*(args + ('UL', )))
        Cell(row=5, col=3)

        >>> _search_same_state(*(args + ('U', )))
        Cell(row=5, col=6)

        >>> _search_same_state(*(args + ('L', )))
        Cell(row=7, col=3)

        >>> args = (True, Cell(5, 3), full_cells, (0, 0), (7, 6))
        >>> _search_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> args = (False, Cell(5, 3), full_cells, (0, 0), (7, 6))
        >>> _search_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> _search_same_state(*(args + ('UL', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=5, col=3) with movement(U)

        >>> args = (True, Cell(5, 6), full_cells, (0, 0), (7, 6))
        >>> _search_same_state(*(args + ('DL', )))
        Cell(row=7, col=4)

    """

    c1 = list(cell)

    for mv in moves:
        c = _search_opposite_state(state, cell, full_cells, up, dn, mv, True)
        dis = _primitive_dir[mv]
        c1 = [i if not k == 0 else j for i, j, k in zip(c, c1, dis)]
    return Cell(*c1)


def expand_range(state, xl_range, full_cells, rng_exp):
    """

    :param state:
    :param up:
    :param dn:
    :param rng:
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :param rng_exp:
    :return:


    Examples::

        >>> full_cells = np.array([
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
        >>> rng_exp = [_repeat_moves('U', times=10)]
        >>> expand_range(True, rng, full_cells, rng_exp)
        [Cell(row=6, col=3), Cell(row=6, col=3)]

        >>> rng = (Cell(row=6, col=3), Cell(row=7, col=3))
        >>> rng_exp = [_repeat_moves('R', times=10)]
        >>> expand_range(True, rng, full_cells, rng_exp)
        [Cell(row=6, col=3), Cell(row=7, col=6)]

        >>> rng = (Cell(row=6, col=3), Cell(row=10, col=3))
        >>> rng_exp = [_repeat_moves('R', times=10)]
        >>> expand_range(True, rng, full_cells, rng_exp)
        [Cell(row=6, col=3), Cell(row=10, col=6)]

        >>> rng = (Cell(row=6, col=5), Cell(row=6, col=5))
        >>> rng_exp = [_repeat_moves('LURD')]
        >>> expand_range(True, rng, full_cells, rng_exp)
        [Cell(row=5, col=3), Cell(row=7, col=6)]

    """
    _m = {
        'L': (0, 1),
        'U': (0, 1),
        'R': (1, 0),
        'D': (1, 0)
    }
    xl_range = [np.array(v) for v in xl_range]
    for moves in rng_exp:
        for directions in moves:
            flag = True
            for d in directions:
                mv = _primitive_dir[d]
                i, j = _m[d]
                st, nd = (xl_range[i], xl_range[j])
                st = st + mv
                nd = [p2 if k == 0 else p1 for p1, p2, k in zip(st, nd, mv)]
                if i == 1:
                    v = full_cells[nd[0]:st[0] + 1, nd[1]:st[1] + 1]
                else:
                    v = full_cells[st[0]:nd[0] + 1, st[1]:nd[1] + 1]
                if (not v.size and state) or (v != state).all():
                    continue
                xl_range[i] = st
                flag = False

            if flag:
                break

    return [Cell(*v) for v in xl_range]


def _capture_range(full_cells, up, dn, sheet_margins, indices, st_cell,
                   nd_cell=None, rng_exp=None):
    """

    :param xlrd.sheet.Sheet sheet:
    :param CellPos st_cell:
    :param CellPos nd_cell:
    :param rng_exp:
    :return:


    Examples::

        >>> full_cells = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 0, 0, 0],
        ...     [0, 0, 0, 0, 1, 1, 1],
        ...     [0, 0, 0, 1, 0, 0, 1],
        ...     [0, 0, 0, 1, 1, 1, 1]
        ... ])
        >>> up, dn = ((0, 0), (7, 6))
        >>> sheet_margins, ind = get_sheet_margins(full_cells)
        >>> st_cell = CellPos(Cell(0, 0), 'DR')
        >>> nd_cell = CellPos(Cell('.', '.'), 'DR')
        >>> _capture_range(full_cells, up, dn, sheet_margins, ind, st_cell, nd_cell)
        (Cell(row=6, col=3), Cell(row=7, col=3))

        >>> nd_cell = CellPos(Cell(7, 6), 'UL')
        >>> _capture_range(full_cells, up, dn, sheet_margins, ind, st_cell, nd_cell)
        (Cell(row=5, col=3), Cell(row=6, col=3))
    """

    st = _make_start_Cell(st_cell.cell, sheet_margins)
    try:
        state = full_cells[st]
    except IndexError:
        state = False

    if st_cell.mov is not None:
        st = _search_opposite_state(state, st, full_cells, up, dn, st_cell.mov)
        state = not state

    if nd_cell is None:
        nd = Cell(*st)
    else:
        nd = _make_start_Cell(nd_cell.cell, sheet_margins, st)

        if nd_cell.mov is not None:
            mov = nd_cell.mov
            if state == full_cells[nd]:
                nd = _search_same_state(state, nd, full_cells, up, dn, mov)
            else:
                nd = _search_opposite_state(
                    not state, nd, full_cells, up, dn, mov)

        c = np.array([st, nd])

        st, nd = (Cell(*list(c.min(0))), Cell(*list(c.max(0))))

    if rng_exp is None:
        return (st, nd)
    else:
        return expand_range(state, (st, nd), full_cells, rng_exp)


def _parse_cell(cell, epoch1904=False):
    """
    Parse a xl-cell.

    :param cell: an excel cell
    :type cell: xlrd.sheet.Cell

    :param epoch1904:
        Which date system was in force when this file was last saved.
        False => 1900 system (the Excel for Windows default).
        True => 1904 system (the Excel for Macintosh default).
    :type epoch1904: bool

    :return: formatted cell value
    :rtype:
        int, float, datetime.datetime, bool, None, str, datetime.time,
        float('nan')


    Examples::

        >>> import xlrd
        >>> from xlrd.sheet import Cell
        >>> _parse_cell(Cell(xlrd.XL_CELL_NUMBER, 1.2))
        1.2

        >>> _parse_cell(Cell(xlrd.XL_CELL_DATE, 1.2))
        datetime.datetime(1900, 1, 1, 4, 48)

        >>> _parse_cell(Cell(xlrd.XL_CELL_TEXT, 'hi'))
        'hi'
    """

    ctype = cell.ctype

    if ctype == XL_CELL_NUMBER:
        # GH5394 - Excel 'numbers' are always floats
        # it's a minimal perf hit and less suprising
        val = int(cell.value)
        if val == cell.value:
            return val
        return cell.value
    elif ctype in (XL_CELL_EMPTY, XL_CELL_BLANK):
        return None
    elif ctype == XL_CELL_TEXT:
        return cell.value
    elif ctype == XL_CELL_BOOLEAN:
        return bool(cell.value)
    elif ctype == XL_CELL_DATE:  # modified from Pandas library
        if xlrd_0_9_3:
            # Use the newer xlrd datetime handling.
            d = xldate.xldate_as_datetime(cell.value, epoch1904)

            # Excel doesn't distinguish between dates and time, so we treat
            # dates on the epoch as times only. Also, Excel supports 1900 and
            # 1904 epochs.
            epoch = (1904, 1, 1) if epoch1904 else (1899, 12, 31)
            if (d.timetuple())[0:3] == epoch:
                d = datetime.time(d.hour, d.minute, d.second, d.microsecond)
        else:
            # Use the xlrd <= 0.9.2 date handling.
            d = xldate.xldate_as_tuple(cell.value, epoch1904)
            if d[0] < datetime.MINYEAR:  # time
                d = datetime.time(*d[3:])
            else:  # date
                d = datetime.datetime(*d)
        return d
    elif ctype == XL_CELL_ERROR:
        return float('nan')

    raise ValueError('invalid cell type %s for %s' % (cell.ctype, cell.value))


def get_xl_table(sheet, xl_range, indices, epoch1904=False):
    """

    :param sheet:
    :param xl_range:
    :param indices:
    :param epoch1904:
    :return:


    Examples::

        >>> import os, tempfile, xlrd, pandas as pd

        >>> os.chdir(tempfile.mkdtemp())
        >>> df = pd.DataFrame([[None, None, None], [5.1, 6.1, 7.1]])
        >>> tmp = 'sample.xlsx'
        >>> writer = pd.ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

        >>> sheet = xlrd.open_workbook(tmp).sheet_by_name('Sheet1')

        >>> sheet_margins, indices = get_sheet_margins(get_full_cells(sheet))

        # minimum matrix in the sheet
        >>> st = _make_start_Cell(Cell('^', '^'), sheet_margins)
        >>> nd = _make_start_Cell(Cell('_', '_'), sheet_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [[None, 0, 1, 2],
         [0, None, None, None],
         [1, 5.1, 6.1, 7.1]]

        # get single value
        >>> get_xl_table(sheet, (Cell(6, 3), Cell(6, 3)), indices)
        [0]

        # get column vector
        >>> st = _make_start_Cell(Cell(0, 3), sheet_margins)
        >>> nd = _make_start_Cell(Cell('_', 3), sheet_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> st = _make_start_Cell(Cell(5, 0), sheet_margins)
        >>> nd = _make_start_Cell(Cell(5, '_'), sheet_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2]

        # get row vector
        >>> st = _make_start_Cell(Cell(5, 0), sheet_margins)
        >>> nd = _make_start_Cell(Cell(5, 10), sheet_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2, None, None, None, None]

    """
    tbl = []
    for r in range(xl_range[0].row, xl_range[1].row + 1):
        row = []
        tbl.append(row)
        for c in range(xl_range[0].col, xl_range[1].col + 1):
            if [r, c] in indices:
                row.append(_parse_cell(sheet.cell(r, c), epoch1904))
            else:
                row.append(None)
    # vector
    if xl_range[1].col == xl_range[0].col:
        tbl = [v[0] for v in tbl]

    # vector
    if xl_range[1].row == xl_range[0].row:
        tbl = tbl[0]

    if isinstance(tbl, list):
        return tbl
    else:
        return [tbl]


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


def redim_captured_values(value, dim_min, dim_max=None):
    """
    Reshapes the output value of get_rect_range function.

    :param value: matrix or vector or value
    :type value: list of lists, list, value

    :param dim_min: minimum dimension
    :type dim_min: int, None

    :param dim_max: maximum dimension
    :type dim_max: int, None

    :return: reshaped value
    :rtype: list of lists, list, value


    Examples::

        >>> redim_captured_values([1, 2], 2)
        [[1, 2]]

        >>> redim_captured_values([[1, 2]], 1)
        [[1, 2]]

        >>> redim_captured_values([[1, 2]], 1, 1)
        [1, 2]

        >>> redim_captured_values([], 2)
        [[]]

        >>> redim_captured_values([[1, 2]], 0, 0)
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


default_range_filters = {
    None: {'fun': lambda x: x},  # TODO: Actually redim_captured_values().
    'df': {'fun': pd.DataFrame},
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}


def process_captured_values(value, type=None, args=(), kws=None, filters=None,
                            available_filters=default_range_filters):
    """
    Processes the output value of get_rect_range function.

    FIXME: Actually use process_captured_values()!

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
            to further process range-values.
    :param dict available_filters:
            Entries of ``<fun_names> --> <callables>`` for pre-configured 
            filters available to post-process range-values.
            The callable for `None` key will be always called
            to the original values to ensure correct dimensionality
    :return: processed range-values
    :rtype: given type, or list of lists, list, value


    Examples::

        >>> value = [[1, 2], [3, 4], [5, 6]]
        >>> res = process_captured_values(value, type='dict')
        >>> sorted(res.items())
        [(1, 2),
         (3, 4),
         (5, 6)]

        >>> value = [[1, 9], [8, 10], [5, 11]]
        >>> process_captured_values(value, 
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
            val = process_captured_values(val, **v)
    return val


#### XLRD HELPER FUNCS ###

def open_xl_workbook(xl_ref_child, xl_ref_parent=None):
    """
    Opens the excel workbook of an excel ref.

    :param dict xl_ref_child: excel ref of the child

    :param xl_ref_parent: excel ref of the parent
    :type xl_ref_parent: dict, None

    """
    url_fl = xl_ref_child['url_file']
    try:
        if url_fl:
            wb = open_workbook(file_contents=urlopen(url_fl).read())
        else:
            wb = xl_ref_parent['xl_workbook']
        xl_ref_child['xl_workbook'] = wb

    except Exception as ex:
        raise ValueError("Invalid excel-file({}) due to:{}".format(url_fl, ex))


def open_xl_sheet(xl_ref_child, xl_ref_parent=None):
    """
    Opens the excel sheet of an excel ref.

    :param dict xl_ref_child: excel ref of the child

    :param xl_ref_parent: excel ref of the parent
    :type xl_ref_parent: dict, None

    """
    try:
        if xl_ref_child['sheet']:
            wb = xl_ref_child['xl_workbook']
            sheet = wb.sheet_by_name(xl_ref_child['sheet'])
        else:
            sheet = xl_ref_parent['xl_sheet']
        xl_ref_child['xl_sheet'] = sheet

    except Exception as ex:
        sh = xl_ref_child['sheet']
        raise ValueError("Invalid excel-sheet({}) due to:{}".format(sh, ex))
