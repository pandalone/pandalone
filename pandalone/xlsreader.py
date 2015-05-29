#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Implements an "Excel-url" format for capturing ranges from sheets.

.. seealso:: Example spreadsheet: :download:`xls_ref.xlsx`

Excel-range addressing
======================

Syntax::

    <1st-cell>[:[<2nd-cell>][:<expansions>]][<filters>]
    :


Annotated Example::

    target-moves───┐
    cell-coords──────┐ │
                    ┌┤┌┴─┐
                    A1(RD):..(RD):L?DR{"fun": "df", "kws": {"header": false}}
                    └─┬──┘ └─┬──┘ └┬─┘└───────────────┬─────────────────────┘
    1st-cell-pos──────┘      │     │                  │
    2nd-cell-pos─────────────┘     │                  │
    range-expansions───────────────┘                  │
    filters───────────────────────────────────────────┘


Definitions
-----------

.. default-role:: term
.. glossary::

    excel-url
    xl-url
        Any url with its fragment abiding to the `xl-range` syntax.

    excel-range
    xl-range
    excel-ref
    xl-ref
        The syntax for `capturing` ranges from excel-sheets, 
        specified within the fragment-part of a `xl-url`.

    cell-pos
    cell-position
        A pair of row/col cell `coordinates` optionally followed by 
        a parenthesized `target-moves`.

    coord
    coords
    coordinate
    coordinates
    cell-coords
    row-coordinate
    row-coord
    column-coordinate
    col-coord
        The cell-column (in letters) and cell-row (number) of a cell.

    absolute-coordinate
    absolute
        Any cell row/col identified with column-characters, row-numbers, or
        the following special-characters:

        - ``^``          The top/Left full-cell `coordinate`.
        - ``_``          The bottom/right full-cell `coordinate`.

    relative-coordinate
    relative
        Any `2nd-cell` `coordinate` identified with a dot(``.``),
        which means that:

            > 2nd-start-cell coordinate = 1st target-cell coordinate

        .. Note:: The cell-pos of the 2nd-cell might contain a "mix" of
            `absolute` and `relative` coordinates.

    target-moves
    targeting
    traverse-directions
    traversing
        A combination of the 4 primitive directions ``LURD``,
        specified with a *single* or a *pair* of the letters inside the 
        `cell-pos` parenthesis.  The pairs ``UD`` and ``LR``, 
        and their inverse, are invalid moves.

    start-cell
    start
        The cell identified by the `coordinates` of the `cell-pos` alone.

    target-cell
    target
        The cell identified after applying `target-moves` on the `start-cell`.
        Failure to identify a target-cell raises an error.

    1st
    1st-cell
    1st-cell-pos
    1st-start-cell
    1st-target-cell
        The `capturing` of a range starts from the `target` of this `cell-pos`.
        It supports `absolute` coordinates only.

    2nd
    2nd-cell
    2nd-cell-pos
    2nd-start-cell
    2nd-target-cell
        The `capturing` of a range stops at the `target` of this `cell-pos`.
        It supports both `absolute` and `relative` coordinates.

    capture
    capturing
    capturing-phase
    capture-moves
    captured-rectangle
    captured-range
    range
        The reading of the excel-sheet values from a rectangular area, 
        bounded by the `1st-target-cell` and the `2nd-target-cell`.

    state
    cell-state
        Whether a cell is empty or full(non-empty).

    termination-rule
    target-termination-rule
        The condition for stopping `target-moves` while searching for 
        a `target-cell`.
        It can be either `search-same` or `search-opposite`.

    search-same
        The `target-cell` is the LAST cell with the SAME `state` as
        the `start-cell`, while `traversing` from it.

    search-opposite
        The `target-cell` is the FIRST cell with OPPOSITE `state` from 
        the `start-cell`, while `traversing` from it.

    range-expansions
    expansions
        How to expand the initial `captured-rectangle`.
        It can be an arbitrary combinations for the ``LURD?`` letters,
        with repetitions.

    filter
    filters
    filter-function
    filter-functions
        Predefined functions to apply for transforming the `captured-range`
        specified as nested *json* dictionaries.



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

  > Start moving *right* till 1st state change, and then 
  > move *down* to the next row, and start traversing right again."


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

To specify a complete `range` we need to identify a 2nd cell.
The 2nd target-cell may be specified:

  - either with `absolute` coordinates, as above, or
  - with `relative` coords, using the dot(``.``) to refer to the 1st cell.


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
    The `capture-moves` from `1st` to `2nd-target-cell` are independent from 
    the implied `target-moves` in the case of `relative` coords.

    More specifically, the `capturing` will always fetch the same values 
    regardless of "row-first" or "column-first" order; this is not the case 
    with `traversing` (``LURD``) moves.

    For instance, to capture ``B4:E5`` in the above sheet we may use 
    ``_5(L):E.(U)``.
    In that case the target cells are ``B5`` and ``E4`` and the `target-moves`
    to reach the 2nd one are ``UR`` which are different from the ``U`` 
    specified on the 2nd cell.


.. Seealso:: `Target-termination rules`_ section


Target-termination rules
--------------------------

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

  > Try expanding Right and Down repeatedly and then try once Left and Up.

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
import re
import json
import datetime
import pandas as pd
import numpy as np
from string import ascii_uppercase
from collections import namedtuple
from itertools import repeat

# noinspection PyUnresolvedReferences
from six.moves.urllib.parse import urldefrag  # @UnresolvedImport
# noinspection PyUnresolvedReferences
from six.moves.urllib.request import urlopen  # @UnresolvedImport

import xlrd

from xlrd import (xldate, XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_TEXT,
                  XL_CELL_BLANK, XL_CELL_ERROR, XL_CELL_BOOLEAN, XL_CELL_NUMBER,
                  open_workbook)

from distutils.version import LooseVersion

if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    xlrd_0_9_3 = True
else:
    xlrd_0_9_3 = False


class _Token(object):

    def __init__(self, name, **kws):
        self.name = name
        vars(self).update(kws)

    def __repr__(self):
        return self.name


CELL_UL = _Token('CELL_UL')
CELL_DR = _Token('CELL_DR')
CELL_REL = _Token('CELL_REL')
MAX_TIME = None

_c_pos = {
    '^': CELL_UL,
    '_': CELL_DR,
    '.': CELL_REL
}

_function_types = {
    None: {'fun': lambda x: x},
    'df': {'fun': pd.DataFrame},
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}

_primitive_dir = {
    'L': np.array([0, -1]),
    'U': np.array([-1, 0]),
    'R': np.array([0, 1]),
    'D': np.array([1, 0])
}

_re_xl_ref_parser = re.compile(
    r"""
    ^\s*(?:(?P<xl_sheet_name>[^!]+)?!)?                  # xl sheet name
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
            (?P<rng_ext>[LURD?\d]+)                      # range extension [opt]
        )?
    )?
    (?P<json>\{.*\})?                                    # any json object [opt]
    \s*$""", re.IGNORECASE | re.X)

_re_rng_ext_splitter = re.compile('([LURD]\d+)', re.IGNORECASE | re.X)

_re_rng_ext_parser = re.compile(
    r"""
    ^(?P<mov>[LURD]+)                                    # primitive moves
    (?P<times>\?|\d+)?                                   # repetition times
    $""", re.IGNORECASE | re.X)


CellPos = namedtuple('CellPos', ['cell', 'mov'])


Cell = namedtuple('Cell', ['row', 'col'])


def row2num(coord):
    """
    Converts the Excel `str` row to a zero-based `int`, reporting invalids.

    :param str, int coord:     excel row coordinate

    :return:    excel row number, >= 0
    :rtype:     int

    Example::

        >>> row2num('1')
        0

        >>> row2num('10') == row2num(10)
        True

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

    :param str coord:     excel column coordinate

    :return:    excel column number, >= 0
    :rtype:     int

    Examples::

        >>> col2num('D')
        3

        >>> col2num('d')
        3

        >>> col2num('AaZ')
        727

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

    try:
        num = 0
        for c in coord:
            num = num * 26 + ascii_uppercase.rindex(c.upper()) + 1

        return num - 1
    except Exception:
        raise ValueError('Invalid column({!r})!'.format(coord))


def fetch_cell_ref(cell_col, cell_row, cell_mov):
    """
    Fetch a cell reference string.

    :param cell_col:
        column reference
    :type cell_col: str, None

    :param cell_row:
        row reference
    :type cell_row: str, None

    :param cell_mov:
        traverse-directions
    :type cell_mov: str, None

    :return:
        a cell-start
    :rtype: CellPos

    Examples::
        >>> fetch_cell_ref('A', '1', 'R')
        CellPos(cell=Cell(row=0, col=0), mov='R')

        >>> fetch_cell_ref('^', '^', 'R').cell
        Cell(row=CELL_UL, col=CELL_UL)

        >>> fetch_cell_ref('_', '_', 'L').cell
        Cell(row=CELL_DR, col=CELL_DR)

        >>> fetch_cell_ref('.', '.', 'D').cell
        Cell(row=CELL_REL, col=CELL_REL)

        >>> fetch_cell_ref(None, None, None)

        >>> fetch_cell_ref('1', '.', None)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='1', row='.') due to: Invalid column('1')!

        >>> fetch_cell_ref('A', 'B', None)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='A', row='B') due to: Invalid row('B')!

        >>> fetch_cell_ref('A', '1', 12)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='A', row='1') due to: 'int' object has no attribute 'upper'
    """

    try:
        if cell_col == cell_row == cell_mov is None:
            return None
        else:
            row = _c_pos[cell_row] if cell_row in _c_pos else row2num(cell_row)
            col = _c_pos[cell_col] if cell_col in _c_pos else col2num(cell_col)
            mov = cell_mov.upper() if cell_mov else None

            return CellPos(cell=Cell(col=col, row=row), mov=mov)
    except Exception as ex:
        msg = 'Invalid cell(col={!r}, row={!r}) due to: {}'
        raise ValueError(msg.format(cell_col, cell_row, ex))


def repeat_moves(moves, times=None):
    # repeat_moves('ABC', 3) --> ABC ABC ABC
    return repeat(moves, int(times)) if times is not None else repeat(moves)


def parse_rng_ext(rng_ext):
    """
    Parses and fetches the contents of range-extension ref.

    :param rng_ext:
        A string with a sequence of primitive moves:
        es. L1U1R1D1
    :type xl_ref: str

    :return:
        A list of primitive move chains.
    :rtype: list

    Examples::

        >>> res = parse_rng_ext('LURD?')

        >>> res
        [repeat('LUR'), repeat('D', 1)]

        # infinite generator
        >>> [next(res[0]) for i in range(10)]
        ['LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR', 'LUR']
        >>> list(res[1])
        ['D']
        >>> parse_rng_ext('1LURD')
        Traceback (most recent call last):
        ...
        ValueError: Invalid range-extension(1LURD) due to: 'NoneType' object has no attribute 'groups'
    """
    try:
        res = _re_rng_ext_splitter.split(rng_ext.replace('?', '1'))

        return [repeat_moves(*_re_rng_ext_parser.match(v).groups())
                for v in res
                if v != '']

    except Exception as ex:
        raise ValueError('Invalid range-extension({}) '
                         'due to: {}'.format(rng_ext, ex))


def parse_xl_ref(xl_ref):
    """
    Parses and fetches the contents of excel ref.

    :param xl_ref:
        a string with the following format:
        <xl_sheet_name>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):
        <rng_ext>{<json>}
        es. xl_sheet_name!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}
    :type xl_ref: str

    :return:
        dictionary containing the following parameters::

        - xl_sheet_name
        - st_cell
        - nd_cell
        - rng_ext
        - json

    :rtype: dict

    Examples::
        >>> from itertools import chain
        >>> xl_ref = 'Sheet1!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}'
        >>> res = parse_xl_ref(xl_ref)

        >>> res['xl_sheet_name']
        'Sheet1'

        >>> res['st_cell']
        CellPos(cell=Cell(row=0, col=0), mov='DR')

        >>> res['nd_cell']
        CellPos(cell=Cell(row=19, col=25), mov='UL')

        >>> list(chain(*res['rng_ext']))
        ['L', 'U', 'U', 'R', 'D']

        >>> res['json'] == {'json': '...'}
        True
    """

    try:
        r = _re_xl_ref_parser.match(xl_ref).groupdict()

        # resolve json
        r['json'] = json.loads(r['json']) if r['json'] else None

        # resolve range extensions
        r['rng_ext'] = parse_rng_ext(r['rng_ext']) if r['rng_ext'] else None

        p = r.pop

        # fetch 1st cell
        r['st_cell'] = fetch_cell_ref(p('st_col'), p('st_row'), p('st_mov'))

        # fetch 2nd cell
        r['nd_cell'] = fetch_cell_ref(p('nd_col'), p('nd_row'), p('nd_mov'))

        return r

    except Exception as ex:
        raise ValueError("Invalid excel-ref({}) due to: {}".format(xl_ref, ex))


def parse_xl_url(url):
    """
    Parses the contents of an excel url.

    :param url:
        a string with the following format:

        <url_file>#<xl_sheet_name>!<cell_up>:<cell_down><json>
        es. file:///path/to/file.xls#xl_sheet_name!UP10:DN20{"json":"..."}
    :type url: str

    :return:
        dictionary containing the following parameters::

        - url_file
        - xl_sheet_name
        - st_col
        - st_row
        - st_mov
        - nd_col
        - nd_row
        - nd_mov
        - json

    :rtype: dict

    Examples::

        >>> url = 'file:///sample.xls#Sheet1!A1{"2": "ciao"}'
        >>> res = parse_xl_url(url)
        >>> sorted(res.items())
        [('json', {'2': 'ciao'}),
         ('nd_cell', None),
         ('rng_ext', None),
         ('st_cell', CellPos(cell=Cell(row=0, col=0), mov=None)),
         ('url_file', 'file:///sample.xls'),
         ('xl_sheet_name', 'Sheet1')]
    """

    try:
        res = {}

        res['url_file'], frag = urldefrag(url)  # parse excel url

        res.update(parse_xl_ref(frag))  # resolve excel reference

        return res

    except Exception as ex:
        raise ValueError("Invalid excel-url({}) due to: {}".format(url, ex))


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
        if xl_ref_child['xl_sheet_name']:
            wb = xl_ref_child['xl_workbook']
            sheet = wb.sheet_by_name(xl_ref_child['xl_sheet_name'])
        else:
            sheet = xl_ref_parent['xl_sheet']
        xl_ref_child['xl_sheet'] = sheet

    except Exception as ex:
        sh = xl_ref_child['xl_sheet_name']
        raise ValueError("Invalid excel-sheet({}) due to:{}".format(sh, ex))


# noinspection PyProtectedMember
def get_no_empty_cells(sheet):
    types = np.array(sheet._cell_types)
    return (types != xlrd.XL_CELL_EMPTY) & (types != xlrd.XL_CELL_BLANK)


def get_xl_abs_margins(no_empty):
    """ Returns upper and lower absolute positions"""
    indices = np.array(np.where(no_empty)).T
    up_r, up_c = indices.min(0)
    dn_r, dn_c = indices.max(0)
    xl_margins = {
        'col': {
            CELL_UL: up_c,
            CELL_DR: dn_c
        },
        'row': {
            CELL_UL: up_r,
            CELL_DR: dn_r
        }
    }
    return xl_margins, indices.tolist()


def get_xl_margins(sheet):
    no_empty = get_no_empty_cells(sheet)

    up = (0, 0)

    dn = (sheet.nrows - 1, sheet.ncols - 1)

    return no_empty, up, dn


def set_coord(cell, xl_margins, parent=None):
    c = {CELL_REL: parent} if parent is not None else {}
    c.update(xl_margins)
    return c[cell] if cell in c else cell


def set_start_cell(cell, xl_margins, parent_cell=None):
    if parent_cell:
        row = set_coord(cell.row, xl_margins['row'], parent_cell.row)
        col = set_coord(cell.col, xl_margins['col'], parent_cell.col)
    else:
        row = set_coord(cell.row, xl_margins['row'], None)
        col = set_coord(cell.col, xl_margins['col'], None)
    return Cell(row=row, col=col)


def search_opposite_state(state, cell, no_empty, up, dn, moves, last=False):
    """

    :param cell:
    :param no_empty:
    :param sheet:
    :param directions:
    :return:

    Examples::
        >>> non_empty = np.array(\
            [[False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False,  True,  True,  True],\
             [False, False, False,  True, False, False, False],\
             [False, False, False,  True,  True,  True,  True]])
        >>> args = (False, Cell(1, 1), non_empty, (0, 0), (7, 6))
        >>> search_opposite_state(*(args + ('DR', )))
        Cell(row=6, col=3)

        >>> search_opposite_state(*(args + ('RD', )))
        Cell(row=5, col=4)

        >>> search_opposite_state(*(args + ('D', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(D)

        >>> search_opposite_state(*(args + ('U', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(U)

        >>> search_opposite_state(*(args + ('R', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(R)

        >>> search_opposite_state(*(args + ('L', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(L)

        >>> search_opposite_state(*(args + ('LU', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=1, col=1) with movement(LU)

        >>> args = (True, Cell(6, 3), non_empty, (0, 0), (7, 6))
        >>> search_opposite_state(*(args + ('D', )))
        Cell(row=8, col=3)

        >>> args = (True, Cell(10, 3), non_empty, (0, 0), (7, 6))
        >>> search_opposite_state(*(args + ('U', )))
        Cell(row=10, col=3)

        >>> args = (False, Cell(10, 10), non_empty, (0, 0), (7, 6))
        >>> search_opposite_state(*(args + ('UL', )))
        Cell(row=7, col=6)

        >>> non_empty = np.array(\
            [[True, True, True],\
             [True, True, True],\
             [True, True, True]])
        >>> args = (True, Cell(0, 2), non_empty, (0, 0), (2, 2))
        >>> search_opposite_state(*(args + ('LD', )))
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
                if no_empty[c1[0], c1[1]] != state:
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


def search_same_state(state, cell, no_empty, up, dn, moves):
    """

    :param cell:
    :param no_empty:
    :param sheet:
    :param directions:
    :return:

    Examples::
        >>> no_empty = np.array(\
            [[False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False,  True,  True,  True],\
             [False, False, False,  True, False, False,  True],\
             [False, False, False,  True,  True,  True,  True]])
        >>> args = (True, Cell(7, 6), no_empty, (0, 0), (7, 6))
        >>> search_same_state(*(args + ('UL', )))
        Cell(row=5, col=3)

        >>> search_same_state(*(args + ('U', )))
        Cell(row=5, col=6)

        >>> search_same_state(*(args + ('L', )))
        Cell(row=7, col=3)

        >>> args = (True, Cell(5, 3), no_empty, (0, 0), (7, 6))
        >>> search_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> args = (False, Cell(5, 3), no_empty, (0, 0), (7, 6))
        >>> search_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> search_same_state(*(args + ('UL', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=5, col=3) with movement(U)

        >>> args = (True, Cell(5, 6), no_empty, (0, 0), (7, 6))
        >>> search_same_state(*(args + ('DL', )))
        Cell(row=7, col=4)

    """

    c1 = list(cell)

    for mv in moves:
        c = search_opposite_state(state, cell, no_empty, up, dn, mv, True)
        dis = _primitive_dir[mv]
        c1 = [i if not k == 0 else j for i, j, k in zip(c, c1, dis)]
    return Cell(*c1)


def extend_range(state, xl_range, no_empty, rng_ext):
    """

    :param state:
    :param up:
    :param dn:
    :param rng:
    :param no_empty:
    :param rng_ext:
    :return:

    Example::

        >>> no_empty = np.array(\
            [[False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False,  True,  True,  True],\
             [False, False, False,  True, False, False,  True],\
             [False, False, False,  True,  True,  True,  True]])


        >>> rng = (Cell(row=6, col=3), Cell(row=6, col=3))
        >>> rng_ext = [repeat_moves('U', times=10)]
        >>> extend_range(True, rng, no_empty, rng_ext)
        [Cell(row=6, col=3), Cell(row=6, col=3)]

        >>> rng = (Cell(row=6, col=3), Cell(row=7, col=3))
        >>> rng_ext = [repeat_moves('R', times=10)]
        >>> extend_range(True, rng, no_empty, rng_ext)
        [Cell(row=6, col=3), Cell(row=7, col=6)]

        >>> rng = (Cell(row=6, col=3), Cell(row=10, col=3))
        >>> rng_ext = [repeat_moves('R', times=10)]
        >>> extend_range(True, rng, no_empty, rng_ext)
        [Cell(row=6, col=3), Cell(row=10, col=6)]

        >>> rng = (Cell(row=6, col=5), Cell(row=6, col=5))
        >>> rng_ext = [repeat_moves('LURD')]
        >>> extend_range(True, rng, no_empty, rng_ext)
        [Cell(row=5, col=3), Cell(row=7, col=6)]

    """
    _m = {
        'L': (0, 1),
        'U': (0, 1),
        'R': (1, 0),
        'D': (1, 0)
    }
    xl_range = [np.array(v) for v in xl_range]
    for moves in rng_ext:
        for directions in moves:
            flag = True
            for d in directions:
                mv = _primitive_dir[d]
                i, j = _m[d]
                st, nd = (xl_range[i], xl_range[j])
                st = st + mv
                nd = [p2 if k == 0 else p1 for p1, p2, k in zip(st, nd, mv)]
                if i == 1:
                    v = no_empty[nd[0]:st[0] + 1, nd[1]:st[1] + 1]
                else:
                    v = no_empty[st[0]:nd[0] + 1, st[1]:nd[1] + 1]
                if (not v.size and state) or (v != state).all():
                    continue
                xl_range[i] = st
                flag = False

            if flag:
                break

    return [Cell(*v) for v in xl_range]


def get_range(no_empty, up, dn, xl_margins, indices, st_cell, nd_cell=None, rng_ext=None):
    """

    :param sheet:
    :type sheet: xlrd.sheet.Sheet
    :param st_cell:
    :type st_cell: CellPos
    :param nd_cell:
    :type nd_cell: CellPos
    :param rng_ext:
    :param epoch1904:
    :return:

    Example::

        >>> no_empty = np.array(\
            [[False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False,  True,  True,  True],\
             [False, False, False,  True, False, False,  True],\
             [False, False, False,  True,  True,  True,  True]])

        >>> up, dn = ((0, 0), (7, 6))
        >>> xl_margins, ind = get_xl_abs_margins(no_empty)
        >>> st_cell = CellPos(Cell(0, 0), 'DR')
        >>> nd_cell = CellPos(Cell(CELL_REL, CELL_REL), 'DR')
        >>> get_range(no_empty, up, dn, xl_margins, ind, st_cell, nd_cell)
        (Cell(row=6, col=3), Cell(row=7, col=3))

        >>> nd_cell = CellPos(Cell(7, 6), 'UL')
        >>> get_range(no_empty, up, dn, xl_margins, ind, st_cell, nd_cell)
        (Cell(row=5, col=3), Cell(row=6, col=3))
    """

    st = set_start_cell(st_cell.cell, xl_margins)
    try:
        state = no_empty[st]
    except IndexError:
        state = False

    if st_cell.mov is not None:
        st = search_opposite_state(state, st, no_empty, up, dn, st_cell.mov)
        state = not state

    if nd_cell is None:
        nd = Cell(*st)
    else:
        nd = set_start_cell(nd_cell.cell, xl_margins, st)

        if nd_cell.mov is not None:
            mov = nd_cell.mov
            if state == no_empty[nd]:
                nd = search_same_state(state, nd, no_empty, up, dn, mov)
            else:
                nd = search_opposite_state(
                    not state, nd, no_empty, up, dn, mov)

        c = np.array([st, nd])

        st, nd = (Cell(*list(c.min(0))), Cell(*list(c.max(0))))

    if rng_ext is None:
        return (st, nd)
    else:
        return extend_range(state, (st, nd), no_empty, rng_ext)


def parse_cell(cell, epoch1904=False):
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

    Example::

        >>> import xlrd
        >>> from xlrd.sheet import Cell
        >>> parse_cell(Cell(xlrd.XL_CELL_NUMBER, 1.2))
        1.2

        >>> parse_cell(Cell(xlrd.XL_CELL_DATE, 1.2))
        datetime.datetime(1900, 1, 1, 4, 48)

        >>> parse_cell(Cell(xlrd.XL_CELL_TEXT, 'hi'))
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

    Example::

        >>> import os, tempfile, xlrd, pandas as pd
        >>> os.chdir(tempfile.mkdtemp())
        >>> df = pd.DataFrame([[None, None, None], [5.1, 6.1, 7.1]])
        >>> tmp = 'sample.xlsx'
        >>> writer = pd.ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

        >>> sheet = xlrd.open_workbook(tmp).sheet_by_name('Sheet1')

        >>> xl_margins, indices = get_xl_abs_margins(get_no_empty_cells(sheet))

        # minimum matrix in the sheet
        >>> st = set_start_cell(Cell(CELL_UL, CELL_UL), xl_margins)
        >>> nd = set_start_cell(Cell(CELL_DR, CELL_DR), xl_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [[None, 0, 1, 2],
         [0, None, None, None],
         [1, 5.1, 6.1, 7.1]]

        # get single value
        >>> get_xl_table(sheet, (Cell(6, 3), Cell(6, 3)), indices)
        [0]

        # get column vector
        >>> st = set_start_cell(Cell(0, 3), xl_margins)
        >>> nd = set_start_cell(Cell(CELL_DR, 3), xl_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> st = set_start_cell(Cell(5, 0), xl_margins)
        >>> nd = set_start_cell(Cell(5, CELL_DR), xl_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2]

        # get row vector
        >>> st = set_start_cell(Cell(5, 0), xl_margins)
        >>> nd = set_start_cell(Cell(5, 10), xl_margins)
        >>> get_xl_table(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2, None, None, None, None]

    """
    tbl = []
    for r in range(xl_range[0].row, xl_range[1].row + 1):
        row = []
        tbl.append(row)
        for c in range(xl_range[0].col, xl_range[1].col + 1):
            if [r, c] in indices:
                row.append(parse_cell(sheet.cell(r, c), epoch1904))
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


def redim_xl_range(value, dim_min, dim_max=None):
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

    Example::

        >>> redim_xl_range([1, 2], 2)
        [[1, 2]]

        >>> redim_xl_range([[1, 2]], 1)
        [[1, 2]]

        >>> redim_xl_range([[1, 2]], 1, 1)
        [1, 2]

        >>> redim_xl_range([], 2)
        [[]]

        >>> redim_xl_range([[1, 2]], 0, 0)
        Traceback (most recent call last):
        ValueError: value cannot be reduced of -2

    """
    val_dim = _get_value_dim(value)
    try:
        if val_dim < dim_min:
            return _redim_value(value, dim_min - val_dim)
        elif dim_max is not None and val_dim > dim_max:
            return _redim_value(value, dim_max - val_dim)
        return value
    except:
        raise ValueError('value cannot be reduced of %d' % (dim_max - val_dim))


def process_xl_table(value, type=None, args=[], kwargs={}, filters=None):
    """
    Processes the output value of get_rect_range function.

    :param value: matrix or vector or value
    :type value: list of lists, list, value

    :param type: reference type
    :type type: str, None

    :param args: additional arguments for the construction function
    :type args: list, optional

    :param kwargs: additional key=value arguments for the construction function
    :type kwargs: dict, optional

    :param filters:
    :type filters: list, optional

    :return: processed output value
    :rtype: given type, or list of lists, list, value

    Example::

        >>> value = [[1, 2], [3, 4], [5, 6]]
        >>> res = process_xl_table(value, type='dict')
        >>> sorted(res.items())
        [(1, 2),
         (3, 4),
         (5, 6)]

        >>> value = [[1, 9], [8, 10], [5, 11]]
        >>> process_xl_table(value, 
        ...     filters=[{'type':'sorted', 'kwargs':{'reverse': True}}])
        [[8, 10],
         [5, 11],
         [1, 9]]
    """
    val = _function_types[type]['fun'](value, *args, **kwargs)
    if filters:
        for v in filters:
            val = process_xl_table(val, **v)
    return val
