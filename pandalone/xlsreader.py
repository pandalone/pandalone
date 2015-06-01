#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
A mini-language to capture rectangular areas with non-empty cells from Excel-sheets.

.. default-role:: term


Introduction
============

The purpose of this module is to use simple traversal operations to
`capture` a rectangular area from excel-sheets when its exact position
is not known beforehand.

The `capturing` depends only on the full/empty `state` of the cells,
and not on their values . Use another library (i.e. "pandas") to examine
the values of the `capture-rect` afterwards.  Nevertheless, the `xl-ref` syntax
provides for specifying `filter` transformations at the end, for setting
the dimensionality and the final type of the captured values.

It is based on **xlrd** library but is also checked for compatibility with
**xlwings** *COM-client* library.


Excel-ref Syntax
----------------
::

    <1st-ref>[:[<2nd-ref>][:<expansions>]][<filters>]
    :


Annotated Syntax
----------------
::

    target-moves────────┐
    landing-cell─────┐  │
                    ┌┤ ┌┤
                    A1(RD):..(RD):L1DR:{"type": "df", "kws": {"header": false}}
                    └─┬──┘ └─┬──┘ └┬─┘ └───────────────┬─────────────────────┘
    1st-target-ref────┘      │     │                   │
    2nd-target-ref───────────┘     │                   │
    rect-expansions────────────────┘                   │
    filters────────────────────────────────────────────┘

Which means:

    1. Identify the first `full-cell` beyond ``A1`` as `target`, by traversing
       right and down;
    2. continue from this point right-down `targeting` the `2nd-ref`;
    3. `capture` the cells between the targets.
    4. try `expansions` on the `capture-rect`, once to the left column
       and then down and right until a full-empty line/row is met, respectively;
    5. finally `filter` captured cells to wrap them up in a pandas DataFrame.


Examples
--------
A typical example is `capturing` a table with a "header" row and
an "index" column.
Below there are (at least) 3 ways to do it, beyond specifying
the exact `coordinates`::


      A B C D E      Β2:E4          ## Exact referencing.
    1  ┌───────┐     ^^.__          ## From top-left full-cell to bottom-right.
    2  │  X X X│     A1(DR):__:U1   ## Start from A1 and move down then right
    3  │X X X X│                    #    until B3; capture till bottom-left;
    3  │X X X X│                    #    expand once upwards (to header row).
    4  │X X X X│     A1(RD):__:L1   ## Start from A1 and move down by row
       └───────┘                    #    until C4; capture till bottom-left;
                                    #    expand once left (to index column).

Note that if ``B1`` were full, the results would still be the same, because
``?`` expands only if any full-cell found in row/column.

In case the bottom-left cell of the sheet does not coincide with table-end,
only the later 2 `xl-ref` would work.
But we can use `dependent` referencing for the `2nd-ref` and start
from the end of the table::

      A B C D E   _^:..(LD):L1      ## Start from top-right full-cell (D2) and
    1  ┌─────┐                      #    move left-down till 1st empty-cell met
    2  │  X X│                      #    at exterior row/col (C3, regardless of
    3  │X X X│                      #    of col/row order); expand once left.
    3  │X X X│    ^_(U):..(UR):U1   ## Target 1st cell starting from B4 and
       └─────┘                      #    moving up; capture till D3;
    4         Χ                     #    expand up once.


In the presence of `empty-cell` breaking the `exterior` row/column of
the `1st-landing-cell`, the capturing becomes more intricate::

      A B C D E
    1  ┌─────┐      Β2:D_
    2  │  X X│      A1(RD):..(RD):L1D
    3  │X X  │      C_:^^
    3  │X    │      A^(DR):C_:U
    4  │  X  │X
       └─────┘


      A B C D E
    1    ┌───┐      ^^(RD):..(RD)
    2    │X X│      _^(R):^.(DR)
    3   X│X  │
         └───┘
    3   X
    4     X   X


      A B C D E
    1  ┌───┐        Β2:C4
    2  │  X│X       A1(RD):^_
    3  │X X│        C_:^^
    3  │X  │        A^(DR):C_:U
    4  │  X│  X     ^^(RD):..(D):D
       └───┘


.. seealso:: Example spreadsheet: :download:`xls_ref.xlsx`


Definitions
===========

.. glossary::

    xl-url
        Any url with its fragment abiding to the `xl-ref` syntax.
        Its file-part should resolve to an excel-file.

    xl-ref
        The syntax for `capturing` rects from excel-sheets,
        specified within the fragment-part of a `xl-url`.

        It is composed of 2 `target-ref` followed by `expansions` and `filters`.

    target-ref
        A pair of row/col cell `coordinates` optionally followed by
        a parenthesized `target-moves`.
        It actually specifies 2 cells, `landing-cell` and `target-cell`.

    landing-cell
        The cell identified by the `coordinates` of the `target-ref` alone.

    target
    target-cell
        The cell identified after applying `target-moves` on the `landing-cell`.
        Failure to identify a target-cell raises an error.

    targeting
        The search for the `target-cell` starts from the `landing-cell`,
        follows the specified `target-moves`, and ends when a `state-change`
        is detected on an `exterior` column or row, according to the enacted
        `termination-rule`.

    target-moves
        Specify the traversing order while `targeting` from `landing-cell`,
        based on `primitive-directions` pairs; so ``DR`` means:

            *"Start going right, column-by-column, traversing each column
            from top to bottom."*

        The pairs ``UD`` and ``LR`` (and their inverse) are invalid.

    directions
    primitive-directions
        The 4 *primitive-directions* that are denoted with one of the letters
        ``LURD``.

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
        row/column and detecting a `state-change`.
        The are 2 rules: `search-same` and `search-opposite`.

        .. seealso::
            Check `Target-termination rules`_ for the enactment of the rules.

    search-same
        The `target-cell` is the LAST cell with the SAME `state` as
        the `landing-cell`, while `targeting` from it.

    search-opposite
        The `target-cell` is the FIRST cell with OPPOSITE `state` from
        the `landing-cell`, while `targeting` from it.

    exterior
        The *column* and the *row* of the `1st-landing-cell`;
        the `termination-rule` gets to be triggered only by `state-change`
        on them.

    coordinates
        The column/row pair of a `landing-cell`, either in ``A1`` format or
        zero-based ``(row, col)`` tuple.

    coordinate
        Either a *cell-column* (in letters) or *cell-row* (number);
        each one might be `absolute` or `dependent`, independently.

    absolute
        Any cell row/col identified with column-characters, row-numbers, or
        the following special-characters:

        - ``^``          The top/Left full-cell `coordinate`.
        - ``_``          The bottom/right full-cell `coordinate`.

    dependent
        Any `2nd-ref` `coordinate` identified with a dot(``.``),
        which means that:

            *"2nd-landing-cell coordinate = 1st target-cell coordinate"*

        The `2nd-ref` might contain a "mix" of `absolute` and *dependent*
        coordinates.

    capture
    capturing
        The *capturing* procedure involves `targeting` the `1st-ref` and
        the `2nd-ref`, and then extracting the values from the
        *rectangular area* between the 2 `target` cells.

        Note that the result-values would always be the same, irrespective of
        row-by row or column-by-column traversal order.

    capture-rect
        The result values from the `capturing`

    1st-ref
    1st-landing-cell
    1st-target-cell
        The `capturing` STARTS from the `target` of *this* `target-ref`.
        It supports `absolute` coordinates only.

    2nd-ref
    2nd-landing-cell
    2nd-target-cell
        The `capturing` STOPS at the `target` of this `target-ref`.
        It supports both `absolute` coordinates, and `dependent` ones from the
        `1st-target-cell`.

    expansions
    rect-expansions
        Due to `state-change` on the 'exterior' cells the `capture-rect`
        might be smaller that a wider contigious but "convex" rectangular area.

        The * expansions* attempt to remedy this by providing for expanding on
        arbitrary `directions` accompanied by a multiplicity for each one.
        If multiplicity is unspecified, infinite assumed, so it expands
        until an empty/full row/column is met.

    filter
    filters
    filter-function
        Predefined functions to apply for transforming the cell-values of
        `capture-rect` specified as nested **json** objects.


Details
=======

Target-moves
---------------

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
the `landing-cell`. For instance, given this xl-sheet below, there are
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


Capturing
---------

To specify a complete `capture-rect` we need to identify a 2nd cell.
The 2nd target-cell may be specified:

  - either with `absolute` coordinates, as above, or
  - with `dependent` coords, using the dot(``.``) to refer to the 1st cell.


In the above example-sheet, here are some ways to specify refs::

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
   Of course, the above rects WILL FAIL since the `target-moves`
   will stop immediately due to ``X`` values being surrounded by empty-cells.

   But the above diagram was to just convey the general idea.
   To make it work, all the in-between cells of the peripheral row and columns
   should have been also non-empty.

.. Note::
    The `capturing` moves from `1st-target-cell` to `2nd-target-cell` are
    independent from the implied `target-moves` in the case of `dependent`
    coords.

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

  - If the `state` of the `2nd-landing-cell` == `1st-target-cell`:
    - Use `search-same` to identify target.

  - Otherwise:
    - Use `search-opposite` to identify target.



Expansions
----------

Captured-rects ("values") may be limited due to empty-cells in the 1st
row/column traversed.  To overcome this, the xl-ref may specify `expansions`
directions using a 3rd ``:``-section like that::

    _5(L):1_(UR):RDL1U1

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
       └┼─────────┼┴──► A1(RD):..(RD):DRL1
    6   │X        │
        └─────────┴───► A1(RD):..(RD):L1DR       A_(UR):^^(RD)
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


import itertools as itt
import numpy as np
import pandas as pd
from six.moves.urllib.parse import urldefrag  # @UnresolvedImport
from six.moves.urllib.request import urlopen  # @UnresolvedImport


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
            (?P<rect_exp>[LURD?\d]+)                     # rect expansion [opt]
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


TargetRef = namedtuple('TargetRef', ['cell', 'mov'])


def _row2num(coord):
    """
    Converts the Excel `str` row to a zero-based `int`, reporting invalids.

    :param str, int coord:  excel-row coordinate or one of ``^_.``

    :return:    excel row number, >= 0
    :rtype:     int


    Examples::

        >>> _row2num('1')
        0

        >>> _row2num('10') == _row2num(10)
        True

        ## "Special" cells are also valid.
        >>> _row2num('_'), _row2num('^')
        ('_', '^')

        >>> _row2num('0')
        Traceback (most recent call last):
        ValueError: Invalid row('0')!

        >>> _row2num('a')
        Traceback (most recent call last):
        ValueError: Invalid row('a')!

        >>> _row2num(None)
        Traceback (most recent call last):
        ValueError: Invalid row(None)!
    """

    if coord in _special_coords:
        return coord

    try:
        row = int(coord) - 1
        if row < 0:
            raise ValueError
        return row
    except Exception:
        raise ValueError('Invalid row({!r})!'.format(coord))


def _col2num(coord):
    """
    Converts the Excel `str` column to a zero-based `int`, reporting invalids.

    :param str coord:     excel-column coordinate or one of ``^_.``

    :return:    excel column number, >= 0
    :rtype:     int


    Examples::

        >>> _col2num('D')
        3

        >>> _col2num('d')
        3

        >>> _col2num('AaZ')
        727

        ## "Special" cells are also valid.
        >>> _col2num('_'), _col2num('^')
        ('_', '^')

        >>> _col2num(None)
        Traceback (most recent call last):
        ValueError: Invalid column(None)!

        >>> _col2num('4')
        Traceback (most recent call last):
        ValueError: Invalid column('4')!

        >>> _col2num(4)
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


def _make_TargetRef(cell_col, cell_row, cell_mov):
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
    :rtype: TargetRef


    Examples::
        >>> _make_TargetRef('A', '1', 'R')
        TargetRef(cell=Cell(row=0, col=0), mov='R')

        >>> _make_TargetRef('^', '^', 'R').cell
        Cell(row='^', col='^')

        >>> _make_TargetRef('_', '_', 'L').cell
        Cell(row='_', col='_')

        >>> _make_TargetRef('.', '.', 'D').cell
        Cell(row='.', col='.')

        >>> _make_TargetRef(None, None, None)

        >>> _make_TargetRef('1', '.', None)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='1', row='.') due to: Invalid column('1')!

        >>> _make_TargetRef('A', 'B', None)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='A', row='B') due to: Invalid row('B')!

        >>> _make_TargetRef('A', '1', 12)
        Traceback (most recent call last):
        ValueError: Invalid cell(col='A', row='1') due to:
            'int' object has no attribute 'upper'
    """

    try:
        if cell_col == cell_row == cell_mov is None:
            return None
        else:
            row = _row2num(cell_row)
            col = _col2num(cell_col)
            mov = cell_mov.upper() if cell_mov else None

            return TargetRef(cell=Cell(col=col, row=row), mov=mov)

    except Exception as ex:
        msg = 'Invalid cell(col={!r}, row={!r}) due to: {}'
        raise ValueError(msg.format(cell_col, cell_row, ex))


def _repeat_moves(moves, times=None):
    """
    Returns an iterator that repeats the primitive-directions for a N times.

    :param str moves: string with a sequence of primitive directions

    :param str times: N of repetitions. If None it means infinite repetitions.

    :return:
        An iterator.
    :rtype: iterator

    Examples::

         >>> list(_repeat_moves('LUR', '3'))
         ['LUR', 'LUR', 'LUR']

         >>> list(_repeat_moves('LUR', '0'))
         []

         >>> _repeat_moves('LUR')  # infinite repetitions
         repeat('LUR')
     """

    args = (moves,)
    if times is not None:
        args += (int(times), )
    return itt.repeat(*args)


def _parse_rect_expansions(rect_exp):
    """
    Parse rect-expansion into a list of dir-letters iterable.

    :param rect_exp:
        A string with a sequence of primitive moves:
        es. L1U1R1D1
    :type rect_exp: str

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

    :param xl_ref:
        a string with the following format:
        <sheet>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):
        <rect_exp>{<json>}
        es. sheet!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}
    :type xl_ref: str

    :return:
        dictionary containing the following parameters::

        - sheet: str
        - st_ref: (TargetRef) the 1st-ref raw, with strings
        - nd_ref: (TargetRef) the 2nd-ref, raw, with strings
        - rect_exp: (list) parsed as list of iterable-letters
        - json: parsed

    :rtype: dict


    Examples::
        >>> from itertools import chain
        >>> xl_ref = 'Sheet1!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}'
        >>> res = parse_xl_ref(xl_ref)

        >>> res['sheet']
        'Sheet1'

        >>> res['st_ref']
        TargetRef(cell=Cell(row=0, col=0), mov='DR')

        >>> res['nd_ref']
        TargetRef(cell=Cell(row=19, col=25), mov='UL')

        >>> list(chain(*res['rect_exp']))
        ['L', 'U', 'U', 'R', 'D']

        >>> res['json'] == {'json': '...'}
        True
    """

    try:
        m = _re_xl_ref_parser.match(xl_ref)
        if not m:
            raise ValueError('Syntax not matched.')
        gs = m.groupdict()

        js = gs['json']
        gs['json'] = json.loads(js) if js else None

        rect = gs['rect_exp']
        gs['rect_exp'] = _parse_rect_expansions(rect) if rect else None

        p = gs.pop

        # fetch 1st cell
        gs['st_ref'] = _make_TargetRef(p('st_col'), p('st_row'), p('st_mov'))

        # fetch 2nd cell
        gs['nd_ref'] = _make_TargetRef(p('nd_col'), p('nd_row'), p('nd_mov'))

        return gs

    except Exception as ex:
        log.debug("Invalid xl-ref(%s) due to: %s", xl_ref, ex, exc_info=1)
        raise ValueError("Invalid xl-ref({}) due to: ".format(xl_ref, ex))


def parse_xl_url(url):
    """
    Parses the contents of an excel url.

    :param str url:
        a string with the following format::

            <url_file>#<sheet>!<1st_ref>:<2nd_ref>:<expand><json>

        Example::

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
         ('nd_ref', TargetRef(cell=Cell(row='^', col='.'), mov='DR')),
         ('rect_exp', [repeat('L'), repeat('U', 1)]),
         ('sheet', 'Sheet1'),
         ('st_ref', TargetRef(cell=Cell(row=0, col=0), mov='UL')),
         ('url_file', 'file:///sample.xlsx')]

    """

    try:
        res = {}

        res['url_file'], frag = urldefrag(url)  # parse excel url

        res.update(parse_xl_ref(frag))  # resolve excel reference

        return res

    except Exception as ex:
        raise ValueError("Invalid xl-url({}) due to: {}".format(url, ex))


def get_sheet_margins(full_cells):
    """
    Returns upper and lower absolute positions.

    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :return:  a 2-tuple with:

              - a `Cell` with zero-based margins for rows/cols,
              - indices for full-cells
    :rtype: tuple

    Examples::

        >>> full_cells = [
        ...    [0, 0, 0],
        ...    [0, 1, 0],
        ...    [0, 1, 1],
        ...    [0, 0, 1],
        ... ]
        >>> sheet_margins, indices = get_sheet_margins(full_cells)
        >>> row, col = sheet_margins
        >>> sorted(row.items())
        [('^', 1), ('_', 3)]

        >>> sorted(col.items())
        [('^', 1), ('_', 2)]

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

    indices = np.array(np.where(full_cells)).T
    up_r, up_c = indices.min(0)
    dn_r, dn_c = indices.max(0)
    sheet_margins = Cell(
        row={'^': up_r, '_': dn_r},
        col={'^': up_c, '_': dn_c})
    return sheet_margins, indices.tolist()


def _get_abs_coord(coord, coord_margins, pcoord=None):
    """
    Translates any special or dependent coord to absolute ones.

    :param int, str coord:    the coord to translate
    :param int, str coord_margins:    the coord to translate
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


def _resolve_cell(cell, sheet_margins, pcell=None):
    """
    Translates any special coords to absolute ones.

    :param Cell cell:    The cell to translate its coords.
    :param Cell pcell:   The cell to base any dependent coords (``.``).
    :rtype: Cell

    Examples::

        >>> margins = Cell(
        ...     row={'^':1, '_':10},
        ...     col={'^':2, '_':6})
        >>> _resolve_cell(Cell(4, 5), margins)
        Cell(row=4, col=5)

        >>> _resolve_cell(Cell('^', '^'), margins)
        Cell(row=1, col=2)

        >>> _resolve_cell(Cell('_', '_'), margins)
        Cell(row=10, col=6)
    """

    row = _get_abs_coord(cell.row, sheet_margins.row, pcell and pcell.row)
    col = _get_abs_coord(cell.col, sheet_margins.col, pcell and pcell.col)

    return Cell(row=row, col=col)


def _target_opposite_state(state, cell, full_cells, dn, moves, last=False):
    """
    Returns the FIRST `cell` with OPPOSITE `state` of the starting-state.

    :param bool state: the starting-state
    :param Cell cell: landing-cell
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :param tuple dn: bottom sheet limits
    :param str moves: string with a sequence of primitive directions
    :param bool last: If True `_target_opposite_state` returns the last cell
    that has the same state of the `starting-state`.
    :rtype: Cell

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
        >>> args = (False, Cell(1, 1), full_cells, (7, 6))
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

        >>> args = (True, Cell(6, 3), full_cells, (7, 6))
        >>> _target_opposite_state(*(args + ('D', )))
        Cell(row=8, col=3)

        >>> args = (True, Cell(10, 3), full_cells, (7, 6))
        >>> _target_opposite_state(*(args + ('U', )))
        Cell(row=10, col=3)

        >>> args = (False, Cell(10, 10), full_cells, (7, 6))
        >>> _target_opposite_state(*(args + ('UL', )))
        Cell(row=7, col=6)

        >>> full_cells = np.array([
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ...     [1, 1, 1],
        ... ])
        >>> args = (True, Cell(0, 2), full_cells, (2, 2))
        >>> _target_opposite_state(*(args + ('LD', )))
        Cell(row=3, col=2)
    """

    up = (0, 0)
    mv = _primitive_dir[moves[0]]  # first move
    c0 = np.array(cell)

    if not state:
        if not c0[0] <= dn[0] and 'U' in moves:
            c0[0] = dn[0]
        if not c0[1] <= dn[1] and 'L' in moves:
            c0[1] = dn[1]

    flag = False
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


def _target_same_state(state, cell, full_cells, dn, moves):
    """
    Returns the LAST `cell` with SAME `state` of the starting-state.

    :param bool state: the starting-state
    :param Cell cell: landing-cell
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :param tuple dn: bottom sheet limits
    :param str moves: string with a sequence of primitive directions
    :rtype: Cell

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
        >>> args = (True, Cell(7, 6), full_cells, (7, 6))
        >>> _target_same_state(*(args + ('UL', )))
        Cell(row=5, col=3)

        >>> _target_same_state(*(args + ('U', )))
        Cell(row=5, col=6)

        >>> _target_same_state(*(args + ('L', )))
        Cell(row=7, col=3)

        >>> args = (True, Cell(5, 3), full_cells, (7, 6))
        >>> _target_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> args = (False, Cell(5, 3), full_cells, (7, 6))
        >>> _target_same_state(*(args + ('DR', )))
        Cell(row=5, col=3)

        >>> _target_same_state(*(args + ('UL', )))
        Traceback (most recent call last):
        ValueError: Invalid Cell(row=5, col=3) with movement(U)

        >>> args = (True, Cell(5, 6), full_cells, (7, 6))
        >>> _target_same_state(*(args + ('DL', )))
        Cell(row=7, col=4)

    """
    up = (0, 0)
    c1 = list(cell)

    for mv in moves:
        c = _target_opposite_state(state, cell, full_cells, dn, mv, True)
        dis = _primitive_dir[mv]
        c1 = [i if not k == 0 else j for i, j, k in zip(c, c1, dis)]
    return Cell(*c1)


def _expand_rect(state, xl_rect, full_cells, rect_exp):
    """

    :param bool state: The starting-state.
    :param tuple xl_rect: A ``(Cell, Cell)`` with the 1st and 2nd bouding-cells
    of the rect.
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`get_full_cells()`.
    :param list rect_exp: A list of primitive-dir chains.
    :rtype: tuple


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
        >>> rect_exp = [_repeat_moves('U', times=10)]
        >>> _expand_rect(True, rng, full_cells, rect_exp)
        [Cell(row=6, col=3), Cell(row=6, col=3)]

        >>> rng = (Cell(row=6, col=3), Cell(row=7, col=3))
        >>> rect_exp = [_repeat_moves('R', times=10)]
        >>> _expand_rect(True, rng, full_cells, rect_exp)
        [Cell(row=6, col=3), Cell(row=7, col=6)]

        >>> rng = (Cell(row=6, col=3), Cell(row=10, col=3))
        >>> rect_exp = [_repeat_moves('R', times=10)]
        >>> _expand_rect(True, rng, full_cells, rect_exp)
        [Cell(row=6, col=3), Cell(row=10, col=6)]

        >>> rng = (Cell(row=6, col=5), Cell(row=6, col=5))
        >>> rect_exp = [_repeat_moves('LURD')]
        >>> _expand_rect(True, rng, full_cells, rect_exp)
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
                    v = full_cells[nd[0]:st[0] + 1, nd[1]:st[1] + 1]
                else:
                    v = full_cells[st[0]:nd[0] + 1, st[1]:nd[1] + 1]
                if (not v.size and state) or (v != state).all():
                    continue
                xl_rect[i] = st
                flag = False

            if flag:
                break

    return [Cell(*v) for v in xl_rect]


def resolve_capture_rect(full_cells, sheet_margins, st_ref,
                         nd_ref=None, rect_exp=None):
    """
    Captures the bouding-cells of the rect.

    :param ndarray full_cells:
            A boolean ndarray with `False` wherever cell are
            blank or empty. Use :func:`get_full_cells()`.
    :param TargetRef st_ref: `1st-target-ref`.
    :param TargetRef nd_ref: `2nd-target-ref`.
    :param rect_exp: A list of primitive-dir chains.

    :return: A ``(Cell, Cell)`` with the 1st and 2nd bouding-cells of the rect.
    :rtype: tuple

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
        >>> sheet_margins, _ = get_sheet_margins(full_cells)
        >>> st_ref = TargetRef(Cell(0, 0), 'DR')
        >>> nd_ref = TargetRef(Cell('.', '.'), 'DR')
        >>> resolve_capture_rect(full_cells, sheet_margins,
        ...         st_ref, nd_ref)
        (Cell(row=6, col=3), Cell(row=7, col=3))

        >>> nd_ref = TargetRef(Cell(7, 6), 'UL')
        >>> resolve_capture_rect(full_cells, sheet_margins,
        ...         st_ref, nd_ref)
        (Cell(row=5, col=3), Cell(row=6, col=3))
    """
    dn = (sheet_margins[0]['_'], sheet_margins[1]['_'])

    st = _resolve_cell(st_ref.cell, sheet_margins)
    try:
        state = full_cells[st]
    except IndexError:
        state = False

    if st_ref.mov is not None:
        st = _target_opposite_state(state, st, full_cells, dn, st_ref.mov)
        state = not state

    if nd_ref is None:
        nd = Cell(*st)
    else:
        nd = _resolve_cell(nd_ref.cell, sheet_margins, st)

        if nd_ref.mov is not None:
            mov = nd_ref.mov
            if state == full_cells[nd]:
                nd = _target_same_state(state, nd, full_cells, dn, mov)
            else:
                nd = _target_opposite_state(
                    not state, nd, full_cells, dn, mov)

        c = np.array([st, nd])

        st, nd = (Cell(*list(c.min(0))), Cell(*list(c.max(0))))

    if rect_exp is None:
        return st, nd
    else:
        return _expand_rect(state, (st, nd), full_cells, rect_exp)


def read_rect_values(sheet, xl_rect, indices, epoch1904=False):
    """
    Reads and parses excel sheet values of the `rect`.

    :param sheet: xlrd sheet.
    :type sheet: xlrd.sheet.Sheet
    :param tuple xl_rect: A ``(Cell, Cell)`` with the 1st and 2nd bouding-cells
    of the rect.
    :param indices: Indices for full-cells.
    :param epoch1904:
        Which date system was in force when this file was last saved.
        False => 1900 system (the Excel for Windows default).
        True => 1904 system (the Excel for Macintosh default).
    :type epoch1904: bool

    :return: A table with excel sheet values of `xl_rect`.
    :rtype: list

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
        >>> st = _resolve_cell(Cell('^', '^'), sheet_margins)
        >>> nd = _resolve_cell(Cell('_', '_'), sheet_margins)
        >>> read_rect_values(sheet, (st, nd), indices)
        [[None, 0, 1, 2],
         [0, None, None, None],
         [1, 5.1, 6.1, 7.1]]

        # get single value
        >>> read_rect_values(sheet, (Cell(6, 3), Cell(6, 3)), indices)
        [0]

        # get column vector
        >>> st = _resolve_cell(Cell(0, 3), sheet_margins)
        >>> nd = _resolve_cell(Cell('_', 3), sheet_margins)
        >>> read_rect_values(sheet, (st, nd), indices)
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> st = _resolve_cell(Cell(5, 0), sheet_margins)
        >>> nd = _resolve_cell(Cell(5, '_'), sheet_margins)
        >>> read_rect_values(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2]

        # get row vector
        >>> st = _resolve_cell(Cell(5, 0), sheet_margins)
        >>> nd = _resolve_cell(Cell(5, 10), sheet_margins)
        >>> read_rect_values(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2, None, None, None, None]

    """

    tbl = []
    st_target = xl_rect[0]
    nd_target = xl_rect[1]
    for r in range(st_target.row, nd_target.row + 1):
        row = []
        tbl.append(row)
        for c in range(st_target.col, nd_target.col + 1):
            if [r, c] in indices:
                row.append(_read_cell(sheet.cell(r, c), epoch1904))
            else:
                row.append(None)
    # vector
    if nd_target.col == st_target.col:
        tbl = [v[0] for v in tbl]

    # vector
    if nd_target.row == st_target.row:
        tbl = tbl[0]

    if isinstance(tbl, list):
        return tbl
    else:
        return [tbl]


def _get_table_dim(table):
    """
    Returns the table dimension.

    :param table: Matrix or vector or value.
    :type table: list of lists, list, value

    :rtype: int
    """
    try:
        if isinstance(table, list):
            return 1 + _get_table_dim(table[0])
    except IndexError:
        return 1
    return 0


def _redim_table(table, n):
    """
    Returns the reshaped table.

    :param table: Matrix or vector or value.
    :type table: list of lists, list, value

    :param n: Delta re-dimension.

    :rtype: list of lists, list, value
    """

    if n > 0:
        return [_redim_table(table, n - 1)]
    elif n < 0:
        if len(table) > 1:
            raise Exception
        return _redim_table(table[0], n + 1)
    return table


def redim_captured_table(table, dim_min, dim_max=None):
    """
    Reshapes the output table of :func:`read_rect_values()`.

    :param table: Matrix or vector or value.
    :type table: list of lists, list, value

    :param dim_min: Minimum dimension.
    :type dim_min: int, None

    :param dim_max: Maximum dimension.
    :type dim_max: int, None

    :return: Reshaped table.
    :rtype: list of lists, list, value

    Examples::

        >>> redim_captured_table([1, 2], 2)
        [[1, 2]]

        >>> redim_captured_table([[1, 2]], 1)
        [[1, 2]]

        >>> redim_captured_table([[1, 2]], 1, 1)
        [1, 2]

        >>> redim_captured_table([], 2)
        [[]]

        >>> redim_captured_table([[1, 2]], 0, 0)
        Traceback (most recent call last):
        ValueError: Cannot reduce Captured-values dimension(2) to (0, 0)!
    """

    val_dim = _get_table_dim(table)
    try:
        if val_dim < dim_min:
            return _redim_table(table, dim_min - val_dim)
        elif dim_max is not None and val_dim > dim_max:
            return _redim_table(table, dim_max - val_dim)
        return table
    except:
        # TODO: Make redimming use np-arrays.
        msg = 'Cannot reduce Captured-values dimension({}) to ({}, {})!'
        raise ValueError(msg.format(val_dim, dim_min, dim_max))


default_filters = {
    None: {'fun': lambda x: x},  # TODO: Actually redim_captured_values().
    'df': {'fun': pd.DataFrame},
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}


def process_captured_values(value, type=None, args=(), kws=None, filters=None,
                            available_filters=default_filters):
    """
    Processes the output value of :func:`read_rect_values()` function.

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


#### XLRD FUNCS ###

import xlrd
from xlrd import (xldate, XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_TEXT,
                  XL_CELL_BLANK, XL_CELL_ERROR, XL_CELL_BOOLEAN, XL_CELL_NUMBER,
                  open_workbook)

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    xlrd_0_9_3 = True
else:
    xlrd_0_9_3 = False


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


def _read_cell(cell, epoch1904=False):
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
        >>> _read_cell(Cell(xlrd.XL_CELL_NUMBER, 1.2))
        1.2

        >>> _read_cell(Cell(xlrd.XL_CELL_DATE, 1.2))
        datetime.datetime(1900, 1, 1, 4, 48)

        >>> _read_cell(Cell(xlrd.XL_CELL_TEXT, 'hi'))
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


# noinspection PyProtectedMember
def get_full_cells(sheet):
    """
    Returns a boolean ndarray with `False` wherever cell are blank or empty.
    """
    types = np.array(sheet._cell_types)
    return (types != XL_CELL_EMPTY) & (types != XL_CELL_BLANK)
