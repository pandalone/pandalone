#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
A mini-language to capture non-empty rectangular areas from Excel-sheets.

Introduction
============

This is a notation for url-fragments accompanied by operations that can
`capture` a rectangular area from excel-sheets using simple traversal operations
when the exact position of the area is not known beforehand.
The goal of this module is to make reading data from excel-workbooks
as practical as when reading from CSVs.

The `capturing` depends only on the full/empty `state` of the cells,
and not on their values . Use some other library (i.e. "pandas") to examine
the values of the `capture-rect` afterwards.  Nevertheless, the `xl-ref` syntax
provides for specifying `filter` transformations at the end, for setting
the dimensionality and the final type of the captured values.

It is based on `xlrd <http://www.python-excel.org/>`_ library but also
checked for compatibility with `xlwings <http://xlwings.org/quickstart/>`_
*COM-client* library.


Excel-ref Syntax
----------------
::

    <1st-edge>[:[<2nd-edge>][:<expansions>]][<filters>]
    :


Annotated Syntax
----------------
::

    target-moves──────┐
    landing-cell───┐  │
                  ┌┤ ┌┤
                  A1(RD):..(RD):L1DR:{"type": "df_num", "kws": {"header": false}}
                  └─┬──┘ └─┬──┘ └┬─┘ └───────────────────┬──────────────────────┘
    1st-edge────────┘      │     │                       │
    2nd-edge───────────────┘     │                       │
    expansions───────────────────┘                       │
    filters──────────────────────────────────────────────┘

Which means:

    1. `Target` the `1st` `edge` by identifying the first `full-cell`
       beyond ``A1`` as traversing right and down;
    2. continue from this point right-down `targeting` the `2nd` `edge`;
    3. `capture` the cells between the targets.
    4. try `expansions` on the `target-rect`, once to the left column
       and then down and right until a full-empty line/row is met, respectively;
    5. finally `filter` the values of the `capture-rect` to wrap them up
       in a pandas DataFrame with *numeric-conversion*.


API
---
- User-facing functionality:

  .. autosummary::

          num2a1_Cell
      parse_xl_ref
      parse_xl_url
      get_sheet_margins
      resolve_capture_rect
      read_capture_rect_values

- **xlrd** back-end functionality:

  .. autosummary::
      open_workbook
      open_sheet
      read_states_matrix
      read_cell

.. default-role:: term


Examples
--------
.. TODO::
    Provide example python-code for reading a `xl-ref`/`xl-url`.
    Till then, read the sources: :file:`tests/test_xlsreader.py`.

A typical case is when a sheet contains a single table with a "header"-row and
a "index"-column.
There are (at least) 3 ways to do it, beyond specifying
the exact `coordinates`::


      A B C D E
    1  ┌───────┐     Β2:E4          ## Exact referencing.
    2  │  X X X│     ^^.__          ## From top-left full-cell to bottom-right.
    3  │X X X X│     A1(DR):__:U1   ## Start from A1 and move down and right
    3  │X X X X│                    #    until B3; capture till bottom-left;
    4  │X X X X│                    #    expand once upwards (to header row).
       └───────┘     A1(RD):__:L1   ## Start from A1 and move down by row
                                    #    until C4; capture till bottom-left;
                                    #    expand once left (to index column).


Note that if ``B1`` were full, the results would still be the same, because
``?`` expands only if any full-cell found in row/column.

In case where the sheet contains more than one *disjoint* tables, the
bottom-left cell of the sheet would not coincide with table-end, so the handy
last two `xl-ref` above would not work.

For that we may resort to `dependent` referencing for the `2nd` `edge`, and
define its position in relation to the `1st` `target`::

      A B C D E
    1  ┌─────┐    _^:..(LD):L1      ## Start from top-right(E2) and target left
    2  │  X X│                      #    left(D2); from there capture left-down
    3  │X X X│                      #    till 1st empty-cell(C4, regardless of
    4  │X X X│                      #    col/row order); expand left once.
       └─────┘    ^_(U):..(UR):U1   ## Start from B5 and target 1st cell up;
    5         Χ                     #    capture from there till D3; expand up.


In the presence of `empty-cell` breaking the `exterior` row/column of
the `1st` `landing-cell`, the capturing becomes more intricate::

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
        A string describing how to `capture` rects from excel-sheets.

        It is composed of 2 `edge` references followed by `expansions` and
        `filters`.
        It is the fragment-part of a `xl-url`.

    edge
        An *edge* might signify:

        - the syntactic construct of the `xl-ref`, composed of a pair
          of row/column `coordinates`, optionally followed by parenthesized
          `target-moves`, like ``A1(LU)``;
        - the bounding cells of the `target-rect`;
        - the bounding cells of the `capture-rect`.

        In all cases above there are 2 instances, the `1st` and `2nd`.

    1st
    2nd
        It may refer to the *1st*/*2nd*:

        - `edge` of some `xl-ref`;
        - `landing-cell` of an `edge`;
        - `target-cell` of an `edge`;
        - `capture-cell` of a `capture-rect`.

        The *1st-edge` supports `absolute` `coordinates` only, while the
        *2nd-edge* supports also `dependent` ones from the *1st* `target-cell`.

    landing-cell
        The cell identified by the `coordinates` of the `edge` alone.

    target-cell
    target-rect
        The bounding *cell* identified after applying `target-moves` on the
        `landing-cell`.


    target
    targeting
        The process of identifying any `target-cell` bounding the
        `target-rect`.

        - The search for the `target-cell` starts from the `landing-cell`,
          follows the specified `target-moves`, and ends when a `state-change`
          is detected on an `exterior` column or row, according to the enacted
          `termination-rule`.
        - Failure to identify a target-cell raises an error.
        - The process is followed by `expansions` to identify the
          `capture-rect`.

        Note that in the case of a `dependent` `2nd` `edge`, the `target-rect`
        would always be the same, irrespective of whether `target-moves`
        denoted a row-by-row or column-by-column traversal.

    capture
    capturing
        It is the overall procedure of:

            1. `targeting` both `edge` refs to come up with the `target-rect`;
            2. performing `expansions` to identify the `capture-rect`;
            3. extracting the values and feed them to `filters`.

    capture-rect
    capture-cell
        The *rectangular-area* of the sheet denoted by the two *capture-cells*
        identified by `capturing`, that is, after applying `expansions` on
        `target-rect`.

    directions
    primitive-directions
        The 4 *primitive-directions* that are denoted with one of the letters
        ``LURD``.
        Thee are used to express both `target-moves` and `expansions`.

    coordinate
    coordinates
        It might refer to:

            - the first part of the `edge` syntax;
            - any pair of a cell/column *coordinates* specifying cell positions,
              i.e. `landing-cell`, `target-cell`, bounds of the `capture-rect`.

        They can be expressed in ``A1`` format or as a zero-based
        ``(row, col)`` tuple (*num*).
        Each *coordinate* might be `absolute` or `dependent`, independently.

    target-moves
        Specify the cell traversing order while `targeting` using
        `primitive-directions` pairs.
        The pairs ``UD`` and ``LR`` (and their inverse) are invalid.
        I.e. ``DR`` means:

            *"Start going right, column-by-column, traversing each column
            from top to bottom."*

    expansions
    expansion-moves
        Due to `state-change` on the 'exterior' cells the `capture-rect`
        might be smaller that a wider contigious but "convex" rectangular area.

        The *expansions* attempt to remedy this by providing for expanding on
        arbitrary `directions` accompanied by a multiplicity for each one.
        If multiplicity is unspecified, infinite assumed, so it expands
        until an empty/full row/column is met.

    absolute
        Any cell row/col identified with column-characters, row-numbers, or
        the following special-characters:

        - ``^``          The top/Left full-cell `coordinate`.
        - ``_``          The bottom/right full-cell `coordinate`.

    dependent
        Any `2nd` `edge` `coordinate` identified with a dot(``.``),
        which means that:

            ``2nd-landing-cell coordinate := 1st target-cell coordinate``

        The `2nd` `edge` might contain a "mix" of `absolute` and *dependent*
        coordinates.

    state
    full-cell
    empty-cell
        A cell is *full* when it is not *empty* / *blank* (in Excel's parlance).

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
        The *column* and the *row* of the `1st` `landing-cell`;
        the `termination-rule` gets to be triggered only by `state-change`
        on them.

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
    The `capturing` moves from `1st` `target-cell` to `2nd` `target-cell` are
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

  - If the `state` of the ``2nd-landing-cell == 1st-target-cell``:
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


TODOs
=====
* Support cubic areas.
* Notation for specifying the "last-sheet".

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

TargetRef = namedtuple('TargetRef', ['cell', 'mov'])
"""
It might be "cooked" or "uncooked" depending on its `Cell`.

- An *uncooked* targetref contains *A1* :data:`Cell`.
- An *cooked* targetref contains a *num* :data:`Cell`.
"""


def num2a1_Cell(row, col):
    """Make *A1* :data:``Cell` from *num* or special coords, with rudimentary error-checking.

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
        col = xlrd.colname(col)
    return Cell(row=row, col=col)


def _uncooked_TargetRef(row, col, mov):
    """
    Make a new `TargetRef` from any non-values supplied, as is capitalized, or nothing.

    :param str, None col:    ie ``A``
    :param str, None row:    ie ``1``
    :param str, None mov:    ie ``RU1D?``

    :return:    a `TargetRef` if any non-None
    :rtype:     TargetRef, None


    Examples::

        >>> tr = _uncooked_TargetRef('1', 'a', 'Rul')
        >>> tr
        TargetRef(cell=Cell(row='1', col='A'), mov='RUL')


    No error checking performed::

        >>> _uncooked_TargetRef('Any', 'foo', 'BaR')
        TargetRef(cell=Cell(row='Any', col='FOO'), mov='BAR')

        >>> print(_uncooked_TargetRef(None, None, None))
        None


    except were coincidental::

        >>> _uncooked_TargetRef(row=0, col=123, mov='BAR')
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'

        >>> _uncooked_TargetRef(row=0, col='A', mov=123)
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'
    """

    if col == row == mov is None:
        return None

    return TargetRef(cell=Cell(col=col and col.upper(), row=row), mov=mov and mov.upper())


def _repeat_moves(moves, times=None):
    """
    Returns an iterator that repeats `moves` x `times`, or infinite if unspecified.

    Used when parsing :term:`primitive-directions`.

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
        - st_ref: (TargetRef, None) the 1st-ref, uncooked, with raw cell
        - nd_ref: (TargetRef, None) the 2nd-ref, uncooked, with raw cell
        - rect_exp: (str) as found on the xl-ref
        - json: parsed

    :rtype: dict


    Examples::

        >>> res = parse_xl_ref('Sheet1!A1(DR):Z20(UL):L1U2R1D1:{"json":"..."}')
        >>> sorted(res.items())
        [('json', {'json': '...'}),
         ('nd_ref', TargetRef(cell=Cell(row='20', col='Z'), mov='UL')),
         ('rect_exp', 'L1U2R1D1'),
         ('sheet', 'Sheet1'),
         ('st_ref', TargetRef(cell=Cell(row='1', col='A'), mov='DR'))]

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
        gs['st_ref'] = _uncooked_TargetRef(
            p('st_row'), p('st_col'), p('st_mov'))
        gs['nd_ref'] = _uncooked_TargetRef(
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
         ('nd_ref', TargetRef(cell=Cell(row='^', col='.'), mov='DR')),
         ('rect_exp', 'LU?'),
         ('sheet', 'Sheet1'),
         ('st_ref', TargetRef(cell=Cell(row='1', col='A'), mov='UL')),
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
    Returns top-left and bottom-down margins and all full-incdices from a :term:`state` matrix.

    Cache its return-value to use it in other functions here needing it.

    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`read_states_matrix()`.
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
        >>> margins, indices = get_sheet_margins(full_cells)
        >>> margins                                         # doctest: +SKIP
        Cell(row={'_': 3, '^': 1}, col={'_': 2, '^': 1})


        >>> indices
         [[1, 1], [2, 1], [2, 2], [3, 2]]


    Note that the botom-left cell is not the same as `full_cells` matrix size::

        >>> full_cells = [
        ...    [0, 0, 0, 0],
        ...    [0, 1, 0, 0],
        ...    [0, 1, 1, 0],
        ...    [0, 0, 1, 0],
        ...    [0, 0, 0, 0],
        ... ]
        >>> margins_2, _ = get_sheet_margins(full_cells)
        >>> margins_2 == margins
        True

    """
    indices = np.array(np.where(full_cells)).T  # XXX: Loads all sheet here?!?
    up_r, up_c = indices.min(0)
    dn_r, dn_c = indices.max(0)
    sheet_margins = Cell(
        row={'^': up_r, '_': dn_r},
        col={'^': up_c, '_': dn_c})
    return sheet_margins, indices.tolist()


def _build_special_dict(cord_bounds, base_coord):
    """Makes a stacked dict of margins and base-coord for resolving all specials coords. """
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
    :param str        cfunc:  the function to convert coord ``str --> int``
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
        ValueError: invalid row(None) due to:
                int() argument must be a string, a bytes-like object or a number, not 'NoneType'

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
                cbounds = _build_special_dict(cbounds, bcoord)
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


def _target_opposite_state(state, cell, full_cells, dn, moves, last=False):
    """

    :param bool state:      the starting-state
    :param cell:
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`read_states_matrix()`.
    :param sheet:
    :param moves:
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

    :param bool state:      the starting-state
    :param cell:
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`read_states_matrix()`.
    :param sheet:
    :param moves:
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

    :param state:
    :param xl_rect:
    :param ndarray full_cells:  A boolean ndarray with `False` wherever cell are
                                blank or empty. Use :func:`read_states_matrix()`.
    :param rect_exp:
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
    Performs :term:`targeting` and applies :term:`expansions` but does not extract values.

    Feed the results into :func:`read_capture_values()`.

    :param ndarray full_cells:
            A boolean ndarray with `False` wherever cell are
            blank or empty. Use :func:`read_states_matrix()`.
    :param TargetRef st_ref:  "uncooked" as matched by regex
    :param TargetRef nd_ref:  "uncooked" as matched by regex
    :param rect_exp:
    :return: a ``(Cell, Cell)`` with the 1st and 2nd :term:`capture-cell`
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
        >>> st_ref = TargetRef(num2a1_Cell(0, 0), 'DR')
        >>> nd_ref = TargetRef(Cell('.', '.'), 'DR')
        >>> resolve_capture_rect(full_cells, sheet_margins,
        ...         st_ref, nd_ref)
        (Cell(row=6, col=3), Cell(row=7, col=3))

        >>> nd_ref = TargetRef(num2a1_Cell(7, 6), 'UL')
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
        return (st, nd)
    else:
        rect_exp = _parse_rect_expansions(rect_exp)
        return _expand_rect(state, (st, nd), full_cells, rect_exp)


def read_capture_rect_values(sheet, xl_rect, indices, epoch1904=False):
    """
    Extracts :term:`capture-rect` values from excel-sheet and apply :term:`filters`.

    :param sheet:
    :param tuple xl_rect:  tuple (num_cell, num_cell) with the edge targets of
                           the capture-rect
    :param indices:
    :param epoch1904:
    :return:


    Examples::

        >>> import os, tempfile, xlrd, pandas as pd

        >>> os.chdir(tempfile.mkdtemp())
        >>> df = pd.DataFrame([
        ... # Cols: 0       1    2
        ...        [None, None, None],
        ...        [5.1,  6.1,  7.1]
        ... ])
        >>> tmp = 'sample.xlsx'
        >>> writer = pd.ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

        >>> sheet = xlrd.open_workbook(tmp).sheet_by_name('Sheet1')

        >>> sheet_margins, indices = get_sheet_margins(read_states_matrix(sheet))

        # minimum matrix in the sheet
        >>> st = _resolve_cell(Cell('^', '^'), sheet_margins)
        >>> nd = _resolve_cell(Cell('_', '_'), sheet_margins)
        >>> read_capture_rect_values(sheet, (st, nd), indices)
        [[None,  0,    1,    2],
         [0,    None, None, None],
         [1,     5.1,  6.1,  7.1]]

        # get single value
        >>> read_capture_rect_values(sheet, (Cell(6, 3), Cell(6, 3)), indices)
        [0]

        # get column vector
        >>> st = _resolve_cell(Cell('1', 'D'), sheet_margins)
        >>> nd = _resolve_cell(Cell('_', 'd'), sheet_margins)
        >>> read_capture_rect_values(sheet, (st, nd), indices)
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> st = _resolve_cell(Cell('6', 'A'), sheet_margins)
        >>> nd = _resolve_cell(Cell('6', '_'), sheet_margins)
        >>> read_capture_rect_values(sheet, (st, nd), indices)
        [None, None, None, None, 0, 1, 2]

        # get row vector
        >>> st = _resolve_cell(Cell('6', 'A'), sheet_margins)
        >>> nd = _resolve_cell(Cell('6', 'K'), sheet_margins)
        >>> read_capture_rect_values(sheet, (st, nd), indices)
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
                row.append(read_cell(sheet.cell(r, c), epoch1904))
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

default_filters = {
    None: {'fun': lambda x: x},  # TODO: Actually _redim_captured_values().
    'df': {'fun': pd.DataFrame},
    'df_num': {'fun': _type_df_with_numeric_conversion},
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}


def _process_captured_values(value, type=None, args=(), kws=None, filters=None,
                             available_filters=default_filters):
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


#### XLRD FUNCS ###

import xlrd
from xlrd import (xldate, XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_TEXT,
                  XL_CELL_BLANK, XL_CELL_ERROR, XL_CELL_BOOLEAN, XL_CELL_NUMBER)

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    xlrd_0_9_3 = True
else:
    xlrd_0_9_3 = False


def open_workbook(xl_ref_child, xl_ref_parent=None):
    """
    Opens the excel workbook of an excel ref.

    :param dict xl_ref_child: excel ref of the child

    :param xl_ref_parent: excel ref of the parent
    :type xl_ref_parent: dict, None

    """
    url_fl = xl_ref_child['url_file']
    try:
        if url_fl:
            wb = xlrd.open_workbook(file_contents=urlopen(url_fl).read())
        else:
            wb = xl_ref_parent['xl_workbook']
        xl_ref_child['xl_workbook'] = wb

    except Exception as ex:
        raise ValueError("Invalid excel-file({}) due to:{}".format(url_fl, ex))


def open_sheet(xl_ref_child, xl_ref_parent=None):
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


def read_cell(cell, epoch1904=False):
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
        >>> read_cell(Cell(xlrd.XL_CELL_NUMBER, 1.2))
        1.2

        >>> read_cell(Cell(xlrd.XL_CELL_DATE, 1.2))
        datetime.datetime(1900, 1, 1, 4, 48)

        >>> read_cell(Cell(xlrd.XL_CELL_TEXT, 'hi'))
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
def read_states_matrix(sheet):
    """
    Returns a boolean ndarray with `False` wherever cell are blank or empty.
    """
    types = np.array(sheet._cell_types)
    return (types != XL_CELL_EMPTY) & (types != XL_CELL_BLANK)
