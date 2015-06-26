#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
A mini-language to capture non-empty rectangular areas from Excel-sheets.

.. default-role:: term

Introduction
============

This modules defines a url-fragment notation for `capturing` rectangular areas
from excel-sheets when their exact position is not known beforehand.
The notation extends the ordinary excel `coordinates`, and provides for
conditionally `traversing` the cells based on their `state`.

The goal is to make the extraction of data-tables from excel-workbooks
as practical as reading CSVs, while keeping it as "cheap" as possible,
by employing state-checks instead of parsing the complete sheet contents.

Since the `capturing` depends only on the full/empty `state` of the cells,
another library would be needed to examine the values of the `capture-rect`
afterwards (i.e. "pandas").  Nevertheless, the `xl-ref` syntax
provides for specifying common `filter` transformations at the end, for setting
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
.. default-role:: obj

- User-facing functionality:

  .. autosummary::

      wrap_sheet

  .. currentmodule:: pandalone.xlref._xlref
  .. autosummary::

      Cell
      Edge
      coords2Cell
      parse_xl_ref
      parse_xl_url
      resolve_capture_rect
      read_capture_rect

- **xlrd** back-end functionality:

  .. currentmodule:: pandalone.xlref._xlrd
  .. autosummary::
      XlrdSheet

.. default-role:: term


Examples
--------
.. ToDo::
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
        The 4 primitive *directions* that are denoted with one of the letters
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

    traversing
    traversal-operations
        Either the `target-moves` or the `expansion-moves`.

    target-moves
        Specify the cell traversing order while `targeting` using
        primitive `directions` pairs.
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

    states-matrix
        A boolean matrix denoting the `state` of the cells, having the same
        size as a sheet it was derived from.

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
letters denoting the 4 primitive `directions`, ``LURD``::

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


.. default-role:: obj
"""

"""

TODOs
=====
* Support cubic areas.
* Notation for specifying the "last-sheet".
"""

from ._xlref import (
    Cell, coords2Cell, Edge,
    parse_xl_url, parse_xl_ref, resolve_capture_rect,
    read_capture_rect, _Spreadsheet
)
from ._xlrd import (
    wrap_sheet as wrap_xlrd_sheet, XlrdSheet
)


def wrap_sheet(backed_sheet, *args, **kws):
    sheet = XlrdSheet(backed_sheet, *args, **kws)

    # Other backends here in try-blocks.

    return sheet
