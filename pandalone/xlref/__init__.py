#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
A mini-language for for "throwing the rope" around rectangular areas of Excel-sheets.

.. default-role:: term

Introduction
============

Overview
--------
This modules defines a url-fragment notation for `capturing` rectangular areas 
of excel-sheets, when their exact position is not known beforehand.
The notation extends the ordinary *A1* and *RC* excel `coordinates` with 
conditional `traversing` operations, based on the cell's empty/full `state`.

For instance, the following `xl-ref` url extracts a DataFrame from 
a contigious table at the top-left of a the 1st sheet of a workbook::

    from pandalone import xlasso

    df = xlasso.lasso('path/to/workbook.xlsx#A1(DR):..(DR):LU:["df"]')


The goal is to make the `capturing` of data-tables from excel-workbooks
as practical as reading CSVs, while keeping it as "cheap" as possible.
Although another library is required for examining the contents of the cells
(i.e. "pandas"), the `xl-ref` syntax provides `filter` transformations 
for some common tasks, such as:
- setting the dimensionality of the result tables,
- creating higher-level objects (dictionaries, *numpy-arrays* & *dataframes*) 
- and applying the `lassoing` on the captured values, recursively .

It is based on `xlrd <http://www.python-excel.org/>`_ library but also
checked for compatibility with `xlwings <http://xlwings.org/quickstart/>`_
*COM-client* library.  It requires *numpy* and optionally *pandas* and is 
developed for python-3 but tested also on python-2 for compatibility.


Xl-Ref Syntax
-------------
::

    [<url>]#[<sheet>!]<1st-edge>[:[<2nd-edge>][:<expansions>]][:<filters>]
    [<url>]#:[<filters>]    # shortcut for ^^:__


Annotated Syntax
----------------
::

  target-moves─────┐
  landing-cell──┐  │
               ┌┤ ┌┤
              #C3(UL):..(RD):L1DR:["pipe": [['dict'], ["recursive"]]]
               └─┬──┘ └─┬──┘ └┬─┘ └─────────────────┬───────────────┘
  1st-edge───────┘      │     │                     │
  2nd-edge──────────────┘     │                     │
  expansions──────────────────┘                     │
  filters───────────────────────────────────────────┘

Which means:

    1. `Target` the `1st` `edge` of the `capture-rect` by starting from ``C3``
       `landing-cell`. If it is a `full-cell`, stop, otherwise start moving
       above and to the left of ``C3`` and stop on the first `full-cell`;
    2. continue from the last `target` and travel the `exterior` row and column
       right and down, stopping on their last `full-cell`;
    3. `capture` all the cells between the 2 targets.
    4. try `expansions` on the `target-rect`, once to the left column
       and then down and right until a full-empty line/row is met, respectively;
    5. finally `filter` the values of the `capture-rect` to wrap them up
       in dictionary, and search its values for `xl-ref` and replace them.


Basic Usage
-----------
To capture a `xl-ref` use the :func:`lasso()`. The simplest example is 
to capture all excel-sheet data but without bordering nulls::

    values = xlasso.lasso('path/to/workbook.xlsx#:')

Assuming that the 1st sheet of the workbook is as shown below, 
the `capture-rect` would be a 2D (nested) list-of-lists with the values
contained in the range ``C2:E4``::

      A B C D E
    1    ┌─────┐
    2    │    X│ 
    3    │X    │ 
    4    │  X  │ 
    5    └─────┘ 

where:
- 'X': the full-cells
- rectangle: the captured-rect


If you do not wish to let the library read your workbooks, you can 
pre-populate a :class:`Ranger' instance with the desired sheets, such as
the sample :class:`ArraySheet`::

    >>> from pandalone import xlref

    >>> ranger = xlref.make_default_Ranger()
    >>> ranger.add_sheet(ArraySheet([[None, None,  'A',   None],
    ...                              [None, 2.2,   'foo', None],
    ...                              [None, None,   2,    None],
    ...                              [None, None,   None, 3.14],
    ... ]))
    >>> ranger.do_lasso('#A1(DR):..(DR):RULD').values
    [[None, 'A'], 
     [2.2, 'foo'], 
     [None, 2]] 

For even more control of the procedure, you can create and use a separate 
:class:`SheetsFactory` instance, which is the backing-store and factory for
all sheets used by the ``Ranger``.


API
---
.. default-role:: obj

- User-facing functionality:

  .. currentmodule:: pandalone.xlref._xlref
  .. autosummary::

      lasso
      Ranger
      Ranger.do_lasso
      SheetsFactory
      make_default_Ranger
      get_default_opts
      get_default_filters
      Lasso
      Cell
      Edge
      coords2Cell

- Major internal functions:
      parse_xlref
      resolve_capture_rect
      ABCSheet.read_rect

- **xlrd** back-end functionality:

  .. currentmodule:: pandalone.xlref._xlrd
  .. autosummary::
      XlrdSheet

.. default-role:: term


More Syntax Examples
--------------------

Another typical case is when a sheet contains a single table 
with a "header"-row and a "index"-column.
There are (at least) 3 ways to do it, beyond specifying
the exact `coordinates`::


      A B C D E
    1  ┌───────┐     Β2:E4          ## Exact referencing.
    2  │  X X X│     ^^.__          ## From top-left full-cell to bottom-right.
    3  │X X X X│     A1(DR):__:U1   ## Start from A1 and move down and right
    3  │X X X X│                    #    until B3; capture till bottom-left;
    4  │X X X X│                    #    expand once upwards (to header row).
       └───────┘     A1(RD):__:L1   ## Start from A1 and move down by row
                                    #    until C1; capture till bottom-left;
                                    #    expand once left (to index column).


Note that if ``B1`` were full, the results would still be the same, because
``?`` expands only if any full-cell found in row/column.

In case where the sheet contains more than one *disjoint* tables, the
bottom-left cell of the sheet would not coincide with table-end, so the handy
last two `xl-ref` above would not work.

For that we may resort to `dependent` referencing for the `2nd` `edge`, and
define its position in relation to the `1st` `target`::

      A B C D E
    1  ┌─────┐    _^:..(LD+):L1     ## Start from top-right(E2) and target left
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
    3  │X X  │      D_:^^
    3  │X    │      A^(DR):D_:U
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
       └───┘        D2(L+):^_


.. seealso:: Example spreadsheet: :download:`xls_ref.xlsx`


Definitions
===========

.. glossary::

    lasso
    lassoing
        It may denote 3 things:
        
        - the whole procedure of `parsing` the `xl-ref` syntax,
          `capturing` values from spreadsheet rect-regions and sending them 
          through any `filters` specified in the xl-ref;
        - the :func:`lasso()` and :meth:`Ranger.lasso()` functions 
          performing the above job;
        - the :class:`Lasso` storing intermediate and final results of the 
          above algorithm.

    xl-ref
        Any url with its fragment abiding to the syntax defined herein.

        - The *fragment* describes how to `capture` rects from excel-sheets, and
          it is composed of 2 `edge` references followed by `expansions` and
          `filters`.
        - The *file-part* should resolve to an excel-file.

    parse
    parsing
        The stage where the input string gets splitted and checked for validity 
        against the `xl-ref` syntax.

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

        They can be expressed in ``A1`` or ``RC`` format or as a zero-based
        ``(row, col)`` tuple (*num*).
        Each *coordinate* might be `absolute` or `dependent`, independently.

    traversing
    traversal-operations
        Either the `target-moves` or the `expansion-moves` that comprise the 
        `capturing`.

    target-moves
        Specify the cell traversing order while `targeting` using
        primitive `directions` pairs.
        The pairs ``UD`` and ``LR`` (and their inverse) are invalid.
        I.e. ``DR`` means:

            *"Start going right, column-by-column, traversing each column
            from top to bottom."*

    move-modifier
        One of ``+`` and ``-`` chars that might trail the `target-moves`
        and define which the `termination-rule` to follow if `landing-cell`
        is `full-cell`, i.e. ``A1(RD+)``

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
    base-cell
        Any `edge` `coordinate` identified with a dot(``.``), meaning that:

            ``landing-cell coordinate := base-cell coordinate``

        where the *base-coordinates* are:
        
        - `1st` edge: the `target-cell` coordinates of the ``context_lasso`` 
          arg given to the :meth:`Ranger.lasso()`; it is an error if ``None``.
        - `2nd` edge: the `target-cell` coordinates of the `1st` edge.

        An `edge` might contain a "mix" of `absolute` and *dependent*
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
        The condition to stop `targeting` while traversing from `landing-cell`.
        The are 2 rules: `search-same` and `search-opposite`.

        .. seealso::
            Check `Target-termination enactment`_ for the enactment of the rules.

    search-opposite
        The `target-cell` is the FIRST `full-cell` found while traveling
        from the `landing-cell` according to the `target-moves`.

    search-same
        The coordinates of the `target-cell` are given by the LAST `full-cell`
        on the `exterior` column/row according to the `target-moves`;
        the order of the moves is insignificant in that case.

    exterior
        The *column* and the *row* of the `landing-cell`; the `search-same`
        `termination-rule` gets to be triggered by 'full-cells' only on them.

    filter
    filters
        The last part of the `xl-ref` specifying predefined functions to 
        apply for transforming the cell-values of `capture-rect`, 
        abiding to the  **json** syntax.

    call-specifier
    call-spec
        The structure to specify some function call in the `filter` part;
        it can either be a json *string*, *list* or *object* like that:
        
        - string: ``"func_name"`` 
        - list:   ``["func_name", ["arg1", "arg2"], {"k1": "v1"}]``
          where the last 2 parts are optional and can be given in any order;
        - object: ``{"func": "func_name", "args": ["arg1"], "kws": {"k":"v"}}`` 
          where the ``args`` and ``kws`` are optional.
          
        If the outer-most filter is a dictionary, a ``'pop'`` kwd is popped-out
        as the `opts`.

    opts
        Key-value pairs affecting the `lassoing` (i.e. opening xlrd-workbooks).
        Read the code to be sure what are the available choices :-( 
        They are a combination of options specified in code (i.e. in the 
        :func:`lasso()` and those extracted from `filters` by the 'opts' key,
        and they are stored in the :class:`Lasso`.


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

.. Seealso:: `Target-termination enactment`_ section


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



Target-termination enactment
----------------------------

The guiding principle for when to enact each rule is to always `capture`
a matrix of `full-cell`.

- If the `landing-cell` is `empty-cell`, always `search-opposite`, that is,
  stop on the first `full-cell`.
- When the `landing-cell` is `full-cell`, it depends on the 'move-modifier':

  - If ``+`` exists, apply `search-same`.
  - If ``-`` exists, stop on `landing-cell`.
  - If no modifier, behave like ```-` (stop on `landing-cell`) except when
    on a `2nd` edge with both its coordinates `dependent` (``..``),
    where the `search-same` is applied

So, both `move-modifier` apply only when `landing-cell` is `full-cell`
, and ``-`` actually makes sense only when `2nd` edge is `dependent`.

If the termination conditions is not met, it is considered an error.



Expansions
----------

Captured-rects ("values") may be limited due to `empty-cell` in the 1st
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
* Support RC with negative coords for counting backwards from the end.
"""

from ._xlref import (
    lasso, Ranger, SheetsFactory, ArraySheet,
    Cell, coords2Cell, Edge, Lasso,
    parse_xlref, resolve_capture_rect,
    make_default_Ranger, get_default_opts, get_default_filters,
    xlwings_dims_call_spec,
)
