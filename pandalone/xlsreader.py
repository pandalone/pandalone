#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Implements an "Excel-url" format for capturing ranges from sheets.
 
Excel-range addressing ("xl-refs")
==================================

<1st-cell>:<2nd-cell>:<extensions>

Definitions
-----------

- 1st cell: is the cell defined before ':'.
- 2nd cell: is the cell defined after the first ':'.
- range-extensions:
- cell-moves: are the 12 scan-directions in the parentheses '(...)'.
- cell-row: is the row coordinate of a cell.
- cell-col: is the column coordinate of a cell.
- cell-ref: is a pair of cell row-col coordinates.
- cell-start: is the initial cell ref before the cell moves.
- cell-target: is the final cell ref after the cell moves.
- cell-state: can be empty or full(non-empty).
- relative position: identified with a dot('.') for each cell-coordinate.
  It can be used only in the 2nd cell-start ref (e.g., '.3', '..', 'B.').
  It assumes that:
        2nd cell-start col/row = 1st cell-target col/row
- absolute position: identified with '^' and '_' for each cell-coordinate.
  * '^': upper full cell-coordinate.
  * '_': lower full cell-coordinate.

Basic search cell-target rules
------------------------------

- Search-opposite-state:
  the cell-target is the first cell founded according to the cell-moves that has
  the opposite state of the cell-start.

- Search-same-state:
  the cell-target is evaluated according to the target-coordinate-moves.

  * target-coordinate-move (cell-row if move='L' or 'R', cell-col if move='U' or
    'D') is the last cell founded from the relative start-coordinate according
    to the move that has the same state of the cell-start.

1st cell target search
----------------------

If the 1st-cell:

  - has not cell-moves: cell-target = cell-start

  - has cell-moves: cell-target is evaluated using the search-opposite-state
      rule.

2nd cell target search
----------------------

If the 2nd-cell:

  - has not cell-moves: cell-target = cell-start

  - has cell-moves and:
    * 2nd-cell-start-state == 1st-cell-target-state:
      cell-target is evaluated using the search-same-state rule.

    * 2nd-cell-start-state != 1st-cell-target-state:
      cell-target is evaluated using the search-opposite-state rule.


Primitive moves
---------------

There are 12 "primitive scan-directions" or excel "cell-moves" named with
a *single* or a *pair* of the the letters `"LURD"`::

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


So a `RD` primitive-move means:

  > Scan cells by rows: start moving *right* till 1st empty/non-empty cell,
  > and then move *down* to the next row, and scan right again."


Target-cells
------------

Using these moves we can identify a "target" xl-cell from a known
"starting" position (ie `A1`, or `^^` and `__` for the start/end of the sheet) 
when the values visited in the 1st row/column *change* from empty to non-empty, 
and vice versa.

For instance, given this xl-sheet below, here some of the ways
to identify (or target) the non-empty values `X`::

      A B C D E F
    1
    2
    3     X        ──► C3    A1(RD)   _^(L)      F3(L)
    4         X    ──► E4    A4(R)    _4(L)      D1(DR)
    5   X          ──► B5    A1(DR)   A_(UR)     _5(L)
    6           X  ──► F6    __       _^(D)      A_(R)

    - The 'X' signify non-empty cells.


So we can target cells with "absolute coordinates" (the usual "A1" notation),
expanded with:

  - undesrcore(`_`) for bottom/right, and
  - accent(`^`) for top/left

columns/rows of the sheet with non-empty values.

When no "LURD"s are specified, the target-cell coinceds with the starting one.


Ranges
------

To specify a complete "range" we need to identify a 2nd cell.
The 2nd target-cell may be specified:

  - either with absolute coordinates, as above, or
  - with "relative" coords, using the dot(`.`) to refer to the 1st cell.


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
   Of course, the above ranges WILL FAIL since the range-traversing moves
   will stop immediately due to `X`s being surrounded by empty-cells.
   
   But the above diagram was to just convey the general idea.
   Under normal circumstances, all the in-between cells must be non-empty.


The required moves for traversing from 1st to 2nd cell are
calculated automatically from the relative positions of the 2 targets cells.
In the case of relative-coords, these moves may or may not concede 
with the moves of the 2nd-cell.

For instance, to capture `B4:E5` in the above sheet we may use `_5(L):E.(U)`.
In that case the target cells are `B5` and `E4` and the traversing-move
to reach the 2nd one is calculated to be `UR` which is different from `U`.


.. Note::
    The traversing-moves from 1st to 2nd target cell will fetch 
    the same values regardless of whether we are traversing "row-first" or 
    "column-first".


Expansion
---------

Captured rect-ranges ("values") may be limited due to empty-cells in the 
1st row/column.
To overcome this, the xl-ref may specify "expansions" directions using 
a 3rd `:`-section like that::

    _5(L):1_(UR):RDL?U?

This particular case means: 

  > Try expanding Right and Down repeatedly and then one Left and Up.

Expansion happens row-by-row or column-by-column basis, and terminates when 
a full empty(or non-empty) line is met.

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
    - The '1' and `2` signify the identified target-cells.


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

StartPos = namedtuple('StartPos', ['cell', 'mov'])
Cell = namedtuple('Cell', ['row', 'col'])


def col2num(col_str):
    """
    Converts the Excel 'str' column ref in a 'int' column ref.

    :param col_str: excel column ref
    :type col_str: str

    :return:  excel column number [0, ...]
    :rtype: int

    Example::

        >>> col2num('D')
        3
        >>> col2num('d')
        3
    """

    num = 0
    for c in col_str:
        num = num * 26 + ascii_uppercase.rindex(c.upper()) + 1

    return num - 1

XL_UP_ABS = object()
XL_BOTTOM_ABS = object()
CELL_RELATIVE = object()
MAX_TIME = None

_c_pos = {
    '^': XL_UP_ABS,
    '_': XL_BOTTOM_ABS,
    '.': CELL_RELATIVE
}


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
        scan-directions
    :type cell_mov: str, None

    :return:
        a cell-start
    :rtype: StartPos

    Example::
        >>> fetch_cell_ref('A', '1', 'R')
        StartPos(cell=Cell(col=0, row=0), mov='R')
        >>> fetch_cell_ref('^', '^', 'R')
        StartPos(cell=Cell(col=XL_UP_ABS, row=XL_UP_ABS), mov='R')
        >>> fetch_cell_ref('_', '_', 'L')
        StartPos(cell=Cell(col=XL_BOTTOM_ABS, row=XL_BOTTOM_ABS), mov='L')
        >>> fetch_cell_ref('.', '.', 'D')
        StartPos(cell=Cell(col=CELL_RELATIVE, row=CELL_RELATIVE), mov='D')
        >>> fetch_cell_ref(None, None, None)

    """
    if cell_col == cell_row == cell_mov is None:
        return None
    elif cell_row != '0':
        row = _c_pos[cell_row] if cell_row in _c_pos else int(cell_row) - 1
        col = _c_pos[cell_col] if cell_col in _c_pos else col2num(cell_col)
        return StartPos(cell=Cell(col=col, row=row), mov=cell_mov)

    raise ValueError('Unsupported row format %s' % cell_row)


def parse_cell(cell, epoch1904=False):
    """
    Parse a xl-cell.

    :param cell: an excel cell
    :type cell: xlrd.sheet.Cell

    :param epoch1904:
        Which date system was in force when this file was last saved.
        False => 1900 system (the Excel for Windows default).
        True => 1904 system (the Excel for Macintosh default).
    :type epoch1904: bool, optional

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


def chain_moves(moves, times=None):
    # chain_moves('ABC', 3) --> A B C A B C A B C
    it = repeat(moves, int(times)) if times is not None else repeat(moves)
    for mv in it:
        for element in mv:
            yield element


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

    Example::

        >>> rng_ext = 'LURD?'
        >>> res = parse_rng_ext(rng_ext)

        >>> res
        [<generator object chain_moves at 0x...>,
         <generator object chain_moves at 0x...>]

        # infinite generator
        >>> [next(res[0]) for i in range(10)]
        ['L', 'U', 'R', 'L', 'U', 'R', 'L', 'U', 'R', 'L']
        >>> list(res[1])
        ['D']

    """
    try:
        res = _re_rng_ext_splitter.split(rng_ext.replace('?', '1'))

        return [chain_moves(*_re_rng_ext_parser.match(v).groups())
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
        <xl_sheet_name>!<st_col><st_row>(<st_mov>):<nd_col><nd_row>(<nd_mov>):<rng_ext>{<json>}
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

    Example::
        >>> from itertools import chain
        >>> xl_ref = 'Sheet1!A1(DR):Z20(UL):L1U2R1D1{"json":"..."}'
        >>> res = parse_xl_ref(xl_ref)

        >>> res['xl_sheet_name']
        'Sheet1'
        >>> res['st_cell']
        Cell(col=0, row=0, mov='DR')
        >>> res['nd_cell']
        Cell(col=25, row=19, mov='UL')
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

    Example::

        >>> url = 'file:///sample.xls#Sheet1!:{"2": "ciao"}'
        >>> res = parse_xl_url(url)
        >>> sorted(res.items())
        [('cell_down', Cell(col=None, row=None)),
         ('cell_up', Cell(col=None, row=None)),
         ('json', {'2': 'ciao'}),
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

    :param xl_ref_child: excel ref of the child
    :type xl_ref_child: dict

    :param xl_ref_parent: excel ref of the parent
    :type xl_ref_parent: dict, None, optional

    Example::

        >>> import tempfile, pandas as pd, xlrd
        >>> from tests.test_utils import chdir
        >>> with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
        ...     df = pd.DataFrame()
        ...     tmp = 'sample.xlsx'
        ...     writer = pd.ExcelWriter(tmp)
        ...     df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        ...     writer.save()
        ...     url = 'file://%s#' % '/'.join([tmpdir, tmp])
        ...     xl_ref = parse_xl_url(url)
        ...     open_xl_workbook(xl_ref)
        ...     isinstance(xl_ref['xl_workbook'], xlrd.book.Book)
        True

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

    :param xl_ref_child: excel ref of the child
    :type xl_ref_child: dict

    :param xl_ref_parent: excel ref of the parent
    :type xl_ref_parent: dict, None, optional

    Example::

        >>> import tempfile, pandas as pd, xlrd
        >>> from tests.test_utils import chdir
        >>> with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
        ...     df = pd.DataFrame()
        ...     tmp = 'sample.xlsx'
        ...     writer = pd.ExcelWriter(tmp)
        ...     df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        ...     writer.save()
        ...     url_parent = 'file://%s#Sheet1!' % '/'.join([tmpdir, tmp])
        ...     xl_ref_parent = parse_xl_url(url_parent)
        ...     open_xl_workbook(xl_ref_parent)
        ...     open_xl_sheet(xl_ref_parent)
        ...     url_child = '#A1:B2'
        ...     xl_ref_child = parse_xl_url(url_child)
        ...     open_xl_workbook(xl_ref_child, xl_ref_parent)
        ...     open_xl_sheet(xl_ref_child, xl_ref_parent)
        ...     isinstance(xl_ref_child['xl_sheet'], xlrd.sheet.Sheet)
        True
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
    :type dim_max: int, None, optional

    :return: reshaped value
    :rtype: list of lists, list, value

    Example::

        >>> redim_xl_range([1, 2], 2)
        [[1, 2]]
        >>> redim_xl_range([[1, 2]], 1)
        [[1, 2]]
        >>> redim_xl_range([[1, 2]], 1, 1)
        [1, 2]
        >>> redim_xl_range([[1, 2]], 0, 0)
        Traceback (most recent call last):
        ...
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


_function_types = {
    None: {'fun': lambda x: x},
    'df': {'fun': pd.DataFrame},
    'nparray': {'fun': np.array},
    'dict': {'fun': dict},
    'sorted': {'fun': sorted}
}


def process_xl_range(value, type=None, args=[], kwargs={}, filters=None):
    """
    Processes the output value of get_rect_range function.

    :param value: matrix or vector or value
    :type value: list of lists, list, value

    :param type: reference type
    :type type: str, None, optional

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
        >>> res = process_xl_range(value, type='dict')
        >>> sorted(res.items())
        [(1, 2),
         (3, 4),
         (5, 6)]
        >>> value = [[1, 9], [8, 10], [5, 11]]
        >>> process_xl_range(value, filters=[{'type':'sorted',\
                                              'kwargs':{'reverse': True}\
                                              }])
        [[8, 10],
         [5, 11],
         [1, 9]]
    """
    val = _function_types[type]['fun'](value, *args, **kwargs)
    if filters:
        for v in filters:
            val = process_xl_range(val, **v)
    return val


_primitive_dir = {
    'L': np.array([0, -1]),
    'U': np.array([-1, 0]),
    'R': np.array([0, 1]),
    'D': np.array([1, 0])
}


def get_no_empty_cells(sheet):
    types = np.array(sheet._cell_types)
    return (types!=xlrd.XL_CELL_EMPTY) & (types!=xlrd.XL_CELL_BLANK)


def get_xl_margins(no_empty):
    """ Returns upper and lower absolute positions"""
    indices = np.array(np.where(no_empty)).T
    up_r, up_c = indices.min(0)
    dn_r, dn_c = indices.max(0)
    xl_margins = {
        'col':{
            XL_UP_ABS: up_c,
            XL_BOTTOM_ABS: dn_c
        },
        'row':{
            XL_UP_ABS: up_r,
            XL_BOTTOM_ABS: dn_r
        }
    }
    return xl_margins, indices


def set_coord(cell, xl_margins, parent=None):
    c = {CELL_RELATIVE: parent} if parent else {}
    c.update(xl_margins)
    return c[cell] if cell in xl_margins else cell


def set_start_cell(cell, xl_margins, parent_cell=None):
    if parent_cell:
        row = set_coord(cell.row, xl_margins['row'], parent_cell.row)
        col = set_coord(cell.col, xl_margins['col'], parent_cell.col)
    else:
        row = set_coord(cell.row, xl_margins['row'], None)
        col = set_coord(cell.col, xl_margins['col'], None)
    return Cell(row=row, col=col)


def search_opposite_state(cell, no_empty, sheet, directions, last=False):
    """

    :param cell:
    :param no_empty:
    :param sheet:
    :param directions:
    :return:

    Example::
        >>> non_empty = np.array(\
            [[False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False, False, False, False],\
             [False, False, False, False,  True,  True,  True],\
             [False, False, False,  True, False, False, False],\
             [False, False, False,  True,  True,  True,  True]])
        >>> Sheet = namedtuple('Sheet', ['ncols', 'nrows'])
        >>> sheet = Sheet(7, 8)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'DR')
        Cell(row=6, col=3)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'RD')
        Cell(row=5, col=4)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'D')
        Cell(row=7, col=1)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'U')
        Cell(row=0, col=1)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'R')
        Cell(row=1, col=6)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'L')
        Cell(row=1, col=0)
        >>> search_opposite_state(Cell(1,1), non_empty, sheet, 'LU')
        Cell(row=0, col=0)
        >>> search_opposite_state(Cell(1,0), non_empty, sheet, 'LU')
        Cell(row=0, col=0)
    """
    state = no_empty[cell]

    up_cell = (0, 0)

    dn_cell = (sheet.nrows - 1, sheet.ncols - 1)

    mv = _primitive_dir[directions[0]]  # first move
    c0 = c1 = np.array(cell)
    while (up_cell <= c0).all() and (c0 <= dn_cell).all():
        c1 = c0
        while (up_cell <= c1).all() and (c1 <= dn_cell).all():
            if no_empty[(c1[0], c1[1])] != state:
                if last:
                    c1 = c1 - mv
                return Cell(*(c1[0], c1[1]))
            c1 = c1 + mv
        c1 = c1 - mv
        try:
            c0 = c0 + _primitive_dir[directions[1]]  # second move
        except IndexError:
            break

    return Cell(*(c1[0], c1[1]))


def search_same_state(cell, no_empty, sheet, directions):
    c1 = list(cell)
    for d in directions:
        c = search_opposite_state(cell, no_empty, sheet, d, True)
        dis = _primitive_dir[d]
        c1 = [i if not k == 0 else j for i, j, k in zip(c, c1, dis)]
    return Cell(*c1)



def get_range(sheet, st_cell, nd_cell=None, rng_ext=None, epoch1904=False):
    """

    :param sheet:
    :type sheet: xlrd.sheet.Sheet
    :param st_cell:
    :type st_cell: StartPos
    :param nd_cell:
    :type nd_cell: StartPos
    :param rng_ext:
    :param epoch1904:
    :return:
    """
    no_empty = get_no_empty_cells(sheet)

    xl_margins = get_xl_margins(no_empty)



    if st_cell.mov is None:
        set_start_cell()


def _xlwings_min_index(it_types, margin, max_i):
    try:
        return next(i for i, c in enumerate(it_types)
                    if c in (XL_CELL_BLANK, XL_CELL_EMPTY)) + margin
    except StopIteration:
        return max_i


def _xlwings_margins(sheet, cell_up, cell_down, up, dn):
    if cell_up.col is not None and cell_up.row is not None and \
            (cell_down.col is None or cell_down.row is None):  # from up
        if cell_down.col is None:
            dn[0] = _xlwings_min_index(sheet.row_types(up[1], up[0]),
                                       up[0], sheet.ncols)

        if cell_down.row is None:
            dn[1] = _xlwings_min_index(sheet.col_types(up[0], up[1]),
                                       up[1], sheet.nrows)
    elif cell_down.col is not None and cell_down.row is not None and \
            (cell_up.col is None or cell_up.row is None):  # from bottom
        _dn = (dn[0] - 1, dn[1] - 1)
        if cell_up.col is None:
            up[0] = -_xlwings_min_index(
                reversed(sheet.row_types(_dn[1], 0, _dn[0])), -_dn[0], 0)

        if cell_up.row is None:
            up[1] = -_xlwings_min_index(
                reversed(sheet.col_types(_dn[0], 0, _dn[1])), -_dn[1], 0)
    return up, dn


def get_rect_range(sheet, cell_up, cell_down=None, epoch1904=False,
                   xlwings=False):
    """
    Discovers a non-empty tabular-shaped region in the xl-sheet from a range.

    :param sheet: a xlrd Sheet object
    :type sheet: xlrd.sheet.Sheet obj

    :param cell_up: up margin
    :type cell_up: Cell

    :param cell_down: bottom margin
    :type cell_down: Cell, optional

    :param epoch1904:
        Which date system was in force when this file was last saved.
        False => 1900 system (the Excel for Windows default).
        True => 1904 system (the Excel for Macintosh default).
    :type epoch1904: bool, optional

    :param xlwings:
        if True get_rect_range has a behavior compatible with xlwings
    :type xlwings: bool, optional

    :return: matrix or vector or value
    :rtype: list of lists, or list, value

    Example::

        >>> import os, tempfile, xlrd, pandas as pd
        >>> os.chdir(tempfile.mkdtemp())
        >>> df = pd.DataFrame([[None, None, None], [5.1, 6.1, 7.1]])
        >>> tmp = 'sample.xlsx'
        >>> writer = pd.ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

        >>> sheet = xlrd.open_workbook(tmp).sheet_by_name('Sheet1')

        # minimum matrix in the sheet
        >>> get_rect_range(sheet,  StartPos(None, None), StartPos(None, None))
        [[None, 0, 1, 2],
         [0, None, None, None],
         [1, 5.1, 6.1, 7.1]]

        # up-left delimited minimum matrix
        >>> get_rect_range(sheet, StartPos(0, 0), StartPos(None, None))
        [[None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, 0, 1, 2],
         [None, None, None, 0, None, None, None],
         [None, None, None, 1, 5.1, 6.1, 7.1]]

        # get single value
        >>> get_rect_range(sheet, StartPos(3, 6))
        0

        # get column vector
        >>> get_rect_range(sheet, StartPos(3, None))
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> get_rect_range(sheet, StartPos(None, 5))
        [None, None, None, None, 0, 1, 2]

        # up-left delimited minimum matrix
        >>> get_rect_range(sheet, StartPos(4, None), StartPos(None, None))
        [[0, 1, 2],
         [None, None, None],
         [5.1, 6.1, 7.1]]

        # delimited matrix
        >>> get_rect_range(sheet, StartPos(3, 5), StartPos(5, 7))
        [[None, 0, 1],
         [0, None, None],
         [1, 5.1, 6.1]]

        # down-right delimited minimum matrix
        >>> get_rect_range(sheet, StartPos(None, None), StartPos(5, 7))
        [[None, 0, 1],
         [0, None, None],
         [1, 5.1, 6.1]]

        # up-down-right delimited minimum matrix
        >>> get_rect_range(sheet, StartPos(None, 6), StartPos(5, 7))
        [[0, None, None],
         [1, 5.1, 6.1]]

        # down delimited minimum vector (i.e., column)
        >>> get_rect_range(sheet, StartPos(5, None), StartPos(5, 7))
        [1, None, 6.1]

        # right delimited minimum vector (i.e., row)
        >>> get_rect_range(sheet, StartPos(2, 5), StartPos(None, 5))
        [None, None, 0, 1, 2]

        # right delimited minimum vector (i.e., row)
        >>> get_rect_range(sheet, StartPos(3, 6), StartPos(_xl_margin, 6))
        [0, None, None, None]

        # right delimited minimum vector (i.e., row)
        >>> get_rect_range(sheet, StartPos(None, None), StartPos(3, 5))
        [[]]
    """

    _pc = lambda cell: parse_cell(cell, epoch1904)

    if cell_down is None:  # vector or cell
        # Set up '_' row/cols as 0.
        _up = {_xl_margin: 0}
        up = [_up.get(i, i) for i in cell_up]

        if up[0] is None:  # return row
            return list(map(_pc, sheet.row(up[1])))
        elif up[1] is None:  # return column
            return list(map(_pc, sheet.col(up[0])))
        else:  # return cell
            if up[1] < sheet.nrows and up[0] < sheet.ncols:
                return _pc(sheet.cell(up[1], up[0]))
            return None
    else:  # table or vector or cell
        # Set up margins.
        #
        _up = dict.fromkeys([None, _xl_margin], 0)
        up = [_up.get(i, i) for i in cell_up]

        # Set bottom margins.
        #
        _dn = [dict.fromkeys([None, _xl_margin], sheet.ncols - 1),
               dict.fromkeys([None, _xl_margin], sheet.nrows - 1)]
        dn = [_dn[i].get(j, j) + 1 for i, j in enumerate(cell_down)]

        nv = lambda x, v=None: [v] * x  # return a None vector  of length x

        # Make a range-sized empty table.
        #
        if up[1] >= sheet.nrows or up[0] >= sheet.ncols:
            ddn = [dn[i] - up[i] if c else 1
                   for i, c in enumerate([cell_down.col is not None,
                                          cell_down.row is not None])]
            return nv(ddn[1], nv(ddn[0]))

        if xlwings:
            up, dn = _xlwings_margins(sheet, cell_up, cell_down, up, dn)

        # Synthesize values for cells outside excel's margins.
        #
        ddn = [max(0, v) for v in (dn[0] - sheet.ncols, dn[1] - sheet.nrows)]

        matrix = [list(map(_pc, sheet.row_slice(r, up[0], dn[0]))) + nv(ddn[0])
                  for r in range(up[1], dn[1] - ddn[1])]

        # Add empty rows.
        #
        if ddn[0] == 0 and ddn[1] > 0:
            matrix += nv(ddn[1], nv(1))
        else:
            matrix += nv(ddn[1], nv(ddn[0]))

        # no empty vector
        ne_vct = lambda vct: any(x is not None for x in vct)

        # return the index of first no empty row in the table
        def ind_row(t, d):
            return next((r for r, v in enumerate(t) if ne_vct(v)), d)

        def reduce_table(t, u, d):
            # Return the minimum vertical table.

            l = len(t)
            m = [ind_row(t, l) if u is None else 0,
                 l - (ind_row(reversed(t), 0) if d is None else 0)]
            return t[m[0]:m[1]] if m[0] != m[1] else [[]]

        # vertical reduction
        #
        if cell_up.row is None or cell_down.row is None:
            matrix = reduce_table(matrix, cell_up.row, cell_down.row)

        # horizontal reduction
        #
        if cell_up.col is None or cell_down.col is None:
            tbl = reduce_table(list(zip(*matrix)), cell_up.col, cell_down.col)
            matrix = [list(r) for r in zip(*tbl)] if tbl != [[]] else [[]]

        # vector
        if cell_down.col is not None and cell_down.col == cell_up.col:
            matrix = [v[0] for v in matrix]

        # vector
        if cell_down.row is not None and cell_down.row == cell_up.row:
            matrix = matrix[0]

        if isinstance(matrix, list):
            return matrix
        else:
            return [matrix]