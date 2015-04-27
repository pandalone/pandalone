#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import re
import json
import datetime
from string import ascii_uppercase
from collections import namedtuple
# noinspection PyUnresolvedReferences
from six.moves.urllib.parse import urldefrag
import xlrd
from xlrd import (xldate, XL_CELL_DATE, XL_CELL_BLANK, XL_CELL_EMPTY,
                  XL_CELL_TEXT, XL_CELL_ERROR, XL_CELL_BOOLEAN, XL_CELL_NUMBER)
from distutils.version import LooseVersion

if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    xlrd_0_9_3 = True
else:
    xlrd_0_9_3 = False

Cell = namedtuple('Cell', ['col', 'row'])


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


def fetch_cell_ref(cell, cell_col, cell_row):
    """
    Fetch a cell reference string.

    :param cell:
        whole cell reference
    :type cell: str, None

    :param cell_col:
        column reference
    :type cell_col: str, None

    :param cell_row:
        row reference
    :type cell_row: str, None

    :return:
        a Cell-tuple
    :rtype: Cell

    Example::
        >>> fetch_cell_ref('A1', 'A', '1')
        Cell(col=0, row=0)
        >>> fetch_cell_ref('_1', '_', '1')
        Cell(col=None, row=0)
        >>> fetch_cell_ref('A_', 'A', '_')
        Cell(col=0, row=None)
        >>> fetch_cell_ref(':', None, None)
        Cell(col=None, row=None)
        >>> fetch_cell_ref(None, None, None)

    """
    if cell is None:
        return None

    if cell_row != '0':
        row = int(cell_row) - 1 if cell_row and cell_row != '_' else None
        col = col2num(cell_col) if cell_col and cell_col != '_' else None
        return Cell(col=col, row=row)

    raise ValueError('unsupported row format %s' % cell_row)


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


def _xlwings_min_index(it_types, margin, max_i):
    try:
        return next(i for i, c in enumerate(it_types)
                      if c in (XL_CELL_BLANK, XL_CELL_EMPTY)) + margin
    except StopIteration:
        return max_i


def _xlwings_margins(sheet, cell_up, cell_down, up, dn):
    if cell_up.col is not None and cell_up.row is not None and \
        (cell_down.col is None or cell_down.row is None): # from up
        if cell_down.col is None:
            dn[0] = _xlwings_min_index(sheet.row_types(up[1], up[0]),
                                       up[0], sheet.ncols)

        if cell_down.row is None:
            dn[1] = _xlwings_min_index(sheet.col_types(up[0], up[1]),
                                       up[1], sheet.nrows)
    elif cell_down.col is not None and cell_down.row is not None and \
        (cell_up.col is None or cell_up.row is None): # from bottom
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
        >>> get_rect_range(sheet,  Cell(None, None), Cell(None, None))
        [[None, 0, 1, 2],
         [0, None, None, None],
         [1, 5.1, 6.1, 7.1]]

        # up-left delimited minimum matrix
        >>> get_rect_range(sheet, Cell(0, 0), Cell(None, None))
        [[None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None],
         [None, None, None, None, 0, 1, 2],
         [None, None, None, 0, None, None, None],
         [None, None, None, 1, 5.1, 6.1, 7.1]]

        # get single value
        >>> get_rect_range(sheet, Cell(3, 6))
        0

        # get column vector
        >>> get_rect_range(sheet, Cell(3, None))
        [None, None, None, None, None, None, 0, 1]

        # get row vector
        >>> get_rect_range(sheet, Cell(None, 5))
        [None, None, None, None, 0, 1, 2]

        # up-left delimited minimum matrix
        >>> get_rect_range(sheet, Cell(4, None), Cell(None, None))
        [[0, 1, 2],
         [None, None, None],
         [5.1, 6.1, 7.1]]

        # delimited matrix
        >>> get_rect_range(sheet, Cell(3, 5), Cell(5, 7))
        [[None, 0, 1],
         [0, None, None],
         [1, 5.1, 6.1]]

        # down-right delimited minimum matrix
        >>> get_rect_range(sheet, Cell(None, None), Cell(5, 7))
        [[None, 0, 1],
         [0, None, None],
         [1, 5.1, 6.1]]

        # up-down-right delimited minimum matrix
        >>> get_rect_range(sheet, Cell(None, 6), Cell(5, 7))
        [[0, None, None],
         [1, 5.1, 6.1]]

        # down delimited minimum vector (i.e., column)
        >>> get_rect_range(sheet, Cell(5, None), Cell(5, 7))
        [1, None, 6.1]

        # right delimited minimum vector (i.e., row)
        >>> get_rect_range(sheet, Cell(2, 5), Cell(None, 5))
        [None, None, 0, 1, 2]
    """

    _pc = lambda cell: parse_cell(cell, epoch1904)

    if cell_down is None:  # vector or cell
        if cell_up.col is None:  # return row
            return list(map(_pc, sheet.row(cell_up.row)))
        elif cell_up.row is None:  # return column
            return list(map(_pc, sheet.col(cell_up.col)))
        else:  # return cell
            if cell_up.row < sheet.nrows and cell_up.col < sheet.ncols:
                return _pc(sheet.cell(cell_up.row, cell_up.col))
            return None
    else:  # table or vector or cell

        # set up margins
        up = [i if i is not None else 0 for i in cell_up]

        # set bottom margins
        dn = [cell_down.col + 1 if cell_down.col is not None else sheet.ncols,
              cell_down.row + 1 if cell_down.row is not None else sheet.nrows]

        nv = lambda x, v=None: [v] * x  # return a None vector  of length x

        if up[1] >= sheet.nrows or up[0] >= sheet.ncols:  # empty table
            ddn = [dn[i] - up[i] if c else 1
                   for i, c in enumerate([cell_down.col is not None,
                                          cell_down.row is not None])]
            return nv(ddn[1], nv(ddn[0]))

        if xlwings:
            up, dn = _xlwings_margins(sheet, cell_up, cell_down, up, dn)

        ddn = [max(0, v) for v in (dn[0] - sheet.ncols, dn[1] - sheet.nrows)]

        matrix = [list(map(_pc, sheet.row_slice(r, up[0], dn[0]))) + nv(ddn[0])
                  for r in range(up[1], dn[1] - ddn[1])]

        # add empty rows
        if ddn[0] == 0 and ddn[1] > 0:
            matrix += nv(ddn[1], nv(1))
        else:
            matrix += nv(ddn[1], nv(ddn[0]))

        # no empty vector
        ne_vct = lambda vct: any(x is not None for x in vct)

        def ind_row(t):  # return the index of first no empty row in the table
            return next((r for r, v in enumerate(t) if ne_vct(v)), 0)

        def reduce_table(t, u, d):  # return the minimum vertical table
            m = [ind_row(t) if u is None else 0,
                 len(t) - (ind_row(reversed(t)) if d is None else 0)]
            return t[m[0]:m[1]]

        if cell_up.row is None or cell_down.row is None:  # vertical reduction
            matrix = reduce_table(matrix, cell_up.row, cell_down.row)

        if cell_up.col is None or cell_down.col is None:  # horizontal reduction
            tbl = reduce_table(list(zip(*matrix)), cell_up.col, cell_down.col)
            matrix = [list(r) for r in zip(*tbl)]

        if cell_down.col is not None and cell_down.col == cell_up.col:  # vector
            matrix = [v[0] for v in matrix]

        if cell_down.row is not None and cell_down.row == cell_up.row:  # vector
            matrix = matrix[0]

        return matrix


_re_xl_ref_parser = re.compile(
    r"""
    ^\s*(?:(?P<xl_sheet_name>[^!]+)?!)?     # xl sheet name
    (?P<cell_up>                            # cell up [opt]
        (?P<u_c>[A-Z]+|_)                   # up col
        (?P<u_r>\d+|_)                      # up row
    )?
    (?P<cell_down>:(?:                      # cell down [opt]
        (?P<d_c>[A-Z]+|_)                   # down col
        (?P<d_r>\d+|_)                      # down row
    )?)?
    (?P<json>\{.*\})?                       # any json object [opt]
    \s*$""", re.IGNORECASE | re.X)


def parse_xl_ref(xl_ref):
    """
    Parses and fetches the contents of excel ref.

    :param xl_ref:
        a string with the following format:
        <xl_sheet_name>!<cell_up>:<cell_down>{<json_kwargs>}
        es. xl_sheet_name!UP10:DN20{"json":"..."}
    :type xl_ref: str

    :return:
        dictionary containing the following parameters::

        - xl_sheet_name
        - cell_up
        - cell_down
        - json

    :rtype: dict

    Example::

        >>> xl_ref = 'Sheet1!:{"1":4,"2":"ciao"}'
        >>> res = parse_xl_ref(xl_ref)

        >>> res['xl_sheet_name']
        'Sheet1'
        >>> res['cell_up']
        Cell(col=None, row=None)
        >>> res['cell_down']
        Cell(col=None, row=None)
        >>> res['json'] == {'2': 'ciao', '1': 4}
        True
    """

    try:
        r = _re_xl_ref_parser.match(xl_ref).groupdict()

        # resolve json
        r['json'] = json.loads(r['json']) if r['json'] else None

        # fetch cell_down
        r['cell_down'] = fetch_cell_ref(r['cell_down'], r.pop('d_c'),
                                        r.pop('d_r'))

        # fetch cell_up
        if r['cell_up'] is None:
            if r['cell_down']:
                r['cell_up'] = Cell(None, None)
            else:
                r['cell_up'] = Cell(0, 0)
                r['cell_down'] = Cell(None, None)

            del r['u_c'], r['u_r']

            return r

        r['cell_up'] = fetch_cell_ref(r['cell_up'], r.pop('u_c'), r.pop('u_r'))

        try:  # check range "crossing"
            if not r['cell_up'] < r['cell_down']:
                raise ValueError('%s < %s' % (r['cell_down'], r['cell_up']))
        except TypeError:
            pass  # Raised when tuples contain None.

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
        - cell_up
        - cell_down
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

