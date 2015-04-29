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
import pandas as pd
import numpy as np
from string import ascii_uppercase
from collections import namedtuple
# noinspection PyUnresolvedReferences
from six.moves.urllib.parse import urldefrag
# noinspection PyUnresolvedReferences
from six.moves.urllib.request import urlopen
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
    ^\s*(?:(?P<xl_sheet_name>[^!]+)?!)?     # xl sheet name
    (?P<cell_up>                            # cell up [opt]
        (?P<u_c>[A-Z]+|_|\*)                # up col
        (?P<u_r>\d+|_|\*)                   # up row
    )?
    (?P<cell_down>:(?:                      # cell down [opt]
        (?P<d_c>[A-Z]+|_|\*)                # down col
        (?P<d_r>\d+|_|\*)                   # down row
    )?)?
    (?P<json>\{.*\})?                       # any json object [opt]
    \s*$""", re.IGNORECASE | re.X)


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

_cell = {
    None: None,
    '*': None,
    '_': 'xl_margin'
}

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
        >>> fetch_cell_ref('*1', '*', '1')
        Cell(col=None, row=0)
        >>> fetch_cell_ref('A*', 'A', '*')
        Cell(col=0, row=None)
        >>> fetch_cell_ref('A_', 'A', '_')
        Cell(col=0, row='xl_margin')
        >>> fetch_cell_ref(':', None, None)
        Cell(col=None, row=None)
        >>> fetch_cell_ref(None, None, None)

    """
    if cell is None:
        return None

    if cell_row != '0':
        row = _cell[cell_row] if cell_row in _cell else int(cell_row) - 1
        col = _cell[cell_col] if cell_col in _cell else col2num(cell_col)
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

        # right delimited minimum vector (i.e., row)
        >>> get_rect_range(sheet, Cell(3, 6), Cell('xl_margin', 6))
        [0, None, None, None]

        # right delimited minimum vector (i.e., row)
        >>> get_rect_range(sheet, Cell(None, None), Cell(3, 5))
        [[]]
    """

    _pc = lambda cell: parse_cell(cell, epoch1904)

    if cell_down is None:  # vector or cell
        # set up margins
        _up = {'xl_margin': 0}
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
        # set up margins
        _up = dict.fromkeys([None, 'xl_margin'], 0)
        up = [_up.get(i, i) for i in cell_up]

        # set bottom margins
        _dn = [dict.fromkeys([None, 'xl_margin'], sheet.ncols - 1),
               dict.fromkeys([None, 'xl_margin'], sheet.nrows - 1)]
        dn = [_dn[i].get(j, j) + 1 for i,j in enumerate(cell_down)]

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

        def ind_row(t, d):  # return the index of first no empty row in the table
            return next((r for r, v in enumerate(t) if ne_vct(v)), d)

        def reduce_table(t, u, d):  # return the minimum vertical table
            l = len(t)
            m = [ind_row(t, l) if u is None else 0,
                 l - (ind_row(reversed(t), 0) if d is None else 0)]
            return t[m[0]:m[1]] if m[0]!=m[1] else [[]]

        if cell_up.row is None or cell_down.row is None:  # vertical reduction
            matrix = reduce_table(matrix, cell_up.row, cell_down.row)

        if cell_up.col is None or cell_down.col is None:  # horizontal reduction
            tbl = reduce_table(list(zip(*matrix)), cell_up.col, cell_down.col)
            matrix = [list(r) for r in zip(*tbl)] if tbl !=[[]] else [[]]

        if cell_down.col is not None and cell_down.col == cell_up.col:  # vector
            matrix = [v[0] for v in matrix]

        if cell_down.row is not None and cell_down.row == cell_up.row:  # vector
            matrix = matrix[0]

        if isinstance(matrix, list):
            return matrix
        else:
            return [matrix]


def parse_xl_ref(xl_ref):
    """
    Parses and fetches the contents of excel ref.

    :param xl_ref:
        a string with the following format:
        <xl_sheet_name>!<cell_up>:<cell_down>{<json>}
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
            pass  # Raised when tuples contain None or str.

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


def open_xl_workbook(xl_ref_child, xl_ref_parent=None):
    """
    Opens the excel workbook of an excel ref.

    :param xl_ref_child: excel ref of the child
    :type xl_ref_child: dict

    :param xl_ref_parent: excel ref of the parent
    :type xl_ref_parent: dict, None, optional
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
    if n>0:
        return [_redim_value(value, n - 1)]
    elif n<0:
        if len(value)>1:
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
    None:{'fun':lambda x: x},
    'df':{'fun': pd.DataFrame},
    'nparray':{'fun': np.array},
    'dict': {'fun': dict},
    'sorted':{'fun': sorted}
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
            val = process_xl_range(val,**v)
    return val
