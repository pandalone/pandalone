#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Implements the *xlrd* backend of :term:`xl-ref` that reads in-file Excel-spreadsheets.

Prefer accessing the public members from the parent module.
"""
import datetime
from distutils.version import LooseVersion

from xlrd import (xldate, XL_CELL_DATE, XL_CELL_EMPTY, XL_CELL_TEXT,
                  XL_CELL_BLANK, XL_CELL_ERROR, XL_CELL_BOOLEAN, XL_CELL_NUMBER)
from xlrd import colname  # @UnusedImport
import xlrd

import numpy as np
from six.moves.urllib.request import urlopen  # @UnresolvedImport


# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    _xlrd_0_9_3 = True
else:
    _xlrd_0_9_3 = False


def open_xlref_workbook(xl_ref_child, xl_ref_parent=None):
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

# noinspection PyProtectedMember


def read_states_matrix(sheet):
    """
    Return a boolean ndarray with `False` wherever cell are blank or empty.
    """
    types = np.array(sheet._cell_types)
    return (types != XL_CELL_EMPTY) & (types != XL_CELL_BLANK)


def read_rect(sheet, states_matrix, st_cell, nd_cell):
    """
    Extract the values enclaved between the 2 edge-cells as a 2D-table.

    :param sheet: a xlrd-sheet to read from
    :param xlref.Cell st_cell: the starting-cell of the rect
    :param xlref.Cell nd_cell: the finishing-cell of the rect
    """
    table = []
    for r in range(st_cell.row, nd_cell.row + 1):
        row = []
        table.append(row)
        for c in range(st_cell.col, nd_cell.col + 1):
            try:
                if states_matrix[r, c]:
                    row.append(_parse_cell(sheet.cell(r, c)))
                    continue
            except IndexError:
                pass
            row.append(None)

    return table


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
        if _xlrd_0_9_3:
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
