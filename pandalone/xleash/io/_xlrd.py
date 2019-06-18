#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2019European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Implements the *xlrd* backend of *xleash* that reads in-file Excel-spreadsheets.

.. currentmodule:: pandalone.xleash
"""

import datetime
from distutils.version import LooseVersion
import logging

from urllib import request
from urllib.parse import urlparse
from pandalone.xleash.io.backend import ABCBackend, ABCSheet, SheetId
from xlrd import (
    xldate,
    XL_CELL_DATE,
    XL_CELL_EMPTY,
    XL_CELL_TEXT,
    XL_CELL_BLANK,
    XL_CELL_ERROR,
    XL_CELL_BOOLEAN,
    XL_CELL_NUMBER,
)
import xlrd

import numpy as np

from .. import EmptyCaptureException, Coords, io_backends
from ... import utils, xlsutils


log = logging.getLogger(__name__)

# noinspection PyUnresolvedReferences
if LooseVersion(xlrd.__VERSION__) >= LooseVersion("0.9.3"):
    _xlrd_0_9_3 = True
else:
    _xlrd_0_9_3 = False


def _parse_cell(xcell, epoch1904=False):
    """
    Parse a xl-xcell.

    :param xlrd.Cell xcell: an excel xcell
    :type xcell: xlrd.sheet.Cell

    :param epoch1904:
        Which date system was in force when this file was last saved.
        False => 1900 system (the Excel for Windows default).
        True => 1904 system (the Excel for Macintosh default).
    :type epoch1904: bool

    :return: formatted xcell value
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

    ctype = xcell.ctype
    cvalue = xcell.value
    if ctype == XL_CELL_NUMBER:
        # GH5394 - Excel 'numbers' are always floats
        # it's a minimal perf hit and less suprising
        cint = int(cvalue)
        if cint == cvalue:
            return cint
        return cvalue
    elif ctype in (XL_CELL_EMPTY, XL_CELL_BLANK):
        return None  # RECT-LOOP NEVER USE THIS
    elif ctype == XL_CELL_TEXT:
        return cvalue
    elif ctype == XL_CELL_BOOLEAN:
        return bool(cvalue)
    elif ctype == XL_CELL_DATE:  # modified from Pandas library
        if _xlrd_0_9_3:
            # Use the newer xlrd datetime handling.
            d = xldate.xldate_as_datetime(cvalue, epoch1904)

            # Excel doesn't distinguish between dates and time, so we treat
            # dates on the epoch as times only. Also, Excel supports 1900 and
            # 1904 epochs.
            epoch = (1904, 1, 1) if epoch1904 else (1899, 12, 31)
            if (d.timetuple())[0:3] == epoch:
                d = datetime.time(d.hour, d.minute, d.second, d.microsecond)
        else:
            # Use the xlrd <= 0.9.2 date handling.
            d = xldate.xldate_as_tuple(xcell.value, epoch1904)
            if d[0] < datetime.MINYEAR:  # time
                d = datetime.time(*d[3:])
            else:  # date
                d = datetime.datetime(*d)
        return d
    elif ctype == XL_CELL_ERROR:
        return float("nan")

    raise ValueError(
        "Invalid XL-cell type(%s) for value(%s)!" % (xcell.ctype, xcell.value)
    )


def _open_sheet_by_name_or_index(xlrd_book, wb_id, sheet_id):
    """
    :param int or str or None sheet_id:
            If `None`, opens 1st sheet.
    """
    if sheet_id is None:
        sheet_id = 0
    if isinstance(sheet_id, int):
        xl_sh = xlrd_book.sheet_by_index(sheet_id)
    else:
        try:
            xl_sh = xlrd_book.sheet_by_name(sheet_id)
        except Exception as xl_ex:
            try:
                sheet_id = int(sheet_id)
            except ValueError:
                raise xl_ex from None
            else:
                xl_sh = xlrd_book.sheet_by_index(sheet_id)
    return XlrdSheet(xl_sh, wb_id)


class XlrdSheet(ABCSheet):
    """
    The *xlrd* workbook wrapper required by xleash library.
    """

    def __init__(self, sheet, book_fname, epoch1904=False):
        if not isinstance(sheet, xlrd.sheet.Sheet):
            raise ValueError("Invalid xlrd-sheet({})".format(sheet))
        self._sheet = sheet
        self._epoch1904 = epoch1904
        self.book_fname = book_fname

    def _close(self):
        """ Override it to release resources for this sheet."""
        self._sheet.book.unload_sheet(self._sheet.name)

    def _close_all(self):
        """ Override it to release resources this and all sibling sheets."""
        self._sheet.book.release_resources()

    def get_sheet_ids(self):
        sh = self._sheet
        return SheetId(self.book_fname or sh.book.filestr, [sh.name, sh.number])

    def open_sibling_sheet(self, sheet_id):
        """Gets by-index only if `sheet_id` is `int`, otherwise tries both by name and index."""
        return _open_sheet_by_name_or_index(self._sheet.book, self.book_fname, sheet_id)

    def list_sheetnames(self):
        return self._sheet.book.sheet_names()

    def _read_states_matrix(self):
        """See super-method. """
        types = np.asarray(self._sheet._cell_types)
        return (types != XL_CELL_EMPTY) & (types != XL_CELL_BLANK)

    def _read_margin_coords(self):
        nrows = self._sheet.nrows - 1
        ncols = self._sheet.ncols - 1
        if nrows < 0 or ncols < 0:
            raise EmptyCaptureException("empty sheet")
        return None, Coords(nrows, ncols)

    def read_rect(self, st, nd):
        """See super-method. """
        sheet = self._sheet

        if nd is None:
            return _parse_cell(sheet.cell(*st), self._epoch1904)

        rect = np.array([st, nd]) + [[0, 0], [1, 1]]
        states_matrix = self.get_states_matrix()

        table = []
        for r in range(*rect[:, 0]):
            row = []
            table.append(row)
            for c in range(*rect[:, 1]):
                try:
                    if states_matrix[r, c]:
                        c = _parse_cell(sheet.cell(r, c), self._epoch1904)
                        row.append(c)
                        continue
                except IndexError:
                    pass
                row.append(None)

        return table


class XlrdBackend(ABCBackend):
    def bid(self, wb_url):
        if wb_url:
            parts = urlparse(wb_url)
            path = utils.urlpath2path(parts.path)
            if xlsutils._xl_extensions_anywhere.search(path):
                return 100

    def open_sheet(self, wb_url, sheet_id):
        """
        Opens the local or remote `wb_url` *xlrd* workbook wrapped as :class:`XlrdSheet`.
        """
        assert wb_url, (wb_url, sheet_id)
        book = self._open_book(wb_url)

        return _open_sheet_by_name_or_index(book, wb_url, sheet_id)

    def list_sheetnames(self, wb_url):
        # TODO: QnD list_sheetnames()!
        book = self._open_book(wb_url)
        return book.sheet_names()

    def _open_book(self, url):
        parts = filename = urlparse(url)
        ropts = parts.params or {}
        if "logfile" not in ropts:
            ropts["logfile"] = utils.LoggerWriter(log, logging.DEBUG)
        if parts.scheme == "file":
            path = utils.urlpath2path(parts.path)
            log.info("Opening book(%r)...", path)
            book = xlrd.open_workbook(path, **ropts)
        else:
            ropts.pop("on_demand", None)
            http_opts = ropts.get("http_opts", {})
            with request.urlopen(url, **http_opts) as response:
                log.info("Opening book(%r)...", filename)
                book = xlrd.open_workbook(filename, file_contents=response, **ropts)

        return book


def load_as_xleash_plugin():
    loaded = [be for be in io_backends if isinstance(be, XlrdBackend)]
    if not loaded:
        io_backends.insert(0, XlrdBackend())
