#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The algorithmic part of :term:`capturing`.

Prefer accessing the public members from the parent module.

.. currentmodule:: pandalone.xleash
"""

from __future__ import unicode_literals

from future.utils import iteritems
from abc import abstractmethod, ABCMeta
from collections import namedtuple

from future.utils import with_metaclass

import itertools as itt
import numpy as np

from .. import Coords
from ...utils import as_list


def margin_coords_from_states_matrix(states_matrix):
    """
    Returns top-left/bottom-down margins of full cells from a :term:`state` matrix.

    May be used by :meth:`ABCSheet.get_margin_coords()` if a backend
    does not report the sheet-margins internally.

    :param np.ndarray states_matrix:
            A 2D-array with `False` wherever cell are blank or empty.
            Use :meth:`ABCSheet.get_states_matrix()` to derrive it.
    :return:    the 2 coords of the top-left & bottom-right full cells
    :rtype:     (Coords, Coords)

    Examples::
        >>> from io._sheets import margin_coords_from_states_matrix

        >>> states_matrix = np.asarray([
        ...    [0, 0, 0],
        ...    [0, 1, 0],
        ...    [0, 1, 1],
        ...    [0, 0, 1],
        ... ])
        >>> margins = margin_coords_from_states_matrix(states_matrix)
        >>> margins
        (Coords(row=1, col=1), Coords(row=3, col=2))


    Note that the botom-left cell is not the same as `states_matrix` matrix size::

        >>> states_matrix = np.asarray([
        ...    [0, 0, 0, 0],
        ...    [0, 1, 0, 0],
        ...    [0, 1, 1, 0],
        ...    [0, 0, 1, 0],
        ...    [0, 0, 0, 0],
        ... ])
        >>> margin_coords_from_states_matrix(states_matrix) == margins
        True

    """
    if not states_matrix.any():
        c = Coords(0, 0)
        return c, c
    indices = np.array(np.where(states_matrix), dtype=np.int16).T

    # return indices.min(0), indices.max(0)
    return Coords(*indices.min(0)), Coords(*indices.max(0))


SheetId = namedtuple('SheetId', ('book', 'ids'))


class ABCSheet(with_metaclass(ABCMeta, object)):
    """
    A delegating to backend factory and sheet-wrapper with utility methods.

    :param np.ndarray _states_matrix:
            The :term:`states-matrix` cached, so recreate object
            to refresh it.
    :param dict _margin_coords:
            limits used by :func:`_resolve_cell`, cached, so recreate object
            to refresh it.

    Resource management is outside of the scope of this class,
    and must happen in the backend workbook/sheet instance.

    *xlrd* examples::

        >>> import xlrd                                       #  doctest: +SKIP
        >>> with xlrd.open_workbook(self.tmp) as wb:          #  doctest: +SKIP
        ...     sheet = xleash.xlrdSheet(wb.sheet_by_name('Sheet1'))
        ...     ## Do whatever

    *win32* examples::

        >>> with dsgdsdsfsd as wb:          #  doctest: +SKIP
        ...     sheet = xleash.win32Sheet(wb.sheet['Sheet1'])
        TODO: Win32 Sheet example
    """

    _states_matrix = None
    _margin_coords = None

    def _close(self):
        """ Override it to release resources for this sheet."""

    def _close_all(self):
        """ Override it to release resources this and all sibling sheets."""

    @abstractmethod
    def get_sheet_ids(self):
        """
        :return: a 2-tuple of its wb-name and a sheet-ids of this sheet i.e. name & indx
        :rtype: SheetId or None
        """

    @abstractmethod
    def open_sibling_sheet(self, sheet_id, opts=None):
        """Return a sibling sheet by the given index or name"""

    @abstractmethod
    def _read_states_matrix(self):
        """
        Read the :term:`states-matrix` of the wrapped sheet.

        :return:   A 2D-array with `False` wherever cell are blank or empty.
        :rtype:     ndarray
        """

    def get_states_matrix(self):
        """
        Read and cache the :term:`states-matrix` of the wrapped sheet.

        :return:   A 2D-array with `False` wherever cell are blank or empty.
        :rtype:     ndarray
        """
        if self._states_matrix is None:
            self._states_matrix = self._read_states_matrix()
        return self._states_matrix

    @abstractmethod
    def read_rect(self, st, nd):
        """
        Fecth the actual values from the backend Excel-sheet.

        :param Coords st:
                the top-left edge, inclusive
        :param Coords, None nd:
                the bottom-right edge, inclusive(!); when `None`,
                must return a scalar value.
        :return:
                Depends on whether both coords are given:
                    - If both given, 2D list-lists with the values of the rect,
                      which might be empty if beyond limits.
                    - If only 1st given, the scalar value, and if
                      beyond margins, raise error!

        :rtype: list
        """

    def _read_margin_coords(self):
        """
        Override if possible to read (any of the) limits directly from the sheet.

        :return:    the 2 coords of the top-left & bottom-right full cells;
                    anyone coords can be None.
                    By default returns ``(None, None)``.
        :rtype:     (Coords, Coords)

        """
        return None, None  # pragma: no cover

    def get_margin_coords(self):
        """
        Extract (and cache) margins either internally or from :func:`margin_coords_from_states_matrix()`.

        :return:    the resolved top-left and bottom-right :class:`.xleash.Coords`
        :rtype:     tuple

        """
        if not self._margin_coords:
            up, dn = self._read_margin_coords()
            if up is None or dn is None:
                sm = self.get_states_matrix()
                up1, dn1 = margin_coords_from_states_matrix(sm)
                up = up or up1
                dn = dn or dn1
            self._margin_coords = up, dn

        return self._margin_coords

    def __repr__(self):
        args = (type(self).__name__, ) + self.get_sheet_ids()
        return '%s(book=%r, sheet_ids=%r)' % args


class ArraySheet(ABCSheet):
    """A sample :class:`ABCSheet` made out of 2D-list or numpy-arrays, for facilitating tests."""

    def __init__(self, arr, ids=SheetId('wb', ['sh', 0])):
        self._arr = np.asarray(arr)
        self._ids = ids

    def open_sibling_sheet(self, sheet_id):
        raise NotImplementedError()

    def get_sheet_ids(self):
        return self._ids

    def _read_states_matrix(self):
        return ~np.equal(self._arr, None)

    def read_rect(self, st, nd):
        if nd is None:
            return self._arr[st]
        rect = np.array([st, nd]) + [[0, 0], [1, 1]]
        return self._arr[slice(*rect[:, 0]), slice(*rect[:, 1])].tolist()

    def __repr__(self):
        return 'ArraySheet(%s, \n%s)' % (self.get_sheet_ids(), self._arr)


class SheetsFactory(object):
    """
    A caching-store of :class:`ABCSheet` instances, serving them based on (workbook, sheet) IDs, optionally creating them from backends.

    :ivar dict _cached_sheets:
            A cache of all _Spreadsheets accessed so far,
            keyed by multiple keys generated by :meth:`_derive_sheet_keys`.

    - To avoid opening non-trivial workbooks, use the :meth:`add_sheet()`
      to pre-populate this cache with them.

    - The last sheet added becomes the *current-sheet*, and will be
      served when :term:`xl-ref` does not specify any workbook and sheet.

      .. Tip::
          For the simplest API usage, try this::

              >>> sf = SheetsFactory()
              >>> sf.add_sheet(some_sheet)              # doctest: +SKIP
              >>> lasso('A1:C3(U)', sf)                 # doctest: +SKIP

    - The *current-sheet* is served only when wokbook-id is `None`, that is,
      the id-pair ``('foo.xlsx', None)`` does not hit it, so those ids
      are send to the cache as they are.

    - To add another backend, modify the opening-sheets logic (ie clipboard),
      override :meth:`_open_sheet()`.

    - It is a resource-manager for contained sheets, wo it can be used wth
      a `with` statement.

    """

    def __init__(self):
        self._cached_sheets = {}

    def _cache_get(self, key):
        wb, sh = key
        if wb in self._cached_sheets:
            shs = self._cached_sheets[wb]
            return shs.get(sh, None)

    def _cache_put(self, key, sheet):
        wb, sh = key
        if wb in self._cached_sheets:
            sh_dict = self._cached_sheets[wb]
        else:
            sh_dict = self._cached_sheets[wb] = {}
        sh_dict[sh] = sheet

    def _build_sheet_key(self, wb, sh):
        assert wb is not None, (wb, sh)
        return (wb, sh)

    def _derive_sheet_keys(self, sheet,  wb_ids=None, sh_ids=None):
        """
        Retuns the product of user-specified and sheet-internal keys.

        :param wb_ids:
                a single or a sequence of extra workbook-ids (ie: file, url)
        :param sh_ids:
                a single or sequence of extra sheet-ids (ie: name, index, None)
        """
        wb_id, sh_ids2 = sheet.get_sheet_ids()
        assert wb_id is not None, (wb_id, sh_ids2)
        wb_ids = [wb_id] + as_list(wb_ids)
        sh_ids = sh_ids2 + as_list(sh_ids)

        key_pairs = itt.product(wb_ids, sh_ids)
        keys = list(set(self._build_sheet_key(*p)
                        for p in key_pairs
                        if p[0]))
        assert keys, (keys, sheet,  wb_ids, sh_ids)

        return keys

    def _close_sheet(self, key):
        sheet = self._cache_get(key)
        if sheet:
            sheet._close()
            for sh_dict in self._cached_sheets.values():
                for sh_id, sh in list(iteritems(sh_dict)):
                    if sh is sheet:
                        del sh_dict[sh_id]

    def close(self):
        """Closes all contained sheets and empties cache."""
        for sh_dict in self._cached_sheets.values():
            for sh in sh_dict.values():
                sh._close_all()
        self._cached_sheets = {}

    def add_sheet(self, sheet, wb_ids=None, sh_ids=None):
        """
        Updates cache.

        :param wb_ids:
                a single or sequence of extra workbook-ids (ie: file, url)
        :param sh_ids:
                a single or sequence of extra sheet-ids (ie: name, index, None)
        """
        assert sheet, (sheet, wb_ids, sh_ids)
        keys = self._derive_sheet_keys(sheet, wb_ids, sh_ids)
        for k in keys:
            old_sheet = self._cache_get(k)
            if old_sheet and old_sheet is not sheet:
                self._close_sheet(k)
            self._cache_put(k, sheet)

    def fetch_sheet(self, wb_id, sheet_id, opts={}, base_sheet=None):
        """
        :param ABCSheet base_sheet:
            The sheet used when unspecified `wb_id`.
        """
        if wb_id is None:
            if not base_sheet:
                msg = "No `base_sheet` given! Specify a Workbook."
                raise ValueError(msg)

            if sheet_id is None:
                return base_sheet

            wb_id, _c_sh_ids = base_sheet.get_sheet_ids()
            assert wb_id is not None, (base_sheet, _c_sh_ids)

            key = self._build_sheet_key(wb_id, sheet_id)
            sheet = self._cache_get(key)

            if not sheet:
                sheet = base_sheet.open_sibling_sheet(sheet_id, opts)
                assert sheet, (wb_id, sheet_id, opts)
                self.add_sheet(sheet, wb_id, sheet_id)
        else:
            key = self._build_sheet_key(wb_id, sheet_id)
            sheet = self._cache_get(key)
            if not sheet:
                sheet = self._open_sheet(wb_id, sheet_id, opts)
                assert sheet, (wb_id, sheet_id, opts)
                self.add_sheet(sheet, wb_id, sheet_id)

        return sheet

    def _open_sheet(self, wb_id, sheet_id, opts):
        """OVERRIDE THIS to change backend."""
        from . import _xlrd
        return _xlrd.open_sheet(wb_id, sheet_id, opts)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()
