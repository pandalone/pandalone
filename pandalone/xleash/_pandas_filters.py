#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The high-level functionality, the filtering and recursive :term:`lassoing`.

Prefer accessing the public members from the parent module.

.. currentmodule:: pandalone.xleash
"""

from __future__ import unicode_literals

from collections import OrderedDict
import pandas as pd
from pandas.io import parsers as pdparsers, excel as pdexcel, common as pdiocom
import pandas.api.types as pdtypes
from . import installed_filters

import logging

log = logging.getLogger(__name__)


def _validate_header_arg(header):
    if isinstance(header, bool):
        raise TypeError("Passing a bool to header is invalid. "
                        "Use header=None for no header or "
                        "header=int or list-like of ints to specify "
                        "the row(s) making up the column names")


def _df_filter(ranger, lasso, header=0, skiprows=None, names=None,
               skip_footer=0, index_col=None, has_index_names=None,
               parse_cols=None, parse_dates=False, date_parser=None,
               na_values=None, thousands=None, convert_float=True,
               verbose=False, squeeze=False, **kwds):
    """
    Converts captured values table as pandas DataFrame

    Doc below copied from :func:`pandas.io.read_excel()`:

    header : int, list of ints, default 0
        Row (0-indexed) to use for the column labels of the parsed
        DataFrame. If a list of integers is passed those row positions will
        be combined into a ``MultiIndex``
    skiprows : list-like
        Rows to skip at the beginning (0-indexed)
    skip_footer : int, default 0
        Rows at the end to skip (0-indexed)
    index_col : int, list of ints, default None
        Column (0-indexed) to use as the row labels of the DataFrame.
        Pass None if there is no such column.  If a list is passed,
        those columns will be combined into a ``MultiIndex``
    names : array-like, default None
        List of column names to use. If file contains no header row,
        then you should explicitly pass header=None
    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the Excel cell content, and return the transformed
        content.
    parse_cols : int or list, default None
        * If None then parse all columns,
        * If int then indicates last column to be parsed
        * If list of ints then indicates list of column numbers to be parsed
        * If string then indicates comma separated list of column names and
          column ranges (e.g. "A:E" or "A,C,E:F")
    squeeze : boolean, default False
        If the parsed data only contains one column then return a Series
    na_values : list-like, default None
        List of additional strings to recognize as NA/NaN
    thousands : str, default None
        Thousands separator for parsing string columns to numeric.  Note that
        this parameter is only necessary for columns stored as TEXT in Excel,
        any numeric columns will automatically be parsed, regardless of display
        format.
    keep_default_na : bool, default True
        If na_values are specified and keep_default_na is False the default NaN
        values are overridden, otherwise they're appended to
    verbose : boolean, default False
        Indicate number of NA values placed in non-numeric columns
    engine: string, default None
        If io is not a buffer or path, this must be set to identify io.
        Acceptable values are None or xlrd
    convert_float : boolean, default True
        convert integral floats to int (i.e., 1.0 --> 1). If False, all numeric
        data will be read in as floats: Excel stores all numbers as floats
        internally
    has_index_names : boolean, default None
        DEPRECATED: for version 0.17+ index names will be automatically
        inferred based on index_col.  To read Excel output from 0.16.2 and
        prior that had saved index names, use True.
    """
    data = lasso.values

    # Copied & adapted from `pandas.io.excel.py` v0.18.1
    #    https://github.com/pydata/pandas/releases/tag/v0.18.1

    skipfooter = kwds.pop('skipfooter', None)
    if skipfooter is not None:
        skip_footer = skipfooter

    _validate_header_arg(header)
    if has_index_names is not None:

        log.warning("\nThe has_index_names argument is deprecated; index names "
                    "will be automatically inferred based on index_col.\n"
                    "This argument is still necessary if reading Excel output "
                    "from 0.16.2 or prior with index names.")

    if 'chunksize' in kwds:
        raise NotImplementedError("chunksize keyword of read_excel "
                                  "is not implemented")
    if parse_dates:
        raise NotImplementedError("parse_dates keyword of read_excel "
                                  "is not implemented")

    if date_parser is not None:
        raise NotImplementedError("date_parser keyword of read_excel "
                                  "is not implemented")

    if not data:
        return pd.DataFrame()

    if pdtypes.is_list_like(header) and len(header) == 1:
        header = header[0]

    # forward fill and pull out names for MultiIndex column
    header_names = None
    if header is not None:
        if pdtypes.is_list_like(header):
            header_names = []
            control_row = [True for _ in data[0]]
            for row in header:
                if pdtypes.is_integer(skiprows):
                    row += skiprows
                try:
                    data[row], control_row = pdexcel._fill_mi_header(data[row], control_row)
                except TypeError:
                    ## Arg `control_row` introduced in pandas-v0.19.0 to fix
                    #  https://github.com/pandas-dev/pandas/issues/12453
                    #  https://github.com/pandas-dev/pandas/commit/67b72e3cbbaeb89a5b9c780b2fe1c8d5eaa9c505
                    data[row] = pdexcel._fill_mi_header(data[row])

                header_name, data[row] = pdexcel._pop_header_name(
                    data[row], index_col)
                header_names.append(header_name)
        else:
            data[header] = pdexcel._trim_excel_header(data[header])

    if pdtypes.is_list_like(index_col):
        # forward fill values for MultiIndex index
        if not pdtypes.is_list_like(header):
            offset = 1 + header
        else:
            offset = 1 + max(header)

        for col in index_col:
            last = data[offset][col]
            for row in range(offset + 1, len(data)):
                if data[row][col] == '' or data[row][col] is None:
                    data[row][col] = last
                else:
                    last = data[row][col]

    if pdtypes.is_list_like(header) and len(header) > 1:
        has_index_names = True

    # Pandaas expect '' instead of `None`!
    data = [['' if c is None else c for c in r] for r in data]

    # GH 12292 : error when read one empty column from excel file
    try:
        parser = pdparsers.TextParser(data, header=header, index_col=index_col,
                                      has_index_names=has_index_names,
                                      na_values=na_values,
                                      thousands=thousands,
                                      parse_dates=parse_dates,
                                      date_parser=date_parser,
                                      skiprows=skiprows,
                                      skip_footer=skip_footer,
                                      squeeze=squeeze,
                                      **kwds)

        output = parser.read()
        if names is not None:
            output.columns = names
        if not squeeze or isinstance(output, pd.DataFrame):
            output.columns = output.columns.set_names(header_names)
    except pdiocom.EmptyDataError:
        # No Data, return an empty DataFrame
        output = pd.DataFrame()

    lasso = lasso._replace(values=output)

    return lasso


def install_filters(filters_dict):
    filters_dict.update({
        'df': {
            'func': _df_filter,
        },
        'sr': {
            'func': lambda ranger, lasso, *args, **kwds: lasso._replace(
                values=pd.Series(OrderedDict(lasso.values), *args, **kwds)),
            'desc': ("Converts a 2-columns list-of-lists into pd.Series.\n" +
                     pd.Series.__doc__),
        }
    })


def load_as_xleash_plugin():
    install_filters(installed_filters)
