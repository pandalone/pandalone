#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2019European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The high-level functionality, the filtering and recursive :term:`lassoing`.

Prefer accessing the public members from the parent module.

.. currentmodule:: pandalone.xleash
"""

import logging
from collections import OrderedDict

import pandas as pd
from pandas.io import parsers as pdparsers

from . import installed_filters

try:
    from pandas.io.excel._util import _fill_mi_header, _pop_header_name
except ImportError:
    # pandas < 0.25.0 (before summer 2019)
    from pandas.io.excel import _fill_mi_header, _pop_header_name


try:
    from pandas.errors import EmptyDataError
except ImportError:
    # From 0.23.4 (Aug 2018):
    #   https://pandas.pydata.org/pandas-docs/version/0.23.4/whatsnew.html#pandas-errors
    from pandas.io.common import EmptyDataError
try:
    from pandas.api.types import is_integer, is_list_like
except ImportError:
    # Moved on 0.19.0 (Oct 2016):
    #   https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.19.0.html#pandas-development-api
    from pandas.core.common import is_integer, is_list_like


log = logging.getLogger(__name__)


def _validate_header_arg(header):
    if isinstance(header, bool):
        raise TypeError(
            "Passing a bool to header is invalid. "
            "Use header=None for no header or "
            "header=int or list-like of ints to specify "
            "the row(s) making up the column names"
        )


def _maybe_convert_usecols(usecols):
    """
    Convert `usecols` into a compatible format for parsing in `parsers.py`.
    Parameters

    COPIED from:
    https://github.com/pandas-dev/pandas/blob/d47fc0cba3cf94ebd289ad3776bf7ff3fe60dfb8/pandas/io/excel/_util.py#L119

    ----------
    usecols : object
        The use-columns object to potentially convert.
    Returns
    -------
    converted : object
        The compatible format of `usecols`.
    """
    if usecols is None:
        return usecols

    if is_integer(usecols):
        import warnings

        warnings.warn(
            (
                "Passing in an integer for `usecols` has been "
                "deprecated. Please pass in a list of int from "
                "0 to `usecols` inclusive instead."
            ),
            FutureWarning,
            stacklevel=2,
        )
        return list(range(usecols + 1))

    if isinstance(usecols, str):
        return _range2cols(usecols)

    return usecols


def _df_filter(
    ranger,
    lasso,
    header=0,
    names=None,
    index_col=None,
    parse_cols=None,
    usecols=None,
    squeeze=False,
    dtype=None,
    engine=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    verbose=False,
    parse_dates=False,
    thousands=None,
    comment=None,
    skipfooter=0,
    convert_float=True,
    mangle_dupe_cols=True,
    **kwds
):
    """
    Converts captured values table as pandas DataFrame

    Most args copied from :func:`pandas.io.read_excel()` except:
    
        sheet_name, skip_footer, converters, date_parser

    Note that ``skip_footer`` has been deprecated by ``skipfooter``.
    """
    data = lasso.values

    # Copied & adapted from `pandas.io.excel.py` v0.24.2+ (Jun 2019)
    #    https://github.com/pandas-dev/pandas/blob/d47fc0c/pandas/io/excel/_base.py#L368

    _validate_header_arg(header)

    invalid_args = (
        set("skip_footer chunksize date_parser converted".split()) & kwds.keys()
    )
    if bool(invalid_args):
        raise NotImplementedError("Cannot implement args: %s" % invalid_args)

    if not data:
        return pd.DataFrame()

    usecols = _maybe_convert_usecols(usecols)

    if is_list_like(header) and len(header) == 1:
        header = header[0]

    # forward fill and pull out names for MultiIndex column
    header_names = None
    if header is not None and is_list_like(header):
        header_names = []
        control_row = [True for _ in data[0]]
        for row in header:
            if is_integer(skiprows):
                row += skiprows
            try:
                data[row], control_row = _fill_mi_header(data[row], control_row)
            except TypeError:
                ## Arg `control_row` introduced in pandas-v0.19.0 to fix
                #  https://github.com/pandas-dev/pandas/issues/12453
                #  https://github.com/pandas-dev/pandas/commit/67b72e3cbbaeb89a5b9c780b2fe1c8d5eaa9c505
                data[row] = _fill_mi_header(data[row])

            if index_col is not None:
                header_name, data[row] = _pop_header_name(data[row], index_col)
                header_names.append(header_name)

    if is_list_like(index_col):
        # forward fill values for MultiIndex index
        if not is_list_like(header):
            offset = 1 + header
        else:
            offset = 1 + max(header)

        # Check if we have an empty dataset
        # before trying to collect data.
        if offset < len(data):
            for col in index_col:
                last = data[offset][col]

                for row in range(offset + 1, len(data)):
                    if data[row][col] == "" or data[row][col] is None:
                        data[row][col] = last
                    else:
                        last = data[row][col]

    has_index_names = is_list_like(header) and len(header) > 1

    # Pandaas expect '' instead of `None`!
    data = [["" if c is None else c for c in r] for r in data]

    # GH 12292 : error when read one empty column from excel file
    try:
        parser = pdparsers.TextParser(
            data,
            names=names,
            header=header,
            index_col=index_col,
            has_index_names=has_index_names,
            squeeze=squeeze,
            dtype=dtype,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            usecols=usecols,
            mangle_dupe_cols=mangle_dupe_cols,
            **kwds
        )

        output = parser.read()

        if not squeeze or isinstance(output, pd.DataFrame):
            if header_names:
                output.columns = output.columns.set_names(header_names)
    except EmptyDataError:
        # No Data, return an empty DataFrame
        output = pd.DataFrame()

    lasso = lasso._replace(values=output)

    return lasso


def install_filters(filters_dict):
    filters_dict.update(
        {
            "df": {"func": _df_filter},
            "sr": {
                "func": lambda ranger, lasso, *args, **kwds: lasso._replace(
                    values=pd.Series(OrderedDict(lasso.values), *args, **kwds)
                ),
                "desc": (
                    "Converts a 2-columns list-of-lists into pd.Series.\n"
                    + pd.Series.__doc__
                ),
            },
        }
    )


def load_as_xleash_plugin():
    install_filters(installed_filters)
