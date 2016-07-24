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
from pandas.io import parsers, excel as pdexcel


def _df_filter(ranger, lasso, *args, **kwds):
    values = lasso.values
    header = kwds.get('header', 'infer')
    if header == 'infer':
        header = kwds['header'] = 0 if kwds.get(
            'names') is None else None
    if header is not None:
        values[header] = pdexcel._trim_excel_header(values[header])
    # , convert_float=True,
    parser = parsers.TextParser(values, **kwds)
    lasso = lasso._replace(values=parser.read())

    return lasso


def install_filters(filters):

    filters.update({
        'df': {
            'func': _df_filter,
            'desc': parsers.TextParser.__doc__,
        },
        'series': {
            'func': lambda ranger, lasso, *args, **kwds: pd.Series(OrderedDict(lasso.values),
                                                                   *args, **kwds),
            'desc': ("Converts a 2-columns list-of-lists into pd.Series.\n" +
                     pd.Series.__doc__),
        }
    })

    return filters
