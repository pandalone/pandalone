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

from collections import namedtuple, OrderedDict
import logging
import functools as fnt


from asteval import Interpreter
from future.utils import iteritems
from past.builtins import basestring
from toolz import dicttoolz as dtz

import numpy as np

from . import Lasso, _parse
from ..utils import LoggerWriter
from ..utils import as_list


log = logging.getLogger(__name__)


def pipe_filter(ranger, lasso, *pipe):
    """
    A :term:`bulk-filter` that applies all call-specifiers one after another on the :term:`capture-rect` values.

    :param list pipe: the call-specifiers
    """

    for call_spec_values in pipe:
        call_spec = _parse.parse_call_spec(call_spec_values)
        lasso = ranger.make_call(lasso, *call_spec)

    return lasso


def _classify_rect_shape(st, nd):
    """
    Identifies rect from its edge-coordinates (row, col, 2d-table)..

    :param Coords st:
            the top-left edge of capture-rect, inclusive
    :param Coords or None nd:
            the bottom-right edge of capture-rect, inclusive
    :return:
            in int based on the input like that:

            - 0: only `st` given
            - 1: `st` and `nd` point the same cell
            - 2: row
            - 3: col
            - 4: 2d-table

    Examples::

        >>> _classify_rect_shape((1,1), None)
        0
        >>> _classify_rect_shape((2,2), (2,2))
        1
        >>> _classify_rect_shape((2,2), (2,20))
        2
        >>> _classify_rect_shape((2,2), (20,2))
        3
        >>> _classify_rect_shape((2,2), (20,20))
        4
    """
    if nd is None:
        return 0
    rows = nd[0] - st[0]
    cols = nd[1] - st[1]
    return 1 + bool(cols) + 2 * bool(rows)


def _decide_ndim_by_rect_shape(shape_idx, ndims_list):
    return ndims_list[shape_idx]


def _updim(values, new_ndim):
    """
    Append trivial dimensions to the left.

    :param values:      The scalar ot 2D-results of :meth:`Sheet.read_rect()`
    :param int new_dim: The new dimension the result should have
    """
    new_shape = (1,) * (new_ndim - values.ndim) + values.shape
    return values.reshape(new_shape)


def _downdim(values, new_ndim):
    """
    Squeeze it, and then flatten it, before inflating it.

    :param values:       The scalar ot 2D-results of :meth:`Sheet.read_rect()`
    :param int new_dim: The new dimension the result should have
    """
    trivial_indxs = [i for i, d in enumerate(values.shape) if d == 1]
    offset = values.ndim - new_ndim
    trivial_ndims = len(trivial_indxs)
    if offset > trivial_ndims:
        values = values.flatten()
    elif offset == trivial_ndims:
        values = values.squeeze()
    else:
        for _, indx in zip(range(offset), trivial_indxs):
            values = values.squeeze(indx)

    return values


def _redim(values, new_ndim):
    """
    Reshapes the :term:`capture-rect` values of :func:`read_capture_rect()`.

    :param values:      The scalar ot 2D-results of :meth:`Sheet.read_rect()`
    :type values: (nested) list, *
    :param new_ndim:
    :type int, (int, bool) or None new_ndim:

    :return: reshaped values
    :rtype: list of lists, list, *


    Examples::

        >>> _redim([1, 2], 2)
        [[1, 2]]

        >>> _redim([[1, 2]], 1)
        [1, 2]

        >>> _redim([], 2)
        [[]]

        >>> _redim([[3.14]], 0)
        3.14

        >>> _redim([[11, 22]], 0)
        [11, 22]

        >>> arr = [[[11], [22]]]
        >>> arr == _redim(arr, None)
        True

        >>> _redim([[11, 22]], 0)
        [11, 22]
    """
    if new_ndim is None:
        return values

    values = np.asarray(values)
    try:
        new_ndim, transpose = new_ndim
        if transpose:
            values = values.T
    except:
        pass
    if new_ndim is not None:
        if values.ndim < new_ndim:
            values = _updim(values, new_ndim)
        elif values.ndim > new_ndim:
            values = _downdim(values, new_ndim)

    return values.tolist()


def xlwings_dims_call_spec():
    """A list :term:`call-spec` for :meth:`_redim_filter` :term:`filter` that imitates results of *xlwings* library."""
    return '["redim", [0, 1, 1, 1, 2]]'


def redim_filter(ranger, lasso,
                 scalar=None, cell=None, row=None, col=None, table=None):
    """
    A :term:`bulk-filter` that reshapes sand/or transpose captured values, depending on rect's shape.

    Each dimension might be a single int or None, or a pair [dim, transpose].
    """
    ndims_list = (scalar, cell, row, col, table)
    shape_idx = _classify_rect_shape(lasso.st, lasso.nd)
    new_ndim = _decide_ndim_by_rect_shape(shape_idx, ndims_list)
    values = lasso.values
    if new_ndim is not None:
        lasso = lasso._replace(values=_redim(values, new_ndim))

    return lasso


Context = namedtuple('Context', ('sheet', 'base_coords'))
"""
Fields extracted from :meth:`do_lasso()` `context_kwds` arg  on recursive invocations.

Practically func:`recursive_filter() preserves these fields if the parsed
ones were `None`.
"""


def _run_filter_elementwise(ranger, lasso, element_func, subfilters,
                         include=None, exclude=None, depth=-1):
    """
    Implement :term:`element-wise` filters strings found by treating values as mappings (dicts, df, series) and/or nested lists.

    - The `include`/`exclude` filter args work only for "indexed" objects
      with ``items()`` or ``iteritems()`` and indexing methods,
      i.e. Mappings, Series and Ddataframes.

      - If no filter arg specified, expands for all keys.
      - If only `include` specified, rejects all keys not explicitly
        contained in this filter arg.
      - If only `exclude` specified, expands all keys not explicitly
        contained in this filter arg.
      - When both `include`/`exclude` exist, only those explicitly
        included are accepted, unless also excluded.

    - Lower the :mod:`logging` level to see other than syntax-errors on
      recursion reported on :data:`log`.
    - Only those in :class:`Context` are passed recursively.

    :param list element_func:
            A function implementing the element-wise :term:`filter`
            and returning a 2-tuple ``(is_proccessed, new_val_or_lasso)``,
            like that::

                def element_func(ranger, lasso, context, vals, **kwds)
                    proced = False
                    try:
                        vals = int(vals)
                        proced = True
                    except ValueError:
                        pass
                    return proced, vals

            The `kwds` may contain the `include`, `exclude` and `depth`.
    :param list subfilters:
            Any :term:`filters` to apply after invoking the `element_func`.
    :param list or str include:
            Items to include when diving into "indexed" values.
            See description above.
    :param list or str exclude:
            Items to exclude when diving into "indexed" values.
            See description above.
    :param int or None depth:
            How deep to dive into nested structures, "indexed" or lists.
            If `< 0`, no limit. If 0, stops completely.
    """
    include = include and as_list(include)
    exclude = exclude and as_list(exclude)

    def is_included(key):
        ok = not include or key in include
        ok &= not exclude or key not in exclude
        return ok

    def new_base_coords(base_coords, cdepth, i):
        if base_coords:
            if cdepth == 0:
                base_coords = base_coords._replace(row=i)
            elif cdepth == 1:
                base_coords = base_coords._replace(col=i)
        return base_coords

    def call_element_func(vals, base_coords, cdepth):
        context_kwds = dtz.keyfilter(lambda k: k in Context._fields,
                                     lasso._asdict())
        context_kwds['base_coords'] = base_coords
        context = Context(**context_kwds)
        proced, rec_lasso = element_func(ranger, lasso, context, vals)
        if proced:
            if not isinstance(rec_lasso, Lasso):
                rec_lasso = lasso._replace(values=rec_lasso)

            if sub_call_spec:
                rec_lasso = ranger.make_call(rec_lasso, *sub_call_spec)
            vals = rec_lasso and rec_lasso.values

        return proced, vals

    def dive_list(vals, base_coords, cdepth):
        proced, vals = call_element_func(vals, base_coords, cdepth)
        if not proced and isinstance(vals, list):
            for i, v in enumerate(vals):
                nbc = new_base_coords(base_coords, cdepth, i)
                vals[i] = dive_indexed(v, nbc, cdepth + 1)

        return vals

    def dive_indexed(vals, base_coords, cdepth):
        if cdepth != depth:
            dived = False
            try:
                items = iteritems(vals)
            except:
                pass  # Just to avoid chained ex.
            else:
                for i, (k, v) in enumerate(items):
                    # Dict is not ordered, so cannot locate `base_coords`!
                    if is_included(k):
                        nbc = (None
                               if isinstance(vals, dict)
                               else new_base_coords(base_coords, cdepth, i))
                        vals[k] = dive_indexed(v, nbc, cdepth + 1)
                dived = True
            if not dived:
                vals = dive_list(vals, base_coords, cdepth)

        return vals

    sub_call_spec = subfilters and _parse.parse_call_spec(subfilters)
    values = dive_indexed(lasso.values, lasso.st, 0)

    return lasso._replace(values=values)


def _recurse_element_func(ranger, lasso, context, vals, **kwds):
    proced = False
    try:
        if isinstance(vals, basestring):
            lasso = ranger.do_lasso(vals, **context._asdict())
            proced = True
    except SyntaxError as ex:
        msg = "Skipped non xl-ref(%s) due to: %s"
        log.debug(msg, vals, ex)
    except Exception as ex:
        msg = "Lassoing  xl-ref(%s) at %s, %s stopped due to: \n  %s"
        msg %= (vals, ) + context + (ex, )
        raise ValueError(msg)

    return proced, lasso


def recursive_filter(ranger, lasso, *subfilters, **kwds):
    """
    A :term:`element-wise-filter` that expand recursively any :term:`xl-ref` strings elements in :term:`capture-rect` values.

    Note that in python-3 the signature woudl be::

        def recursive_filter(ranger, lasso, element_func, subfilters,
                             include=None, exclude=None, depth=-1):
    """
    include = kwds.pop('include', None)
    exclude = kwds.pop('exclude', None)
    depth = kwds.pop('depth', -1)
    return _run_filter_elementwise(ranger, lasso, _recurse_element_func, subfilters,
                                include=include, exclude=exclude, depth=depth)


ast_log_writer = LoggerWriter(logging.getLogger('%s.eval' % __name__),
                              logging.INFO)


def eval_filter(ranger, lasso):
    """
    A :term:`element-wise-filter` that uses :mod:`asteval` to evaluate string values as python expressions.

    The `expr` fecthed from `term:`capturing` may access read-write
    all :func:`locals()` of this method(`ranger`, `lasso`), :mod:`numpy` funcs,
    and the :mod:`pandalone.xleash` module under the `xlash` variable.

    The `expr` may return either:
        - the processed values, or
        - an instance of the :class:`Lasso`, in which case only its `opt`
          field is checked and replaced with original if missing.
          So better user :func:`namedtuple._replace()` on the current `lasso`
          which exists in the globals.


    Example::

        >>> expr = '''
        ... res = array([[0.5, 0.3, 0.1, 0.1]])
        ... res * res.T
        ... '''
        >>> lasso = Lasso(values=expr, opts={})
        >>> ranger = Ranger(None)
        >>> eval_filter(ranger, lasso).values
        array([[ 0.25,  0.15,  0.05,  0.05],
               [ 0.15,  0.09,  0.03,  0.03],
               [ 0.05,  0.03,  0.01,  0.01],
               [ 0.05,  0.03,  0.01,  0.01]])
    """
    expr = str(lasso.values)
    symtable = locals()
    from .. import xleash
    symtable.update({'xleash': xleash})
    aeval = Interpreter(symtable, writer=ast_log_writer)
    res = aeval.eval(expr)
    if aeval.error:
        msg = "While py-eval %r: %s(%s)"
        if lasso.opts.get('lax', False):
            for e in aeval.error:
                log.error(msg, expr, *e.get_error())
        else:
            msg_args = (expr,) + aeval.error[0].get_error()
            raise ValueError(msg % msg_args)
    if isinstance(res, Lasso):
        lasso = res._replace(opts=lasso.opts) if res.opts is None else res
    else:
        lasso = lasso._replace(values=res)

    return lasso


def get_default_filters(overrides=None):
    """
   The default available :term:`filters` used by :func:`lasso()` when constructing its internal :class:`Ranger`.

    :param dict or None overrides:
            Any items to update the default ones.

    :return:
            a dict-of-dicts with 2 items:

            - *func*: a function with args: ``(Ranger, Lasso, *args, **kwds)``
            - *desc*:  help-text replaced by ``func.__doc__`` if missing.

    :rtype:
            dict
    """
    filters = {
        'pipe': {
            'func': pipe_filter,
        },
        'peval': {
            'func': eval_filter,
        },
        'recurse': {
            'func': recursive_filter,
        },
        'redim': {
            'func': redim_filter,
        },
        'numpy': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=np.array(lasso.values, *args, **kwds)),
            'desc': np.array.__doc__,
        },
        'dict': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=dict(lasso.values, *args, **kwds)),
            'desc': dict.__doc__,
        },
        'odict': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=OrderedDict(lasso.values, *args, **kwds)),
            'desc': OrderedDict.__doc__,
        },
        'sorted': {
            'func': lambda ranger, lasso, * args, **kwds: lasso._replace(
                values=sorted(lasso.values, *args, **kwds)),
            'desc': sorted.__doc__,
        },
    }

    try:
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
    except ImportError as ex:
        msg = "The 'df' and 'series' filters were notinstalled, due to: %s"
        log.info(msg, ex)

    if overrides:
        filters.update(overrides)

    return filters
