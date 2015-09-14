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


from asteval import Interpreter
from future.utils import iteritems
from past.builtins import basestring
from toolz import dicttoolz as dtz

import numpy as np

from . import Lasso, _parse
from ..utils import LoggerWriter
from ..utils import as_list


log = logging.getLogger(__name__)


def pipe_filter(ranger, lasso, *filters, **kwds):
    """
    A :term:`bulk-filter` that applies all call-specifiers one after another on the :term:`capture-rect` values.

    :param list filters:
            the json-parsed :term:`call-spec`
    """
    for filt in filters:
        call_spec = _parse.parse_call_spec(filt)
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


XLocation = namedtuple('XLocation',
                       ('sheet', 'st', 'nd', 'base_coords'))
"""
Fields denoting the position of a sheet/cell while running a :term:`element-wise-filter`.

Practically func:`run_filter_elementwise() preserves these fields if the
processed ones were `None`.
"""


def run_filter_elementwise(ranger, lasso, element_func, filters,
                           include=None, exclude=None, depth=-1,
                           *args, **kwds):
    """
    Runner of all :term:`element-wise` :term:`filters`.

    It applies the `element_func` on elements extracted from ``lasso.values``
    by treating the later first as "indexed" objects
    (Mappings, Series and Dataframes.), and if that fails, as nested lists.

    - The `include`/`exclude` filter args work only for "indexed" objects
      with ``items()`` or ``iteritems()`` and indexing methods.

        - If no filter arg specified, expands for all keys.
        - If only `include` specified, rejects all keys not explicitly
          contained in this filter arg.
        - If only `exclude` specified, expands all keys not explicitly
          contained in this filter arg.
        - When both `include`/`exclude` exist, only those explicitly
          included are accepted, unless also excluded.

    - Lower the :mod:`logging` level to see other than syntax-errors on
      recursion reported on :data:`log`.
    - Only those in :class:`XLocation` are passed recursively.

    :param list element_func:
            A function implementing the element-wise :term:`filter`
            and returning a 2-tuple ``(is_proccessed, new_val_or_lasso)``,
            like that::

                def element_func(ranger, lasso, context, elval)
                    proced = False
                    try:
                        elval = int(elval)
                        proced = True
                    except ValueError:
                        pass
                    return proced, elval

            Its `kwds` may contain the `include`, `exclude` and `depth` args.
            Any exception raised from `element_func` will cancel the diving.
    :param list filters:
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
    :params args:
            To be relayed to 'element_func'.
    :params kwds:
            To be relayed to 'element_func'.
    """
    include = include and as_list(include)
    exclude = exclude and as_list(exclude)

    def is_included(elval, key, cdepth):
        ok = True
        if cdepth == 0 or isinstance(elval, dict):
            ok &= not include or key in include
            ok &= not exclude or key not in exclude
        return ok

    def upd_base_coords(elval, cdepth, base_coords, i):
        if base_coords and not isinstance(elval, dict):
            row, col = base_coords
            try:
                import pandas as pd
            except ImportError:
                if cdepth == 0:
                    row += i
                elif cdepth == 1:
                    col += + i
            else:
                if isinstance(elval, pd.DataFrame):
                    col += i
                elif isinstance(elval, pd.Series):
                    row += i

            return row, col

    def call_element_func(elval, cdepth, base_coords):
        context_kwds = dtz.keyfilter(lambda k: k in XLocation._fields,
                                     lasso._asdict())
        context_kwds['base_coords'] = base_coords
        context = XLocation(**context_kwds)
        try:
            proced, res_lasso = element_func(ranger, lasso, context, elval,
                                             *args, **kwds)
        except Exception as ex:
            msg_args = (elval, context, ex)
            raise ValueError("Value(%r) at %s: \n    %s" % msg_args)

        if proced:
            if not isinstance(res_lasso, Lasso):
                res_lasso = lasso._replace(values=res_lasso)

            for call_spec in sub_call_specs:
                res_lasso = ranger.make_call(res_lasso, *call_spec)
            elval = res_lasso and res_lasso.values

        return proced, elval

    def dive_list(elval, cdepth, base_coords):
        proced, elval = call_element_func(elval, cdepth, base_coords)
        if not proced and isinstance(elval, list):
            for i, v in enumerate(elval):
                nbc = upd_base_coords(elval, cdepth, base_coords, i)
                elval[i] = dive_indexed(v, cdepth + 1, nbc)

        return elval

    def dive_indexed(elval, cdepth, base_coords):
        if cdepth != depth:
            dived = False
            try:
                items = iteritems(elval)
            except:
                pass  # Just to avoid chained ex.
            else:
                for i, (k, v) in enumerate(items):
                    # Dict is not ordered, so cannot locate `base_coords`!
                    if is_included(elval, k, cdepth):
                        nbc = upd_base_coords(elval, cdepth, base_coords, i)
                        elval[k] = dive_indexed(v, cdepth + 1, nbc)
                dived = True
            if not dived:
                elval = dive_list(elval, cdepth, base_coords)

        return elval

    sub_call_specs = [_parse.parse_call_spec(f) for f in filters]
    values = dive_indexed(lasso.values, 0, lasso.st)

    return lasso._replace(values=values)


def _recurse_element_func(ranger, lasso, context, elval):
    proced = False
    try:
        if isinstance(elval, basestring):
            lasso = ranger.do_lasso(elval, **context._asdict())
            proced = True
    except SyntaxError as ex:
        msg = "Skipped non `xl-ref` value(%r) \n  ++at %s \n  ++while lassoing %r \n  ++due to: %s"
        msg_args = (elval, context, lasso.xl_ref, ex)
        log.debug(msg, *msg_args)
    except Exception as ex:
        msg = "Lassoing  `xl-ref` failed due to: %s"
        raise ValueError(msg % ex)

    return proced, lasso


def recursive_filter(ranger, lasso, *filters, **kwds):
    """
    A :term:`element-wise-filter` that expand recursively any :term:`xl-ref` strings elements in :term:`capture-rect` values.

    :param list filters:
            Any :term:`filters` to apply after invoking the `element_func`.
    :param list or str include:
            Items to include when diving into "indexed" values.
            See :func:`run_filter_elementwise()`.
    :param list or str exclude:
            Items to exclude when diving into "indexed" values.
            See :func:`run_filter_elementwise()`.
    :param int or None depth:
            How deep to dive into nested structures, "indexed" or lists.
            If `< 0`, no limit. If 0, stops completely.
            See :func:`run_filter_elementwise()`.

    Note that in python-3 the signature would be::

        def recursive_filter(ranger, lasso, element_func, filters,
                             include=None, exclude=None, depth=-1):
    """
    include = kwds.pop('include', None)
    exclude = kwds.pop('exclude', None)
    depth = kwds.pop('depth', -1)
    return run_filter_elementwise(ranger, lasso, _recurse_element_func,
                                  filters,
                                  include=include,
                                  exclude=exclude,
                                  depth=depth)


ast_log_writer = LoggerWriter(logging.getLogger('%s.pyeval' % __name__),
                              logging.INFO)


def _pyeval_element_func(ranger, lasso, context, elval, eval_all):
    proced = False
    if isinstance(elval, basestring):
        expr = str(elval)
        symtable = locals()
        from .. import xleash
        symtable.update({'xleash': xleash})
        aeval = Interpreter(symtable, writer=ast_log_writer)
        res = aeval.eval(expr)
        if aeval.error:
            error = aeval.error[0].get_error()
            if eval_all:
                msg = "%i errors while py-evaluating %r: %s: %s"
                msg_args = (len(aeval.error), expr) + error
                raise ValueError(msg % msg_args)
            else:
                msg = "Skipped py-evaluating value(%r) \n  ++at %s \n  ++while lassoing %r \n  ++due to %i errors: %s: %s"
                msg_args = (elval, context, lasso.xl_ref,
                            len(aeval.error)) + error
                log.warning(msg, *msg_args)
        else:
            if isinstance(res, Lasso):
                lasso = (res._replace(opts=lasso.opts)
                         if res.opts is None
                         else res)
            else:
                lasso = lasso._replace(values=res)
            proced = True

    return proced, lasso


def pyeval_filter(ranger, lasso, *filters, **kwds):
    """
    A :term:`element-wise-filter` that uses :mod:`asteval` to evaluate string values as python expressions.

    The `expr` fecthed from `term:`capturing` may access read-write
    all :func:`locals()` of this method (ie: `ranger`, `lasso`),
    the :mod:`numpy` funcs, and the :mod:`pandalone.xleash` module under
    the `xleash` variable.

    The `expr` may return either:
        - the processed values, or
        - an instance of the :class:`Lasso`, in which case only its `opt`
          field is checked and replaced with original if missing.
          So better use :func:`namedtuple._replace()` on the current `lasso`
          which exists in the expr's namespace.

    :param bool eval_all:
            If `True` raise on 1st error and stop diving cells.
            Defaults to `False`.
    :param list filters:
            Any :term:`filters` to apply after invoking the `element_func`.
    :param list or str include:
            Items to include when diving into "indexed" values.
            See :func:`run_filter_elementwise()`.
    :param list or str exclude:
            Items to exclude when diving into "indexed" values.
            See :func:`run_filter_elementwise()`.
    :param int or None depth:
            How deep to dive into nested structures, "indexed" or lists.
            If `< 0`, no limit. If 0, stops completely.
            See :func:`run_filter_elementwise()`.

    Note that in python-3 the signature woudl be::

        def pyeval_filter(ranger, lasso, element_func, filters,
                             include=None, exclude=None, depth=-1):

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
    include = kwds.pop('include', None)
    exclude = kwds.pop('exclude', None)
    depth = kwds.pop('depth', -1)
    eval_all = kwds.pop('eval_all', False)
    return run_filter_elementwise(ranger, lasso, _pyeval_element_func,
                                  filters,
                                  include=include,
                                  exclude=exclude,
                                  depth=depth,
                                  eval_all=eval_all)


def py_filter(ranger, lasso, expr):
    """
    A :term:`bulk-filter` that passes values through a python-expression using :mod:`asteval` library.

    The `expr` may access read-write all :func:`locals()` of this method
    (`ranger`, `lasso`), the :mod:`numpy` funcs, and the :mod:`pandalone.xleash`
    module under the `xleash` variable.

    The `expr` may return either:
        - the processed values, or
        - an instance of the :class:`Lasso`, in which case only its `opt`
          field is checked and replaced with original if missing.
          So better use :func:`namedtuple._replace()` on the current `lasso`
          which exists in the expr's namespace.

    :param str expr:
            The python-expression, which may comprise of multiple statements.
    """
    symtable = locals()
    from .. import xleash
    symtable.update({'xleash': xleash})
    aeval = Interpreter(symtable, writer=ast_log_writer)
    res = aeval.eval(expr)
    if aeval.error:
        error = aeval.error[0].get_error()
        msg = "%i errors while py-evaluating %r: %s: %s"
        msg_args = (len(aeval.error), expr) + error
        raise ValueError(msg % msg_args)
    else:
        if isinstance(res, Lasso):
            lasso = (res._replace(opts=lasso.opts)
                     if res.opts is None
                     else res)
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
        'pyeval': {
            'func': pyeval_filter,
        },
        'py': {
            'func': py_filter,
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
        msg = "The 'df' and 'series' filters were not installed, due to: %s"
        log.info(msg, ex)

    if overrides:
        filters.update(overrides)

    return filters
