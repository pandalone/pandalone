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

from copy import deepcopy
import inspect
import logging
import textwrap

from collections import ChainMap
from toolz import dicttoolz as dtz

from . import installed_filters

from . import Lasso, EmptyCaptureException, _parse, _capture
from pandalone.xleash.io import backend


log = logging.getLogger(__name__)


def _build_call_help(name, func, desc):
    sig = func and inspect.formatargspec(*inspect.getfullargspec(func))
    desc = textwrap.indent(textwrap.dedent(desc), "    ")
    return "\n\nFilter: %s%s:\n%s" % (name, sig, desc)


def _Lasso_to_edges_str(lasso):
    st = lasso.st_edge if lasso.st_edge else ""
    nd = lasso.nd_edge if lasso.nd_edge else ""
    s = st if st and not nd else "%s:%s" % (st, nd)
    exp = ":%s" % lasso.exp_moves.upper() if lasso.exp_moves else ""
    return s + exp


class Ranger(object):
    """
    The director-class that performs all stages required for "throwing the lasso" around rect-values.

    Use it when you need to have total control of the procedure and
    configuration parameters, since no defaults are assumed.

    The :meth:`do_lasso()` does the job.

    :ivar SheetsFactory sheets_factory:
            Factory of sheets from where to parse rect-values; does not
            close it in the end.
            Maybe `None`, but :meth:`do_lasso()` will scream unless invoked
            with a `context_lasso` arg containing a concrete :class:`ABCSheet`.
    :ivar dict base_opts:
            The :term:`opts` that are deep-copied and used as the defaults
            for every :meth:`do_lasso()`, whether invoked directly or
            recursively by :func:`recursive_filter()`.
            If unspecified, no opts are used, but this attr is set to an
            empty dict.
            See :func:`get_default_opts()`.
    :ivar dict or None available_filters:
            The :term:`filters` available for a :term:`xl-ref` to use.
            If `None`, then uses :data:`xleash.installed_filters`.
            Use an empty dict not to use any filters.
    :ivar Lasso intermediate_lasso:
            A ``('stage', Lasso)`` pair with the last :class:`Lasso` instance
            produced during the last execution of the :meth:`do_lasso()`.
            Used for inspecting/debuging.
    """

    def __init__(self, sheets_factory, base_opts=None, available_filters=None):
        if not sheets_factory:
            raise ValueError("Please specify a non-null sheets-factory!")
        self.sheets_factory = sheets_factory
        if base_opts is None:
            base_opts = {}
        self.base_opts = base_opts
        self.available_filters = (
            installed_filters if available_filters is None else available_filters
        )
        self.intermediate_lasso = None

    def _relasso(self, lasso, stage, **kwds):
        """Replace lasso-values and updated :attr:`intermediate_lasso`."""
        lasso = lasso._replace(**kwds)
        self.intermediate_lasso = (stage, lasso)

        return lasso

    def make_call(self, lasso, func_name, args, kwds):
        """
        Executes a :term:`call-spec` respecting any `lax` argument popped from `kwds`.

        :param bool lax:
                After overlaying it on :term:`opts`, it governs whether to
                raise on errors.
                Defaults to `False` (scream!).
        """

        def parse_avail_func_rec(func, desc=None):
            if not desc:
                desc = func.__doc__
            return func, desc

        # Just to update intermediate_lasso.
        lasso = self._relasso(lasso, func_name)

        verbose = lasso.opts.get("verbose", False)
        lax = kwds.pop("lax", lasso.opts.get("lax", False))
        func, func_desc = "", ""
        try:
            func_rec = self.available_filters[func_name]
            func, func_desc = parse_avail_func_rec(**func_rec)
            lasso = func(self, lasso, *args, **kwds)
            assert isinstance(lasso, Lasso), "Filter(%r) returned not a Lasso(%r)!" % (
                func_name,
                lasso,
            )
        except Exception as ex:
            if verbose:
                func_desc = _build_call_help(func_name, func, func_desc)
            msg = "While invoking(%s, args=%s, kwds=%s): %s%s"
            help_msg = func_desc if verbose else ""
            if lax:
                log.warning(msg, func_name, args, kwds, ex, help_msg, exc_info=1)
            else:
                raise ValueError(msg % (func_name, args, kwds, ex, help_msg))

        return lasso

    def _make_init_Lasso(self, **context_kwds):
        """Creates the lasso to be used for each new :meth:`do_lasso()` invocation."""
        context_kwds["opts"] = ChainMap(deepcopy(self.base_opts))
        init_lasso = Lasso(**context_kwds)

        return init_lasso

    def _parse_and_merge_with_context(self, xlref, init_lasso):
        """
        Merges xl-ref parsed-parsed_fields with `init_lasso`, reporting any errors.

        :param Lasso init_lasso:
                Default values to be overridden by non-nulls.

        :return: a Lasso with any non `None` parsed-fields updated
        """
        assert isinstance(init_lasso.opts, ChainMap), init_lasso

        try:
            parsed_fields = _parse.parse_xlref(xlref)
            filled_fields = dtz.valfilter(lambda v: v is not None, parsed_fields)
            init_lasso = init_lasso._replace(**filled_fields)
        except SyntaxError:
            raise
        except Exception as ex:
            msg = "Parsing xl-ref(%r) failed due to: %s"
            log.debug(msg, xlref, ex, exc_info=1)
            raise ValueError(msg % (xlref, ex)) from ex

        return init_lasso

    def _open_sheet(self, lasso):
        try:
            sheet = self.sheets_factory.fetch_sheet(
                lasso.url_file, lasso.sh_name, base_sheet=lasso.sheet
            )
        except Exception as ex:
            msg = "Loading sheet([%s]%s) failed due to: %s"
            raise ValueError(msg % (lasso.url_file, lasso.sh_name, ex)) from ex
        return sheet

    # TODO: Move _resolve_capture_rect into ABCSheet
    def _resolve_capture_rect(self, lasso, sheet):
        """Also handles :class:`EmptyCaptureException` in case ``opts['no_empty'] != False``."""

        try:
            st, nd = _capture.resolve_capture_rect(
                sheet.get_states_matrix(),
                sheet.get_margin_coords(),
                lasso.st_edge,
                lasso.nd_edge,
                lasso.exp_moves,
                lasso.base_coords,
            )
        except EmptyCaptureException:
            if lasso.opts.get("no_empty", False):
                raise
            st = nd = None
        except Exception as ex:
            msg = "Resolving capture-rect(%r) failed due to: %s"
            raise ValueError(msg % (_Lasso_to_edges_str(lasso), ex))
        return st, nd

    def _run_filters(self, lasso):
        if lasso.call_spec:
            try:
                lasso = self.make_call(lasso, *lasso.call_spec)
                # relasso(values) invoked internally.
            except Exception as ex:
                msg = "Filtering xl-ref(%r) failed due to: %s"
                raise ValueError(msg % (lasso.xl_ref, ex))
        return lasso

    def do_lasso(self, xlref, **context_kwds):
        """
        The director-method that does all the job of hrowing a :term:`lasso`
        around spreadsheet's rect-regions according to :term:`xl-ref`.

        :param str xlref:
            a string with the :term:`xl-ref` format::

                <url_file>#<sheet>!<1st_edge>:<2nd_edge>:<expand><js_filt>

            i.e.::

                file:///path/to/file.xls#sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}

        :param Lasso context_kwds:
                Default :class:`Lasso` fields in case parsed ones are `None`
        :return:
                The final :class:`Lasso` with captured & filtered values.
        :rtype: Lasso
        """
        if not isinstance(xlref, str):
            raise ValueError("Expected a string as `xl-ref`: %s" % xlref)
        self.intermediate_lasso = None

        lasso = self._make_init_Lasso(**context_kwds)
        lasso = self._relasso(lasso, "context")

        lasso = self._parse_and_merge_with_context(xlref, lasso)
        lasso = self._relasso(lasso, "parse")

        sheet = self._open_sheet(lasso)
        lasso = self._relasso(lasso, "open", sheet=sheet)

        st, nd = self._resolve_capture_rect(lasso, sheet)
        lasso = self._relasso(lasso, "capture", st=st, nd=nd)

        if st or nd:
            values = sheet.read_rect(st, nd)
        else:
            values = []
        lasso = self._relasso(lasso, "read_rect", values=values)

        lasso = self._run_filters(lasso)
        # relasso(values) invoked internally.

        return lasso


def get_default_opts(overrides=None):
    """
    Default :term:`opts` used by :func:`lasso()` when constructing its internal :class:`Ranger`.

    :param dict or None overrides:
            Any items to update the default ones.
    """
    opts = {"lax": False, "verbose": False, "read": {"on_demand": True}}

    if overrides:
        opts.update(overrides)

    return opts


def make_default_Ranger(sheets_factory=None, base_opts=None, available_filters=None):
    """
    Makes a defaulted :class:`Ranger`.

    :param sheets_factory:
            Factory of sheets from where to parse rect-values; if unspecified,
            a new :class:`SheetsFactory` is created.
            Remember to invoke its :meth:`SheetsFactory.close()` to clear
            resources from any opened sheets.

    :param base_opts:
            Default opts to affect the lassoing, to be merged with defaults;
            uses :func:`get_default_opts()`.

            Read the code to be sure what are the available choices :-(.
    :type base_opts:
            dict or None

    :param available_filters:
            The :term:`filters` available for a :term:`xl-ref` to use.
            (:data:`xleash.installed_filters` used if unspecified).
    :type available_filters:
            dict or None


    For instance, to make you own sheets-factory and override options,
    yoummay do this::

        >>> from pandalone import xleash

        >>> with xleash.SheetsFactory() as sf:
        ...     xleash.make_default_Ranger(sf, base_opts={'lax': True})
        <pandalone.xleash._lasso.Ranger object at
        ...
    """
    return Ranger(
        sheets_factory or backend.SheetsFactory(),
        base_opts or get_default_opts(),
        available_filters,
    )


def lasso(
    xlref,
    sheets_factory=None,
    base_opts=None,
    available_filters=None,
    return_lasso=False,
    **context_kwds
):
    """
    High-level function to :term:`lasso` around spreadsheet's rect-regions
    according to :term:`xl-ref` strings by using internally a :class:`Ranger` .

    :param str xlref:
        a string with the :term:`xl-ref` format::

            <url_file>#<sheet>!<1st_edge>:<2nd_edge>:<expand><js_filt>

        i.e.::

            file:///path/to/file.xls#sheet_name!UPT8(LU-):_.(D+):LDL1{"dims":1}

    :param sheets_factory:
            Factory of sheets from where to parse rect-values; if unspecified,
            the new :class:`SheetsFactory` created is closed afterwards.
            Delegated to :func:`make_default_Ranger()`, so items override
            default ones; use a new :class:`Ranger` if that is not desired.

    :ivar base_opts:
            Opts affecting the lassoing procedure that are deep-copied and used
            as the base-opts for every :meth:`Ranger.do_lasso()`, whether invoked
            directly or recursively by :func:`recursive_filter()`.
            Read the code to be sure what are the available choices.
            Delegated to :func:`make_default_Ranger()`, so items override
            default ones; use a new :class:`Ranger` if that is not desired.
    :type base_opts:
            dict or None

    :param available_filters:
            Delegated to :func:`make_default_Ranger()`, so items override
            default ones; use a new :class:`Ranger` if that is not desired.
    :type available_filters:
            dict or None

    :param bool return_lasso:
            If `True`, values are contained in the returned Lasso instance,
            along with all other artifacts of the :term:`lassoing` procedure.

            For more debugging help, create a :class:`Range` yourself and
            inspect the :attr:`Ranger.intermediate_lasso`.
    :param Lasso context_kwds:
            Default :class:`Lasso` fields in case parsed ones are `None`
            (i.e. you can specify the sheet like that).

    :return:
            Either the captured & filtered values or the final :class:`Lasso`,
            depending on the `return_lassos` arg.

    Example::

        sheet = _
    """
    factory_is_mine = not sheets_factory
    if base_opts is None:
        base_opts = get_default_opts()

    try:
        ranger = make_default_Ranger(
            sheets_factory=sheets_factory,
            base_opts=base_opts,
            available_filters=available_filters,
        )
        lasso = ranger.do_lasso(xlref, **context_kwds)
    finally:
        if factory_is_mine:
            ranger.sheets_factory.close()

    return lasso if return_lasso else lasso.values
