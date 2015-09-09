#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
The syntax-parsing part *xleash*.

Prefer accessing the public members from the parent module.

.. currentmodule:: pandalone.xleash
"""

from __future__ import unicode_literals

from collections import namedtuple
import json
import re

from future.builtins import str
from future.moves.urllib.parse import urldefrag
from past.builtins import basestring

import itertools as itt
import numpy as np


class Cell(namedtuple('Cell', ('row', 'col', 'brow', 'bcol'))):
    """
    A pair of 1-based strings, denoting the "A1" coordinates of a cell.

    The "num" coords (numeric, 0-based) are specified using numpy-arrays
    (:class:`Coords`).
    """

    def __new__(cls, row, col, brow=None, bcol=None):

        return super(cls, Cell).__new__(cls,
                                        row and row.upper(),
                                        col and col.upper(),
                                        brow and brow.upper(),
                                        bcol and bcol.upper())

    def __str__(self):
        r = self.row
        c = self.col
        try:
            c = int(c)
            s = 'R%sC%s' % (r, c, )
        except:
            s = '%s%s' % (c, r)
        return s

    def __repr__(self):
        if self.brow or self.bcol:
            s = super(Cell, self).__repr__()
        else:
            s = "Cell(row=%r, col=%r)" % (self.row, self.col)
        return s

Cell.__new__.__defaults__ = (None, None)
"""Make :class:`Cell` construct with missing 'brow', 'bcol' fields as `None`."""


class Edge(namedtuple('Edge', ('land', 'mov', 'mod'))):
    """
    All the infos required to :term:`target` a cell.

    An :term:`Edge` contains *A1* :class:`Cell` as `land`.

    :param Cell land: the :term:`landing-cell`
    :param str mov: use None for missing moves.
    :param str mod: one of (`+`, `-` or `None`)
    """
    __slots__ = ()

    def __new__(cls, land, mov=None, mod=None):
        return super(cls, Edge).__new__(cls, land, mov, mod)

    def __str__(self):
        return ('%s(%s%s)' % (self.land, self.mov, self.mod or '')
                if self.mov
                else str(self.land))

    # def __repr__(self):
    #     return "Edge('%s')" % str(self)

_topleft_Edge = Edge(Cell('^', '^'))
_bottomright_Edge = Edge(Cell('_', '_'))


def Edge_new(row, col, mov=None, mod=None, default=None):
    """
    Make a new `Edge` from any non-values supplied, as is capitalized, or nothing.

    :param str, None col:    ie ``A``
    :param str, None row:    ie ``1``
    :param str, None mov:    ie ``RU``
    :param str, None mod:    ie ``+``

    :return:    a `Edge` if any non-None
    :rtype:     Edge, None


    Examples::

        >>> Edge_new('1', 'a', 'Rul', '-')
        Edge(land=Cell(row='1', col='A'), mov='RUL', mod='-')
        >>> print(Edge_new('5', '5'))
        R5C5


    No error checking performed::

        >>> Edge_new('Any', 'foo', 'BaR', '+_&%')
        Edge(land=Cell(row='ANY', col='FOO'), mov='BAR', mod='+_&%')

        >>> print(Edge_new(None, None, None, None))
        None


    except were coincidental::

        >>> Edge_new(row=0, col=123, mov='BAR', mod=None)
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'

        >>> Edge_new(row=0, col='A', mov=123, mod=None)
        Traceback (most recent call last):
        AttributeError: 'int' object has no attribute 'upper'
    """

    if col == row == mov == mod is None:
        return default

    return Edge(land=Cell(row, col),
                mov=mov and mov.upper(), mod=mod)

_encase_regex = re.compile(r'^\s*(?P<q>[/\\"$%&])(.+)(?P=q)\s*$', re.DOTALL)
_regular_xlref_regex = re.compile(
    r"""
    ^\s*(?:(?P<sh_name>[^!]+)?!)?                            # xl sheet name
    (?:                                                      # 1st-edge
        (?:
            (?:
                (?P<st_col>[A-Z]+|[_^])                      # A1-col
                (?P<st_row>[123456789]\d*|[_^])              # A1-row
            ) | (?:
                R(?P<st_row2>-?[123456789]\d*|[_^.])         # RC-row
                C(?P<st_col2>-?[123456789]\d*|[_^.])         # RC-col
            )
        )
        (?:\(
            (?P<st_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)      # moves
            (?P<st_mod>[+?])?                                # move modifiers
            \)
        )?
    )?
    (?:(?P<colon>:)                                          # ':' needed if 2nd
        (?:                                                  # 2nd-edge
            (?:                                              # target
                (?:
                    (?P<nd_col>[A-Z]+|[_^.])                 # A1-col
                    (?P<nd_row>[123456789]\d*|[_^.])         # A1-row
                ) | (?:
                    R(?P<nd_row2>-?[123456789]\d*|[_^.])     # RC-row
                    C(?P<nd_col2>-?[123456789]\d*|[_^.])     # RC-col
                )
            )
            (?:\(
                (?P<nd_mov>L|U|R|D|LD|LU|UL|UR|RU|RD|DL|DR)  # moves
                (?P<nd_mod>[+?])?                            # move-modifiers
                \)
            )?
        )?
        (?:
            :(?P<exp_moves>[LURD?123456789]+)                # expansion moves
        )?
    )?
    (?:
        :\s*(?P<js_filt>[[{"].*)                             # filters
    )?$
    """,
    re.IGNORECASE | re.X | re.DOTALL)
"""The regex for parsing regular :term:`xl-ref`. """

_re_exp_moves_splitter = re.compile('([LURD]\d+)', re.IGNORECASE)

# TODO: Make exp_moves `?` work different from numbers.
_re_exp_moves_parser = re.compile(
    r"""
    ^(?P<moves>[LURD]+)                                  # primitive moves
    (?P<times>\?|\d+)?                                   # repetition times
    $""",
    re.IGNORECASE | re.X)


_excel_str_translator = str.maketrans('“”', '""')  # @UndefinedVariable
"""Excel use these !@#% chars for double-quotes, which are not valid JSON-strings!!"""

CallSpec = namedtuple('CallSpec', ['func', 'args', 'kwds'])
"""The :term:`call-specifier` for holding the parsed json-filters."""

CallSpec.__new__.__defaults__ = ([], {})
"""Make optional the last 2 fields of :class:`CallSpec` ``(args', kwds)`` ."""


def parse_call_spec(call_spec_values):
    """
    Parse :term:`call-specifier` from json-filters.

    :param call_spec_values:
        This is a *non-null* structure specifying some function call
        in the `filter` part, which it can be either:

        - string: ``"func_name"``
        - list:   ``["func_name", ["arg1", "arg2"], {"k1": "v1"}]``
          where the last 2 parts are optional and can be given in any order;
        - object: ``{"func": "func_name", "args": ["arg1"], "kwds": {"k":"v"}}``
          where the `args` and `kwds` are optional.

    :return:
        the 3-tuple ``func, args=(), kwds={}`` with the defaults as shown
        when missing.
    """
    def boolarr(l):
        return np.fromiter(l, dtype=bool)

    def parse_list(func, item1=None, item2=None):
        items = (item1, item2)
        isargs = boolarr(isinstance(a, list) for a in items)
        iskwds = boolarr(isinstance(a, dict) for a in items)
        isnull = boolarr(a is None for a in items)

        if isargs.all() or iskwds.all() or not (
                isargs ^ iskwds ^ isnull).all():
            msg = "Cannot decide `args`/`kwds` for call_spec(%s)!"
            raise ValueError(msg.format(call_spec_values))
        args, kwds = None, None
        if isargs.any():
            args = items[isargs.nonzero()[0][0]]
        if iskwds.any():
            kwds = items[iskwds.nonzero()[0][0]]
        return func, args, kwds

    def parse_object(func, args=None, kwds=None):
        return func, args, kwds

    try:
        if isinstance(call_spec_values, basestring):
            func, args, kwds = call_spec_values, None, None
        # elif isinstance(call_spec_values, (tuple, list)): ???
        elif isinstance(call_spec_values, list):
            func, args, kwds = parse_list(*call_spec_values)
        elif isinstance(call_spec_values, dict):
            func, args, kwds = parse_object(**call_spec_values)
        else:
            msg = "One of str, list or dict expected for call-spec({})!"
            raise ValueError(msg.format(call_spec_values))
    except ValueError:
        raise
    except Exception as ex:
        msg = "Cannot parse call-spec({}) due to: {}"
        raise ValueError(msg.format(call_spec_values, ex))

    if not isinstance(func, basestring):
        msg = "Expected a `string` for func({}) for call-spec({})!"
        raise ValueError(msg.format(func, call_spec_values))
    if args is None:
        args = []
    elif not isinstance(args, list):
        msg = "Expected a `list` for args({}) for call-spec({})!"
        raise ValueError(msg.format(args, call_spec_values))
    if kwds is None:
        kwds = {}
    elif not isinstance(kwds, dict):
        msg = "Expected a `dict` for kwds({}) for call-spec({})!"
        raise ValueError(msg.format(kwds, call_spec_values))

    return CallSpec(func, args, kwds)


def _repeat_moves(moves, times=None):
    """
    Returns an iterator that repeats `moves` x `times`, or infinite if unspecified.

    Used when parsing primitive :term:`directions`.

   :param str moves: the moves to repeat ie ``RU1D?``
   :param str times: N of repetitions. If `None` it means infinite repetitions.
   :return:    An iterator of the moves
   :rtype:     iterator

    Examples::

         >>> list(_repeat_moves('LUR', '3'))
         ['LUR', 'LUR', 'LUR']
         >>> list(_repeat_moves('ABC', '0'))
         []
         >>> _repeat_moves('ABC')  ## infinite repetitions
         repeat('ABC')
     """
    args = (moves,)
    if times is not None:
        args += (int(times), )
    return itt.repeat(*args)


def parse_expansion_moves(exp_moves):
    """
    Parse rect-expansion into a list of dir-letters iterables.

    :param exp_moves:
        A string with a sequence of primitive moves:
        es. L1U1R1D1
    :type xl_ref: str

    :return:
        A list of primitive-dir chains.
    :rtype: list


    Examples::

        >>> res = parse_expansion_moves('lu1urd?')
        >>> res
        [repeat('L'), repeat('U', 1), repeat('UR'), repeat('D', 1)]

        # infinite generator
        >>> [next(res[0]) for i in range(10)]
        ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']

        >>> list(res[1])
        ['U']

        >>> parse_expansion_moves('1LURD')
        Traceback (most recent call last):
        ValueError: Invalid rect-expansion(1LURD) due to:
                'NoneType' object has no attribute 'groupdict'

    """
    try:
        res = _re_exp_moves_splitter.split(exp_moves.upper().replace('?', '1'))

        return [_repeat_moves(**_re_exp_moves_parser.match(v).groupdict())
                for v in res
                if v != '']

    except Exception as ex:
        msg = 'Invalid rect-expansion({}) due to: {}'
        raise ValueError(msg.format(exp_moves, ex))


def _parse_edge(gs, prefix, default_edge):
    row_a1, row_rc = gs.pop('%s_row' % prefix), gs.pop('%s_row2' % prefix)
    col_a1, col_rc = gs.pop('%s_col' % prefix), gs.pop('%s_col2' % prefix)
    return Edge_new(row_a1 or row_rc, col_a1 or col_rc,
                    gs.pop('%s_mov' % prefix),
                    gs.pop('%s_mod' % prefix), default_edge)


def _parse_xlref_fragment(xlref_fragment):
    """
    Parses a :term:`xl-ref` fragment.

    :param str xlref_fragment:
            the url-fragment part of the :term:`xl-ref` string,
            including the ``'#'`` char.
    :return:
        dictionary containing the following parameters:

        - sheet: (str, int, None) i.e. ``sheet_name``
        - st_edge: (Edge, None) the 1st-ref, with raw cell
          i.e. ``Edge(land=Cell(row='8', col='UPT'), mov='LU', mod='-')``
        - nd_edge: (Edge, None) the 2nd-ref, with raw cell
          i.e. ``Edge(land=Cell(row='_', col='.'), mov='D', mod='+')``
        - exp_moves: (sequence, None), as i.e. ``LDL1`` parsed by
          :func:`parse_expansion_moves()`
        - js_filt: dict i.e. ``{"dims: 1}``

    :rtype: dict


    Examples::

        >>> res = _parse_xlref_fragment('Sheet1!A1(DR+):Z20(UL):L1U2R1D1:'
        ...                             '{"opts":{}, "func": "foo"}')
        >>> sorted(res.items())
        [('call_spec', CallSpec(func='foo', args=[], kwds={})),
         ('exp_moves', 'L1U2R1D1'),
         ('nd_edge', Edge(land=Cell(row='20', col='Z'), mov='UL', mod=None)),
         ('opts', {}),
         ('sh_name', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='DR', mod='+'))]

    Shortcut for all sheet from top-left to bottom-right full-cells::

        >>> res=_parse_xlref_fragment(':')
        >>> sorted(res.items())
        [('call_spec', None),
         ('exp_moves', None),
         ('nd_edge', Edge(land=Cell(row='_', col='_'), mov=None, mod=None)),
         ('opts', None),
         ('sh_name', None),
         ('st_edge', Edge(land=Cell(row='^', col='^'), mov=None, mod=None))]


    Errors::

        >>> _parse_xlref_fragment('A1(DR)Z20(UL)')
        Traceback (most recent call last):
        SyntaxError: Not an `xl-ref` syntax: A1(DR)Z20(UL)

    """

    m = _regular_xlref_regex.match(xlref_fragment)
    if not m:
        raise SyntaxError('Not an `xl-ref` syntax: %s' % xlref_fragment)

    gs = m.groupdict()

    is_colon = gs.pop('colon')
    gs['st_edge'] = _parse_edge(gs, 'st', is_colon and _topleft_Edge)
    gs['nd_edge'] = _parse_edge(gs, 'nd', is_colon and _bottomright_Edge)
    assert is_colon or not gs['nd_edge'], (xlref_fragment, gs['nd_edge'])

    exp_moves = gs['exp_moves']
    gs['exp_moves'] = exp_moves

    js = gs.pop('js_filt', None)
    if js:
        try:
            js = json.loads(js)
        except ValueError as ex:
            msg = 'Filters are not valid JSON: %s\n  JSON: \n%s'
            raise ValueError(msg % (ex, js))

    opts = js.pop('opts', None) if isinstance(js, dict) else None
    if opts is not None and not isinstance(opts, dict):
        msg = 'Filter-opts({}) must be a json-object(dictionary)!'
        raise ValueError(msg.format(opts))
    gs['opts'] = opts
    gs['call_spec'] = parse_call_spec(js) if js else None

    return gs


def parse_xlref(xlref):
    try:
        res = _parse_xlref(xlref)
    except SyntaxError as ex:
        try:
            m = _encase_regex.match(xlref)
        except SyntaxError:
            raise ex
        else:
            if m:
                print(m.group(2))
                res = _parse_xlref(m.group(2))
                res['xl-ref'] = xlref
            else:
                raise ex

    return res


def _parse_xlref(xlref):
    """
    Parse a :term:`xl-ref` into a dict.

    :param str xlref:
            A url-string abiding to the :term:`xl-ref` syntax.
    :return:
            A dict with all fields, with None with those missing.
    :rtype: dict


    Examples::

        >>> res = parse_xlref('workbook.xlsx#Sheet1!A1(DR+):Z20(UL):L1U2R1D1:'
        ...                             '{"opts":{}, "func": "foo"}')
        >>> sorted(res.items())
         [('call_spec', CallSpec(func='foo', args=[], kwds={})),
         ('exp_moves', 'L1U2R1D1'),
         ('nd_edge', Edge(land=Cell(row='20', col='Z'), mov='UL', mod=None)),
         ('opts', {}),
         ('sh_name', 'Sheet1'),
         ('st_edge', Edge(land=Cell(row='1', col='A'), mov='DR', mod='+')), ('url_file', 'workbook.xlsx'), ('xl_ref', 'workbook.xlsx#Sheet1!A1(DR+):Z20(UL):L1U2R1D1:{"opts":{}, "func": "foo"}')]

    Shortcut for all sheet from top-left to bottom-right full-cells::

        >>> res=parse_xlref('#:')
        >>> sorted(res.items())
        [('call_spec', None),
         ('exp_moves', None),
         ('nd_edge', Edge(land=Cell(row='_', col='_'), mov=None, mod=None)),
         ('opts', None),
         ('sh_name', None),
         ('st_edge', Edge(land=Cell(row='^', col='^'), mov=None, mod=None)),
         ('url_file', None),
         ('xl_ref', '#:')]


    Errors::

        >>> parse_xlref('A1(DR)Z20(UL)')
        Traceback (most recent call last):
        SyntaxError: No fragment-part (starting with '#'): A1(DR)Z20(UL)

        >>> parse_xlref('#A1(DR)Z20(UL)')          ## Missing ':'.
        Traceback (most recent call last):
        SyntaxError: Not an `xl-ref` syntax: A1(DR)Z20(UL)

    But as soon as syntax is matched, subsequent errors raised are
    :class:`ValueErrors`::

        >>> parse_xlref("#A1:B1:{'Bad_JSON_str'}")
        Traceback (most recent call last):
        ValueError: Filters are not valid JSON:
        Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
          JSON:
        {'Bad_JSON_str'}
    """
    xlref = xlref.translate(_excel_str_translator)
    url_file, frag = urldefrag(xlref)
    if not frag:
        raise SyntaxError(
            "No fragment-part (starting with '#'): %s" % xlref)
    res = _parse_xlref_fragment(frag)
    frag = frag.strip()
    res['url_file'] = url_file.strip() or None
    res['xl_ref'] = xlref

    return res
