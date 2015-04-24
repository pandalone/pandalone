#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Defines the building-blocks of a "model":

components and assemblies:
    See :class:`Component`, :class:`FuncComponent` and :class:`Assembly`

paths and path-mappings (pmods):
    See :func:`pmods_from_tuples`, :class:`Pstep`
"""

from __future__ import division, unicode_literals

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from copy import copy
import logging
import re
from unittest.mock import MagicMock

import functools as ft
import itertools as itt
from pandalone.pandata import iter_jsonpointer_parts
import pandas as pd


__commit__ = ""

log = logging.getLogger(__name__)


class Pmod(object):

    """
    A path-step mapping forming the pmods-hierarchy.

    - The :term:`pmods` denotes the hierarchy of all :term:`mappings`,
      that either *rename* or *relocate* path-steps.

    - A single :term:`mapping` transforms an "origin" path to 
      a "destination" one (also called as "from" and "to" paths).

    - A mapping always transforms the *final* path-step, like that::

        FROM_PATH       TO_PATH       RESULT_PATH
        ---------       -------       -----------
        /rename/path    foo       --> /rename/foo        ## renaming
        /relocate/path  foo/bar   --> /relocate/foo/bar  ## relocation
        /root           a/b/c     --> /a/b/c             ## Relocates all /root sub-paths.

    - The :term:`pmod` is the mapping of that single path-step.

    - It is possible to match fully on path-steps using regular-expressions,
      and then to use any captured-groups from the *final* step into 
      the mapped value::

        (/all(.*)/path, foo)   + all_1/path --> /all_1/foo
                               + allXXX     --> /allXXX          ## no change
        (/all(.*)/path, foo\1) + all_1/path --> /all_1/foo_1

      If more than one regex match, they are merged in the order declared
      (the latest one overrides a previous one).

    - Any exact child-name matches are applied and merged after regexs.

    - Use :func:`pmods_from_tuples()` to construct the pmods-hierarchy.

    - The pmods are used internally by class:`Pstep` to correspond
      the component-paths of their input & output onto the actual
      value-tree paths.

    :ivar str _alias:          (optional) the mapped-name of the pstep for
                               this pmod
    :ivar dict _steps:         {original_name --> pmod}
    :ivar OrderedDict _regxs:  {regex_on_originals --> pmod}


    Example:

    .. Note:: 
        Do not manually construct instances from this class!
        To construct a hierarchy use the :func:`pmods_from_tuples()` and/or 
        the :func:`convert_df_as_pmods_tuples()`. 

    You can either use it for mass-converting paths like that::

        >>> pmods = pmods_from_tuples([
        ...         ('/a',           'A/AA'),
        ...         ('/b(.*)',      r'BB\\1'),
        ...         ('/b.*/(c.*)',  r'C/\\1'),
        ... ])
        >>> pmods.map_paths(['/a', '/a/foo', '/big/stuff', '/be/courageous'])
        ['/A/AA', '/A/AA/foo', '/BBig/stuff',  '/BBe/C/courageous']


    Or for a single-step searching::

        >>> pmods.descend('a')
        (pmod('A/AA'), 'A/AA')
    """

    __slots__ = ['_alias', '_steps', '_regxs']

    def __init__(self, _alias=None, _steps={}, _regxs={}):
        """
        Args passed only for testing, remember `_regxs` to be (k,v) tuple-list!

        .. Note:: Volatile arg-defaults (empty dicts) are knowingly used , 
            to preserve memory; should never append in them!

        """
        self._alias = _alias
        self._steps = _steps
        if _regxs:
            self._regxs = OrderedDict(
                (re.compile(k), v) for k, v in _regxs)
        else:
            self._regxs = _regxs

    def _append_into_steps(self, key):
        """ 
        Inserts a child-mappings into `_steps` dict. 

        :param str key:    the step-name to add
        """

        cpmod = None
        d = self._steps
        if not d:
            self._steps = d = {}  # Do not modify init-defaults.
        else:
            cpmod = d.get(key)
        if not cpmod:
            d[key] = cpmod = Pmod()

        return cpmod

    def _append_into_regxs(self, key):
        """
        Inserts a child-mappings into `_steps` dict. 

        :param str key:    the regex-pattern to add
        """
        key = re.compile(key)
        cpmod = None
        d = self._regxs
        if not d:
            self._regxs = d = OrderedDict()  # Do not modify init-defaults.
        else:
            cpmod = d.get(key)
            if cpmod:
                # Remove it, to append it at then end.
                del d[key]
        if not cpmod:
            cpmod = Pmod()
        d[key] = cpmod

        return cpmod

    def _override_steps(self, other):
        """
        Override this pmod's '_steps' dict with other's, recursively.

        Same as :meth:`_override_regxs()` but without caring for order.
        """

        opmods = other._steps
        if opmods:
            spmods = self._steps
            if spmods:
                # Like ``spmods.copy().update()`` but
                # recursive `_merge()` on common items.
                #
                spmods = spmods.copy()
                for name, opmod in opmods.items():
                    spmod = spmods.get(name)
                    if spmod:
                        opmod = spmod._merge(opmod)
                    spmods[name] = opmod  # Share other-pmod if not mine.
                opmods = spmods

            # Share other dict if self hadn't its own.
            self._steps = opmods

    def _override_regxs(self, other):
        """
        Override this pmod's `_regxs` dict with other's, recursively.

        - It may "share" (crosslink) the dict and/or its child-pmods
          between the two pmod args (`self` and `other`).
        - No dict is modified (apart from self, which must have been cloned
          previously by :meth:`Pmod._merge()`), to avoid side-effects
          in case they were "shared".
        - It preserves dict-ordering so that `other` order takes precedence
          (its elements are the last ones).

        :param Pmod self:    contains the dict that would be overridden
        :param Pmod other:   contains the dict with the overrides
        """

        opmods = other._regxs
        if opmods:
            spmods = self._regxs
            if spmods:
                # Like ``spmods.copy().update()`` but
                # with recursive `_merge()` on common items,
                # and preserve order.
                #
                opairs = []
                for name, opmod in opmods.items():
                    spmod = spmods.get(name)
                    if spmod:
                        mpmod = spmod._merge(opmod)
                    else:
                        mpmod = opmod  # Share other-pmod.
                    opairs.append((name, mpmod))

                okeys = opmods.keys()
                spairs = [(name, spmod)  # Share self-pmod.
                          for name, spmod in spmods.items()
                          if name not in okeys]

                opmods = type(spmods)(spairs + opairs)

            # Share other dict if self hadn't its own.
            self._regxs = opmods

    def _merge(self, other):
        """
        Clone and override all its props with props from other-pmod, recursively.

        Although it does not modify this, the `other` or their children pmods,
        it may "share" (crosslink) them, so pmods MUST NOT be modified later.

        :param Pmod other: contains the dicts with the overrides
        :return:           the cloned merged pmod
        :rtype:            Pmod

        Examples:

        Look how `_steps` are merged::

            >>> pm1 = Pmod(_alias='pm1', _steps={
            ...     'a':Pmod(_alias='A'), 'c':Pmod(_alias='C')})
            >>> pm2 = Pmod(_alias='pm2', _steps={
            ...     'b':Pmod(_alias='B'), 'a':Pmod(_alias='AA')})
            >>> pm = pm1._merge(pm2)
            >>> sorted(pm._steps.keys())
            ['a', 'b', 'c']


        And here it is `_regxs` merging, which preserves order::

            >>> pm1 = Pmod(_alias='pm1',
            ...            _regxs=[('d', Pmod(_alias='D')),
            ...                    ('a', Pmod(_alias='A')),
            ...                    ('c', Pmod(_alias='C'))])
            >>> pm2 = Pmod(_alias='pm2',
            ...            _regxs=[('b', Pmod(_alias='BB')),
            ...                    ('a', Pmod(_alias='AA'))])

            >>> pm1._merge(pm2)
            pmod('pm2', OrderedDict([(re.compile('d'), pmod('D')),
                       (re.compile('c'), pmod('C')),
                       (re.compile('b'), pmod('BB')),
                       (re.compile('a'), pmod('AA'))]))

            >>> pm2._merge(pm1)
            pmod('pm1', OrderedDict([(re.compile('b'), pmod('BB')),
                        (re.compile('d'), pmod('D')),
                        (re.compile('a'), pmod('A')),
                        (re.compile('c'), pmod('C'))]))
        """
        self = copy(self)
        if other._alias:
            self._alias = other._alias
        self._override_steps(other)
        self._override_regxs(other)

        return self

    def _match_regxs(self, cstep):
        """Return (pmod, regex.match) for those child-pmods matching `cstep`."""

        return [(rpmod, match)
                for rpmod, match
                in ((rpmod, regex.fullmatch(cstep))
                    for regex, rpmod
                    in self._regxs.items())
                if match]

    def descend(self, cstep):
        """
        Return child-pmod with merged any exact child with all matched regexps, along with its alias regex-expaned.

        :param str cstep:   the child path-step cstep of the pmod to return
        :return:            the merged-child pmod, along with the alias; 
                            both might be None, if nothing matched, or no alias.
        :rtype:             tuple(Pmod, str)

        Example::

            >>> pm = Pmod(
            ...     _steps={'a': Pmod(_alias='A')},
            ...     _regxs=[('a\w*', Pmod(_alias='AWord')),
            ...              ('a(\d*)', Pmod(_alias=r'A_\\1')),
            ...    ])
            >>> pm.descend('a')
            (pmod('A'), 'A')

            >>> pm.descend('abc')
            (pmod('AWord'), 'AWord')

            >>> pm.descend('a12')
            (pmod('A_\\\\1'), 'A_12')

            >>> pm.descend('BAD')
            (None, None)


        Notice how children of regexps are merged together::

            >>> pm = Pmod(
            ...     _steps={'a':
            ...        Pmod(_alias='A', _steps={1: 11})},
            ...     _regxs=[
            ...        ('a\w*', Pmod(_alias='AWord', 
            ...                      _steps={2: Pmod(_alias=22)})),
            ...        ('a\d*', Pmod(_alias='ADigit', 
            ...                     _steps={3: Pmod(_alias=33)})),
            ...    ])
            >>> sorted(pm.descend('a')[0]._steps)    ## All children and regexps match.
            [1, 2, 3]

            >>> pm.descend('aa')[0]._steps           ## Only 'a\w*' matches.
            {2: pmod(22)}

            >>> sorted(pm.descend('a1')[0]._steps )  ## Both regexps matches.
            [2, 3]

        So it is possible to say::

            >>> pm.descend('a1')[0].alias(2)
            22
            >>> pm.descend('a1')[0].alias(3)
            33
            >>> pm.descend('a1')[0].descend('BAD')
            (None, None)
            >>> pm.descend('a$')
            (None, None)

        but it is better to use :meth:`map_path()` for this.
        """
        alias = None

        cpmod = self._steps.get(cstep)
        pmods = self._match_regxs(cstep)

        if cpmod and cpmod._alias:
            alias = cpmod._alias
        else:
            for rpmod, match in reversed(pmods):
                if rpmod._alias:
                    alias = match.expand(rpmod._alias)
                    break
        pmods = [pmod for pmod, _ in pmods]
        if cpmod:
            pmods.append(cpmod)

        if pmods:
            return (ft.reduce(Pmod._merge, pmods), alias)
        return (None, None)

    def alias(self, cstep):
        """Like :meth:`descend()` but without merging child-pmods."""
        cpmod = self._steps.get(cstep)
        if cpmod and cpmod._alias:
            if cpmod._alias:
                return cpmod._alias

        pmods = self._match_regxs(cstep)

        for rpmod, match in reversed(pmods):
            if rpmod._alias:
                return match.expand(rpmod._alias)

    def map_path(self, path):
        """
        Maps a '/rooted/path' using all aliases while descending its child pmods.

        It uses any aliases on all child pmods if found.

        :param str path: a rooted path to transform
        :return:         the rooted mapped path or '/' if path was '/'
        :rtype           str or None

        Example:

            >>> pmods = pmods_from_tuples([
            ...         ('/a',             'A/AA'),
            ...         ('/a(\\w*)',       'BB\\1'),
            ...         ('/a(\\d+)/(c.*)', 'C/\\1'),
            ... ])
            >>> #pmods.map_path('/a'
        """
        steps = list(iter_jsonpointer_parts(path))
        if not steps:
            nsteps = self._alias or ''
        else:
            nsteps = []
            pmod = self
            for step in steps[:-1]:
                alias = None
                if pmod:
                    pmod, alias = pmod.descend(step)
                nsteps.append(alias or step)
            final_step = steps[-1]
            if pmod:
                final_step = pmod.alias(final_step) or final_step
            nsteps.append(final_step)

        return '/%s' % '/'.join(nsteps)

    def map_paths(self, paths):
        return [self.map_path(p) for p in paths]

    def __repr__(self):
        args = [repr(a)
                for a in [self._alias, self._steps, self._regxs]
                if a]

        args = ', '.join(args)
        return 'pmod({})'.format(args)


def pmods_from_tuples(pmods_tuples):
    """
    Turns a list of 2-tuples into a *pmods* hierarchy.

    Each tuple defines the renaming-or-relocation of the *final* part
    of some component path onto another one into value-trees, such as::

        (rename/path, foo)           --> rename/foo
        (relocate/path, foo/bar)     --> relocate/foo/bar


    In case the the "from" path contains any of the `[].*()` chars,
    it is assumed to be a regular-expression::

        (all(.*)/path, foo)
        (some[\d+]/path, foo\1)


    :return: a root pmod
    :rtype:  Pmod


    Example::

        >>> pmods_tuples = [
        ...     ('/a', 'A1/A2'),
        ...     ('/a/b', 'B'),
        ... ]
        >>> pmods = pmods_from_tuples(pmods_tuples)
        >>> pmods
        pmod({'a': pmod('A1/A2', {'b': pmod('B')})})

        >>> pmods_tuples = [
        ...     ('/a*', 'A1/A2'),
        ...     ('/a/b[123]', 'B'),
        ... ]
        >>> pmods = pmods_from_tuples(pmods_tuples)
        >>> pmods
        pmod({'a': pmod(OrderedDict([(re.compile('b[123]'), pmod('B'))]))}, 
             OrderedDict([(re.compile('a*'), pmod('A1/A2'))]))

    """
    root = Pmod()
    for i, (f, t) in enumerate(pmods_tuples):
        if not (f and t):
            msg = 'pmod-tuple(%i): `source(%s)` and/or `to(%s)` were empty!'
            log.warning(msg, i, f, t)
            continue

        pmod = root
        for srcstep in iter_jsonpointer_parts(f):
            is_regex = any(set('[]().*+?') & set(srcstep))
            if is_regex:
                pmod = pmod._append_into_regxs(srcstep)
            else:
                pmod = pmod._append_into_steps(srcstep)

        pmod._alias = t

    return root


def convert_df_as_pmods_tuples(df_pmods, col_from='from', col_to='to'):
    """
    Turns a a dataframe with `col_from`, `col_to` columns into a list of 2-tuples.

    :return: a list of 2-tuples that can be fed into :func:`pmods_from_tuples`.
    :rtype: list

    Example::

        >>> pmods_tuples = [
        ...     ('/a', 'A1/A2'),
        ...     ('/a/b', 'B'),
        ... ]
        >>> df_pmods = pd.DataFrame(pmods_tuples)
        >>> res = convert_df_as_pmods_tuples(df_pmods)
        >>> res
        rec.array([('/a', 'A1/A2'), ('/a/b', 'B')],
              dtype=[('from', 'O'), ('to', 'O')])

        >>> df_pmods.columns = ['Rename from', 'Rename to']
        >>> df_pmods['extra columns'] = ['not', 'used']
        >>> res = convert_df_as_pmods_tuples(
        ...         df_pmods, col_from='Rename from', col_to='Rename to')
        >>> res
        rec.array([('/a', 'A1/A2'), ('/a/b', 'B')],
              dtype=[('Rename from', 'O'), ('Rename to', 'O')])
        """
    if df_pmods.empty:
        return []
    cols_df = set(df_pmods.columns)
    if col_from not in cols_df or col_to not in cols_df:
        if df_pmods.shape[1] != 2:
            cols_miss = cols_df - set([col_from, col_to])
            msg = "Missing pmods-columns%s, and shape%s is not just 2 columns!"
            raise ValueError(msg % (cols_miss, df_pmods.shape))
        else:
            df_pmods.columns = [col_from, col_to]
    df = df_pmods[[col_from, col_to]]

    return df.to_records(index=False)


_NONE = object()
"""Denotes non-existent json-schema attribute in :class:`JSchema`."""


class JSchema(object):

    """
    Facilitates the construction of json-schema-v4 nodes on :class:`PStep` code.

    It does just rudimentary args-name check.   Further validations
    should apply using a proper json-schema validator.

    :param type: if omitted, derived as 'object' if it has children
    :param kws:  for all the rest see http://json-schema.org/latest/json-schema-validation.html

    """
    type = _NONE,  # @ReservedAssignment
    items = _NONE,  # @ReservedAssignment
    required = _NONE,
    title = _NONE,
    description = _NONE,
    minimum = _NONE,
    exclusiveMinimum = _NONE,
    maximum = _NONE,
    exclusiveMaximum = _NONE,
    patternProperties = _NONE,
    pattern = _NONE,
    enum = _NONE,
    allOf = _NONE,
    oneOf = _NONE,
    anyOf = _NONE,

    def todict(self):
        return {k: v for k, v in vars(self).items() if v is not _NONE}


class Pstep(str):

    """
    Automagically-constructed *renamable* paths for accessing data-tree.

    The "magic" autocreates psteps as they referenced, making writting code
    that access data-tree paths, natural, while at the same time the "model"
    of those tree-data gets discovered.

    Each pstep keeps internaly the *name* of a data-tree step, which, when
    created through recursive referencing, coincedes with parent's branch
    leading to this step.  That name can be modified with :class:`Pmod`
    so the same data-accessing code can consume differently-named data-trees.

    :param str pname:    this pstep's name (stored at super-str object)
    :ivar Pstep _csteps: the child-psteps
    :ivar dict _pmods:   path-modifications used to construct this and
                         relayed to children
    :ivar int _lock:     one of
                         - :const:`Pstep.CAN_RELOCATE`(default, reparenting allowed),
                         - :const:`Pstep.CAN_RENAME`,
                         - :const:`Pstep.LOCKED' (neither from the above).
    :ivar dict _schema:  jsonschema data.


    Usage:

    .. Warning::
        String's slicing operations do not work on this string-subclass!

    - Just by referencing (non_private) attributes, they are created.

    - It raises :exc:`AssertionError` if any non-pstep value gets assigned
      as dict-item or as non-private attribute (ie `_name` is indeed allowed).

    - Use :meth:`_paths()` to get all defined paths so far.

    - Construction::

        >>> Pstep()
        `.`
        >>> Pstep('a')
        `a`

    - Paths are created implicitely as they are referenced::

        >>> m = {'a': 1, 'abc': 2, 'cc': 33}
        >>> p = Pstep('a')
        >>> assert m[p] == 1
        >>> assert m[p.abc] == 2
        >>> assert m[p['321'].cc] == 33

        >>> sorted(p._paths)
        ['a/321/cc', 'a/abc']

    - Its is possible to define "path-renames" on construction::

        >>> pmods = {'root':'/deeper/ROOT',
        ...    '_child_': {'abc': 'ABC', '_child_': {'foo': 'BAR'}}}
        >>> p = Pstep('root', pmods=pmods)
        >>> p.abc.foo
        `BAR`
        >>> p._paths
        ['/deeper/ROOT/ABC/BAR']

    - but if exceptions are thrown if marked as "locked":


    - Assignments are allowed only to special attributes::

        >>> p.assignments = 'FAIL!'
        Traceback (most recent call last):
        AssertionError: Cannot assign 'FAIL!' to '/deeper/ROOT/assignments'!  Only other psteps allowed.

        >>> p._but_hidden = 'Ok'


    TODO: Use __slot__ on Pstep.
    """

    CAN_RELOCATE = 3
    CAN_RENAME = 1
    LOCKED = 0

    @staticmethod
    def lockstr(lock):
        if lock >= Pstep.CAN_RELOCATE:
            return 'CAN_RELOCATE'
        if Pstep.LOCKED <= lock < Pstep.CAN_RELOCATE:
            return 'LOCKED'
        return 'LOCKED'

    def __new__(cls, pname='.', pmods=None):
        orig = pname
        if pmods:
            pname = pmods.get(pname, pname)
        self = str.__new__(cls, pname)
        self._orig = orig

        self._csteps = {}
        self._pmods = pmods
        vars(self)['_lock'] = Pstep.CAN_RELOCATE

        return self

    def __missing__(self, cpname):
        try:
            cpname = self._pmods.get(cpname, cpname)
            pmods = self._pmods[_PMOD_CHILD]
        except:
            pmods = None
        child = Pstep(cpname, pmods=pmods)
        self._csteps[cpname] = child
        return child

    def __getitem__(self, cpname):
        child = self._csteps.get(cpname, None)
        return child or self.__missing__(cpname)

    def __setitem__(self, cpname, value):
        raise self._ex_invalid_assignment(cpname, value)

    def __getattr__(self, cpname):
        if cpname.startswith('_'):
            msg = "'%s' object has no attribute '%s'"
            raise AttributeError(msg % (self, cpname))
        return self.__missing__(cpname)

    def __setattr__(self, cpname, value):
        if cpname.startswith('_'):
            str.__setattr__(self, cpname, value)
        else:
            raise self._ex_invalid_assignment(cpname, value)

    def _ex_invalid_assignment(self, cpname, value):
        msg = "Cannot assign '%s' to '%s/%s'!  Only other psteps allowed."
        return AssertionError(msg % (value, self, cpname))

    def __repr__(self):
        return '`%s`' % self

    @property
    def _lock(self):
        """One of `CAN_RELOCATE`, `CAN_RENAME`, `LOCKED'

        :raise: ValueError when setting stricter lock-value on a renamed/relocated pstep
        """
        return vars(self)['_lock']

    @_lock.setter
    def _lock(self, lock):
        if self != self._orig:
            if lock < Pstep.CAN_RENAME or (lock < Pstep.CAN_RELOCATE and '/' in self):
                msg = "Cannot rename/relocate '%s'-->'%s' due to %s!"
                raise ValueError(msg % (self._orig, self, Pstep.lockstr(lock)))
        vars(self)['_lock'] = int(lock)

    @property
    def _paths(self):
        p = []
        self._paths_(p)
        return p

    def _paths_(self, paths, prefix=None):
        """:return: all child/steps constructed so far, in a list"""
        if prefix:
            prefix = '%s/%s' % (prefix, self)
        else:
            prefix = self
        if self._csteps:
            for _, v in self._csteps.items():
                v._paths_(paths, prefix)
        else:
            paths.append(prefix)

    @property
    def _schema(self):
        """Updates json-schema-v4 on this pstep (see :class:`JSchema`)."""

        # Lazy create it
        #    (clients should check before`_schema_exists()`)
        #
        jschema = vars(self).get('_schema')
        if jschema is None:
            jschema = JSchema()
            vars(self)['_schema'] = jschema
        return jschema

    def _schema_exists(self):
        """Always use this to avoid needless schema-instantiations."""
        return '_schema' in vars(self)


name_uniqueizer = None  # TODO: Handle clashes on component-names.


class Component(object, metaclass=ABCMeta):

    """
    Encapsulates a function and its its inputs/outputs dependencies.

    It should be callable, and when executed it may read/modify
    the data-tree given as its 1st input.

    An opportunity to fix the internal-state (i.e. inputs/output/name)
    is when the  :meth:`_build()` is invoked.

    :ivar list _name:    identifier
    :ivar list _inp:     list/of/paths required on the data-tree (must not overlap with `out`)
    :ivar list _out:     list/of/paths modified on the data-tree (must not overlap with `inp`)

    Mostly defined through *cfuncs*, which provide for defining a component
    with a single function with a special signature, see :class:`FuncComponent`.
    """

    def __init__(self, name):
        if name_uniqueizer:
            name = name_uniqueizer(name)
        self._name = name
        self._inp = None
        self._out = None

    @abstractmethod
    def __call__(self, *agrs, **kws):
        pass

    @abstractmethod
    def _build(self, pmods=None):
        """Invoked once before run-time and should apply `pmaps` when given."""
        pass

    def _iter_validations(self):
        """ Yields a msg for each failed validation rule.

        Invoke it after :meth:`_build()` component.
        """
        # TODO: Implement Component's validation.

        if False:
            yield
#         expected_attrs = ['name', 'inp', 'out']
#         for attr in expected_attrs:
#             if not hasattr(self, attr):
#                 yield "`%s` is unset!" % attr


class FuncComponent(Component):

    """
    Converts a "cfunc" into a component.

    A cfunc is a function that modifies the values-tree with this signature::

        cfunc_XXXX(comp, vtree)

    where:

    comp:
        the  :class:`FuncComponent` associated with the cfunc

    vtree:
        the part of the data-tree involving the values to be modified
        by the cfunc

    It works also as a utility to developers of a cfuncs, since it is passed
    as their 1st arg.

    The cfuncs may use :meth:`pinp` and :meth:`pout` when accessing
    its input and output data-tree values respectively.
    Note that accessing any of those attributes from outside of cfunc,
    would result in an error.

    If a cfunc access additional values with "fixed' paths, then it has to
    manually add those paths into the :attr:`_inp` and :attr:`_out`
    lists.


    Example:

    This would be a fully "relocatable" cfunc::

        >>> def cfunc_calc_foobar_rate(comp, value_tree):
        ...     pi = comp.pinp()
        ...     po = comp.pout()
        ...
        ...     df = value_tree.get(pi)
        ...
        ...     df[po.Acc] = df[pi.V] / df[pi.T]

    To get the unmodified component-paths, use::

        >>> comp = FuncComponent(cfunc_calc_foobar_rate)
        >>> comp._build()
        >>> assert list(comp._iter_validations()) == []
        >>> sorted(comp._inp + comp._out)
        ['calc_foobar_rate/Acc', 'calc_foobar_rate/T', 'calc_foobar_rate/V']

    To get the path-modified component-paths, use::

        >>> pmods = {'calc_foobar_rate': '/A/B', '_child_':{'foo': 'FOO'}}
        >>> comp._build(pmods)
        >>> sorted(comp._inp + comp._out)
        ['/A/B/Acc', '/A/B/T', '/A/B/V']

        >>> comp._build(pmods)
        >>> sorted(comp._inp + comp._out)
        ['/A/B/Acc', '/A/B/T', '/A/B/V']

    """

    def __init__(self, cfunc, name=None):
        self._cfunc = cfunc
        if name is None:
            name = cfunc.__name__
            prefix = 'cfunc_'
            if name.startswith(prefix):
                name = name[len(prefix):]
        Component.__init__(self, name=name)

        # The following are initialized in _build():
        # self._inp = None
        # self._out = None
        # self._pmods = None
        # self._pinp = None
        # self._pout = None

    def __call__(self, *args, **kws):
        self._cfunc(self, *args, **kws)

    def _fetch_all_paths(self, pstep):
        return pstep._paths if pstep else []

    def pinp(self, path=None):
        """The suggested :class:`Pstep` for cfunc to use to access inputs."""
        p = self._pinp
        if p is None:
            p = Pstep(path or self._name, pmods=self._pmods)
            self._pinp = p
        return p

    def pout(self, path=None):
        """The suggested :class:`Pstep` for cfunc to use to access outputs."""
        p = self._pout
        if p is None:
            p = Pstep(path or self._name, pmods=self._pmods)
            self._pout = p
        return p

    def _build(self, pmods=None):
        """Extracts inputs/outputs from cfunc. """
        vtree = MagicMock()
        self._inp = []
        self._out = []
        self._pinp = None
        self._pout = None
        self._pmods = pmods

        self._cfunc(self, vtree)

        self._inp.extend(self._fetch_all_paths(self._pinp))
        self._out.extend(self._fetch_all_paths(self._pout))


class Assembly(Component):  # TODO: Assembly inherit Component

    """
    Example:

        >>> def cfunc_f1(comp, value_tree):
        ...     comp.pinp().A
        ...     comp.pout().B
        >>> def cfunc_f2(comp, value_tree):
        ...     comp.pinp().B
        ...     comp.pout().C
        >>> ass = Assembly(FuncComponent(cfunc) for cfunc in [cfunc_f1, cfunc_f2])
        >>> ass._build()
        >>> assert list(ass._iter_validations()) == []
        >>> ass._inp
        ['f1/A', 'f2/B']
        >>> ass._out
        ['f1/B', 'f2/C']

        >>> pmods = {'f1':'/root', 'f2':'/root'}
        >>> ass._build(pmods)
        >>> sorted(ass._inp + ass._out)
        ['/root/A', '/root/B', '/root/B', '/root/C']

    """

    def __init__(self, components, name=None):
        Component.__init__(self, name=name or 'assembly')
        self._comps = list(components)

    def __call__(self, *args, **kws):
        pass  # TODO: Invoke Dispatcher with Assembly's child-components.

    def _build(self, pmods=None):
        inp = set()
        out = set()
        for c in self._comps:
            c._build(pmods)
            inp.update(c._inp)
            out.update(c._out)
        self._inp = sorted(inp)
        self._out = sorted(out)


if __name__ == '__main__':
    raise "Not runnable!"
