#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functionality for mapping (*renaming* or *relocating*) paths.

See:

- :class:`Pmod`, 
- :func:`pmods_from_tuples`, and 
- :func:`convert_df_as_pmods_tuples()`.

"""

from __future__ import division, unicode_literals

from collections import OrderedDict
from copy import copy
import logging
import re

import functools as ft
from pandalone.pandata import iter_jsonpointer_parts,\
    _iter_jsonpointer_parts_relaxed


__commit__ = ""

log = logging.getLogger(__name__)


class Pmod(object):

    r"""
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
        ''              a/b/c     --> /a/b/c             ## Relocate all paths.
        /               a/b/c     --> /a/b/c             ## Relocates 1st "empty-str" step.

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
        the :func:`df_as_pmods_tuples()`. 

    You can either use it for mass-mapping paths, either for *renaming* them::

        >>> pmods = pmods_from_tuples([
        ...         ('/a',           'A'),
        ...         ('/b.*',        r'BB\g<0>'),
        ...         ('/b.*/c.(.*)', r'W\1ER'),
        ... ])
        >>> pmods.map_paths(['/a', '/a/foo', '/big/stuff', '/born/child'])
        ['/A', '/A/foo', '/BBbig/stuff',  '/BBborn/WildER']


    or to *relocate* them::

        >>> pmods = pmods_from_tuples([
        ...         ('/a',           'A/AA'),
        ...         ('/b.*/c(.*)',  r'../C/\1'),
        ...         ('/b.*/.*/r.*', r'/\g<0>'),
        ... ])
        >>> pmods.map_paths(['/a/foo', '/big/child', '/begin/from/root'])
        ['/A/AA/foo', '/C/hild',  '/root']

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
        r"""
        Return child-pmod with merged any exact child with all matched regexps, along with its alias regex-expaned.

        :param str cstep:   the child path-step cstep of the pmod to return
        :return:            the merged-child pmod, along with the alias; 
                            both might be None, if nothing matched, or no alias.
        :rtype:             tuple(Pmod, str)

        Example::

            >>> pm = Pmod(
            ...     _steps={'a': Pmod(_alias='A')},
            ...     _regxs=[('a\w*', Pmod(_alias='AWord')),
            ...              ('a(\d*)', Pmod(_alias=r'A_\1')),
            ...    ])
            >>> pm.descend('a')
            (pmod('A'), 'A')

            >>> pm.descend('abc')
            (pmod('AWord'), 'AWord')

            >>> pm.descend('a12')
            (pmod('A_\\1'), 'A_12')

            >>> pm.descend('BAD')
            (None, None)


        Notice how children of regexps are merged together::

            >>> pm = Pmod(
            ...     _steps={'a':
            ...        Pmod(_alias='A', _steps={1: 11})},
            ...     _regxs=[
            ...        (r'a\w*', Pmod(_alias='AWord', 
            ...                      _steps={2: Pmod(_alias=22)})),
            ...        (r'a\d*', Pmod(_alias='ADigit', 
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
        """
        Like :meth:`descend()` but without merging child-pmods.

        :return: the expanded alias from child/regexs or None
        """
        cpmod = self._steps.get(cstep)
        if cpmod and cpmod._alias:
            if cpmod._alias:
                return cpmod._alias

        pmods = self._match_regxs(cstep)

        for rpmod, match in reversed(pmods):
            if rpmod._alias:
                return match.expand(rpmod._alias)

    @staticmethod
    def _append_path(steps, path):
        """
        Joins `steps`-list with `path`, respecting '/', '..', '.', ''.

        :return: the new or updated steps-list.
        :rtype:  list

        Example::

            >>> Pmod._append_path([], 'a')
            ['a']

            >>> Pmod._append_path([], '../a')
            ['a']
            >>> Pmod._append_path(['a', 'b'], '../c')
            ['a', 'c']
            >>> Pmod._append_path(['a', 'b'], '../../c')
            ['c']

            >>> Pmod._append_path(['a', 'b'], '')
            ['a', 'b']
            >>> Pmod._append_path(['a', 'b'], '.')
            ['a', 'b']

            >>> Pmod._append_path(['a', 'b'], './c')
            ['a', 'b', 'c']

            >>> Pmod._append_path(['a', 'b'], '/r')
            ['r']


        """
        if path:
            if path.startswith('/'):
                steps = []
            else:
                path = '/{}'.format(path)
        for step in iter_jsonpointer_parts(path):
            if not step or step == '.':
                continue
            if step == '..':
                steps = steps[:-1]
                continue
            steps.append(step)

        return steps

    def map_path(self, path):
        r"""
        Maps a '/rooted/path' using all aliases while descending its child pmods.

        It uses any aliases on all child pmods if found.

        :param str path: a rooted path to transform
        :return:         the rooted mapped path or '/' if path was '/'
        :rtype           str or None

        Examples::

            >>> pmods = pmods_from_tuples([
            ...         ('/a',              'A/AA'),
            ...         ('/a(\\w*)',       r'BB\1'),
            ...         ('/a\\w*/d.*',     r'D \g<0>'),
            ...         ('/a(\\d+)',       r'C/\1'),
            ...         ('/a(\\d+)/(c.*)', r'CC-/\1'), # The 1st group is ignored!
            ...         ('/a\\d+/e.*',     r'/newroot/\g<0>'), # Rooted mapping.
            ... ])

            >>> pmods.map_path('/a')
            '/A/AA'

            >>> pmods.map_path('/a_hi')
            '/BB_hi'

            >>> pmods.map_path('/a12')
            '/C/12'

            >>> pmods.map_path('/a12/etc')
            '/newroot/etc'

        Notice how children from *all* matching prior-steps are merged::

            >>> pmods.map_path('/a12/dow')
            '/C/12/D dow'
            >>> pmods.map_path('/a12/cow')
            '/C/12/CC-/cow'


        Use this to map *root*::

            >>> pmods = pmods_from_tuples([('', 'New/Root'),])
            >>> pmods
            pmod('New/Root')

            >>> pmods.map_path('/for/plant')
            '/New/Root/for/plant'

        .. NOTE:: 
            Using slash('/') for "from" path will NOT map *root*::

                >>> pmods = pmods_from_tuples([('/', 'New/Root'),])
                >>> pmods
                pmod({'': pmod('New/Root')})

                >>> pmods.map_path('/for/plant')
                '/for/plant'

                >>> pmods.map_path('//for/plant')
                '/New/Root/for/plant'

                '/root'

        but '' always remains unchanged (whole document)::

            >>> pmods.map_path('')
            ''

        """
        steps = list(iter_jsonpointer_parts(path))
        nsteps = [self._alias] if self._alias else []
        if steps:
            pmod = self
            for step in steps[:-1]:
                alias = None
                if pmod:
                    pmod, alias = pmod.descend(step)
                nsteps = Pmod._append_path(nsteps, alias or step)

            # On last step, the merging of child-pmods is a waste,
            #    so make it outside above-loop to
            #    avoid calling expensive `descend`.
            #
            final_step = steps[-1]
            if pmod:
                final_step = pmod.alias(final_step) or final_step
            nsteps = Pmod._append_path(nsteps, final_step)

            return '/%s' % '/'.join(nsteps)
        return ''

    def map_paths(self, paths):
        return [self.map_path(p) for p in paths]

    def __repr__(self):
        args = [repr(a)
                for a in [self._alias, self._steps, self._regxs]
                if a]

        args = ', '.join(args)
        return 'pmod({})'.format(args)

    def __eq__(self, o):
        try:
            return (self._alias, self._steps, self._regxs) == (o._alias, o._steps, o._regxs)
        except:
            return False


def pmods_from_tuples(pmods_tuples):
    """
    Turns a list of 2-tuples into a *pmods* hierarchy.

    - Each tuple defines the renaming-or-relocation of the *final* part
      of some component path onto another one into value-trees, such as::

          (/rename/path, foo)          --> rename/foo
          (/relocate/path, foo/bar)    --> relocate/foo/bar


    - The "from" must be an absolute path.

    - In case the "from" path contains any of the `[].*()` chars,
      it is assumed to be a regular-expression::

          (/all(.*)/path, foo)
          (/some[\d+]/path, foo\1)


    :return: a root pmod
    :rtype:  Pmod


    Example::

        >>> pmods_from_tuples([
        ...     ('/a', 'A1/A2'),
        ...     ('/a/b', 'B'),
        ... ])
        pmod({'a': pmod('A1/A2', {'b': pmod('B')})})

        >>> pmods_from_tuples([
        ...     ('/a*', 'A1/A2'),
        ...     ('/a/b[123]', 'B'),
        ... ])
        pmod({'a': 
            pmod(OrderedDict([(re.compile('b[123]'), pmod('B'))]))}, 
             OrderedDict([(re.compile('a*'), pmod('A1/A2'))]))


    This is how you map *root*::

        >>> pmods = pmods_from_tuples([
        ...     ('', 'New/Root'),
        ...     ('/a/b', '/B'),
        ... ])
        >>> pmods
        pmod('New/Root', {'a': pmod({'b': pmod('/B')})})

        >>> pmods.map_path('/a/c')
        '/New/Root/a/c'

        >>> pmods.map_path('/a/b')
        '/B'


    But note that '/' maps the 1st "empty-str" step after root::

        >>> pmods_from_tuples([
        ...     ('/', 'New/Root'),
        ... ])
        pmod({'': pmod('New/Root')})

    """
    root = Pmod()
    for i, (f, t) in enumerate(pmods_tuples):
        if not t:
            msg = 'pmod-tuple(%i): `to(%s)` were empty!'
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


def df_as_pmods_tuples(df_pmods, col_from='from', col_to='to'):
    """
    Turns a a dataframe with `col_from`, `col_to` columns into a list of 2-tuples.

    :return: a list of 2-tuples that can be fed into :func:`pmods_from_tuples`.
    :rtype: list

    Example::

        >>> import pandas as pd

        >>> pmods_tuples = [
        ...     ('/a', 'A1/A2'),
        ...     ('/a/b', 'B'),
        ... ]
        >>> df_pmods = pd.DataFrame(pmods_tuples)
        >>> res = df_as_pmods_tuples(df_pmods)
        >>> res
        rec.array([('/a', 'A1/A2'), ('/a/b', 'B')],
              dtype=[('from', 'O'), ('to', 'O')])

        >>> df_pmods.columns = ['Rename from', 'Rename to']
        >>> df_pmods['extra columns'] = ['not', 'used']
        >>> res = df_as_pmods_tuples(
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


if __name__ == '__main__':
    raise "Not runnable!"
