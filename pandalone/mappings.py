#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Hierarchical string-like objects useful for indexing, that can be rename/relocated at a later stage.

.. autosummary::

    Pstep
    pmods_from_tuples
    Pmod

*Example*::

    >>> from pandalone.mappings import pmods_from_tuples

    >>> pmods = pmods_from_tuples([
    ...     ('',         'deeper/ROOT'),
    ...     ('/abc',     'ABC'),
    ...     ('/abc/foo', 'BAR'),
    ... ])
    >>> p = pmods.step()
    >>> p.abc.foo
    `BAR`
    >>> p._paths()
    ['deeper/ROOT/ABC/BAR']

- TODO: Implements "anywhere" pmods(`//`).
"""

from __future__ import division, unicode_literals

from collections import OrderedDict
from copy import copy
import logging
import re

from past.builtins import basestring

import functools as ft
from pandalone import utils
from pandalone.pandata import (
    iter_jsonpointer_parts_relaxed, JSchema, unescape_jsonpointer_part,
    escape_jsonpointer_part)


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
                               + all_XYZ    --> /all_XYZ        ## no change
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
        To construct a hierarchy use the :func:`pmods_from_tuples()`.

    You can either use it for massively map paths, either for *renaming* them::

        >>> pmods = pmods_from_tuples([
        ...         ('/a',           'A'),
        ...         ('/~b.*',        r'BB\g<0>'),  ## Previous match.
        ...         ('/~b.*/~c.(.*)', r'W\1ER'),    ## Capturing-group(1)
        ... ])
        >>> pmods.map_paths(['/a', '/a/foo'])     ## 1st rule
        ['/A', '/A/foo']

        >>> pmods.map_path('/big/stuff')          ## 2nd rule
        '/BBbig/stuff'

        >>> pmods.map_path('/born/child')         ## 2nd & 3rd rule
        '/BBborn/WildER'


    or to *relocate* them::

        >>> pmods = pmods_from_tuples([
        ...         ('/a',           'A/AA'),
        ...         ('/~b.*/~c(.*)',  r'../C/\1'),
        ...         ('/~b.*/~.*/~r.*', r'/\g<0>'),
        ... ])
        >>> pmods.map_paths(['/a/foo', '/big/child', '/begin/from/root'])
        ['/A/AA/foo', '/big/C/hild', '/root']


    Here is how you relocate "root"
    (notice that the `''` path is the root)::

        >>> pmods = pmods_from_tuples([('', '/NEW/ROOT')])
        >>> pmods.map_paths(['/a/foo', ''])
        ['/NEW/ROOT/a/foo', '/NEW/ROOT']

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

    def step(self, pname='', alias=None):
        """
        Create a new :class:`Pstep` having as mappings this pmod.

        If no `pname` specified, creates a *root* pstep.

        Delegates to :meth:`Pstep.__new__()`.
        """
        return Pstep(pname, _proto_or_pmod=self, alias=alias)

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
        if other._alias is not None:
            self._alias = other._alias
        self._override_steps(other)
        self._override_regxs(other)

        return self

    def _match_regxs(self, cstep):
        """Return (pmod, regex.match) for those child-pmods matching `cstep`."""

        return [(rpmod, match)
                for rpmod, match
                in ((rpmod, utils.fullmatch(regex, cstep))
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

        if cpmod and cpmod._alias is not None:
            alias = cpmod._alias
        else:
            for rpmod, match in reversed(pmods):
                if rpmod._alias is not None:
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
        if cpmod and cpmod._alias is not None:
            return cpmod._alias

        pmods = self._match_regxs(cstep)

        for rpmod, match in reversed(pmods):
            if rpmod._alias is not None:
                return match.expand(rpmod._alias)

    def map_path(self, path):
        r"""
        Maps a '/rooted/path' using all aliases while descending its child pmods.

        It uses any aliases on all child pmods if found.

        :param str path: a rooted path to transform
        :return:         the rooted mapped path or '/' if path was '/'
        :rtype:          str or None

        Examples::

            >>> pmods = pmods_from_tuples([
            ...         ('/a',              'A/AA'),
            ...         ('/~a(\\w*)',       r'BB\1'),
            ...         ('/~a\\w*/~d.*',     r'D \g<0>'),
            ...         ('/~a(\\d+)',       r'C/\1'),
            ...         ('/~a(\\d+)/~(c.*)', r'CC-/\1'), # The 1st group is ignored!
            ...         ('/~a\\d+/~e.*',     r'/newroot/\g<0>'), # Rooted mapping.
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


        To map *root* use '' which matches before the 1st slash('/')::

            >>> pmods = pmods_from_tuples([('', 'New/Root'),])  ## Relative
            >>> pmods
            pmod({'': pmod('New/Root')})

            >>> pmods.map_path('/for/plant')
            'New/Root/for/plant'

            >>> pmods_from_tuples([('', '/New/Root'),]).map_path('/for/plant')
            '/New/Root/for/plant'

        .. Note::
            Using slash('/') for "from" path will NOT map *root*::

                >>> pmods = pmods_from_tuples([('/', 'New/Root'),])
                >>> pmods
                pmod({'': pmod({'': pmod('New/Root')})})

                >>> pmods.map_path('/for/plant')
                '/for/plant'

                >>> pmods.map_path('//for/plant')
                '/New/Root/for/plant'

                '/root'

        but '' always remains unchanged (whole document)::

            >>> pmods.map_path('')
            ''

        """
        is_folder = len(path) > 1 and path.endswith('/')
        if is_folder:
            path = path[:-1]

        steps = tuple(iter_jsonpointer_parts_relaxed(path))
        if self._alias is None:
            nsteps = ()
        else:
            nsteps = tuple(iter_jsonpointer_parts_relaxed(self._alias))

        if steps:
            pmod = self
            # Separate last-step from loop below, since
            #    merged child-pmods in `descend` are not needed.
            #
            for step in steps[:-1]:
                if pmod:
                    pmod, alias = pmod.descend(step)
                if alias is not None:
                    if alias.startswith('.'):
                        nsteps += (step, )
                    step = alias
                # XXX: Monkey business here.
                if len(step) > 1 and step.endswith('/'):
                    step = step[:-1]
                nsteps += tuple(iter_jsonpointer_parts_relaxed(step))

            final_step = steps[-1]
            if pmod:
                alias = pmod.alias(final_step)
                if alias is not None:
                    if alias.startswith('.'):
                        nsteps += (final_step, )
                    final_step = alias
            # XXX: Monkey business here.
            is_folder = len(final_step) > 1 and final_step.endswith('/')
            if is_folder:
                final_step = final_step[:-1]
            nsteps += tuple(iter_jsonpointer_parts_relaxed(final_step))

        npath = _join_paths(*nsteps)

        if is_folder:
            path += '%s/' % path

        return npath

    def map_paths(self, paths):
        return [self.map_path(p) for p in paths]

    def __repr__(self):
        args = [repr(a)
                for a in [self._alias, self._steps, self._regxs]
                if a or a == '']

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
          (relocate/path, foo/bar)    --> relocate/foo/bar


    - The "from" path may be:
      - relative,
      - absolute(starting with `/`), or
      - "anywhere"(starting with `//`).

    - In case a "step" in the "from" path starts with tilda(`~`),
      it is assumed to be a regular-expression, and it is removed from it.
      The "to" path can make use of any "from" capture-groups::

          ('/~all(.*)/path', 'foo')
          ('~some[\d+]/path', 'foo\1')
          ('//~all(.*)/path', 'foo')



    :param list(tuple(str, str) pmods_tuples:
    :return: a root pmod
    :rtype:  Pmod


    Example::

        >>> pmods_from_tuples([
        ...     ('/a', 'A1/A2'),
        ...     ('/a/b', 'B'),
        ... ])
        pmod({'': pmod({'a': pmod('A1/A2', {'b': pmod('B')})})})

        >>> pmods_from_tuples([
        ...     ('/~a*', 'A1/A2'),
        ...     ('/a/~b[123]', 'B'),
        ... ])
        pmod({'': pmod({'a':
                pmod(OrderedDict([(re.compile('b[123]'), pmod('B'))]))},
                     OrderedDict([(re.compile('a*'), pmod('A1/A2'))]))})


    This is how you map *root*::

        >>> pmods = pmods_from_tuples([
        ...     ('', 'relative/Root'),        ## Make all paths relatives.
        ...     ('/a/b', '/Rooted/B'),        ## But map `b` would be "rooted".
        ... ])
        >>> pmods
        pmod({'':
                pmod('relative/Root',
                        {'a': pmod({'b':
                                pmod('/Rooted/B')})})})

        >>> pmods.map_path('/a/c')
        'relative/Root/a/c'

        >>> pmods.map_path('/a/b')
        '/Rooted/B'


    But note that '/' maps the 1st "empty-str" step after root::

        >>> pmods_from_tuples([
        ...     ('/', 'New/Root'),
        ... ])
        pmod({'': pmod({'': pmod('New/Root')})})


    TODO: Implement "anywhere" matches.
    """
    root = Pmod()
    for i, (f, t) in enumerate(pmods_tuples):
        if (f, t) == ('', '') or f is None or t is None:
            msg = 'pmod-tuple #%i of %i: Invalid "from-to" tuple (%r, %r).'
            log.warning(msg, i + 1, len(pmods_tuples), f, t)
            continue

        pmod = root
        for srcstep in iter_jsonpointer_parts_relaxed(f):
            is_regex = srcstep.startswith('~')
            if is_regex:
                pmod = pmod._append_into_regxs(srcstep[1:])
            else:
                pmod = pmod._append_into_steps(srcstep)

        pmod._alias = t

    return root


def _append_step(steps, step):
    """
    Joins `step` at the right of `steps`, respecting '/', '..', '.', ''.

    :param tuple steps:  where to append into
                         ("absolute" when 1st-element is '')
    :param str step:     what to append
                         (may be: ``'foo', '.', '..', ''``)
    :rtype:  tuple

    .. Note::
        The empty-string('') is the "root" for both `steps` and `step`.
        An empty-tuple `steps` is considered "relative", equivalent to dot(`.`).


    Example::

        >>> _append_step((), 'a')
        ('a',)

        >>> _append_step(('a', 'b'), '..')
        ('a',)

        >>> _append_step(('a', 'b'), '.')
        ('a', 'b')


    Not that an "absolute" path has the 1st-step empty(`''`),
    (so the previous paths above were all "relative")::

        >>> _append_step(('a', 'b'), '')
        ('',)

        >>> _append_step(('',), '')
        ('',)

        >>> _append_step((), '')
        ('',)


    Dot-dots preserve "relative" and "absolute" paths, respectively,
    and hence do not coalesce when at the left::

        >>> _append_step(('',), '..')
        ('',)

        >>> _append_step(('',), '.')
        ('',)

        >>> _append_step(('a',), '..')
        ()

        >>> _append_step((), '..')
        ('..',)

        >>> _append_step(('..',), '..')
        ('..', '..')

        >>> _append_step((), '.')
        ()



    Single-dots('.') just dissappear::

        >>> _append_step(('.',), '.')
        ()

        >>> _append_step(('.',), '..')
        ('..',)

    """
    assert isinstance(steps, tuple), (steps, step)
    assert not step or isinstance(step, basestring), (steps, step)

    if step == '':
        return ('',)

    _last_pair_choices = {
        ('.',): (),
        ('..',): ('..',),
        ('.', '.'): (),
        ('.', '..',): ('..',),
        ('..', '.'): ('..',),
        ('..', '..'): ('..', '..'),
        ('', '.'): ('',),
        ('', '..'): ('',),
    }

    try:
        last_pair = steps[-1:] + (step,)
        steps = steps[:-1] + _last_pair_choices[last_pair]
    except KeyError:
        if step == '.':
            pass
        elif step == '..':
            steps = steps[:-1]
        else:
            steps += (step,)

    return steps


def _join_paths(*steps):
    """
    Joins all path-steps in a single string, respecting ``'/', '..', '.', ''``.

    :param str steps:  single json-steps, from left to right
    :rtype: str

    .. Note::
        If you use :func:`iter_jsonpointer_parts_relaxed()` to generate
        path-steps, the "root" is signified by the empty(`''`) step;
        not the slash(`/`)!

        Hence a lone slash(`/`) gets splitted to an empty step after "root"
        like that: ``('', '')``, which generates just "root"(`''`).

        Therefore a "folder" (i.e. `some/folder/`) when splitted equals
        ``('some', 'folder', '')``, which results again in the "root"(`''`)!


    Examples::

        >>> _join_paths('r', 'a', 'b')
        'r/a/b'

        >>> _join_paths('', 'a', 'b', '..', 'bb', 'cc')
        '/a/bb/cc'

        >>> _join_paths('a', 'b', '.', 'c')
        'a/b/c'


    An empty-step "roots" the remaining path-steps::

        >>> _join_paths('a', 'b', '', 'r', 'aa', 'bb')
        '/r/aa/bb'


    All `steps` have to be already "splitted"::

        >>> _join_paths('a', 'b', '../bb')
        'a/b/../bb'


    Dot-doting preserves "relative" and "absolute" paths, respectively::

        >>> _join_paths('..')
        '..'

        >>> _join_paths('a', '..')
        '.'

        >>> _join_paths('a', '..', '..', '..')
        '../..'

        >>> _join_paths('', 'a', '..', '..')
        ''


    Some more special cases::

        >>> _join_paths('..', 'a')
        '../a'

        >>> _join_paths('', '.', '..', '..')
        ''

        >>> _join_paths('.', '..')
        '..'

        >>> _join_paths('..', '.', '..')
        '../..'

    .. seealso:: _append_step
    """
    nsteps = ft.reduce(_append_step, steps, ())
    if not nsteps:
        return '.'
    else:
        return '/'.join(nsteps)

_forbidden_pstep_attrs = ('get_values', 'Series')
"""
Psteps attributes excluded from magic-creation, because searched by pandas's indexing code.
"""

def _clone_attrs(obj):
    """Clone deeply any collection attributes of the passed-in object."""
    attrs = vars(obj).copy()
    ccsteps = attrs.get('_csteps', None)
    if ccsteps:
        attrs['_csteps'] = ccsteps.copy()
    ctags = attrs['_tags']
    if ctags:
        attrs['_tags'] = attrs['_tags'].copy()
    return attrs


class Pstep(str):

    """
    Automagically-constructed *relocatable* paths for accessing data-tree.

    The "magic" autocreates psteps as they referenced, making writing code
    that access data-tree paths, natural, while at the same time the "model"
    of those tree-data gets discovered.

    Each pstep keeps internally the *name* of a data-tree step, which,
    when created through recursive referencing, concedes with parent's
    branch leading to this step.  That name can be modified with :class:`Pmod`
    so the same data-accessing code can refer to differently-named values
    int the data-tree.

    :ivar dict _csteps:  the child-psteps by their name (default `None`)
    :ivar dict _pmod:    path-modifications used to construct this and
                         relayed to children (default `None`)
    :ivar int _locked:   one of
                         - :const:`Pstep.CAN_RELOCATE` (default),
                         - :const:`Pstep.CAN_RENAME`,
                         - :const:`Pstep.LOCKED` (neither from the above).
    :ivar set _tags:     A set of strings (default `()`)
    :ivar dict _schema:  json-schema data.

    See :meth:`__new__()` for interal constructor.

    **Usage:**

    - Use a :meth:`Pmod.pstep()` to construct a *root* pstep from mappings.
      Specify a string argument to construct a relative pstep-hierarchy.

    - Just referencing (non_private) attributes, creates them.

    - Private attributes and functions (starting with ``_``) exist for
      specific operations (ie for specifying json-schema, or
      for collection all paths).

    - Assignments are only allowed for string-values, or to private attributes::

        >>> p = Pstep()
        >>> p.assignments = 12
        Traceback (most recent call last):
        AssertionError: Cannot assign '12' to '/assignments!

        >>> p._but_hidden = 'Ok'

    - Use :meth:`_paths()` to get all defined paths so far.

    - Construction::

        >>> Pstep()
        ``
        >>> Pstep('a')
        `a`

      Notice that pstesps are surrounded with the back-tick char('`').

    - Paths are created implicitely as they are referenced::

        >>> m = {'a': 1, 'abc': 2, 'cc': 33}
        >>> p = Pstep('a')
        >>> assert m[p] == 1
        >>> assert m[p.abc] == 2
        >>> assert m[p.a321.cc] == 33

        >>> sorted(p._paths())
        ['a/a321/cc', 'a/abc']


    - Any "path-mappings" or "pmods" maybe specified during construction::

        >>> from pandalone.mappings import pmods_from_tuples

        >>> pmods = pmods_from_tuples([
        ...     ('',         'deeper/ROOT'),
        ...     ('/abc',     'ABC'),
        ...     ('/abc/foo', 'BAR'),
        ... ])
        >>> p = pmods.step()
        >>> p.abc.foo
        `BAR`
        >>> p._paths()
        ['deeper/ROOT/ABC/BAR']

    - but exceptions are thrown if mapping any step marked as "locked":

        >>> p.abc.foo._locked  ## 3: CAN_RELOCATE
        3

        >>> p.abc.foo._lock    ## Screams, because `foo` is already mapped.
        Traceback (most recent call last):
        ValueError: Cannot rename/relocate 'foo'-->'BAR' due to LOCKED!

    - .. Warning::
          Creating an empty(`''`) step in some paths will "root" the path::

              >>> p = Pstep()
              >>> _ = p.a1.b
              >>> _ = p.A2
              >>> p._paths()
              ['/A2', '/a1/b']

              >>> _ = p.a1.a2.c
              >>> _ = p.a1.a2 = ''
              >>> p._paths()
              ['/A2', '/a1/b', '/c']

    """

    CAN_RELOCATE = 3
    CAN_RENAME = 1
    LOCKED = 0

    @staticmethod
    def _lockstr(lock):
        if lock >= Pstep.CAN_RELOCATE:
            return 'CAN_RELOCATE'
        if Pstep.LOCKED <= lock < Pstep.CAN_RELOCATE:
            return 'LOCKED'
        return 'LOCKED'

    def __new__(cls, pname='', _proto_or_pmod=None, alias=None):
        """
        Constructs a string with str-content which may comes from the mappings.

        These are the valid argument combinations::

            pname='attr_name',
            pname='attr_name', _alias='Mass [kg]'

            pname='attr_name', _proto_or_pmod=Pmod

            pname='attr_name', _proto_or_pmod=Pstep
            pname='attr_name', _proto_or_pmod=Pstep, _alias='Mass [kg]'


        :param str pname:
                this pstep's name which must coincede with the name of
                the parent-pstep's attribute holding this pstep.
                It is stored at `_orig` and if no `alias` and unmapped by pmod,
                this becomes the `alias`.
        :param Pmod or Pstep _proto_or_pmod:
                It can be either:

                - the mappings for this pstep,
                - another pstep to clone attributes from
                  (used when replacing an existing child-pstep), or
                - None.

                The mappings will apply only if :meth:`Pmod.descend()`
                match `pname` and will derrive the alias.
        :param str alias:
                Will become the super-str object when no mappaings specified
                (`_proto_or_pmod` is a dict from some prototype pstep)
                It gets jsonpointer-escaped if it exists
                (see :func:`pandata.escape_jsonpointer_part()`)
        """
        pmod = None
        attrs = None
        if _proto_or_pmod:
            if isinstance(_proto_or_pmod, Pmod):
                pmod, m_alias = _proto_or_pmod.descend(pname)
                if alias is None and m_alias:
                    # TODO: Add Escape-path TCs.
                    alias = unescape_jsonpointer_part(m_alias)
            else:
                assert isinstance(_proto_or_pmod, Pstep), (
                    'Invalid type(%s) for `_proto_or_pmod`!' % _proto_or_pmod)
                attrs = _clone_attrs(_proto_or_pmod)
        if alias is None:
            alias = pname
        self = str.__new__(cls, alias)
        self.__dict__ = attrs or {
            '_orig': pname,
            '_pmod': pmod,
            '_csteps': None,
            '_locked': Pstep.CAN_RELOCATE,
            '_tags': ()
        }

        return self

    def __make_cstep(self, ckey, alias=None, existing_cstep=None):
        csteps = self._csteps
        if not csteps:
            self._csteps = csteps = {}
        else:
            if existing_cstep is None:
                existing_cstep = csteps.get(ckey, None)
                # Update my mappings for `b` when ``self.b = "foo"``.
                #
                if alias is not None:
                    pmod = self._pmod
                    if pmod:
                        pmod._alias = alias
                    else:
                        self._pmod = Pmod(_alias=alias)
        csteps[ckey] = child = Pstep(ckey, existing_cstep or self._pmod, alias)

        return child

    def __getattr__(self, attr):
        try:
            return str.__getattr__(self, attr)
        except AttributeError as ex:
            if attr.startswith('_') or attr in _forbidden_pstep_attrs:
                raise
            csteps = self._csteps
            child = csteps and csteps.get(attr, None)
            return child or self.__make_cstep(attr)

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            str.__setattr__(self, attr, value)
        elif isinstance(value, Pstep):
            self.__make_cstep(attr, existing_cstep=value)
        elif isinstance(value, basestring):
            self.__make_cstep(attr, alias=value)
        else:
            raise self._ex_invalid_assignment(attr, value)

    def __dir__(self):
        d = super(str, self).__dir__()
        if self._csteps:
            d = sorted(d + list(self._csteps.keys()))
        return d

    def _ex_invalid_assignment(self, cpname, value):
        msg = "Cannot assign '%s' to '%s/%s!"
        return AssertionError(msg % (value, self, cpname))

    def __repr__(self):
        return '`%s`' % self

    @property
    def _locked(self):
        """
        Gets `_locked` internal flag or scream on set, when step already renamed/relocated

        Prefer using one of :attr:`_fix` or :attr:`_lock` instead.

        :param locked:  One of :attr:`CAN_RELOCATE`, :attr:`CAN_RENAME`,
                        :attr:`LOCKED`.
        :raise: ValueError when stricter lock-value on a renamed/relocated pstep
        """
        return vars(self)['_locked']

    @_locked.setter
    def _locked(self, lock_state):
        if self != self._orig:
            if lock_state < Pstep.CAN_RENAME or (
                    lock_state < Pstep.CAN_RELOCATE and '/' in self):
                msg = "Cannot rename/relocate '%s'-->'%s' due to %s!"
                raise ValueError(
                    msg % (self._orig, self, Pstep._lockstr(lock_state)))
        vars(self)['_locked'] = int(lock_state)

    @property
    def _fix(self):
        """Sets :attr:`locked` = `CAN_RENAME`.
        :return: self
        :raise: ValueError if step has been relocated pstep
        """
        self._locked = Pstep.CAN_RENAME
        return self

    @property
    def _lock(self):
        """Set :attr:`locked` = `LOCKED`.
        :return: self, for chained use
        :raise: ValueError if step has been renamed/relocated pstep
        """
        self._locked = Pstep.LOCKED
        return self

    def _tag(self, tag):
        """Add a "tag" for this pstep.

        :return: self, for chained use
        """
        tags = self._tags
        if tags:
            tags.add(tag)
        else:
            self._tags = set([tag])

        return self

    def _tag_remove(self, tag):
        """Delete a "tag" from this pstep.

        :return: self, for chained use
        """
        tags = self._tags
        if tags:
            tags.discard(tag)

        return self

    def _steps(self, keys=False, tag=None):
        csteps = self._csteps
        if not csteps:
            return []
        if keys:
            return csteps.keys()
        return csteps.values()

    def _paths(self, with_orig=False, tag=None):
        """
        Return all children-paths (str-list) constructed so far, in a list.

        :param bool with_orig: wheter to include also orig-path, for debug.
        :param str tag:        If not 'None', fetches all paths with `tag`
                               in their last step.
        :rtype: [str]


        Examples::

              >>> p = Pstep()
              >>> _ = p.a1._tag('inp').b._tag('inp').c
              >>> _ = p.a2.b2

              >>> p._paths()
              ['/a1/b/c', '/a2/b2']

              >>> p._paths(tag='inp')
              ['/a1', '/a1/b']


        For debugging set `with_orig` to `True`::

            >>> pmods = pmods_from_tuples([
            ...     ('',         'ROOT'),
            ...     ('/a',     'A/AA'),
            ... ])
            >>> p = pmods.step()
            >>> _ = p.a.b
            >>> p._paths(with_orig=True)
             ['(-->ROOT)/(a-->A/AA)/b']

        """
        def append_orig(s):
            orig = s._orig
            return '(%s-->%s)' % (orig, s) if s != orig else s

        paths = []
        for path in self._iter_hierarchy():
            send = path[-1]
            if (tag and tag in send._tags) or (not tag and not send._csteps):
                if with_orig:
                    path = [append_orig(s) for s in path]
                paths.append(_join_paths(*path))

        return sorted(set(paths))

    def _derrive_map_tuples(self):
        """
        Recursively extract ``(cmap --> alias)`` pairs from the pstep-hierarchy.

        :param list pairs:             Where to append subtree-paths built.
        :param tuple prefix_steps:     branch currently visiting
        :rtype: [(str, str)]
        """
        def orig_paths(psteps):
            return [p._orig for p in psteps]

        map_pairs = ((_join_paths(*orig_paths(p)), str(p[-1]))
                     for p in self._iter_hierarchy())
        return sorted(map_pairs, key=lambda p: p[0])

    def _iter_hierarchy(self, prefix_steps=()):
        """
        Breadth-first traversing of pstep-hierarchy.

        :param tuple prefix_steps:     Builds here branch currently visiting.
        :return: yields the visited pstep along with its path (including it)
        :rtype: (Pstep, [Pstep])
        """
        prefix_steps += (self, )
        yield prefix_steps

        csteps = self._csteps
        if csteps:
            for v in csteps.values():
                for p in v._iter_hierarchy(prefix_steps):
                    yield p

    @property
    def _schema(self):
        """Updates json-schema-v4 on this pstep (see :class:`JSchema`)."""

        # Lazy create it
        #    (clients should check before`_schema_exists()`)
        #
        sdict = vars(self)
        jschema = sdict.get('_schema')
        if jschema is None:
            sdict['_schema'] = jschema = JSchema()

        return jschema

    def _schema_exists(self):
        """Always use this to avoid needless schema-instantiations."""
        return '_schema' in vars(self)


if __name__ == '__main__':  # pragma: no cover
    raise NotImplementedError
