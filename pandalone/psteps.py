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
    See :class:`Component`, :class:`FuncComponent` and :class:`Assembly`.

paths and path-mappings (pmods):
    See :class:`Pmod`, :func:`pmods_from_tuples` and :class:`Pstep`.
"""

from __future__ import division, unicode_literals

import logging

from pandalone.pandata import _iter_jsonpointer_parts_relaxed, JSchema


__commit__ = ""

log = logging.getLogger(__name__)


def _append_step(steps, step):
    """
    Joins `steps`-list with `path`, respecting '/', '..', '.', ''.

    :return: the new or updated steps-list.
    :rtype:  list

    Example::

        >>> _append_step([], 'a')
        ['a']

        >>> _append_step([], '..)
        ['']
        >>> _append_step(['a', 'b'], '..')
        ['a']

        >>> _append_step(['a', 'b'], '.')
        ['a', 'b']

        >>> _append_step(['a', 'b'], '')
        ['']

    """
    if step == '':
        steps = []
    elif step == '.':
        pass
    elif step == '..':
        steps = steps[:-1]
    else:
        steps.append(step)

    return steps


def _append_path(steps, path):
    """
    Joins `steps`-list with `path`, respecting '/', '..', '.', ''.

    :return: the new or updated steps-list.
    :rtype:  list

    Example::

        >>> _append_path([], 'a')
        ['a']

        >>> _append_path([], '../a')
        ['a']
        >>> _append_path(['a', 'b'], '../c')
        ['a', 'c']
        >>> _append_path(['a', 'b'], '../../c')
        ['c']

        >>> _append_path(['a', 'b'], '.')
        ['a', 'b']

        >>> _append_path(['a', 'b'], './c')
        ['a', 'b', 'c']

        >>> _append_path(['a', 'b'], '/r')
        ['r']

        >>> _append_path(['a', 'b'], '')
        []

    """
    for step in _iter_jsonpointer_parts_relaxed(path):
        steps = _append_step(steps, step)

    return steps


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

    :ivar Pstep _csteps: the child-psteps
    :ivar dict _pmod:   path-modifications used to construct this and
                         relayed to children
    :ivar int _lock:     one of
                         - :const:`Pstep.CAN_RELOCATE`(default, reparenting allowed),
                         - :const:`Pstep.CAN_RENAME`,
                         - :const:`Pstep.LOCKED' (neither from the above).
                         - TODO: Split pstep._lock into _fix/_lock property. 
    :ivar dict _schema:  json-schema data.


    Usage:

    - Just referencing (non_private) attributes, creates them.

    - Private attributes and functions (starting with '_') exist for 
      specific operations (ie for specifying json-schema, or 
      for collection all paths).

    - Assignments are only allowed to private attributes::

          >>> p.assignments = 'FAIL!'
          Traceback (most recent call last):
          AssertionError: Cannot assign 'FAIL!' to '/deeper/ROOT/assignments'!      
                  Only other psteps allowed.

          >>> p._but_hidden = 'Ok'

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


    - Any "path-mappings" or "pmods" maybe specified during construction::

        >>> pmods = pmods_from_tuples([
        ...     ('root',         'deeper/ROOT'),
        ...     ('root/abc',     'ABC'),
        ...     ('root/abc/foo', 'BAR'),
        ... ])
        >>> p = Pstep('root', pmods=pmods)
        >>> p.abc.foo
        `BAR`
        >>> p._paths
        ['/deeper/ROOT/ABC/BAR']

    - but exceptions are thrown if mapping any step marked as "locked":

        >>> p.abc.foo._lock

    - .. Warning::
          String's slicing operations do not work on this string-subclass!

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

    def __new__(cls, pname='', pmod=None, alias=None):
        """
        Constructs a string with str-content which may be mapped from pmods.

        :param str pname:   this pstep's name; it is stored at `_orig` and
                            if unmapped by pmod, becomes super-str object.
        :param PMod pmod:   the mappings for the children of this pstep, which
                            contains the un-expanded `_alias` for this pstep,
                            or None
        :param str alias:   the regex-expanded alias for this pstep, or None
        """
        alias = pmod._alias if pmod and not alias else alias
        self = str.__new__(cls, alias or pname)
        self._orig = pname
        self._pmod = pmod
        self._csteps = {}
        vars(self)['_lock'] = Pstep.CAN_RELOCATE

        return self

    def __missing__(self, cpname):
        try:
            cpmod, alias = self._pmod.descend(cpname)
        except:
            cpmod, alias = (None, None)
        child = Pstep(cpname, cpmod, alias)
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
                raise ValueError(
                    msg % (self._orig, self, Pstep._lockstr(lock)))
        vars(self)['_lock'] = int(lock)

    @property
    def _paths(self):
        """
        Return all children-paths (str-list) constructed so far, in a list. 

        :rtype: [str]
        """
        paths = []
        self._append_children(paths)
        return ['/'.join(p) for p in paths]

    def _append_children(self, paths, prefix_steps=[]):
        """
        Append all child-steps in the `paths` list. 

        :param list prefix_steps: default-value always copied
        :rtype: [[str]]
        """
        nprefix = prefix_steps.copy()
        nprefix.append(self)
        if self._csteps:
            for v in self._csteps.values():
                v._append_children(paths, nprefix)
        else:
            paths.append(nprefix)

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


if __name__ == '__main__':
    raise "Not runnable!"
