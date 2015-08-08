#! python
#-*- coding: utf-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Defines the building-blocks of a "model":

components and assemblies:
    See :class:`Component`, :class:`FuncComponent` and :class:`Assembly`.

paths and path-mappings (pmods):
    See :class:`Pmod`, :func:`pmods_from_tuples` and :class:`Pstep`.

TODO
----

1. Assembly use ComponentLoader collecting components with:

   - `gatattr()` and
   - `filter_predicate` default to ``attr.__name__.startswith('cfunc_')``.
   - enforce a `disable` flag on them.

2. Component/assembly should have a stackable or common cwd?

3. Components should be easy to run without "framework".
   - `_build()` --> `run()`
   - pmods on init OR `run()`?
   - As ContextManager?

4. Imply a default Assembly.
"""

from __future__ import division, unicode_literals

from abc import ABCMeta, abstractmethod
import logging
from pandalone.mappings import Pstep

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock  # @UnusedImport


__commit__ = ""

log = logging.getLogger(__name__)


name_uniqueizer = None  # TODO: Handle clashes on component-names.


class Component(object):

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

    __metaclass__ = ABCMeta

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
    def _build(self, pmod=None):
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

        >>> from pandalone.mappings import pmods_from_tuples

        >>> pmods = pmods_from_tuples([
        ...     ('~.*', '/A/B'),
        ... ])
        >>> comp._build(pmods)

        >>> sorted(comp.pinp()._paths())
        ['/A/B/T', '/A/B/V']

        >>> comp.pout()._paths()
        ['/A/B/Acc']

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
        # self._pmod = None
        # self._pinp = None
        # self._pout = None

    def __call__(self, *args, **kws):
        self._cfunc(self, *args, **kws)

    def _fetch_all_paths(self, pstep):
        return pstep._paths() if pstep else []

    def pinp(self, path=None):
        """The suggested :class:`Pstep` for cfunc to use to access inputs."""
        p = self._pinp
        if p is None:
            self._pinp = p = Pstep(path or self._name,
                                   _proto_or_pmod=self._pmod)
        return p

    def pout(self, path=None):
        """The suggested :class:`Pstep` for cfunc to use to access outputs."""
        p = self._pout
        if p is None:
            self._pout = p = Pstep(path or self._name,
                                   _proto_or_pmod=self._pmod)
        return p

    def _build(self, pmod=None):
        """Extracts inputs/outputs from cfunc. """
        vtree = MagicMock()
        self._inp = []
        self._out = []
        self._pinp = None
        self._pout = None
        self._pmod = pmod

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

        >>> from pandalone.mappings import pmods_from_tuples

        >>> pmod = pmods_from_tuples([
        ...     ('~.*',  '/root'),
        ... ])
        >>> ass._build(pmod)
        >>> sorted(ass._inp + ass._out)
        ['/root/A', '/root/B', '/root/B', '/root/C']

    """

    def __init__(self, components, name=None):
        Component.__init__(self, name=name or 'assembly')
        self._comps = list(components)

    def __call__(self, *args, **kws):
        pass  # TODO: Invoke Dispatcher with Assembly's child-components.

    def _build(self, pmod=None):
        inp = set()
        out = set()
        for c in self._comps:
            c._build(pmod)
            inp.update(c._inp)
            out.update(c._out)
        self._inp = sorted(inp)
        self._out = sorted(out)


if __name__ == '__main__':  # pragma: no cover
    raise NotImplementedError
