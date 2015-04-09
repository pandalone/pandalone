#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, unicode_literals

from abc import ABCMeta, abstractmethod, abstractproperty
from unittest.mock import MagicMock

import itertools as itt

from .pandata import Pstep, resolve_jsonpointer


__commit__ = ""


def component_from_cfunc(cfunc):
    comp = Component(cfunc)
    mm = MagicMock()
    cfunc(comp, mm)

    # Component's validation pending now.


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
    """

#     def __init__(self, func, **kws):
#         self.func = func
#         self.inp = kws.get('inp')
#         self.out = kws.get('out')
#         self.name = str(kws.get('name', func.__name__))

    @abstractmethod
    def __call__(self, *agrs, **kws):
        pass

    @abstractproperty
    def _name(self):
        pass

    @abstractproperty
    def _inp(self):
        pass

    @abstractproperty
    def _out(self):
        pass

    def _build(self):
        pass

    def _iter_validations(self):
        """ Yields a msg for each failed validation rule. 

        Invoke it after :meth:`_build()` component.
        """
        if False:
            yield
#         expected_attrs = ['name', 'inp', 'out']
#         for attr in expected_attrs:
#             if not hasattr(self, attr):
#                 yield "`%s` is unset!" % attr


class FuncComponent(Component):

    """
    A utility to developers of a cfuncs, which is passed as their 1st arg.

    CFuncs may use :meth:`pinp` and :meth:`pout` when accessing 
    their input and output data-tree values respectively.  
    But if CFuncs access additional values with "fixed' paths, then they 
    have to manually add those paths into the :attr:`inp` and :attr:`out`
    lists.

    Example:

    This would be a fully "relocatable" cfunc::

        >>> from pandalone.pandata import resolve_jsonpointer
        >>> def cfunc_calc_rate(comp, vtree):
        ...     pi = comp.pinp()
        ...     po = comp.pout('deep/results')
        ...
        ...     df1 = comp.walk(vtree, pi.foo)
        ...     df2 = comp.walk(vtree, pi.bar)
        ...     df = comp.walk(vtree, po)
        ...
        ...     df[po.R] = df2[pi.V] / df2[pi.A]

    After build time, it will be a valid component::

        >>> comp = FuncComponent(cfunc_calc_rate)
        >>> comp._build()
        >>> assert list(comp._iter_validations()) == []


    """

    def __init__(self, cfunc):
        self.__inp = []
        self.__out = []
        self._cfunc = cfunc
        self.__name = str(cfunc)  # TODO: Remove prefix from cfunc-names.
        self.pinps = {}
        self.pouts = {}

    def __call__(self, *args, **kws):
        self._cfunc(self, *args, **kws)

    @property
    def _name(self):
        self.__name

    @property
    def _inp(self):
        self.__inp

    @property
    def _out(self):
        self.__out

    def _build(self):
        """
        Invoked once before run-time to extract inputs/outputs from cfunc.

        Example:


        """
#             >>> from unittest.mock import MagicMock
#             >>> comp = FuncComponent(cfunc_calc_rate)
#             >>> vtree = MagicMock()
#             >>> cfunc_calc_rate(comp, vtree)
        vtree = MagicMock()
        self._cfunc(self, vtree)
        self.__inp.extend(self._fetch_all_paths(self.pinps))
        self.__out.extend(self._fetch_all_paths(self.pouts))

    def _fetch_all_paths(self, psteps_map):
        return itt.chain(p._paths for p in psteps_map.values())

    def _getadd_pstep(self, psteps_map, path):
        p = psteps_map.get(path)
        if p is None:
            p = Pstep(path)
            psteps_map[path] = p
        return p

    def pinp(self, path=None):
        """The suggested :class:`Pstep` for accessing inputs."""
        return self._getadd_pstep(self.pinps, path)

    def pout(self, path=None):
        """The suggested :class:`Pstep` for accessing outputs."""
        return self._getadd_pstep(self.pouts, path)

    def walk(self, doc, path):
        # FIXME: javadoc hack for absolute-path.
        return resolve_jsonpointer(doc, '/' + path)


if __name__ == '__main__':
    raise "Not runnable!"
