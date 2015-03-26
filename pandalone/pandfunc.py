#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
'''A best-effort attempt to build computation dependency-graphs from method with dict-like objects (such as pandas),
inspired by XForms: http://lib.tkk.fi/Diss/2007/isbn9789512285662/article3.pdf

.. seealso::
    Dependencies
    @calculation
    @calculation_factory [TODO]
    execute_funcs_map()
'''
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Iterable
import inspect
import logging
import re

from networkx.exception import NetworkXError

import itertools as it
import networkx as nx
import pandas as pd


from pandalone._mock import MagicMock

__commit__       = "$Id$"
DEBUG= False

_root_name = 'R'
_root_len = len(_root_name)+1
log = logging.getLogger(__name__)

class DependenciesError(Exception):
    def __init__(self, msg, item=None):
        super(Exception, self).__init__(msg)
        self.item = item

def _make_mock(*args, **kwargs):
    return MagicMock(*args, **kwargs)


def harvest_funcs_factory(funcs_factory, root=None, func_rels=None):
    new_func_rels = []

    ## Wrap and invoke funcs_factory with "rooted" mockups as args
    #    to collect mock_calls.
    #
    funcs_factory = _wrap_funcs_factory(funcs_factory)
    (root, mocks) = funcs_factory.mockup_func_args(root=root)

    cfuncs = funcs_factory(*mocks) ## The cfuncs are now wrapped children.

    ## Harvest cfunc deps as a list of 3-tuple (item, deps, funx)
    #    by inspecting root after each cfunc-call.
    #
    for cfunc in cfuncs:
        root.reset_mock()
        tmp = cfunc()
        try: tmp += 2   ## Force dependencies from return values despite compiler-optimizations.
        except:
            pass
        _harvest_mock_calls(root.mock_calls, cfunc, new_func_rels)
    funcs_factory.reset()

    _validate_func_relations(new_func_rels)

    if func_rels is not None:
        func_rels.extend(new_func_rels)

    return new_func_rels

def harvest_func(func, root=None, func_rels=None):
    """Invokes `func` with mock and harvests dependencies in func_rels."""
    new_func_rels = []

    func = _wrap_standalone_func(func)
    (root, mocks) = func.mockup_func_args(root=root)

    ret = func(*mocks)
    if ret is not None:
        msg = 'Dependency-func(%s) should not return value! \n  value=%s ' % (func, ret)
        raise DependenciesError(msg)
        pass
    func.reset()
    _harvest_mock_calls(root.mock_calls, func, new_func_rels)

    _validate_func_relations(new_func_rels)

    if func_rels is not None:
        func_rels.extend(new_func_rels)

    return new_func_rels


def _harvest_mock_calls(mock_calls, func, func_rels):
    ## A map from 'pure.dot.paths --> call.__paths__
    #  filled-in and consumed )mostly) by _harvest_mock_call().
    deps_set = OrderedDict()

    #last_path = None Not needed!
    for call in mock_calls:
        last_path = _harvest_mock_call(call, func, deps_set, func_rels)

    ## Any remaining deps came from a last not-assignment (a statement) in func.
    #  Add them as non-dependent items.
    #
    if deps_set:
        deps_set[last_path] = None  ## We don't care about call dep-subprefixes(the value) anymore.
#         if ret is None:
#             item = last_path
#         else:
#             item = _parse_mock_arg(ret)
        for dep in _filter_common_prefixes(deps_set.keys()):
            _append_func_relation(dep, [], func, func_rels)

def _parse_mock_arg(mock):
    mpath = _parse_mock_str(mock)[0]
    return (_strip_magic_tail(mpath), mpath)

def _harvest_mock_call(mock_call, func, deps_set, func_rels):
    '''Adds a 2-tuple (indep, [deps]) into indeps with all deps collected so far when it visits a __setartr__. '''

    def parse_mock_path(mock):
        mpath = _parse_mock_str(mock)[0]
        try:
            ## Hack to consolidate 'dot.__getitem__.com' --> fot.Xt.com attributes.
            #  Just search if previous call is subprefix of this one.
            prev_path = next(reversed(deps_set))
            prev_call = deps_set[prev_path]
            if (prev_call+'()' == call[:len(prev_call)+2]):
                mpath = prev_path + mpath[len(prev_call)+_root_len+2:] # 4 = R.()
        except (KeyError, StopIteration):
            pass
        return _strip_magic_tail(mpath)

    (call, args, kw) = mock_call

    deps_set.update((_parse_mock_arg(arg) for arg in args        if isinstance(arg, MagicMock)))
    deps_set.update((_parse_mock_arg(arg) for arg in kw.values() if isinstance(arg, MagicMock)))


    path = parse_mock_path(mock_call.parent)

    tail = call.split('.')[-1]
    if (tail == '__getitem__'):
        for item in _harvest_indexing(args[0]):
            new_path = '%s.%s'%(path, item)
            deps_set[new_path] = call
    elif (tail == '__setitem__'):
        deps = list(deps_set.keys())
        deps_set.clear()

        for item in _harvest_indexing(args[0]):
            new_path ='%s.%s'%(path, item)
            _append_func_relation(new_path, deps, func, func_rels)

    return path

def _append_func_relation(item, deps, func, func_rels):
    func_rels.append((item, deps, func))

def _harvest_indexing(index):
    '''Harvest any strings, slices, etc, assuming to be DF's indices. '''

    if isinstance(index, slice):
        deps = _harvest_indexing(index.start) + _harvest_indexing(index.stop) + _harvest_indexing(index.step)
    elif isinstance(index, str):
        deps = [index]
    elif isinstance(index, Iterable):
        deps = [ii for i in index for ii in _harvest_indexing(i)]
    else:
        deps = []
    return deps


def _strip_magic_tail(path):
    '''    some.path___with_.__magics__ --> some.path '''
    pa_th = path.split('.')
    while (pa_th[-1].startswith('__')):
        del pa_th[-1]
        if (not pa_th):
            return []
    path = '.'.join(pa_th)
    return path[:-2] if path.endswith('()') else path


_mock_id_regex = re.compile(r"name='([^']+)' id='(\d+)'")
def _parse_mock_str(m):
    return _mock_id_regex.search(m.__repr__()).groups()

def _validate_func_relations(func_rels):
    try:
        bad_items = [item for (item, _, _) in func_rels if not item.startswith(_root_name)]
        if bad_items:
            raise DependenciesError("Not all dependency-items are prefixed with root(%s): %s"%(_root_name, bad_items), bad_items)

        bad_deps = [dep for (_, deps, _) in func_rels for dep in deps if not dep.startswith(_root_name)]
        if bad_deps:
            raise DependenciesError("Not all dependency-data are prefixed with root(%s): %s"%(_root_name, bad_deps), bad_deps)

        ## NO, check only for algo-generated errors.
        # bad_funcs = [func for (_, _, func) in func_rels if not isinstance(func, _DepFunc)]
        # if bad_funcs:
        #     raise DependenciesError("Not all dependency-funcs are _DepFunc instances: %s"%bad_funcs, bad_funcs)
    except DependenciesError:
        raise
    except Exception as ex:
        raise DependenciesError("Bad explicit func_relations(%s) (item not a string, or deps not a tuple, etc) due to: %s"%(func_rels, ex), func_rels) from ex


def _build_func_dependencies_graph(func_rels, graph = None):
    '''
    :param func_rels: {(item, dep): funcs_set, ...}
    '''
    
    if graph is None:
        graph = nx.DiGraph()

    func_rels = _consolidate_relations(func_rels)

    for ((path, dep), funcs) in func_rels.items():
        graph.add_edge(path, dep, {'funcs': funcs})

    cycles = list(nx.simple_cycles(graph))
    if cycles:
        log.warning('Cyclic dependencies! %s', cycles)

    return graph


def _consolidate_relations(relations):
    '''[(item, deps, func), ...] --> {(item, dep): funcs_set, ...}'''

    rels = defaultdict()
    rels.default_factory = set #funcs-set

    ## Join all item's  deps & funcs, and strip root-name.
    #
    for (item, deps, func) in relations:
        item = item[_root_len:]
        for dep in deps:
            if dep:
                dep = dep[_root_len:]
                rels[(item, dep)].add(func)

    return rels


def _filter_common_prefixes(deps):
    '''deps: not-empty set

    example::
        deps = ['a', 'a.b', 'b.cc', 'a.d', 'b', 'ac', 'a.c']
        res = _filter_common_prefixes(deps)
        assert res == ['a.b', 'a.c', 'a.d', 'ac', 'b.cc']
    '''

    deps = sorted(deps)
    (it1, it2) = it.tee(deps)
    s2 = next(it2)
    ndeps = []
    try:
        while(True):
            s1=next(it1)
            s2=next(it2)
            if s1+'.' != s2[:len(s1)+1]:
                ndeps.append(s1)
    except StopIteration:
        ndeps.append(s2)

    return ndeps


def _keep_calculation_routes(graph, sources, dests):
    ## Trim irrelevant nodes.
    #
    graph.nbunch_iter(dests + _all_predecessors(graph, dests))
    
def _research_calculation_routes(graph, sources, dests):
    '''Find nodes reaching 'dests' but not 'sources'.

    :param graph: dependency-graph, to be cloned (not modified)
    :param sources: a list of nodes (existent or not) to search for all paths originating from them
    :param dests:   a list of nodes to search for all paths leading to them them
    :return: a 4-tuple (see source)
    '''

    ## Remove dests already present in sources.
    #
    calc_out_nodes = set(dests)
    calc_out_nodes -= set(sources)

    calc_inp_nodes = set(graph.nbunch_iter(sources))

    ## Deps graph: all INPUT's deps broken
    #    To be used for planing functions_to_run.
    #
    deps_graph = graph.copy()
    deps_graph.remove_edges_from(deps_graph.out_edges(calc_inp_nodes))

    ## Data_to_be_calced: all INPUTs erased
    #    To be used for topological-sorting.
    #
    data_graph = graph.copy()
    data_graph.remove_nodes_from(calc_inp_nodes)
    
    ## Report (and why remove?) isolated islands.
    #
    isolates = nx.isolates(data_graph)
    log.debug('Isolates: %s', isolates)
    data_graph.remove_nodes_from(isolates)
    
    try:
        io_pairs = [(o, i) for o in calc_out_nodes for i in calc_inp_nodes]
        routes = set(it.chain(nx.all_simple_paths(deps_graph, pair) for pair in io_pairs))
        calc_nodes = set(list(calc_out_nodes) + _all_predecessors(data_graph, calc_out_nodes))
    except (KeyError, NetworkXError) as ex:
        unknown = [node for node in calc_out_nodes if node not in graph]
        raise DependenciesError('Unknown OUT-args(%s)!' % unknown, (graph, unknown)) from ex
    else:
        return (calc_inp_nodes, calc_out_nodes, calc_nodes, deps_graph)

def _all_predecessors(graph, nodes):
    return [k for node in nodes for k in nx.bfs_predecessors(graph, node).keys()]
def _all_successors(graph, nodes):
    return [k for node in nodes for k in nx.bfs_successors(graph, node).keys()]


def _find_calculation_order(graph, calc_nodes):
    subgraph = graph.subgraph(calc_nodes)
    ordered_calc_nodes = list(reversed(nx.topological_sort(subgraph)))

    return ordered_calc_nodes


def _find_missing_input(calc_inp_nodes, graph):
    '''Search for *tentatively* missing data.'''
    calc_inp_nodes = set(calc_inp_nodes) # for efficiency below
    missing_input_nodes = []
    for node in nx.dfs_predecessors(graph):
        if ( node not in calc_inp_nodes and graph.out_degree(node) == 0):
            missing_input_nodes.append(node)
    return missing_input_nodes


def _extract_funcs_from_edges(graph, ordered_nodes):
    # f=list(fs[0]['funcs'])[0]
    funcs = [f for (_, _, d) in graph.edges_iter(ordered_nodes, True) if d
            for f in d['funcs']] # a list of sets


    ## Remove duplicates and No-funcs whilist preserving order.
    #
    od = OrderedDict.fromkeys(funcs)
    od.pop(None, None)
    funcs = list(od)

    return funcs




def _wrap_standalone_func(func):
    return _DepFunc(func=func, is_funcs_factory=False)
def _wrap_funcs_factory(funcs_factory):
    return _DepFunc(func=funcs_factory, is_funcs_factory=True)
def _wrap_child_func(funcs_factory, child_index):
    if not isinstance(funcs_factory, _DepFunc):
        funcs_factory = _wrap_funcs_factory(funcs_factory)
    return _DepFunc(func=funcs_factory, _child_index=child_index)
class _DepFunc:
    '''A wrapper for functions explored for relations, optionally allowing them to form a hierarchy of factories and produced functions.

    It can be in 3 types:
        * 0, standalone function: args given to function invocation are used immediatelt,
        * 10, functions-factory: args are stored and will be used by the child-funcs returned,
        * 20, child-func: created internally, and no args given when invoked, will use parent-factory's args.

    Use factory methods to create one of the first 2 types:
        * pdcalc._wrap_standalone_func()
        * pdcalc._wrap_funcs_factory()
    '''
    TYPES = ['standalone', 'funcs_fact', 'child']

    def __init__(self, func, is_funcs_factory=False, _child_index=None):
        self.func = func
        if is_funcs_factory:            ## Factory
            self._type = 1
            self.child_funcs = None

            assert _child_index == None, self
        elif _child_index is not None:  ## Child
            self._type = 2
            self.child_index = _child_index

            assert func.is_funcs_factory(), self
        else:                           ## Standalone
            self._type = 0

            assert _child_index == None, self

        if not callable(func):
            raise DependenciesError('Cannot create a _DepFunc for a non-callable(%s)!'%func, func)


    def get_type(self):
        return _DepFunc.TYPES[self._type]

    def is_standalone_func(self):
        return self._type == 0
    def is_funcs_factory(self):
        return self._type == 1
    def is_child_func(self):
        return self._type == 2

    def reset(self):
        if self.is_funcs_factory():
            self.child_funcs = None
    def is_reset(self):
        assert self.is_funcs_factory(), self

        return self.child_funcs is not None

    def mockup_func_args(self, root=None):
        assert not self.is_child_func(), self

        if not root:
            root = _make_mock(name=_root_name)

        sig = inspect.signature(self.func)
        mocks = []
        for (name, param) in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                log.warning('Any dependencies from **%s will be ignored for %s!', name, self)
                break
            mock = _make_mock()
            mocks.append(mock)
            root.attach_mock(mock, name)
        return (root, mocks)


    def __call__(self, *args, **kwargs):
        if self.is_standalone_func():           ## Standalone
            return self.func(*args, **kwargs)

        elif self.is_funcs_factory():           ## Factory
            cfuncs = self.func(*args, **kwargs)
            if not cfuncs:
                raise DependenciesError('%s returned %s as child-functions!'%(self.func, cfuncs), self.func)
            self.child_funcs = cfuncs
            return [_DepFunc(func=self, _child_index=i) for i in range(len(cfuncs))]

        else:                                   ## Child
            parent_fact = self.func
            assert parent_fact.is_funcs_factory(), self

            ## Use new args only if parent has previously been reset.
            #
            if (args or kwargs) and not parent_fact.is_reset():
                parent_fact(*args, **kwargs) ## Ignore returned depfuncs, we are the children!

            cfunc = parent_fact.child_funcs[self.child_index]
            return cfunc()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        try:
            if (self.is_child_func()):
                return '_DepFunc<child>(%s, %s)'%(self.func.func, self.child_index)
            return '_DepFunc<%s>(%s)'%(self.get_type(), self.func)
        except:
            return '_DepFunc<BAD_STR>(%s)'%self.func


##############################
## User Utilities
##############################
##

def default_arg_paths_extractor(arg_name, arg, paths):
    '''
    Dig recursively objects for indices to build the `sources`/`dests` sequences for :meth:`Dependencies.build_plan()`

    It extracts indices recursively from maps and pandas
    The inner-ones paths are not added, ie ``df.some.key``, but not ``df`` or ``df.some``.

    .. Note:: for pandas-series their index (their infos-axis) gets appended only the 1st time
        (not if invoked in recursion, from DataFrame columns).

    :param list paths: where to add the extracted paths into
    '''
    try:
        for key in arg.keys():
            path = '%s.%s'%(arg_name, key)
            value = arg[key]
            if (isinstance(value, pd.Series)):
                paths.append(path) # do not recurse into series.
            else:
                default_arg_paths_extractor(path, value, paths)
    except (AttributeError, KeyError):
        paths.append(arg_name)


def tell_paths_from_named_args(named_args, arg_paths_extractor_func=default_arg_paths_extractor, paths=None):
    '''
    Builds the `sources` or the `dests` sequences params of :meth:`Dependencies.build_plan()` from a map of function-arguments

    :param named_args: an ordered map ``{param_name --> param_value}`` similar to that returned by
            ``inspect.signature(func).bind(*args).arguments: BoundArguments``.
            Use the utility :func:`name_all_func_args()` to generate such a map for some function.
    :return: the `paths` updated with all ''dotted.vars'' found
    '''

    if paths is None:
        paths = []
    for (name, arg) in named_args.items():
        arg_paths_extractor_func(name, arg, paths)

    return paths


def name_all_func_args(func, *args, **kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)

    return bound_args.arguments



##############################
## Classes
##############################
##

class Dependencies:
    '''
    Discovers and stores the rough functions-relationships needed to produce ExecutionPlanner

    The relation-tuples are "rough" in the sense that they may contain duplicates etc
    requiring cleanup by _consolidate_relations().

    **Usage:**
    

    Use the `harvest_{XXX}()` methods or the `add_func_rel()` method
    to gather dependencies from functions and function-facts
    and then use the `build_plan()` method to freeze them into a `plan`.
    '''

    @classmethod
    def from_funcs_map(cls, funcs_map, deps = None):
        '''
        Factory method for building `Dependencies` by harvesting multiple functions and func_factories, at once

        :param funcs_map: a mapping or a sequence of pairs ``(what --> bool_or_null)`` with values:
                
                True
                    when `what` is a funcs_factory, a function returning a sequence of functions processing the data
                False
                    when `what` is a standalone_function, or
                None
                    when `what` is an explicit relation 3-tuple (item, deps, func) to be fed
                    directly to :func:`Dependencies.add_func_rel()`, where:
                    
                    item 
                        a ''dotted.varname'' string,
                    deps
                        a string or a sequence of ''dotted.varname'' strings, and
                    func
                        a standalone func or a funcs_factory as a 2-tuple (func, index)
                        
                    Both items and deps are dot-separated sequence of varnames, such as 'foo.bar'
            
        :param deps: if none, a new `Dependencies` instance is created to harvest relations
        :return: a new (even inherited) or an updated `Dependencies` instance

        .. Important:: All functions must accept exactly the same args, or else the results
                will be undetermined and it will most-likely scream on execute_funcs_map.
        '''

        if deps is None:
            deps = cls()

        if isinstance(funcs_map, Mapping):
            pairs = funcs_map.items()
        else:
            pairs = funcs_map

        for (func, is_factory) in list(pairs):
            try:
                if is_factory is None:          ## relation-tuple
                    funcs_map.pop(func)
                    deps.add_func_rel(*func)
                elif is_factory:                ## funcs_factory
                    deps.harvest_funcs_factory(func)
                else:                           ## standalone-function
                    deps.harvest_func(func)
            except DependenciesError as ex:
                raise
            except Exception as ex:
                raise DependenciesError("Failed harvesting dependencies for %s due to: %s"%(func, ex), func) from ex

        return deps


    def __init__(self):
        self._relation_tuples = []

    def harvest_funcs_factory(self, funcs_factory):
        root = _make_mock(name=_root_name)
        harvest_funcs_factory(funcs_factory, root=root, func_rels=self._relation_tuples)
        log.debug('DEPS collected(%i): %s', len(self._relation_tuples), self._relation_tuples)

    def harvest_func(self, func):
        root = _make_mock(name=_root_name)
        harvest_func(func, root=root, func_rels=self._relation_tuples)
        log.debug('DEPS collected(%i): %s', len(self._relation_tuples), self._relation_tuples)

    def add_func_rel(self, item, deps=None, func=None):
        '''
        :param str item: a dotted.var
        :param deps: a string or a sequence of dotted.vars
        :param func: a standalone func, or a funcs_factory as a 2-tuple (func, index)
        '''

        ## Prepare args and do some basic argument-checking.
        #
        try:
            if func is not None:
                if isinstance(func, tuple):
                    (funcs_factory, child_index) = func
                    func = _wrap_child_func(funcs_factory, child_index)
                else:
                    func = _wrap_standalone_func(func)
        except (TypeError, ValueError) as ex:
            raise DependenciesError("Failed adding explicit func_relation(%s) due to bad functions\n  Function must either be a standalone-function or a 2-tuple(funcS_factory, child_index)."%item) from ex
        try:
            item = _root_name + '.' + item
            if deps is None:
                deps = []
            else:
                if isinstance(deps, str):
                    deps = [deps]
                deps = [_root_name + '.' + dep for dep in deps]
        except TypeError as ex:
            raise DependenciesError("Failed adding explicit func_relation(%s) due to bad item or dependencies strings!"%item) from ex

        new_func_rels = []
        _append_func_relation(item, deps, func, new_func_rels)

        _validate_func_relations(new_func_rels)

        self._relation_tuples.extend(new_func_rels)

    def _build_deps_graph(self):
        graph = _build_func_dependencies_graph(self._relation_tuples)
        return graph

    def _make_empty_plan(self):
        return pd.Series(dict(calc_inp_nodes=[], calc_out_nodes=[], calc_nodes=[],
            missing_data=[] if DEBUG else None, deps_graph=[]))

    def build_plan(self, sources, dests):
        '''
        Builds an execution-plan that when executed with :func:`execute_plan()` it will produce `dests` from `sources`.

        Turns any stored-relations into a the full dependencies-graph then trims it
        only to those ''dotted.varname'' nodes reaching from `sources` to `dests`.

        :param sequence dests:   a list of ''dotted.varname''s that will be the outcome of the calculations
        :param sources: a list of ''dotted.varname''s (existent or not) that are assumed to exist
                when the execution wil start
        :return: the new `plan` that can be fed to :func:`execute_plan()`
        :rtype: pd.Series
        '''

        log.debug('EXISTING data(%i): %s', len(sources), sources)
        log.debug('REQUESTED data(%i): %s', len(dests), dests)

        plan        = self._make_empty_plan()
        plan.dests  = dests

        graph       = self._build_deps_graph()
        log.debug('GRAPH constructed(%i): %s', graph.size(), graph.edges(data=True))

        (calc_inp_nodes, calc_out_nodes, unordered_calc_nodes, deps_graph) = \
                                _research_calculation_routes(graph, sources, dests)
        plan.deps_graph     = deps_graph
        plan.calc_out_nodes = calc_out_nodes
        plan.calc_inp_nodes = calc_inp_nodes

        calc_nodes          = _find_calculation_order(graph, unordered_calc_nodes)
        plan.calc_nodes     = calc_nodes
        plan.deps           = deps_graph.edges(calc_nodes, data=True)

        if DEBUG:
            missing_inp_nodes       = _find_missing_input(calc_inp_nodes, deps_graph)
            plan.missing_inp_nodes  = missing_inp_nodes
        else:
            plan.missing_inp_nodes  = None

        funcs               = _extract_funcs_from_edges(deps_graph, calc_nodes)
        plan.funcs          = funcs

        return plan




##############################
## Top-level methods
##############################
##

## Used if no default-dependencies specified in the `calculation` decorator, below.
calculation_deps = Dependencies()

def calculation(deps = None):
    '''A decorator that extracts dependencies from a function.

    :param deps: if None, any :subscript:`2` extracted deps are added in the `calculation_deps` module-variable.

    Example::

        @calculation
        def foo(a, r):
            a['bc'] = a.aa
            r['r1'] = a.bb

        @calculation
        def bar(a, r):
            r['r2'] = a.bc

        plan    = calculation_deps.build_plan(['a.ab', 'a.bc', 'r'], ['r.r2'])
        planner.execute_funcs_map(plan, 1, 2)
    '''

    if deps is None:
        deps = calculation_deps

    def decorator(func):
        deps.harvest_func(func)
        return func
    return decorator


def execute_plan(plan, *args, **kwargs):
    results = []
    for func in plan.funcs:
        try:
            results.append(func(*args, **kwargs))
        except Exception as ex:
            raise DependenciesError("Failed executing %s due to: %s"%(func, ex), func) from ex

    return results


def execute_funcs_factory(funcs_fact, dests, *args, **kwargs):
    '''A one-off way to run calculations from a funcions-factory (see :func:`execute_funcs_map()`)'''
    return execute_funcs_map({funcs_fact: True}, dests, *args, **kwargs)

def execute_funcs_map(funcs_map, dests, *args, **kwargs):
    '''
    A one-off way to run calculations from a funcs_map as defined in :meth:`Dependencies:from_funcs_map()`.

    :param funcs_map: a dictionary similar to the one used by :meth:`Dependencies.from_funcs_map()`
    :param seq dests: what is required as the final outcome of the execution,
            ie: for the func-functory::

                 def some_funcs(foo, bar):
                     def func_1():
                         foo['a1'] = ...
                     ...
                     def func_n(): ...
                         foo['a2'] = ...
                         bar['b1'] = ...

                     return [func_1, ..., func_n]

            the `dests` may be::

                ['foo.a1', 'foo.a2', 'bar.b1']

    :param args: the actual args to use for invoking the functions in the map,
            and the names of these args would come rom the first function
            to be invoked (which ever that might be!).


    .. Note:: Do not use it to run the calculations repeatedly.  Preffer to cache the function-relations
            into an intermediate :class:`Dependencies` instance, instead.
    '''

    ## Find the first func in the map and
    #    use it for naming the args.
    #
    a_func = None
    for func in funcs_map.keys():
        if callable(func):
            a_func = func
            break
    else:
        raise DependenciesError('No function found in funcs_map(%s)!'%funcs_map, funcs_map)
    named_args  = name_all_func_args(a_func, *args, **kwargs)
    sources     = tell_paths_from_named_args(named_args)

    deps        = Dependencies.from_funcs_map(funcs_map)
    plan        = deps.build_plan(sources, dests)

    return execute_plan(plan, *args, **kwargs)

