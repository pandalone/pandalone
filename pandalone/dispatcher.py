from heapq import heappush, heappop
from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt

try:
    from cPickle import dump, load, HIGHEST_PROTOCOL
except ImportError:
    from pickle import dump, load, HIGHEST_PROTOCOL


class DispatcherMap(object):
    """
    Base class for dispatcher maps.

    A DispatcherMap stores data and function nodes with optional attributes as
    NetworkX DiGraphs.

    :ivar graph: A directed graph that maps the workflow of the used functions.
    :type graph: NetworkX DiGraph

    :ivar default_values: A dictionary of the data node default values.
    :type default_values: dict

    :ivar start: label used has starting node.
    :type start: hashable Python object

    Example:

    Create an empty graph structure (a "null graph") with no nodes and
    no edges::

        >>> dmap = DispatcherMap()

        # Add data node to the dispatcher map.
        >>> add_data(dmap, data_id='/a')
        '/a'
        >>> add_data(dmap, data_id='/c')
        '/c'

    Add data node with a default value to the dispatcher map.
        >>> add_data(dmap, data_id='/b', default_value=1)
        '/b'

    Create a function node.
        >>> def diff_function(a, b):
        ...     return b - a

        >>> add_function(dmap, function=diff_function, inputs=['/a', '/b'], \
                              outputs=['/c'])
        '...dispatcher:diff_function<0>'

    Create a function node with domain.
        >>> from math import log

        >>> def log_domain(x):
        ...     return x > 0

        >>> add_function(dmap, function=log, inputs=['/c'], outputs=['/d'], \
                              input_domain=log_domain)
        'math:log<0>'

    Create a data node with function estimation and callback function.
        - function estimation:
            estimate one unique output from multiple estimations.
        - callback function:
            is invoked after computing the output.
        >>> def average_fun(kwargs):
        ...     x = kwargs.values()
        ...     return sum(x) / len(x)

        >>> def callback_fun(x):
        ...     print('(log(1)+1)/2=%.1f'%x)

        >>> add_data(dmap, data_id='/d', default_value=1, wait_inputs=True, \
                          function=average_fun, callback=callback_fun)
        '/d'

    Run the dispatcher output assuming that data node '/a' has value 1.
        >>> outputs, dmap_out = run_output(dmap, input_values={'/a': 0})
        (log(1)+1)/2=0.5
        >>> outputs == {'/a': 0, '/b': 1, '/d': 0.5, '/c': 1}
        True
    """

    def __init__(self, *args, **kwargs):
        self.graph = nx.DiGraph()
        self.counter = Counter()
        self.start = 'start'
        self.default_values = {}


def add_data(dmap, data_id=None, default_value=None, wait_inputs=False,
             wait_exceptions=None, function=None, callback=None, **kwargs):
    """
    Add a single data node to dispatcher map.

    :param dmap: dispatcher map that identifies the model to be adopted.
    :type dmap: DispatcherMap

    :param data_id: data node id (dfl=next 'int' not in id list of graph nodes).
    :type data_id: any hashable Python object except None, optional

    :param default_value: data node default value
    :type default_value: any object, optional

    :param wait_inputs:
        if True Dijkstra algorithm stops on the node until the gets all input
        estimations.
    :type wait_inputs: bool, optional

    :param wait_exceptions: flag (requires wait_inputs=True)
    :type wait_exceptions: list of function node ids, optional

    :param function: data node estimation function (requires wait_inputs=True)
    :type function: any function that takes only keywords (key=function node id)
                    as inputs and return one value, optional

    :param callback: callback function to be called after node estimation
    :type callback: any function that takes only one argument (data node value)

    :param kwargs:  Set or change node attributes using key=value.
    :type callback: keyword arguments, optional
        
    Example::

        >>> dmap = DispatcherMap()

        # data to be calculated (i.e., result data node)
        >>> add_data(dmap, data_id='/a')
        '/a'

        # data with a default value (i.e., input data node)
        >>> add_data(dmap, data_id='/b', default_value='value of the data')
        '/b'

        >>> def average_fun(*x):
        ...     return sum(x) / len(x)

        # data node that is estimated as the average of all function node
        # estimations
        >>> add_data(dmap, data_id='/c', wait_inputs=True, function=average_fun)
        '/c'

        # initial data that is calculated as the average of all estimations
        >>> add_data(dmap, data_id='/d', default_value='value of the data', \
                          wait_inputs=True, function=average_fun)
        '/d'

        # create an internal data and return the generated id
        >>> add_data(dmap)
        0

    See Also
    --------
    add_function
    load_dmap_from_lists

    Notes
    -----
    A hashable object is one that can be used as a key in a Python
    dictionary. This includes strings, numbers, tuples of strings
    and numbers, etc.

    On many platforms hashable items also include mutables such as
    NetworkX Graphs, though one should be careful that the hash
    doesn't change on mutables.
    """

    attr_dict = {'type': 'data', 'wait_inputs': wait_inputs}

    if function is not None:
        attr_dict['function'] = function

    if callback is not None:
        attr_dict['callback'] = callback

    if wait_exceptions is not None:
        attr_dict['wait_exceptions'] = wait_exceptions

    attr_dict.update(kwargs)

    if data_id is None:
        data_id = dmap.counter()
        while dmap.graph.has_node(data_id):
            data_id = dmap.counter()

    if default_value is not None:
        dmap.default_values[data_id] = default_value
    elif data_id in dmap.default_values:
        dmap.default_values.pop(data_id)

    dmap.graph.add_node(data_id, attr_dict=attr_dict)

    return data_id


def add_function(dmap, function_id = None, function=lambda x: None, inputs=None,
                 outputs=None, input_domain=None, weight=None, edge_weight=None,
                 **kwargs):
    """
    Example::

        >>> dmap = DispatcherMap()

        >>> def my_function(a, b):
        ...     c = a + b
        ...     d = a - b
        ...     return c, d

        >>> add_function(dmap, function=my_function, inputs=['/a', '/b'], \
                              outputs=['/c', '/d'])
        '...dispatcher:my_function<0>'

        >>> from math import log
        >>> def my_log(a, b):
        ...     log(b - a)

        >>> def my_domain(a, b):
        ...     return a < b

        >>> add_function(dmap, function=my_log, inputs=['/a', '/b'], \
                              outputs=['/e'], input_domain=my_domain)
        '...dispatcher:my_log<0>'
    """

    if outputs is None:
        outputs = [add_data(dmap)]

    attr_dict = {'type': 'function',
                 'inputs': inputs,
                 'outputs': outputs,
                 'function': function,
                 'wait_inputs': True}

    if input_domain:
        attr_dict['input_domain'] = input_domain

    n = Counter()

    if function_id is None:
        # noinspection PyUnresolvedReferences
        function_name = '%s:%s' % (function.__module__, function.__name__)
    else:
        function_name = function_id
    fun_id = function_name

    while dmap.graph.has_node(fun_id):
        fun_id = '%s<%d>' % (function_name, n())

    if weight is not None:
        attr_dict['weight'] = weight

    attr_dict.update(kwargs)

    dmap.graph.add_node(fun_id, attr_dict=attr_dict)

    if edge_weight is not None:
        def add_edge(*e):
            if e not in edge_weight:
                dmap.graph.add_edge(*e)
            else:
                dmap.graph.add_edge(*e, attr_dict={'weight': edge_weight[e]})
    else:
        def add_edge(*e):
            dmap.graph.add_edge(*e)

    for u in inputs:
        if not dmap.graph.has_node(u):
            add_data(dmap, data_id=u)
        add_edge(u, function_id)

    for v in outputs:
        if not dmap.graph.has_node(v):
            add_data(dmap, data_id=v)
        add_edge(function_id, v)

    return function_id


def load_dmap_from_lists(dmap, data_list=None, fun_list=None):
    """
    Example::

        >>> dmap = DispatcherMap()
        >>> data_list = [
        ...     {'data_id': '/a'},
        ...     {'data_id': '/b'},
        ...     {'data_id': '/c'},
        ... ]

        >>> def fun(a, b):
        ...     return a + b

        >>> fun_list = [
        ...     {'function': fun, 'inputs': ['/a', '/b'], 'outputs': ['/c']},
        ...     {'function': fun, 'inputs': ['/c', '/d'], 'outputs': ['/a']},
        ... ]
        >>> dmap = load_dmap_from_lists(dmap, data_list, fun_list)
    """
    if data_list:
        for v in data_list:
            add_data(dmap, **v)

    if fun_list:
        for v in fun_list:
            add_function(dmap, **v)
    return dmap


def set_default_value(dmap, data_id=None, value=None, **kwargs):
    """
    Example::

        >>> dmap = DispatcherMap()
        >>> add_data(dmap, data_id='/a')
        '/a'
        >>> set_default_value(dmap, data_id='/a', value='value of the data')

        >>> set_default_value(dmap, data_id='/b', value='value of the data')

        >>> add_function(dmap, function=max, inputs=['/a', '/b'], \
                              outputs=['/c'])
        'builtins: <0>'
        >>> set_default_value(dmap, data_id='builtins:max<0>', \
                                    value='value of the data')
        Traceback (most recent call last):
            ...
        ValueError: ('Input error:', 'builtins:max<0> is not a data node')
    """

    if not data_id in dmap.graph.node:
        add_data(dmap, data_id=data_id, default_value=value, **kwargs)
    else:
        if dmap.graph.node[data_id]['type'] == 'data':
            dmap.default_values[data_id] = value
        else:
            raise ValueError('Input error:',
                             '%s is not a data node' % data_id)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    from itertools import tee

    a, b = tee(iterable)

    next(b, None)

    return zip(a, b)


def rename_function(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


class Counter(object):
    def __init__(self, value=-1):
        self.value = value

    def __call__(self):
        self.value += 1
        return self.value


# modified from NetworkX library
def scc_fun(graph, nbunch=None):
    """
    Return nodes in strongly connected components (SCC) of the reachable graph.

    Recursive version of algorithm.

    :param graph: An directed graph.
    :type graph: NetworkX DiGraph

    :param nbunch: A container of nodes which will be iterated through once.
    :type nbunch: list, iterable

    :return comp: A list of nodes for each SCC of the reachable graph.
    :rtype: list

    Notes
    -----
    Uses Tarjan's algorithm with Nuutila's modifications.

    References
    ----------
    .. [1] Depth-first search and linear graph algorithms, R. Tarjan
       SIAM Journal of Computing 1(2):146-160, (1972).

    .. [2] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soinen
       Information Processing Letters 49(1): 9-14, (1994)..

    .. [3] http://networkx.lanl.gov/_modules/networkx/algorithms/components/
       strongly_connected.html#strongly_connected_components
    """

    p_ord, l_link, scc_found, scc_queue = ({}, {}, {}, [])
    pre_ord_n = Counter()  # Pre-order counter
    for source in (nbunch if nbunch else graph):
        if source not in scc_found:
            q = [source]  # queue
            while q:
                v = q[-1]

                if v not in p_ord:
                    p_ord[v] = pre_ord_n()

                v_nbrs = graph[v]

                if next((q.append(w) for w in v_nbrs if w not in p_ord), True):
                    l_link[v] = [l_link[w] if p_ord[w] > p_ord[v] else p_ord[w]
                                 for w in v_nbrs
                                 if w not in scc_found]
                    l_link[v].append(p_ord[v])
                    l_link[v] = min(l_link[v])
                    q.pop()
                    if l_link[v] == p_ord[v]:
                        scc_found[v], scc = (True, [v])
                        while scc_queue and p_ord[scc_queue[-1]] > p_ord[v]:
                            scc_found[scc_queue[-1]] = True
                            scc.append(scc_queue.pop())
                        yield scc
                    else:
                        scc_queue.append(v)


# modified from NetworkX library
def dijkstra(graph, source, targets=None, cutoff=None):
    """
    Compute shortest paths and lengths in a weighted graph.

    Uses Dijkstra's algorithm for shortest paths.

    :param graph: An directed graph.
    :type graph: NetworkX DiGraph

    :param source: Starting node for path.
    :type source: node label

    :param targets: Ending nodes for paths.
    :type targets: iterable node labels, optional

    :param cutoff:
        Depth to stop the search. Only paths of length <= cutoff are returned.
    :type cutoff: integer or float, optional

    :return  distance,path:
        Returns a tuple of two dictionaries keyed by node.
        The first dictionary stores distance from the source.
        The second stores the path from the source to that node.
    :rtype: dictionaries

    Example::

        >>> graph = nx.path_graph(5)
        >>> length, path = dijkstra(graph, 0)
        >>> print(length[4])
        4
        >>> print(length)
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        >>> path[4]
        [0, 1, 2, 3, 4]

    Notes
    ---------
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    Based on the NetworkX library at
    http://networkx.lanl.gov/reference/generated/networkx.algorithms.\
    shortest_paths.weighted.single_source_dijkstra.html#networkx.algorithms.\
    shortest_paths.weighted.single_source_dijkstra

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    """

    if targets is not None:

        if len(targets) == 1 and source == targets[0]:
            return {source: 0}, {source: [source]}

        targets_copy = targets.copy()

        def check_targets(n):
            if n in targets_copy:
                targets_copy.remove(n)
                if not targets_copy:
                    return True

    else:
        check_targets = lambda n: False

    dist = {}  # dictionary of final distances
    paths = {source: [source]}  # dictionary of paths
    seen = {source: 0}
    c = Counter()
    fringe = [(0, c(), source)]  # use heapq with (distance,label) tuples
    while fringe:
        (d, _, v) = heappop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if check_targets(v):
            break

        for w, edgedata in graph[v].items():
            vw_dist = dist[v] + edgedata.get('weight', 1)
            if cutoff is not None:
                if vw_dist > cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist
                heappush(fringe, (vw_dist, c(), w))
                paths[w] = paths[v] + [w]
    return dist, paths


def set_data_node_output(graph, graph_output, node_id, node_attr, empty_fun,
                         data_output):
    # args is a list of all estimations available for the given node
    kwargs = {k: v['value'] for k, v in graph_output.pred[node_id].items()}

    # final estimation of the node and node status
    value, status = evaluate_node_fun(node_id, node_attr, (kwargs, ), empty_fun)

    if not status:  # is missing estimation function of data node
        return False

    if 'callback' in node_attr:  # invoke callback function of data node
        node_attr['callback'](value)

    # set data output
    data_output[node_id] = value

    # set graph_output
    for u in graph.successors_iter(node_id):
        graph_output.add_edge(node_id, u, attr_dict={'value': value})

    # return True, i.e. that the output have been evaluated correctly
    return True


def set_function_node_output(graph, graph_output, node_id, node_attr, empty_fun,
                             visited_nodes):
    # list of nodes that can still be estimated by the function node
    output_nodes = [u for u in node_attr['outputs']
                    if (not u in visited_nodes) and (u in graph.node)]

    if not output_nodes:  # this function is not needed
        graph_output.remove_node(node_id)
        return False

    args = graph_output.pred[node_id]  # list of the function's arguments
    args = [args[k]['value'] for k in node_attr['inputs']]

    # list of function results and node status
    results, status = evaluate_node_fun(node_id, node_attr, args, empty_fun)

    if not status:
        # is missing function of the node or args are not respecting the domain
        return False

    # set graph_output
    for u, value in zip(output_nodes, results):
        graph_output.add_edge(node_id, u, attr_dict={'value': value})

    # return True, i.e. that the output have been evaluated correctly
    return True


def evaluate_node_fun(node_id, node_attr, args, empty_fun):
    n_output = len(node_attr['outputs']) if 'outputs' in node_attr else 2

    if not node_attr['wait_inputs']:
        # it is a data node that has just one estimation value
        return list(args[0].values())[0], True
    elif empty_fun:
        return [None] * n_output, True
    elif not 'function' in node_attr:
        Warning('Missing node attribute:',
                'estimation function of data node %s' % node_id)
        return None, False
    elif 'input_domain' in node_attr and not node_attr['input_domain'](*args):
        # args are not respecting the domain
        return None, False
    else:  # use the estimation function of data node
        results = node_attr['function'](*args)
        return results if n_output > 1 else [results], True


def set_node_output(graph, graph_output, data_output, node_id,
                    node_attr, visited_nodes, empty_fun):
    """Return the status of the node.

    Recursive version of algorithm.

    Parameters
    ----------
    graph : NetworkX DiGraph
        An Dispatcher graph.

    graph_output : NetworkX DiGraph
        An directed graph that has nodes' inputs and outputs in edges
        attributes.

    data_output : dict
        A container of data nodes outputs.

    Returns
    -------
    status : boolean
        A flag that says if the node output have been set.

    Sets
    -------
    graph_output : NetworkX Graph
        A flag that says if the node output have been set.
    """
    node_type = node_attr['type']

    if node_type == 'data':
        return set_data_node_output(graph, graph_output, node_id, node_attr,
                                    empty_fun, data_output)
    elif node_type == 'function':
        return set_function_node_output(graph, graph_output, node_id, node_attr,
                                        empty_fun, visited_nodes)


def set_starting_node(graph, graph_output, start_id, data_sources):
    if graph.has_node(start_id):
        graph.remove_node(start_id)

    if graph_output.has_node(start_id):
        graph_output.remove_node(start_id)

    graph.add_node(start_id, attr_dict={'type': 'start'})
    graph_output.add_node(start_id, attr_dict={'type': 'start'})

    for k, v in data_sources.items():
        graph.add_edge(start_id, k)
        graph_output.add_edge(start_id, k, attr_dict={'value': v})


def populate_output(dmap, data_sources, targets, cutoff, empty_fun,
                    weight=False):
    graph_output = nx.DiGraph()

    data_output = {}

    trg_except_counter = {}

    if targets:
        targets_copy = targets.copy()

        def check_targets(n):

            if n in targets_copy:
                counter = trg_except_counter.get(n, 0)
                if counter > 0:
                    targets_copy.remove(n)
                    if not targets_copy:
                        return True
                else:
                    if counter == 1:
                        trg_except_counter.pop(n)
                    else:
                        trg_except_counter[n] = counter - 1

    else:
        check_targets = lambda n: False

    if cutoff is not None:
        check_cutoff = lambda d: d > cutoff
    else:
        check_cutoff = lambda d: False

    def check_wait_input_flag(wait_in, node, visited_nodes):
        return wait_in and not set(dmap.graph.pred[node]).issubset(visited_nodes)

    if weight:
        def edge_weight(edge_data, node_out):
            return edge_data.get('weight', 1) + node_out.get('weight', 0)
    else:
        def edge_weight(*args):
            return 1

    set_starting_node(dmap.graph, graph_output, dmap.start, data_sources)

    dist = {}  # dict of final distances
    c = Counter()

    # use heapq with (distance,wait,counter,label)
    fringe = [(0, False, c(), dmap.start)]

    seen = {dmap.start: 0}  # dict of seen distances

    while fringe:
        (d, _, _, v) = heappop(fringe)

        if v in dist:
            continue  # already searched this node.

        dist[v] = d

        if check_targets(v):
            break

        for w, edge_data in dmap.graph[v].items():
            node = dmap.graph.node[w]

            vw_dist = dist[v] + edge_weight(edge_data, node)

            wait_in = node['wait_inputs']

            if check_cutoff(vw_dist) or check_wait_input_flag(wait_in, w, dist):
                if v not in node.get('wait_exceptions', []):
                    continue
                elif targets:
                    trg_except_counter[w] = trg_except_counter.get(w, 0) + 1

            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist

                if set_node_output(dmap.graph, graph_output, data_output, w,
                                   node, seen, empty_fun):
                    heappush(fringe, (vw_dist, wait_in, c(), w))

    return graph_output, data_output


def get_dmap_without_cycles(dispatcher, data_sources):
    """
    Example::

        >>> dmap = DispatcherMap()
        >>> add_data(dmap, data_id='/a')
        '/a'
        >>> add_data(dmap, data_id='/b')
        '/b'
        >>> add_function(dmap, function=max, inputs=['/a', '/b'], outputs=['/c'])
        'builtins:max<0>'
        >>> add_function(dmap, function=max, inputs=['/c', '/b'], outputs=['/a'])
        'builtins:max<1>'

        >>> dmap_woc = get_dmap_without_cycles(dmap, {'/a': None, '/b': None})
        >>> edges = dmap_woc.graph.edges()
        >>> edges.sort()
        >>> edges
        [('/a', 'builtins:max<0>'), ('/b', 'builtins:max<0>'),
         ('builtins:max<0>', '/c'), ('start', '/a'), ('start', '/b')]

        >>> dmap_woc = get_dmap_without_cycles(dmap, {'/c': None, '/b': None})
        >>> edges = dmap_woc.graph.edges()
        >>> edges.sort()
        >>> edges
        [('/b', 'builtins:max<1>'), ('/c', 'builtins:max<1>'),
         ('builtins:max<1>',  '/a'), ('start', '/b'), ('start', '/c')]
    """
    from heapq import heappush

    active_nodes = set()
    edge_deleted = []

    def order_nodes_by_relevance(graph, scc):
        function_nodes, data_nodes = ([], [])

        no_cycles_to_be_deleted = True

        for u, n in ((u, graph.node[u]) for u in scc):

            node_type = n['type']

            if node_type == 'function':
                heappush(function_nodes, (graph.out_degree(u), u))
            elif node_type == 'data':
                if not n['wait_inputs']:
                    no_cycles_to_be_deleted = False
                heappush(data_nodes, (graph.in_degree(u), u))

        return no_cycles_to_be_deleted, function_nodes, data_nodes

    def min_cycles_ord_by_relevance(data_nodes, fun_nodes, graph):
        min_d = []
        c = Counter()
        for in_d, i in data_nodes:
            f_n = [j for _, j in fun_nodes if i in graph[j]]
            dist, path = dijkstra(graph, i, f_n, None)
            for j in (j for j in f_n if j in dist):
                d = dist[j]
                p = path[j] + [i]
                heappush(min_d, (d, len(p), 1 / in_d, c(), list(pairwise(p))))
        return [p[-1] for p in min_d]

    def delete_cycles(graph, nbunch):
        for scc in scc_fun(graph, nbunch):
            active_nodes.update(scc)

            if len(scc) < 2:  # single node
                continue  # not a cycle

            no_cycles, function_n, data_n = order_nodes_by_relevance(graph, scc)

            if no_cycles:  # no cycles to be deleted
                continue  # cycles are deleted by populate_output algorithm

            sub_g = graph.subgraph(scc)

            data_n.reverse()

            cycles = min_cycles_ord_by_relevance(data_n, function_n, sub_g)

            e_del = []

            for e in (c[-1] for c in cycles if set(c).isdisjoint(e_del)):
                e_del.append(e)
                sub_g.remove_edge(*e)
                edge_deleted.append(e)

            delete_cycles(sub_g, list(i[1] for i in data_n))

    dmap = deepcopy(dispatcher)

    set_starting_node(dmap.graph, nx.DiGraph(), dmap.start, data_sources)

    delete_cycles(dmap.graph, [dmap.start])

    dmap = sub_dispatcher(dmap, active_nodes)

    for edge in edge_deleted:
        dmap.graph.remove_edge(*edge)
        if not dmap.graph.out_degree(edge[0]):
            dmap.graph.remove_node(edge[0])

    return dmap


def sub_dispatcher(dispatcher, nodes, edges=None):
    dmap = deepcopy(dispatcher)

    dmap.graph = dmap.graph.subgraph(nodes)

    dmap.default_values = {k: dmap.default_values[k]
                          for k in nodes
                          if k in dmap.default_values}

    if edges is not None:
        for e in list(dmap.graph.edges_iter()):
            if not e in edges:
                dmap.graph.remove_edge(*e)

    return dmap


def data_function_from_dmap(dispatcher, dmap_inputs, dmap_outputs, fun_name=None,
                           rm_cycles=False):
    """
    Example::

        >>> dmap = DispatcherMap()
        >>> add_function(dmap, function=max, inputs=['/a', '/b'], outputs=['/c'])
        'builtins:max<0>'
        >>> add_function(dmap, function=max, inputs=['/c', '/b'], outputs=['/a'])
        'builtins:max<1>'
        >>> res = data_function_from_dmap(dmap, ['/c', '/b'], ['/a'], \
                                         fun_name='myF')
        >>> res['inputs'] == ['/c', '/b']
        True
        >>> res['outputs'] == ['/a']
        True
        >>> res['function'].__name__
        'myF'
        >>> res['function'](1, 2)
        [2]
    """

    dmap_new = resolve_route(dispatcher, dmap_inputs, dmap_outputs, rm_cycles)

    @rename_function(fun_name)
    def dsp_fun(*args):
        cazz=0
        o = run_output(dmap_new, input_values=dict(zip(dmap_inputs, args)),
                       output_targets=dmap_outputs)[0]
        return [o[k] for k in dmap_outputs]

    return {'function': dsp_fun, 'inputs': dmap_inputs, 'outputs': dmap_outputs}


def resolve_route(dmap, input_values=None, output_targets=None,
                  rm_cycles=False):
    """
    Example::

        >>> dmap = DispatcherMap()
        >>> add_function(dmap, function=max, inputs=['/a', '/b'], outputs=['/c'])
        'builtins:max<0>'
        >>> add_function(dmap, function=max, inputs=['/b', '/d'], outputs=['/e'])
        'builtins:max<1>'
        >>> add_function(dmap, function=max, inputs=['/d', '/e'], \
                             outputs=['/c','/f'])
        'builtins:max<2>'
        >>> add_function(dmap, function=max, inputs=['/d', '/f'], outputs=['/g'])
        'builtins:max<3>'
        >>> dmap_route = resolve_route(dmap, input_values=['/a', '/b', '/d'], \
                                          output_targets=['/c', '/e', '/f'])
        >>> nodes = dmap_route.graph.nodes()
        >>> nodes.sort()
        >>> nodes
        ['/a', '/b', '/c', '/d', '/e', '/f', 'builtins:max<0>', \
    'builtins:max<1>', 'builtins:max<2>', 'start']
        >>> edges = dmap_route.graph.edges()
        >>> edges.sort()
        >>> edges
        [('/a', 'builtins:max<0>'), ('/b', 'builtins:max<0>'), \
    ('/b', 'builtins:max<1>'), ('/d', 'builtins:max<1>'), \
    ('/d', 'builtins:max<2>'), ('/e', 'builtins:max<2>'), \
    ('builtins:max<0>', '/c'), ('builtins:max<1>', '/e'), \
    ('builtins:max<2>', '/f'), ('start', '/a'), ('start', '/b'), ('start', '/d')]
    """

    data_values = dict.fromkeys(dmap.default_values, None)

    if input_values is not None:
        data_values.update({k: None for k in input_values})

    if rm_cycles:
        dmap_copy = get_dmap_without_cycles(dmap, data_values)
    else:
        dmap_copy = deepcopy(dmap)

    graph_output = \
        populate_output(dmap_copy, data_values, output_targets, None, True)[0]

    nodes = nx.topological_sort(graph_output.reverse(), output_targets,
                                True)

    edges = list(graph_output.edges_iter())

    return sub_dispatcher(dmap_copy, nodes, edges)


def run_output(dmap, input_values=None, output_targets=None, cutoff=None,
               rm_cycles=False):
    """
    Example::

        >>> dmap = DispatcherMap()
        >>> from math import log
        >>> add_data(dmap, data_id='/a', default_value=0)
        '/a'
        >>> add_data(dmap, data_id='/b', default_value=1)
        '/b'

        >>> def my_log(a, b):
        ...     return log(b - a)

        >>> def my_domain(a, b):
        ...     return a < b

        >>> add_function(dmap, function=my_log, inputs=['/a', '/b'], \
                             outputs=['/c'], input_domain=my_domain)
        '...dispatcher:my_log<0>'
        >>> outputs, dmap_output = run_output(dmap, input_values={}, \
                                                 output_targets=['/c'])
        >>> outputs
        {'/c': 0.0}
        >>> nodes = dmap_output.graph.nodes()
        >>> nodes.sort()
        >>> nodes
        ['/a', '/b', '/c', '...dispatcher:my_log<0>', 'start']
        >>> edges = dmap_output.graph.edges()
        >>> edges.sort()
        >>> edges
        [('/a', '...dispatcher:my_log<0>'), ('/b', '...dispatcher:my_log<0>'), \
    ('...dispatcher:my_log<0>', '/c'), ('start', '/a'), ('start', '/b')]

        >>> outputs, dmap_output = run_output(dmap, input_values={'/b': 0}, \
                                                 output_targets=['/c'])
        >>> outputs
        {}
        >>> nodes = dmap_output.graph.nodes()
        >>> nodes.sort()
        >>> nodes
        ['/a', '/b', '...dispatcher:my_log<0>', 'start']
        >>> edges = dmap_output.graph.edges()
        >>> edges.sort()
        >>> edges
        [('/a', ...dispatcher:my_log<0>'), ('/b', '...dispatcher:my_log<0>'),
         ('start', '/a'), ('start', '/b')]
    """

    data_values = dmap.default_values.copy()

    if input_values is not None:
        data_values.update(input_values)

    if rm_cycles:
        dmap_copy = get_dmap_without_cycles(dmap, data_values)
    else:
        dmap_copy = dmap

    graph_output, data_outputs = \
        populate_output(dmap_copy, data_values, output_targets, cutoff, False)

    nodes = graph_output.nodes()

    edges = list(graph_output.edges_iter())

    dmap_output = sub_dispatcher(dmap_copy, nodes, edges)

    dmap_copy.graph.remove_node(dmap.start)

    if output_targets is not None:
        data_outputs = {k: data_outputs[k]
                        for k in output_targets
                        if k in data_outputs}

    return data_outputs, dmap_output


@nx.utils.open_file(1, mode='wb')
def save_dispatcher(dmap, path):
    """
    Example::

        >>> from tempfile import gettempdir
        >>> dmap = DispatcherMap()
        >>> tmp = '/'.join([gettempdir(),'test.dispatcher'])
        >>> save_dispatcher(dmap, tmp)
    """
    # noinspection PyArgumentList
    dump(dmap, path, HIGHEST_PROTOCOL)


@nx.utils.open_file(0, mode='rb')
def load_dispatcher(path):
    """
    Example::

        >>> from tempfile import gettempdir
        >>> dmap = DispatcherMap()
        >>> add_data(dmap)
        0
        >>> tmp = '/'.join([gettempdir(),'test.dispatcher'])
        >>> save_dispatcher(dmap, tmp)
        >>> dmap_loaded = load_dispatcher(tmp)
        >>> dmap.graph.node[0]['type']
        'data'
    """
    # noinspection PyArgumentList
    return load(path)


@nx.utils.open_file(1, mode='wb')
def save_default_values(dmap, path):
    """
    Example::

        >>> from tempfile import gettempdir
        >>> dmap = DispatcherMap()
        >>> tmp = '/'.join([gettempdir(),'test.dispatcher_default'])
        >>> save_default_values(dmap, tmp)
    """
    # noinspection PyArgumentList
    dump(dmap.default_values, path, HIGHEST_PROTOCOL)


@nx.utils.open_file(1, mode='rb')
def load_default_values(dmap, path):
    """
    Example::

        >>> from tempfile import gettempdir
        >>> tmp = '/'.join([gettempdir(),'test.dispatcher_default'])
        >>> dmap = DispatcherMap()
        >>> add_data(dmap, default_value=5)
        0
        >>> save_default_values(dmap, tmp)
        >>> dmap_loaded = DispatcherMap()
        >>> load_default_values(dmap_loaded, tmp)
        >>> dmap_loaded.default_values == dmap.default_values
        True
    """
    # noinspection PyArgumentList
    dmap.default_values = load(path)


def save_graph(dmap, path):
    """
    Example::

        >>> from tempfile import gettempdir
        >>> tmp = '/'.join([gettempdir(),'test.dispatcher_graph'])
        >>> dmap = DispatcherMap()
        >>> save_graph(dmap, tmp)
    """
    nx.write_gpickle(dmap.graph, path)


def load_graph(dmap, path):
    """
    Example::

        >>> from tempfile import gettempdir
        >>> tmp = '/'.join([gettempdir(),'test.dispatcher_graph'])
        >>> dmap = DispatcherMap()
        >>> fun_node = add_function(dmap, function=max, inputs=['/a'])
        >>> fun_node
        'builtins:max<0>'
        >>> save_graph(dmap, tmp)
        >>> dmap_loaded = DispatcherMap()
        >>> load_graph(dmap_loaded, tmp)
        >>> dmap_loaded.graph.degree(fun_node) == dmap.graph.degree(fun_node)
        True
    """
    dmap.graph = nx.read_gpickle(path)


def plot_dmap(dmap, *args, **kwargs):
    """
    Example::
        >>> dmap = DispatcherMap()
        >>> add_function(dmap, function=max, inputs=['/a', '/b'], \
                               outputs=['/c'])
        'builtins:max<0>'
        >>> plot_dmap(dmap)
    """
    plt.figure(*args, **kwargs)

    pos = nx.spring_layout(dmap.graph)

    start, data, function = ([], [], [])

    for k, v in dmap.graph.nodes_iter(True):
        eval(v['type']).append(k)

    label_nodes = {k: '%s' % k for k in dmap.graph.nodes_iter()}
    label_nodes.update({k: '%s:%s' % (str(k), str(v))
                        for k, v in dmap.default_values.items()})

    if dmap.start in dmap.graph.node:
        label_nodes[dmap.start] = 'start'

    nx.draw_networkx_nodes(dmap.graph, pos, node_shape='^', nodelist=start,
                           node_color='b')
    nx.draw_networkx_nodes(dmap.graph, pos, node_shape='o', nodelist=data,
                           node_color='r')
    nx.draw_networkx_nodes(dmap.graph, pos, node_shape='s',
                           nodelist=function, node_color='y')
    nx.draw_networkx_labels(dmap.graph, pos, labels=label_nodes)

    label_edges = {k: '' for k in dmap.graph.edges_iter()}
    label_edges.update({(u, v): '%s' % (str(a['value']))
                        for u, v, a in dmap.graph.edges_iter(data=True)
                        if 'value' in a})

    nx.draw_networkx_edges(dmap.graph, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(dmap.graph, pos, edge_labels=label_edges)

    plt.axis('off')
