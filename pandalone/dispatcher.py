from heapq import heappush, heappop
from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt

try:
    from cPickle import dump, load, HIGHEST_PROTOCOL
except ImportError:
    from pickle import dump, load, HIGHEST_PROTOCOL


class Dispatcher(object):
    """
    Example:
    >>> dsp = Dispatcher()

    >>> def diff_function(a, b):
    ...     return b - a

    >>> add_function(dsp, function=diff_function, inputs=['/a', '/b'], \
                         outputs=['/c'])
    '...dispatcher:diff_function<0>'
    >>> from math import log

    >>> def log_domain(x):
    ...     return x > 0

    >>> add_function(dsp, function=log, inputs=['/c'], outputs=['/d'], \
                         input_domain=log_domain)
    'math:log<0>'
    >>> add_data(dsp, data_id='/a', default_value=0)
    '/a'
    >>> add_data(dsp, data_id='/b', default_value=1)
    '/b'
    >>> def average_fun(kwargs):
    ...     x = kwargs.values()
    ...     return sum(x) / len(x)

    >>> def callback_fun(*x):
    ...     print('(log(1)+1)/2=%.1f'%x)

    >>> add_data(dsp, data_id='/d', default_value=1, wait_inputs=True, \
                     function=average_fun, callback=callback_fun)
    '/d'
    >>> outputs, dsp_out = run_output(dsp, {})
    (log(1)+1)/2=0.5
    >>> outputs == {'/a': 0, '/b': 1, '/d': 0.5, '/c': 1}
    True
    """

    def __init__(self, *args, **kwargs):
        self.graph = nx.DiGraph(*args, **kwargs)
        self.counter = Counter()
        self.start = 'start'
        self.default_values = {}


def add_data(dsp, data_id=None, default_value=None, wait_inputs=False,
             function=None, callback=None, wait_exceptions=None, **kwargs):
    """
    Example:
    >>> dsp = Dispatcher()

    # data to be calculated, i.e., internal data
    >>> add_data(dsp, data_id='/a')
    '/a'

    # data with a initial value, i.e., initial data
    >>> add_data(dsp, data_id='/b', default_value='value of the data')
    '/b'

    >>> def average_fun(*x):
    ...     return sum(x) / len(x)

    # internal data that is calculated as the average of all estimations
    >>> add_data(dsp, data_id='/c', wait_inputs=True, function=average_fun)
    '/c'

    # initial data that is calculated as the average of all estimations
    >>> add_data(dsp, data_id='/d', default_value='value of the data', \
                     wait_inputs=True, function=average_fun)
    '/d'

    # create an internal data and return the generated id
    >>> add_data(dsp, )
    0
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
        data_id = dsp.counter()
        while dsp.graph.has_node(data_id):
            data_id = dsp.counter()

    if default_value is not None:
        dsp.default_values[data_id] = default_value
    elif data_id in dsp.default_values:
        dsp.default_values.pop(data_id)

    dsp.graph.add_node(data_id, attr_dict=attr_dict)

    return data_id


def add_function(dsp, function=lambda x: None, outputs=None, inputs=None,
                 input_domain=None, weight=None, edge_weight=None,
                 **kwargs):
    """
    Example:
    >>> dsp = Dispatcher()

    >>> def my_function(a, b):
    ...     c = a + b
    ...     d = a - b
    ...     return c, d

    >>> add_function(dsp, function=my_function, inputs=['/a', '/b'], \
                         outputs=['/c', '/d'])
    '...dispatcher:my_function<0>'

    >>> from math import log
    >>> def my_log(a, b):
    ...     log(b - a)

    >>> def my_domain(a, b):
    ...     return a < b

    >>> add_function(dsp, function=my_log, inputs=['/a', '/b'], \
                         outputs=['/e'], input_domain=my_domain)
    '...dispatcher:my_log<0>'
    """

    if outputs is None:
        outputs = [add_data(dsp)]

    attr_dict = {'type': 'function',
                 'inputs': inputs,
                 'outputs': outputs,
                 'function': function,
                 'wait_inputs': True}

    if input_domain:
        attr_dict['input_domain'] = input_domain

    n = Counter()

    # noinspection PyUnresolvedReferences
    function_name = '%s:%s' % (function.__module__, function.__name__)

    function_id = '%s<%d>' % (function_name, n())

    while dsp.graph.has_node(function_id):
        function_id = '%s<%d>' % (function_name, n())

    if weight is not None:
        attr_dict['weight'] = weight

    attr_dict.update(kwargs)

    dsp.graph.add_node(function_id, attr_dict=attr_dict)

    if edge_weight is not None:
        def add_edge(*e):
            if e not in edge_weight:
                dsp.graph.add_edge(*e)
            else:
                dsp.graph.add_edge(*e, attr_dict={'weight': edge_weight[e]})
    else:
        def add_edge(*e):
            dsp.graph.add_edge(*e)

    for u in inputs:
        if not dsp.graph.has_node(u):
            add_data(dsp, data_id=u)
        add_edge(u, function_id)

    for v in outputs:
        if not dsp.graph.has_node(v):
            add_data(dsp, data_id=v)
        add_edge(function_id, v)

    return function_id


def set_default_value(dsp, data_id=None, value=None, **kwargs):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> add_data(dsp, data_id='/a')
    '/a'
    >>> set_default_value(dsp, data_id='/a', value='value of the data')

    >>> set_default_value(dsp, data_id='/b', value='value of the data')

    >>> add_function(dsp, function=sum, inputs=['/a', '/b'], \
                         outputs=['/c', '/d'])
    'builtins:sum<0>'
    >>> set_default_value(dsp, data_id='builtins:sum<0>', \
                                value='value of the data')
    Traceback (most recent call last):
        ...
    ValueError: ('Input error:', 'builtins:sum<0> is not a data node')
    """

    if not data_id in dsp.graph.node:
        add_data(dsp, data_id=data_id, default_value=value, **kwargs)
    else:
        if dsp.graph.node[data_id]['type'] == 'data':
            dsp.default_values[data_id] = value
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


# modified from networkx library
def scc_fun(graph, nbunch = None):
    """Return nodes in strongly connected components of the reachable graph.

    Recursive version of algorithm.

    Parameters
    ----------
    G : NetworkX Graph
       An directed graph.

    nbunch : list, iterable
            A container of nodes which will be iterated through once.

    Returns
    -------
    comp : list of lists
       A list of nodes for each component of G.

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


# modified from networkx library
def dijkstra(graph, source, targets=None, cutoff=None):
    """Compute shortest paths and lengths in a weighted graph G.

    Uses Dijkstra's algorithm for shortest paths.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    targets : node labels, optional
       Ending nodes for paths

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the path from the source to that node.


    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = dijkstra(G, 0)
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

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path()
    single_source_dijkstra_path_length()
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
    graph : NetworkX Graph
        An Dispatcher graph.

    graph_output : NetworkX Graph
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


def populate_output(dsp, data_sources, targets, cutoff, empty_fun, weight=False):
    graph_output = nx.DiGraph()

    data_output = {}

    trg_except_counter = {}

    if targets:
        targets_copy = targets.copy()

        def check_targets(n):

            if n in targets_copy:
                e = trg_except_counter.get(n, 0)
                if e > 0:
                    targets_copy.remove(n)
                    if not targets_copy:
                        return True
                else:
                    if e == 1:
                        trg_except_counter.pop(n)
                    else:
                        trg_except_counter[n] = e - 1

    else:
        check_targets = lambda n: False

    if cutoff is not None:
        check_cutoff = lambda d: d > cutoff
    else:
        check_cutoff = lambda d: False

    def check_wait_input_flag(wait_in, node, visited_nodes):
        return wait_in and not set(dsp.graph.pred[node]).issubset(visited_nodes)

    if weight:
        def edge_weight(edge_data, node_out):
            return edge_data.get('weight', 1) + node_out.get('weight', 0)
    else:
        def edge_weight(*args):
            return 1

    set_starting_node(dsp.graph, graph_output, dsp.start, data_sources)

    dist = {}  # dict of final distances
    c = Counter()

    # use heapq with (distance,wait,counter,label)
    fringe = [(0, False, c(), dsp.start)]

    seen = {dsp.start: 0}  # dict of seen distances

    while fringe:
        (d, _, _, v) = heappop(fringe)

        if v in dist:
            continue  # already searched this node.

        dist[v] = d

        if check_targets(v):
            break

        for w, edge_data in dsp.graph[v].items():
            node = dsp.graph.node[w]

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

                if set_node_output(dsp.graph, graph_output, data_output, w,
                                   node, seen, empty_fun):
                    heappush(fringe, (vw_dist, wait_in, c(), w))

    return graph_output, data_output


def get_dsp_without_cycles(dispatcher, data_sources):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> add_data(dsp, data_id='/a')
    '/a'
    >>> add_data(dsp, data_id='/b')
    '/b'
    >>> add_function(dsp, function=sum, inputs=['/a', '/b'], outputs=['/c'])
    'builtins:sum<0>'
    >>> add_function(dsp, function=sum, inputs=['/c', '/b'], outputs=['/a'])
    'builtins:sum<1>'

    >>> dsp_woc = get_dsp_without_cycles(dsp, {'/a': None, '/b': None})
    >>> edges = dsp_woc.graph.edges()
    >>> edges.sort()
    >>> edges
    [('/a', 'builtins:sum<0>'), ('/b', 'builtins:sum<0>'), ('builtins:sum<0>',\
 '/c'), ('start', '/a'), ('start', '/b')]

    >>> dsp_woc = get_dsp_without_cycles(dsp, {'/c': None, '/b': None})
    >>> edges = dsp_woc.graph.edges()
    >>> edges.sort()
    >>> edges
    [('/b', 'builtins:sum<1>'), ('/c', 'builtins:sum<1>'), ('builtins:sum<1>',\
 '/a'), ('start', '/b'), ('start', '/c')]
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

    dsp = deepcopy(dispatcher)

    set_starting_node(dsp.graph, nx.DiGraph(), dsp.start, data_sources)

    delete_cycles(dsp.graph, [dsp.start])

    dsp = sub_dispatcher(dsp, active_nodes)

    for edge in edge_deleted:
        dsp.graph.remove_edge(*edge)
        if not dsp.graph.out_degree(edge[0]):
            dsp.graph.remove_node(edge[0])

    return dsp


def sub_dispatcher(dispatcher, nodes, edges=None):
    dsp = deepcopy(dispatcher)

    dsp.graph = dsp.graph.subgraph(nodes)

    dsp.default_values = {k: dsp.default_values[k]
                          for k in nodes
                          if k in dsp.default_values}

    if edges is not None:
        for e in list(dsp.graph.edges_iter()):
            if not e in edges:
                dsp.graph.remove_edge(*e)

    return dsp


def data_function_from_dsp(dispatcher, dsp_inputs, dsp_outputs, fun_name=None,
                           rm_cycles=False):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> add_function(dsp, function=sum, inputs=['/a', '/b'], outputs=['/c'])
    'builtins:sum<0>'
    >>> add_function(dsp, function=sum, inputs=['/c', '/b'], outputs=['/a'])
    'builtins:sum<1>'
    >>> res = data_function_from_dsp(dsp, ['/c', '/b'], ['/a'], fun_name='myF')
    >>> res['inputs'] == ['/c', '/b']
    True
    >>> res['outputs'] == ['/a']
    True
    >>> res['function'].__name__
    'myF'
    """

    dsp = resolve_route(dispatcher, dsp_inputs, dsp_outputs, rm_cycles)

    @rename_function(fun_name)
    def dsp_fun(*args):
        o = run_output(dsp, input_values=dict(zip(dsp_inputs, args)),
                       output_targets=dsp_outputs)[0]
        return [o[k] for k in dsp_outputs]

    return {'function': dsp_fun, 'inputs': dsp_inputs, 'outputs': dsp_outputs}


def resolve_route(dsp, input_values=None, output_targets=None,
                  rm_cycles=False):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> add_function(dsp, function=sum, inputs=['/a', '/b'], outputs=['/c'])
    'builtins:sum<0>'
    >>> add_function(dsp, function=sum, inputs=['/b', '/d'], outputs=['/e'])
    'builtins:sum<1>'
    >>> add_function(dsp, function=sum, inputs=['/d', '/e'], \
                         outputs=['/c','/f'])
    'builtins:sum<2>'
    >>> add_function(dsp, function=sum, inputs=['/d', '/f'], outputs=['/g'])
    'builtins:sum<3>'
    >>> dsp_route = resolve_route(dsp, input_values=['/a', '/b', '/d'], \
                                      output_targets=['/c', '/e', '/f'])
    >>> nodes = dsp_route.graph.nodes()
    >>> nodes.sort()
    >>> nodes
    ['/a', '/b', '/c', '/d', '/e', '/f', 'builtins:sum<0>', \
'builtins:sum<1>', 'builtins:sum<2>', 'start']
    >>> edges = dsp_route.graph.edges()
    >>> edges.sort()
    >>> edges
    [('/a', 'builtins:sum<0>'), ('/b', 'builtins:sum<0>'), \
('/b', 'builtins:sum<1>'), ('/d', 'builtins:sum<1>'), \
('/d', 'builtins:sum<2>'), ('/e', 'builtins:sum<2>'), \
('builtins:sum<0>', '/c'), ('builtins:sum<1>', '/e'), \
('builtins:sum<2>', '/f'), ('start', '/a'), ('start', '/b'), ('start', '/d')]
    """

    data_values = dict.fromkeys(dsp.default_values, None)

    if input_values is not None:
        data_values.update({k: None for k in input_values})

    if rm_cycles:
        dsp_copy = get_dsp_without_cycles(dsp, data_values)
    else:
        dsp_copy = deepcopy(dsp)

    graph_output = \
        populate_output(dsp_copy, data_values, output_targets, None, True)[0]

    nodes = nx.topological_sort(graph_output.reverse(), output_targets,
                                True)

    edges = list(graph_output.edges_iter())

    return sub_dispatcher(dsp_copy, nodes, edges)


def run_output(dsp, input_values=None, output_targets=None, cutoff=None,
               rm_cycles=False):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> from math import log
    >>> add_data(dsp, data_id='/a', default_value=0)
    '/a'
    >>> add_data(dsp, data_id='/b', default_value=1)
    '/b'

    >>> def my_log(a, b):
    ...     return log(b - a)

    >>> def my_domain(a, b):
    ...     return a < b

    >>> add_function(dsp, function=my_log, inputs=['/a', '/b'], \
                         outputs=['/c'], input_domain=my_domain)
    '...dispatcher:my_log<0>'
    >>> outputs, dsp_output = run_output(dsp, input_values={}, \
                                             output_targets=['/c'])
    >>> outputs
    {'/c': 0.0}
    >>> nodes = dsp_output.graph.nodes()
    >>> nodes.sort()
    >>> nodes
    ['/a', '/b', '/c', '...dispatcher:my_log<0>', 'start']
    >>> edges = dsp_output.graph.edges()
    >>> edges.sort()
    >>> edges
    [('/a', '...dispatcher:my_log<0>'), ('/b', '...dispatcher:my_log<0>'), \
('...dispatcher:my_log<0>', '/c'), ('start', '/a'), ('start', '/b')]

    >>> outputs, dsp_output = run_output(dsp, input_values={'/b': 0}, \
                                             output_targets=['/c'])
    >>> outputs
    {}
    >>> nodes = dsp_output.graph.nodes()
    >>> nodes.sort()
    >>> nodes
    ['/a', '/b', '...dispatcher:my_log<0>', 'start']
    >>> edges = dsp_output.graph.edges()
    >>> edges.sort()
    >>> edges
    [('/a', '...dispatcher:my_log<0>'), ('/b', '...dispatcher:my_log<0>'), \
('start', '/a'), ('start', '/b')]
    """

    data_values = dsp.default_values.copy()

    if input_values is not None:
        data_values.update(input_values)

    if rm_cycles:
        dsp_copy = get_dsp_without_cycles(dsp, data_values)
    else:
        dsp_copy = dsp

    graph_output, data_outputs = \
        populate_output(dsp_copy, data_values, output_targets, cutoff, False)

    nodes = graph_output.nodes()

    edges = list(graph_output.edges_iter())

    dsp_output = sub_dispatcher(dsp_copy, nodes, edges)

    dsp_copy.graph.remove_node(dsp.start)

    if output_targets is not None:
        data_outputs = {k: data_outputs[k]
                        for k in output_targets
                        if k in data_outputs}

    return data_outputs, dsp_output


def load_dsp_from_lists(dsp, data_list=None, fun_list=None):
    """
    Example:

    >>> dsp = Dispatcher()
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
    >>> dsp = load_dsp_from_lists(dsp, data_list, fun_list)
    """
    if data_list:
        for v in data_list:
            add_data(dsp, **v)

    if fun_list:
        for v in fun_list:
            add_function(dsp, **v)
    return dsp


@nx.utils.open_file(1, mode='wb')
def save_dispatcher(dsp, path):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> save_dispatcher(dsp, "test.dispatcher")
    """
    # noinspection PyArgumentList
    dump(dsp, path, HIGHEST_PROTOCOL)


@nx.utils.open_file(0, mode='rb')
def load_dispatcher(path):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> add_data(dsp)
    0
    >>> save_dispatcher(dsp, "test.dispatcher")
    >>> dsp_loaded = load_dispatcher("test.dispatcher")
    >>> dsp.graph.node[0]['type']
    'data'
    """
    # noinspection PyArgumentList
    return load(path)


@nx.utils.open_file(1, mode='wb')
def save_default_values(dsp, path):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> save_dispatcher(dsp, "test.dispatcher")
    """
    # noinspection PyArgumentList
    dump(dsp.default_values, path, HIGHEST_PROTOCOL)


@nx.utils.open_file(1, mode='rb')
def load_default_values(dsp, path):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> save_default_values(dsp, "test.dispatcher_data")
    >>> add_data(dsp, default_value=5)
    0
    >>> dsp_loaded = Dispatcher()
    >>> load_default_values(dsp_loaded, "test.dispatcher_data")
    >>> dsp.default_values[0]
    5
    """
    # noinspection PyArgumentList
    dsp.default_values = load(path)


def save_graph(dsp, path):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> save_graph(dsp, "test.dispatcher_graph")
    """
    nx.write_gpickle(dsp.graph, path)


def load_graph(dsp, path):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> save_graph(dsp, "test.dispatcher_graph")
    >>> add_data(dsp)
    0
    >>> dsp_loaded = Dispatcher()
    >>> load_graph(dsp_loaded, "test.dispatcher_graph")
    >>> dsp.graph.node[0]['type']
    'data'
    """
    dsp.graph = nx.read_gpickle(path)


def plot_dsp(dsp, *args, **kwargs):
    """
    Example:
    >>> dsp = Dispatcher()
    >>> add_function(dsp, function=sum, inputs=['/a', '/b'], \
                         outputs=['/c', '/d'])
    'builtins:sum<0>'
    >>> plot_dsp(dsp)
    """
    plt.figure(*args, **kwargs)

    pos = nx.spring_layout(dsp.graph)

    start, data, function = ([], [], [])

    for k, v in dsp.graph.nodes_iter(True):
        eval(v['type']).append(k)

    label_nodes = {k: '%s' % k for k in dsp.graph.nodes_iter()}
    label_nodes.update({k: '%s:%s' % (str(k), str(v))
                        for k, v in dsp.default_values.items()})

    if dsp.start in dsp.graph.node:
        label_nodes[dsp.start] = 'start'

    nx.draw_networkx_nodes(dsp.graph, pos, node_shape='^', nodelist=start,
                           node_color='b')
    nx.draw_networkx_nodes(dsp.graph, pos, node_shape='o', nodelist=data,
                           node_color='r')
    nx.draw_networkx_nodes(dsp.graph, pos, node_shape='s',
                           nodelist=function, node_color='y')
    nx.draw_networkx_labels(dsp.graph, pos, labels=label_nodes)

    label_edges = {k: '' for k in dsp.graph.edges_iter()}
    label_edges.update({(u, v): '%s' % (str(a['value']))
                        for u, v, a in dsp.graph.edges_iter(data=True)
                        if 'value' in a})

    nx.draw_networkx_edges(dsp.graph, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(dsp.graph, pos, edge_labels=label_edges)

    plt.axis('off')
