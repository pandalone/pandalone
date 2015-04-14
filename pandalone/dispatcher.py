import networkx as nx
import matplotlib.pyplot as plt


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    from itertools import tee

    a, b = tee(iterable)

    next(b, None)

    return zip(a, b)


class Counter(object):
    def __init__(self, value=-1):
        self.value = value

    def __call__(self):
        self.value += 1
        return self.value


def strongly_connected_components_function(graph, nbunch=list):
    pre_order, low_link, scc_found, scc_queue = ({}, {}, {}, [])
    pre_order_counter = Counter()  # Pre-order counter
    for source in (nbunch if nbunch else graph):
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]

                if v not in pre_order:
                    pre_order[v] = pre_order_counter()

                v_nbrs = graph[v]

                if next((queue.append(w) for w in v_nbrs if w not in pre_order), True):
                    low_link[v] = [low_link[w] if pre_order[w] > pre_order[v] else pre_order[w]
                                   for w in v_nbrs
                                   if w not in scc_found]
                    low_link[v].append(pre_order[v])
                    low_link[v] = min(low_link[v])
                    queue.pop()
                    if low_link[v] == pre_order[v]:
                        scc_found[v], scc = (True, [v])
                        while scc_queue and pre_order[scc_queue[-1]] > pre_order[v]:
                            scc_found[scc_queue[-1]] = True
                            scc.append(scc_queue.pop())
                        yield scc
                    else:
                        scc_queue.append(v)


def set_node_output(graph, graph_output, data_output, node_id, node_type, node_attr, seen, boolean_functions=False):
    if node_type == 'data':
        args = [v['value'] for v in graph_output.pred[node_id].values()]
        if node_attr['wait_inputs']:
            if 'function' in node_attr:
                value = True if boolean_functions else node_attr['function'](*args)
            else:
                raise ValueError('Missing node attribute:', 'estimation function of data node %s' % node_id)
        else:
            value = args[0]

        if 'callback' in node_attr:
            node_attr['callback'](value)

        data_output[node_id] = value

        for u in graph.successors_iter(node_id):
            graph_output.add_edge(node_id, u, attr_dict={'value': value})

    elif node_type == 'function':
        args = graph_output.pred[node_id]

        args = [args[k]['value'] for k in node_attr['inputs']]

        n_output = len(node_attr['outputs'])

        if boolean_functions:
            results = [True] * n_output
        elif 'input_domain' in node_attr and not node_attr['input_domain'](*args):
            return False
        else:
            results = node_attr['function'](*args) if n_output > 1 else [node_attr['function'](*args)]

        for u, value in zip(node_attr['outputs'], results):
            if not u in seen:
                graph_output.add_edge(node_id, u, attr_dict={'value': value})

    return True


def set_starting_node(graph, graph_output, start_id, data_inputs, sources, boolean_functions):
    graph_output.clear()

    if graph.has_node(start_id):
        graph.remove_node(start_id)

    graph.add_node(start_id, attr_dict={'type': 'start'})

    for k, v in (data_inputs if not boolean_functions else {k: True for k in sources}).items():
        graph.add_edge(start_id, k)
        graph_output.add_edge(start_id, k, attr_dict={'value': v})

    return boolean_functions


def populate_output(dsp, sources=None, targets=None, cutoff=None, boolean_functions=False):
    from heapq import heappush, heappop

    graph_output = nx.DiGraph()

    if targets is not None:
        targets_copy = targets.copy()

    data_output = {}

    set_starting_node(dsp.graph, graph_output, dsp.start, dsp.data_inputs, sources, boolean_functions)

    dist, paths = ({}, {dsp.start: [dsp.start]})  # dictionaries of final distances and of paths

    fringe, seen = ([], {dsp.start: 0})  # use heapq with (distance,label) tuples

    heappush(fringe, (0, False, dsp.start))

    while fringe:
        (d, _, v) = heappop(fringe)

        if v in dist:
            continue  # already searched this node.

        dist[v] = d

        if targets is not None:
            if v in targets_copy:
                targets_copy.remove(v)
            if not targets_copy:
                break

        for w, edge_data in dsp.graph[v].items():
            vw_dist = dist[v] + edge_data.get('weight', 1)

            node = dsp.graph.node[w]

            node_type, wait_inputs = (node['type'], node.get('wait_inputs', False))

            if (cutoff is not None and vw_dist > cutoff) or \
                    (wait_inputs and not set(dsp.graph.predecessors(w)).issubset(dist)):
                continue

            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:', 'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w], paths[w] = (vw_dist, paths[v] + [w])

                if set_node_output(dsp.graph, graph_output, data_output, w, node_type, node, seen,boolean_functions):
                    heappush(fringe, (vw_dist, wait_inputs, w))

    return graph_output,data_output


def get_dispatcher_without_cycles(dispatcher, sources=None):
    """
    Example:
        >>> dsp = Dispatcher()
        >>> dsp.add_data(data_id='/a',value=0)
        '/a'
        >>> dsp.add_data(data_id='/b',value=1)
        '/b'
        >>> dsp.add_function(function=sum,inputs=['/a','/b'],outputs=['/c'])
        'builtins:sum<0>'
        >>> dsp.add_function(function=sum,inputs=['/c','/b'],outputs=['/a'])
        'builtins:sum<1>'
        >>> dsp_woc = get_dispatcher_without_cycles(dsp)
        >>> edges = dsp_woc.graph.edges(); edges.sort()
        >>> edges
        [('/a', 'builtins:sum<0>'), ('/b', 'builtins:sum<0>'), ('builtins:sum<0>', '/c'), ('start', '/a'), ('start', '/b')]
        >>> dsp_woc = get_dispatcher_without_cycles(dsp, sources=['/c','/b'])
        >>> edges = dsp_woc.graph.edges(); edges.sort()
        >>> edges
        [('/b', 'builtins:sum<1>'), ('/c', 'builtins:sum<1>'), ('builtins:sum<1>', '/a'), ('start', '/b'), ('start', '/c')]

    """
    from itertools import product
    from heapq import heappush

    def delete_cycles(graph, actv_nodes=set(), edge_del=list(), nbunch=list(['start'])):
        for v in strongly_connected_components_function(graph, nbunch):
            actv_nodes.update(v)

            if len(v) < 2:
                continue

            function_nodes, data_nodes, min_distance, e_del = ([], [], [], [])

            no_delete = True

            for u, n, node_type in ((u, n, n['type']) for u, n in ((u, graph.node[u]) for u in v)):
                if node_type == 'function':
                    heappush(function_nodes, (graph.out_degree(u), u))
                elif node_type == 'data':
                    if not n.get('wait_inputs', False):
                        no_delete = False
                    heappush(data_nodes, (graph.in_degree(u), u))

            if no_delete:
                continue

            sub_network = graph.subgraph(v)
            data_nodes.reverse()

            for i, (in_degree,j) in ((j[1], i) for i, j in product(data_nodes,function_nodes) if i[1] in sub_network[j[1]]):
                distance, path = nx.single_source_dijkstra(sub_network, j, i, None, None)
                distance = distance.get(i, None)
                if distance is not None:
                    path = path[i] + [j]
                    heappush(min_distance, (distance, len(path), 1/in_degree, list(pairwise(path))))

            for e in (e for e, p in map(lambda x: (x[3][-1], set(x[3])), min_distance) if p.isdisjoint(e_del)):
                e_del.append(e)
                sub_network.remove_edge(*e)
                edge_del.append(e)

            delete_cycles(sub_network, actv_nodes, edge_del, list(i[1] for i in data_nodes))

        return actv_nodes, edge_del

    dsp = dispatcher.copy()

    boolean_functions = sources is not None

    set_starting_node(dsp.graph, nx.DiGraph(), dsp.start, dsp.data_inputs, sources, boolean_functions)

    active_nodes, edge_deleted = delete_cycles(dsp.graph, nbunch=[dsp.start])

    dsp = sub_dispatcher(dsp, active_nodes)

    for edge in edge_deleted:
        dsp.graph.remove_edge(*edge)
        if not dsp.graph.out_degree(edge[0]):
            dsp.graph.remove_node(edge[0])

    return dsp


def sub_dispatcher(dispatcher, nodes, edges = None):
    dsp = dispatcher.copy()

    dsp.graph = dsp.graph.subgraph(nodes)

    dsp.data_inputs = {k:dsp.data_inputs[k] for k in nodes if k in dsp.data_inputs}

    if edges is not None:
        for e in list(dsp.graph.edges_iter()):
            if not e in edges:
                dsp.graph.remove_edge(*e)

    return dsp


class Dispatcher(object):
    """
    Example:
        >>> dsp = Dispatcher()

        >>> def diff_function(a, b):
        ...     return b - a

        >>> dsp.add_function(function=diff_function,inputs=['/a','/b'],outputs=['/c'])
        'dispatcher:diff_function<0>'
        >>> from math import log

        >>> def log_domain(x):
        ...     return x > 0

        >>> dsp.add_function(function=log,inputs=['/c'],outputs=['/d'], input_domain=log_domain)
        'math:log<0>'
        >>> dsp.add_data(data_id='/a',value=0)
        '/a'
        >>> dsp.add_data(data_id='/b',value=1)
        '/b'
        >>> def average_fun(*x):
        ...     return sum(x) / len(x)

        >>> def callback_fun(*x):
        ...     print('(log(1)+1)/2=%.1f'%x)

        >>> dsp.add_data(data_id='/d',value=1,wait_inputs=True,function=average_fun,callback=callback_fun)
        '/d'
        >>> dsp_out = dsp.run_output()
        (log(1)+1)/2=0.5
        >>> dsp.plot()
        >>> dsp_out.plot()
    """

    def __init__(self, *args, **kwargs):
        self.graph = nx.DiGraph(*args, **kwargs)
        self.counter = Counter()
        self.start = 'start'
        self.data_inputs = {}

    def add_data(self, data_id=None, value=None, wait_inputs=False, function=None, callback=None, **kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_data(data_id='/a') # data to be calculated, i.e., internal data
            '/a'
            >>> dsp.add_data(data_id='/b',value='value of the data') # data with a initial value, i.e., initial data
            '/b'
            >>> def average_fun(*x):
            ...     return sum(x) / len(x)

            # internal data that is calculated as the average of all function estimations
            >>> dsp.add_data(data_id='/c',wait_inputs=True,function=average_fun)
            '/c'

            # initial data that is calculated as the average of all function estimations + initial data
            >>> dsp.add_data(data_id='/d',value='value of the data',wait_inputs=True,function=average_fun)
            '/d'
            >>> dsp.add_data() # create an internal data, and return the generated id
            0
        """

        attr_dict = {'type': 'data', 'wait_inputs': wait_inputs}

        if function is not None:
            attr_dict['function'] = function

        if callback is not None:
            attr_dict['callback'] = callback

        attr_dict.update(kwargs)

        if data_id is None:
            data_id = self.counter()
            while self.graph.has_node(data_id):
                data_id = self.counter()

        if value is not None:
            self.data_inputs[data_id] = value

        self.graph.add_node(data_id, attr_dict=attr_dict)

        return data_id

    def add_function(self, function=lambda x: None, outputs=None, inputs=None, input_domain=None, **kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> def my_function(a, b):
            ...     return 'result for /c data','result for /d data'
            >>> dsp.add_function(function=my_function,inputs=['/a','/b'],outputs=['/c','/d'])
            'dispatcher:my_function<0>'
            >>> from math import log
            >>> def my_log(a, b):
            ...     log(b - a)
            >>> def my_domain(a, b):
            ...     return a < b
            >>> dsp.add_function(function=my_log,inputs=['/a','/b'],outputs=['/e'], input_domain=my_domain)
            'dispatcher:my_log<0>'
        """

        if outputs is None:
            outputs = [self.add_data()]

        attr_dict = {'type': 'function',
                     'inputs': inputs,
                     'outputs': outputs,
                     'function': function,
                     'wait_inputs': True}

        if input_domain:
            attr_dict['input_domain'] = input_domain

        n = Counter()

        function_name = '%s:%s' % (function.__module__, function.__name__)

        function_id = '%s<%d>' % (function_name, n())

        while self.graph.has_node(function_id):
            function_id = '%s<%d>' % (function_name, n())

        attr_dict.update(kwargs)

        self.graph.add_node(function_id, attr_dict=attr_dict)

        for u in inputs:
            if not self.graph.has_node(u):
                self.add_data(data_id=u)
            self.graph.add_edge(u, function_id)

        for v in outputs:
            if not self.graph.has_node(v):
                self.add_data(data_id=v)
            self.graph.add_edge(function_id, v)

        return function_id

    def set_data_node_value(self, data_id=None, value=None, **kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_data(data_id='/a')
            '/a'
            >>> dsp.set_data_node_value(data_id='/a',value='value of the data')

            >>> dsp.set_data_node_value(data_id='/b',value='value of the data')

            >>> dsp.add_function(function=sum,inputs=['/a','/b'],outputs=['/c','/d'])
            'builtins:sum<0>'
            >>> dsp.set_data_node_value(data_id='builtins:sum<0>',value='value of the data')
            Traceback (most recent call last):
                ...
            ValueError: ('Input error:', 'builtins:sum<0> is not a data node')
        """

        if not data_id in self.graph.node:
            self.add_data(data_id=data_id, value=value, **kwargs)
        else:
            if self.graph.node[data_id]['type'] == 'data':
                self.data_inputs[data_id] = value
            else:
                raise ValueError('Input error:', '%s is not a data node' % data_id)

    def resolve_route(self, sources=None, targets=None, cutoff=None, remove_cycles=False):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_function(function=sum,inputs=['/a','/b'],outputs=['/c'])
            'builtins:sum<0>'
            >>> dsp.add_function(function=sum,inputs=['/b','/d'],outputs=['/e'])
            'builtins:sum<1>'
            >>> dsp.add_function(function=sum,inputs=['/d','/e'],outputs=['/c','/f'])
            'builtins:sum<2>'
            >>> dsp.add_function(function=sum,inputs=['/d','/f'],outputs=['/g'])
            'builtins:sum<3>'
            >>> dsp_route = dsp.resolve_route(sources=['/a','/b','/d'], targets=['/c','/e','/f'])
            >>> nodes = dsp_route.graph.nodes(); nodes.sort()
            >>> nodes
            ['/a', '/b', '/c', '/d', '/e', '/f', 'builtins:sum<0>', 'builtins:sum<1>', 'builtins:sum<2>', 'start']
            >>> edges = dsp_route.graph.edges(); edges.sort()
            >>> edges
            [('/a', 'builtins:sum<0>'), ('/b', 'builtins:sum<0>'), ('/b', 'builtins:sum<1>'), ('/d', 'builtins:sum<1>'), ('/d', 'builtins:sum<2>'), ('/e', 'builtins:sum<2>'), ('builtins:sum<0>', '/c'), ('builtins:sum<1>', '/e'), ('builtins:sum<2>', '/f'), ('start', '/a'), ('start', '/b'), ('start', '/d')]
        """

        dsp = get_dispatcher_without_cycles(self, sources) if remove_cycles else self.copy()

        graph_output = populate_output(dsp, sources, targets, cutoff, True)[0]

        nodes = nx.topological_sort(graph_output.reverse(),nbunch=targets,reverse=True)

        edges = list(graph_output.edges_iter())

        return sub_dispatcher(dsp, nodes, edges)

    def run_output(self, targets=None, cutoff=None, remove_cycles=False):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> from math import log
            >>> dsp.add_data(data_id='/a',value=0)
            '/a'
            >>> dsp.add_data(data_id='/b',value=1)
            '/b'
            >>> def my_log(a, b):
            ...     return log(b-a)

            >>> def my_domain(a, b):
            ...     return a < b

            >>> dsp.add_function(function=my_log,inputs=['/a','/b'],outputs=['/c'], input_domain=my_domain)
            'dispatcher:my_log<0>'
            >>> dsp_output = dsp.run_output(targets=['/c'])
            >>> nodes = dsp_output.graph.nodes(); nodes.sort()
            >>> nodes
            ['/a', '/b', '/c', 'dispatcher:my_log<0>', 'start']
            >>> edges = dsp_output.graph.edges(); edges.sort()
            >>> edges
            [('/a', 'dispatcher:my_log<0>'), ('/b', 'dispatcher:my_log<0>'), ('dispatcher:my_log<0>', '/c'), ('start', '/a'), ('start', '/b')]
            >>> dsp.set_data_node_value('/b',0)
            >>> dsp_output = dsp.run_output(targets=['/c'])
            >>> nodes = dsp_output.graph.nodes(); nodes.sort()
            >>> nodes
            ['/a', '/b', 'dispatcher:my_log<0>', 'start']
            >>> edges = dsp_output.graph.edges(); edges.sort()
            >>> edges
            [('/a', 'dispatcher:my_log<0>'), ('/b', 'dispatcher:my_log<0>'), ('start', '/a'), ('start', '/b')]
        """
        dsp = get_dispatcher_without_cycles(self, sources=self.data_inputs) if remove_cycles else self.copy()

        graph_output,dsp.data_inputs = populate_output(dsp, self.data_inputs, targets, cutoff, False)

        nodes = graph_output.nodes()

        edges = list(graph_output.edges_iter())

        return sub_dispatcher(dsp, nodes, edges)

    def copy(self):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp_copy=dsp.copy()
        """
        from copy import deepcopy

        return deepcopy(self)

    def plot(self, *args, **kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_function(function=sum,inputs=['/a','/b'],outputs=['/c','/d'])
            'builtins:sum<0>'
            >>> dsp.plot()
        """
        plt.figure(*args, **kwargs)

        pos = nx.spring_layout(self.graph)

        start, data, function = ([], [], [])

        for k, node_type in ((k, v['type']) for k, v in self.graph.nodes_iter(True)):
            eval(node_type).append(k)

        label_nodes = {k: '%s' % k for k in self.graph.nodes_iter()}

        label_nodes.update({k: '%s:%s' % (str(k), str(v)) for k, v in self.data_inputs.items()})

        if self.start in self.graph.node:
            label_nodes[self.start] = 'start'

        label_edges = {k: '' for k in self.graph.edges_iter()}

        label_edges.update({(u,v): '%s' %(str(a['value'])) for u, v, a in self.graph.edges_iter(data=True) if 'value' in a})

        nx.draw_networkx_nodes(self.graph, pos, node_shape='^', nodelist=start, node_color='b')

        nx.draw_networkx_nodes(self.graph, pos, node_shape='o', nodelist=data, node_color='r')

        nx.draw_networkx_nodes(self.graph, pos, node_shape='s', nodelist=function, node_color='y')

        nx.draw_networkx_labels(self.graph, pos, labels=label_nodes)

        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)

        nx.draw_networkx_edge_labels(self.graph, pos,edge_labels =label_edges)

        plt.axis('off')

    def import_graph_from_lists(self, data_list, fun_list):
        """
        Example:
            >>> def fun(a, b):
            ...     return a + b
            >>> dsp = Dispatcher()
            >>> data_list = [
            ...     {'data_id': '/a'},
            ...     {'data_id': '/b'},
            ...     {'data_id': '/c'},
            ... ]
            >>> fun_list = [
            ...     {
            ...         'function': fun,
            ...         'inputs': ['/a', '/b'],
            ...         'outputs': ['/c'],
            ...     },
            ... ]
            >>> dsp.import_graph_from_lists(data_list, fun_list)
        """

        for v in data_list: self.add_data(**v)
        for v in fun_list: self.add_function(**v)

    @nx.utils.open_file(1,mode='wb')
    def save_dispatcher(self,path):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.save_dispatcher("test.dispatcher")
        """
        from networkx import write_gpickle
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        pickle.dump(self, path, pickle.HIGHEST_PROTOCOL)

    @nx.utils.open_file(1,mode='rb')
    def load_dispatcher(self,path):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.save_dispatcher("test.dispatcher")
            >>> dsp_loaded = Dispatcher()
            >>> dsp_loaded.load_dispatcher("test.dispatcher")
        """
        from networkx import write_gpickle
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        pickle.load(path)

    def write_graph_pickle(self, path):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.write_graph_pickle("test.dispatcher_graph")
        """
        from networkx import write_gpickle

        write_gpickle(self.graph, path)

    def load_graph_from_pickled_graph(self, path):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.write_graph_pickle("test.dispatcher_graph")
            >>> dsp_loaded = Dispatcher()
            >>> dsp_loaded.load_graph_from_pickled_graph("test.dispatcher_graph")
        """
        from networkx import read_gpickle

        self.graph = read_gpickle(path)
