from networkx import DiGraph
import matplotlib.pyplot as plt

from functools import reduce
from collections import OrderedDict


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


def strongly_connected_components_function(graph, sources=None):
    pre_order, low_link, scc_found, scc_queue = ({}, {}, {}, [])
    pre_order_counter = Counter()  # Pre-order counter
    for source in (sources if sources else graph):
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in pre_order: pre_order[v] = pre_order_counter()
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


def set_node_input(node_values,from_node_id,to_node_id,to_node_type):
    if not to_node_id in node_values: node_values[to_node_id]={'inputs':{}}

    to_node = node_values[to_node_id]

    from_node = node_values[from_node_id]

    if not 'inputs' in to_node:to_node['inputs'] = {}

    to_node['inputs'][from_node_id] = from_node['outputs'][to_node_id] if to_node_type == 'data' else from_node['outputs']


def set_node_output(node_values,node_id,node_type,node_attr):
    node=node_values[node_id]

    if node_type == 'data':
        args = list(node['inputs'].values())
        if node_attr['wait_inputs']:
            if 'function' in node_attr:
                node['outputs'] = node_attr['function'](*args)
            else:
                raise ValueError('Missing node attribute:', 'estimation function of data node %s'%(node_id))
        else:
            node['outputs'] = args[0]
    elif node_type == 'function':
        args = [node['inputs'][k] for k in node_attr['inputs']]
        if 'input_domain' in node and not node_attr['input_domain'](*args): return
        results = node_attr['function'](*args) if len(node_attr['outputs'])>1 else [node_attr['function'](*args)]
        node['outputs'] = {k: res for res, k in zip(results, node_attr['outputs'])}

def set_starting_node(graph,start_id, sources):

    if graph.has_node(start_id): graph.remove_node(start_id)

    graph.add_node(start_id, attr_dict={'type': 'start'})

    for v in sources: graph.add_edge(start_id, v)


class Dispatcher(object):
    def __init__(self, *args, **kwargs):
        self.graph = DiGraph(*args, **kwargs)
        self.counter = Counter()
        self.start = 'start'
        self.node_values = {}

    def add_data(self, data_id=None, value=None, wait_inputs=False,function=None, **kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_data(data_id='/a') # data to be calculated, i.e., internal data
            >>> dsp.add_data(data_id='/b',value='value of the data') # data with a initial value, i.e., initial data
            >>> average_fun= lambda x: sum(x)/len(x)
            >>> dsp.add_data(data_id='/c',wait_inputs=True,function=average_fun) # internal data that is calculated as the average of all function estimations
            >>> dsp.add_data(data_id='/d',value='value of the data',wait_inputs=True,function=average_fun) # initial data that is calculated as the average of all function estimations + initial data
        """

        attr_dict = {'type': 'data', 'wait_inputs': wait_inputs}
        if function is not None: attr_dict['function']=function
        attr_dict.update(**kwargs)
        if data_id is None:
            data_id = self.counter()
            while self.graph.has_node(data_id):
                data_id = self.counter()

        if value is not None: self.node_values[data_id]={'outputs':value}

        self.graph.add_node(data_id, attr_dict=attr_dict)

    def add_function(self, function=lambda x: None, outputs=None, inputs=None, input_domain=None, **kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> my_function = lambda a,b:('result for /c data','result for /d data')
            >>> dsp.add_function(function=my_function,inputs=['/a','/b'],outputs=['/c','/d'])

            >>> from math import log
            >>> my_log= lambda a,b:log(b-a)
            >>> my_domain = lambda a,b: a<b
            >>> dsp.add_function(function=my_log,inputs=['/a','/b'],outputs=['/e'], input_domain=my_domain)
        """

        if outputs is None: outputs = {}
        attr_dict = {'type': 'function',
                     'inputs': inputs,
                     'outputs': outputs,
                     'function': function,
                     'wait_inputs': True}

        if input_domain: attr_dict['input_domain'] = input_domain

        n = Counter()

        function_name='%s:%s' %(function.__module__,function.__name__)

        function_id = '%s<%d>' %(function_name,n())

        while self.graph.has_node(function_id):
            function_id = '%s<%d>' %(function_name,n())

        self.graph.add_node(function_id, attr_dict=attr_dict)

        for u in inputs:
            if not self.graph.has_node(u): self.add_data(data_id=u)
            self.graph.add_edge(u, function_id)

        for v in outputs:
            if not self.graph.has_node(v): self.add_data(data_id=v)
            self.graph.add_edge(function_id, v)

    def set_data_node_value(self,data_id=None,value=None,**kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_data(data_id='/a')
            >>> dsp.set_data_node_value(data_id='/a',value='value of the data')

            >>> dsp.set_data_node_value(data_id='/b',value='value of the data')
        """

        if not data_id in self.graph.node:
            self.add_data(data_id=data_id,value=value,**kwargs)
        else:
            if self.graph.node[data_id].get('type', None) == 'data':
                self.node_values[data_id]={'outputs':value}
            else:
                raise ValueError('Input error:','%s is not a data node'%(data_id))

    def get_data_sources(self):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.set_data_node_value(data_id='/a',value='value of the data')
            >>> dsp.add_data(data_id='/b')
            >>> dsp.get_data_sources()
            ['/a']
        """

        return [k for k, v in self.graph.node.items()
                if (v.get('type', None) == 'data')
                and k in self.node_values
                and self.node_values[k].get('outputs', None) is not None]

    def populate_output(self, sources=None, targets=None, cutoff=None,dist=None, paths=None):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> from math import log
            >>> dsp.add_data(data_id='/a',value=0)
            >>> dsp.add_data(data_id='/b',value=1)
            >>> my_log= lambda a,b:log(b-a)
            >>> my_domain = lambda a,b: a<b
            >>> dsp.add_function(function=my_log,inputs=['/a','/b'],outputs=['/c'], input_domain=my_domain)
            >>> dsp.populate_output()
        """

        from heapq import heappush, heappop

        set_starting_node(self.graph,self.start,sources if sources else self.get_data_sources())

        dist, paths = ({}, {self.start: [self.start]})  # dictionaries of final distances and of paths
        fringe, seen = ([], {self.start: 0})  # use heapq with (distance,label) tuples
        heappush(fringe, (0, False, self.start))
        start = True
        while fringe:
            (d, _, v) = heappop(fringe)
            if v in dist: continue  # already searched this node.
            dist[v] = d
            if targets is not None:
                if v in targets: targets.remove(v)
                if not targets: break

            for w, edge_data in self.graph[v].items():
                vw_dist = dist[v] + edge_data.get('weight', 1)

                node = self.graph.node[w]
                node_type, wait_inputs = (node.get('type', None), node.get('wait_inputs', False))

                if not start and (node_type in ('data', 'function')):
                    set_node_input(self.node_values,v,w,node_type)

                if (cutoff is not None and vw_dist > cutoff) or \
                   (wait_inputs and not set(self.graph.predecessors(w)).issubset(dist)):
                    continue

                if w in dist:
                    if vw_dist < dist[w]:
                        raise ValueError('Contradictory paths found:', 'negative weights?')
                elif w not in seen or vw_dist < seen[w]:
                    seen[w], paths[w] = (vw_dist, paths[v] + [w])
                    heappush(fringe, (vw_dist, wait_inputs, w))
                    if start: continue
                    set_node_output(self.node_values,w,node_type,node)

            start = False

    def get_dispatcher_without_cycles(self,sources=None):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_data(data_id='/a',value=0)
            >>> dsp.add_data(data_id='/b',value=1)
            >>> dsp.add_function(function=sum,inputs=['/a','/b'],outputs=['/c'])
            >>> dsp.add_function(function=sum,inputs=['/c','/b'],outputs=['/a'])
            >>> dsp_woc = dsp.get_dispatcher_without_cycles()
            >>> dsp_woc = dsp.get_dispatcher_without_cycles(sources=['/a','/b'])
        """

        from networkx import single_source_dijkstra
        from itertools import product
        from heapq import heappush

        def delete_cycles(graph, active_nodes=set(), edge_deleted=[], sources=['start']):
            for v in strongly_connected_components_function(graph, sources):
                active_nodes.update(v)

                if len(v) < 2: continue

                function_nodes, data_nodes, min_distance, e_del, no_delete = ([], [], [], [], True)

                for u, n, node_type in ((u, n, n.get('type', None)) for u, n in ((u, graph.node[u]) for u in v)):
                    if node_type == 'function':
                        heappush(function_nodes, (graph.out_degree(u), u))
                    elif node_type == 'data':
                        if not n.get('wait_inputs', False): no_delete = False
                        heappush(data_nodes, (graph.in_degree(u), u))

                if no_delete: continue

                sub_network = graph.subgraph(v)
                data_nodes.reverse()

                for i, j in ((i[1], j[1]) for i, j in product(function_nodes, data_nodes) if j[1] in sub_network[i[1]]):
                    distance, path = single_source_dijkstra(sub_network, j, i, None, None)
                    distance = distance.get(i, None)
                    if distance is not None:
                        path = path[i] + [j]
                        heappush(min_distance, (distance, len(path), list(pairwise(path))))

                for e in (e for e, p in map(lambda x: (x[2][-1], set(x[2])), min_distance) if p.isdisjoint(e_del)):
                    e_del.append(e)
                    sub_network.remove_edge(*e)
                    edge_deleted.append(e)

                delete_cycles(sub_network, active_nodes, edge_deleted, list(i[1] for i in data_nodes))

            return active_nodes, edge_deleted

        network = self.copy()

        set_starting_node(network.graph,network.start,sources if sources else network.get_data_sources())

        active_nodes, edge_deleted = delete_cycles(network.graph, sources=[self.start])

        network.graph = network.graph.subgraph(active_nodes)

        for e in edge_deleted:
            network.graph.remove_edge(*e)
            if not network.graph.out_degree(e[0]): network.graph.remove_node(e[0])

        return network

    def copy(self):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp_copy=dsp.copy()
        """
        from copy import deepcopy

        return deepcopy(self)

    def plot(self,*args,**kwargs):
        """
        Example:
            >>> dsp = Dispatcher()
            >>> dsp.add_function(function=sum,inputs=['/a','/b'],outputs=['/c','/d'])
            >>> dsp.plot()
        """
        plt.figure(*args,**kwargs)
        from networkx import spring_layout, draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels

        pos = spring_layout(self.graph)

        start, data, function = ([], [], [])

        for k, node_type in ((k, v.get('type', None)) for k, v in self.graph.node.items()):
            eval(node_type).append(k)

        labels = {k:'%s'%(k) for k in self.graph.node.keys()}

        labels.update({k: '%s:%s' % (str(k), str(v)) for k, v in self.node_values.items()})

        if self.start in self.graph.node: labels[self.start] = 'start'

        draw_networkx_nodes(self.graph, pos, node_shape='^', nodelist=start, node_color='b')

        draw_networkx_nodes(self.graph, pos, node_shape='o', nodelist=data, node_color='r')

        draw_networkx_nodes(self.graph, pos, node_shape='s', nodelist=function, node_color='y')

        draw_networkx_labels(self.graph, pos, labels=labels)

        draw_networkx_edges(self.graph, pos, alpha=0.5)

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

    def write_gpickle(self, path):
        from networkx import write_gpickle

        write_gpickle(self.graph, path)

    def import_graph_from_gpickle(self, path):
        from networkx import read_gpickle

        self.graph = read_gpickle(path)


def define_network():
    from random import random, randint, shuffle, choice

    network = Dispatcher()
    # define data list
    n_data = 7
    data_list = []
    for i in range(n_data):
        # print('data',network.add_data(data=None if i>2 else random()*6,wait_inputs=False if i!=4 else True))
        v = i < 2
        data_list.append({'data': i + 1 if v else None, 'wait_inputs': False if v else choice([True, False])})

    def fun(n):
        def mul_(*args):
            return [reduce(lambda x, y: x * y, args)] * n

        def sum_(*args):
            return [reduce(lambda x, y: x + y, args)] * n

        return sum_

    # define functions list
    n_function = 6
    functions_list = []
    '''
    for i in range(n_function):
        data = list(range(n_data));shuffle(data)

        outputs = {data.pop():{'weight':1} for v in range(2)}
        inputs = OrderedDict([(data.pop(),{'weight':1}) for v in range(choice([1,2]))])
        functions_list[i]={'function':fun(len(outputs)),'outputs':outputs,'inputs':inputs}

    '''
    for v in [(fun(1), [3], [1, 2]),
              (fun(1), [4], [2, 3]),
              (fun(2), [2, 4], [0, 1]),
              (fun(1), [5],[4]),
              (fun(2), [3, 0], [5]),
              (fun(2), [0, 1], [2])]:
        functions_list.append({'function': v[0], 'outputs': v[1], 'inputs': v[2]})

    network.import_graph_from_lists(data_list, functions_list)

    return network


if __name__ == '__main__':

    network = define_network()
    network_without_cycles = network.get_dispatcher_without_cycles()


    network.populate_output()
    network.plot()

    network_without_cycles.populate_output()
    network_without_cycles.plot()

    for (k, v) in network_without_cycles.node_values.items():
        print(k, v)

    plt.show()