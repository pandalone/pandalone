from networkx import DiGraph
import matplotlib.pyplot as plt

from functools import reduce
from collections import OrderedDict
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee
    a, b = tee(iterable); next(b, None)
    return zip(a, b)

class Counter(object):
    def __init__(self, value=-1):
        self.value = value

    def __call__(self, *args, **kwargs):
        self.value = self.value + 1
        return self.value

def strongly_connected_components_function(G,sources=None):
    preorder,lowlink,scc_found,scc_queue=({},{},{},[])
    preorder_counter=Counter()     # Preorder counter
    for source in (sources if sources else G):
        if source not in scc_found:
            queue=[source]
            while queue:
                v=queue[-1]
                if v not in preorder: preorder[v]=preorder_counter()
                v_nbrs=G[v]

                if next((queue.append(w) for w in v_nbrs if w not in preorder),True):
                    lowlink[v]=[lowlink[w] if preorder[w]>preorder[v] else preorder[w] for w in v_nbrs if w not in scc_found]
                    lowlink[v].append(preorder[v]);lowlink[v]=min(lowlink[v])
                    queue.pop()
                    if lowlink[v]==preorder[v]:
                        scc_found[v],scc=(True, [v])
                        while scc_queue and preorder[scc_queue[-1]]>preorder[v]:
                            k=scc_queue.pop()
                            scc_found[k]=True
                            scc.append(k)
                        yield scc
                    else:
                        scc_queue.append(v)

class Dispatcher(object):
    def __init__(self,*args,**kwargs):
        self.graph = DiGraph(*args,**kwargs)
        self.counter = Counter()
        self.start = 'start'

    def add_data(self,id=None,data=None,wait_inputs=False,**kwargs):
        attr_dict={'type':'data','outputs':data,'wait_inputs':wait_inputs};attr_dict.update(**kwargs)
        if id is None:
            id = self.counter()
            while self.graph.has_node(id):id = self.counter()
        self.graph.add_node(id,attr_dict=attr_dict)
        return id

    def add_function(self,function=lambda x:None,outputs={},inputs=OrderedDict(),input_domain=None,**kwargs):
        attr_dict = {'type':'function',
                     'inputs':OrderedDict([(k,None) for k in inputs.keys()]),
                     'outputs':{k:None for k in outputs.keys()},
                     'function':function,
                     'wait_inputs':True}

        if input_domain: attr_dict['input_domain'] = input_domain

        id = self.counter()
        while self.graph.has_node(id):id = self.counter()
        self.graph.add_node(id,attr_dict=attr_dict)

        for u,attr in inputs.items():
            if not self.graph.has_node(u): self.graph.add_data(id=u)
            self.graph.add_edge(u,id,attr_dict=attr)

        for v,attr in outputs.items():
            if not self.graph.has_node(v): self.graph.add_data(id=v)
            self.graph.add_edge(id,v,attr_dict=attr)

    def get_data_sources(self):
        return [k for k,v in self.graph.node.items() if (v.get('type',None)=='data') and (v.get('outputs',None) is not None)]

    def set_starting_node(self,sources=None):
        if self.graph.has_node(self.start): self.graph.remove_node(self.start)

        self.graph.add_node(self.start,attr_dict={'type':'start'})

        for v in (sources if sources else self.get_data_sources()):self.graph.add_edge(self.start,v,{'weight':1})

    def populate_output(self,targets=None,cutoff=None):
        from heapq import heappush,heappop
        from numpy import mean
        self.set_starting_node()
        dist, paths = ({},{self.start:[self.start]})  # dictionaries of final distances and of paths
        fringe,seen = ([],{self.start:0}) # use heapq with (distance,label) tuples
        heappush(fringe,(0,False,self.start))
        start = True
        while fringe:
            (d,_,v)=heappop(fringe)
            if v in dist: continue # already searched this node.
            dist[v] = d
            if targets is not None:
                if v in targets: targets.remove(v)
                if not targets: break

            for w,edgedata in self.graph[v].items():
                vw_dist = dist[v] + edgedata.get('weight',1)

                node = self.graph.node[w]; node_type, wait_inputs = (node.get('type',None), node.get('wait_inputs',False))

                if not start and (node_type in ('data','function')):
                    node['inputs'] = node.get('inputs',{}); output=self.graph.node[v]['outputs']
                    node['inputs'][v] = output[w] if node_type == 'data' else output

                if (cutoff is not None and vw_dist>cutoff) or \
                   (wait_inputs and not set(self.graph.predecessors(w)).issubset(dist)): continue

                if w in dist:
                    if vw_dist < dist[w]:
                        raise ValueError('Contradictory paths found:', 'negative weights?')
                elif w not in seen or vw_dist < seen[w]:
                    seen[w],paths[w] = (vw_dist,paths[v]+[w])
                    heappush(fringe,(vw_dist,wait_inputs,w))
                    if start: continue
                    args=list(node['inputs'].values())
                    if node_type == 'data':
                        node['outputs'] = (node['function'](*args) if 'function' in node else mean(args)) if wait_inputs else args[0]
                    elif node_type == 'function':
                        if 'input_domain' in node and not node['input_domain'](*args): continue
                        node['outputs'] = {k:res for res,(k,v) in zip(node['function'](*args),node['outputs'].items())}

            start = False

        return (dist,paths)

    def remove_cycles(self):
        from networkx import single_source_dijkstra
        from itertools import product
        from heapq import heappush

        def delete_cycles(network,active_nodes=set(),edge_deleted=[],sources=['start']):
            for v in strongly_connected_components_function(network,sources):
                active_nodes.update(v)

                if len(v)<2: continue

                function_nodes,data_nodes,min_distance,e_del,no_delete = ([],[],[],[],True)

                for u,n,type in ((u,n,n.get('type',None)) for u,n in ((u,network.node[u]) for u in v)):
                    if type == 'function':
                        heappush(function_nodes,(network.out_degree(u),u))
                    elif type == 'data':
                        if not n.get('wait_inputs',False):no_delete = False
                        heappush(data_nodes,(network.in_degree(u),u))

                if no_delete: continue

                sub_network = network.subgraph(v); data_nodes.reverse()

                for i,j in ((i[1],j[1]) for i,j in product(function_nodes,data_nodes) if j[1] in sub_network[i[1]]):
                    distance,path = single_source_dijkstra(sub_network,j,i,None,None); distance = distance.get(i,None)
                    if distance is not None:
                        path = path[i]+[j]; heappush(min_distance,(distance,len(path),list(pairwise(path))))

                for e in (e for e,p in map(lambda x:(x[2][-1],set(x[2])),min_distance) if p.isdisjoint(e_del)):
                    e_del.append(e); sub_network.remove_edge(*e); edge_deleted.append(e)

                delete_cycles(sub_network,active_nodes,edge_deleted,list(i[1] for i in data_nodes))

            return active_nodes,edge_deleted

        network = self.copy()
        network.set_starting_node()
        active_nodes,edge_deleted=delete_cycles(network.graph)

        network.graph=network.graph.subgraph(active_nodes)

        for e in edge_deleted:
            network.graph.remove_edge(*e)
            if not network.graph.out_degree(e[0]): network.graph.remove_node(e[0])

        return network

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def plot(self):
        from networkx import spring_layout,draw_networkx_nodes,draw_networkx_edges,draw_networkx_labels,get_node_attributes
        pos=spring_layout(self.graph)
        start,data,function=([],[],[])
        for k,type in ((k,v.get('type',None)) for k,v in self.graph.node.items()):
            eval(type).append(k)
        labels = get_node_attributes(self.graph,'outputs')
        labels = {k:'%s:%s'%(str(k),str(v)) for k,v in labels.items()}
        labels[self.start] = 'start'
        draw_networkx_nodes(self.graph,pos,node_shape='^',nodelist = start,node_color='b')
        draw_networkx_nodes(self.graph,pos,node_shape='o',nodelist = data,node_color='r')
        draw_networkx_nodes(self.graph,pos,node_shape='s',nodelist=function,node_color='y')
        draw_networkx_labels(self.graph,pos,labels=labels)
        draw_networkx_edges(self.graph,pos,alpha=0.5)
        plt.axis('off')

    def import_graph_from_lists(self,data_list,fun_list):
        for v in data_list: self.add_data(**v)
        for v in fun_list: self.add_function(**v)

    def write_gpickle(self,path):
        from networkx import write_gpickle
        write_gpickle(self.graph,path)

    def import_graph_from_gpickle(self,path):
        from networkx import read_gpickle
        self.graph=read_gpickle(path)

def define_network():
    from random import random,randint,shuffle,choice
    network = Dispatcher()
    # define data list
    n_data = 7
    data_list = []
    for i in range(n_data):
        #print('data',network.add_data(data=None if i>2 else random()*6,wait_inputs=False if i!=4 else True))
        v = i<2
        data_list.append({'data':i+1 if v else None, 'wait_inputs':False if v else choice([True,False])})


    def fun(n):
        def mul_(*args):
            return [reduce(lambda x, y: x*y, args)]*n

        def sum_(*args):
            return [reduce(lambda x, y: x+y, args)]*n

        return sum_

    # define functions list
    n_function = 6
    functions_list=[]
    '''
    for i in range(n_function):
        data = list(range(n_data));shuffle(data)

        outputs = {data.pop():{'weight':1} for v in range(2)}
        inputs = OrderedDict([(data.pop(),{'weight':1}) for v in range(choice([1,2]))])
        functions_list[i]={'function':fun(len(outputs)),'outputs':outputs,'inputs':inputs}

    '''
    for v in [(fun(1),{v:{'weight':1} for v in [3]}, OrderedDict([(v,{'weight':1}) for v in [1,2]])),
              (fun(1),{v:{'weight':1} for v in [4]}, OrderedDict([(v,{'weight':1}) for v in [2,3]])),
              (fun(2),{v:{'weight':1} for v in [2,4]}, OrderedDict([(v,{'weight':1}) for v in [0,1]])),
              (fun(1),{v:{'weight':1} for v in [5]}, OrderedDict([(v,{'weight':1}) for v in [4]])),
              (fun(2),{v:{'weight':1} for v in [3,0]}, OrderedDict([(v,{'weight':1}) for v in [5]])),
              (fun(2),{v:{'weight':1} for v in [0,1]}, OrderedDict([(v,{'weight':1}) for v in [2]]))]:

        functions_list.append({'function':v[0],'outputs':v[1],'inputs':v[2]})

    network.import_graph_from_lists(data_list,functions_list)

    return network

if __name__ == '__main__':

    network = define_network()
    network_without_cycles = network.remove_cycles()

    plt.figure()
    network.populate_output()
    network.plot()

    plt.figure()
    network_without_cycles.populate_output()
    network_without_cycles.plot()

    for (k,v) in network_without_cycles.graph.node.items():
        print(k,v)

    plt.show()