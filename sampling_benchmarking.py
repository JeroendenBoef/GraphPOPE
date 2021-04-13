import timeit

import_module = '''
import argparse
import os.path as osp
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Flickr
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
'''


# G = to_networkx(data)
# print(nx.pagerank_scipy(G))
print('testing scipy...')
scipy = '''
G = to_networkx(data)
pagerank = nx.pagerank_scipy(G)
'''
print('scipy version: ', timeit.timeit(stmt=scipy, setup=import_module))

print('testing numpy...')
numpy = '''
G = to_networkx(data)
pagerank = nx.pagerank_numpy(G)
'''
print('numpy version: ' , timeit.timeit(stmt=numpy, setup=import_module))

print('testing normal version...')
normal = '''
G = to_networkx(data)
pagerank = nx.pagerank(G)
'''
print('normal version: ', timeit.timeit(stmt=normal, setup=import_module))