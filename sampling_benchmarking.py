# import timeit

# import_module = '''
# import argparse
# import os.path as osp
# import networkx as nx
# import torch
# from torch_geometric.utils import to_networkx
# from torch_geometric.datasets import Flickr
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# dataset = Flickr(path)
# data = dataset[0]
# '''


# G = to_networkx(data)
# # print(nx.pagerank_scipy(G))
# print('testing scipy...')
# scipy = '''
# G = to_networkx(data)
# pagerank = nx.pagerank_scipy(G)
# '''
# print('scipy version: ', timeit.timeit(stmt=scipy, setup=import_module, number=20

# print('testing normal version...')
# normal = '''
# G = to_networkx(data)
# pagerank = nx.pagerank(G)
# '''
# print('normal version: ', timeit.timeit(stmt=normal, setup=import_module, number=20))


import argparse
import os.path as osp
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Flickr
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]

G = to_networkx(data)
pagerank = nx.pagerank_scipy(G)

sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])}
sampled = list(sorted_pagerank.keys())[:32]