import timeit

import_module_as = '''
import os.path as osp
import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree

from utils import attach_distance_embedding

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
'''


print('testing async...')
async_anchors = '''
attach_distance_embedding(data, num_anchor_nodes=64, use_cache=False, sampling_method='stochastic')
'''
print('async: ', timeit.timeit(stmt=async_anchors, setup=import_module_as, number=3))

import_module_normal = '''
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import multiprocessing as mp
from tqdm import tqdm
import os.path as osp
import os
import random
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree

def sample_anchor_nodes(data, num_anchor_nodes, sampling_method):
    """
    Returns num_anchor_nodes amount of randomly sampled anchor nodes 
    """
    if sampling_method == 'stochastic':
        node_indices = np.arange(data.num_nodes)
        sampled_anchor_nodes = np.random.choice(node_indices, num_anchor_nodes)

    if sampling_method == 'pagerank':
        G = to_networkx(data)
        pagerank = nx.pagerank_scipy(G)
        sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_pagerank.keys())[:32]

    return sampled_anchor_nodes

def shortest_path_length(G, anchor_nodes):
    """
    Calculate shortest path distance to every sampled anchor node and normalize by 1/distance. No path = 0
    """
    dists_dict = {}
    for node in G.nodes:
        distances = []
        for anchor_node in anchor_nodes:
            try:
                distances.append(1/len(nx.shortest_path(G, source=node, target=anchor_node)))

            except nx.NetworkXNoPath:
                distances.append(0)
        dists_dict[node] = distances.copy()


        #dists_dict[node] = nx.single_source_shortest_path_length(G, node)
    return dists_dict

def get_simple_distance_vector(data):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []
    G = to_networkx(data)

    dist_dict = shortest_path_length(G, data.anchor_nodes)
    distance_embedding = torch.as_tensor(list(dist_dict.values()))
    return distance_embedding


def concat_into_features(embedding_matrix, data, caching):
    """
    Merge features and embedding matrix, returns combined feature matrix
    """
    embedding_tensor = torch.as_tensor(embedding_matrix)
    if caching == True:
        torch.save(embedding_tensor, 'cached_node_embeddings.pt')
    combined = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
    return combined

def attach_distance_embedding(data, num_anchor_nodes, sampling_method, caching=False, use_cache=False):
    if use_cache == False:
        print('sampling anchor nodes')
        data.anchor_nodes = sample_anchor_nodes(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)
        print('deriving shortest paths to anchor nodes')
        embedding_matrix = get_simple_distance_vector(data=data)
        extended_features = concat_into_features(embedding_matrix=embedding_matrix, data=data, caching=True)
        data.x = extended_features
        print('feature matrix is blessed by the POPE')
    else:
        print('loading cached distance embeddings')
        embedding_tensor = torch.load('cached_node_embeddings.pt')
        extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
        data.x = extended_features
        print('feature matrix is blessed by the POPE')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
'''

print('testing normal...')
normal = '''
attach_distance_embedding(data, num_anchor_nodes=64, use_cache=False, sampling_method='stochastic')
'''
print('normal: ', timeit.timeit(stmt=normal, setup=import_module_normal, number=3))

# print('testing networkx...')
# networkx_version = '''
# G = to_networkx(data)
# dists_dict = {}
# for node in range(20):
#     distances = []
#     for anchor_node in anchor_nodes:
#         try:
#             distances.append(1/len(nx.shortest_path(G, source=node, target=anchor_node)))

#         except nx.NetworkXNoPath:
#             distances.append(0)
#     dists_dict[node] = distances.copy()

# '''
# print('networkx version: ', timeit.timeit(stmt=networkx_version, setup=import_module, number=20))
# #print('networkx version: 324.49460455030203')

# print('testing cugraph...')
# cugraph_version = '''
# G = to_networkx(data)
# dists_dict = {}
# for node in range(20):
#     distances = []
#     for anchor_node in anchor_nodes:
#         try:
#             distances.append(1/shortest_path_length(G, source=node, target=anchor_node))

#         except nx.NetworkXNoPath:
#             distances.append(0)
#     dists_dict[node] = distances.copy()

# '''
# print('cugraph version: ', timeit.timeit(stmt=cugraph_version, setup=import_module, number=20))

print('testing cugraph...')
cugraph_version = '''
G = to_networkx(data)
centrality = betweenness_centrality(G, k=int(data.x.size(0)/2))
'''
print('unapproximated betweenness centrality speed: 1651.2824539281428')
print('approximated betweenness centrality speed: ', timeit.timeit(stmt=cugraph_version, setup=import_module, number=3))


print('testing cugraph pagerank...')
cugraph_pagerank='''
G = to_networkx(data)
rank = pagerank(G)
'''
print('cugraph pagerank speed:', timeit.timeit(stmt=cugraph_pagerank, setup=import_module, number=3))

print('testing networkx pagerank...')
nx_pagerank='''
G = to_networkx(data)
nx_pr = nx.pagerank_scipy(G)
'''
print('networkx pagerank speed:', timeit.timeit(stmt=nx_pagerank, setup=import_module, number=3))


# print('testing normal version...')
# normal = '''
# G = to_networkx(data)
# pagerank = nx.pagerank(G)
# '''
# print('normal version: ', timeit.timeit(stmt=normal, setup=import_module, number=20))


# import argparse
# import os.path as osp
# import networkx as nx
# import torch
# from torch_geometric.utils import to_networkx
# from torch_geometric.datasets import Flickr
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# dataset = Flickr(path)
# data = dataset[0]

# G = to_networkx(data)
# pagerank = nx.pagerank_scipy(G)

# sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])}
# sampled = list(sorted_pagerank.keys())[:32]


# import numpy as np
# import networkx as nx
# import cugraph as cg
# import torch
# from torch_geometric.utils import to_networkx
# import multiprocessing as mp
# from tqdm import tqdm

# def sample_anchor_nodes(data, num_anchor_nodes, sampling_method):
#     """
#     Returns num_anchor_nodes amount of randomly sampled anchor nodes 
#     """
#     if sampling_method == 'stochastic':
#         node_indices = np.arange(data.num_nodes)
#         sampled_anchor_nodes = np.random.choice(node_indices, num_anchor_nodes)

#     if sampling_method == 'pagerank':
#         G = to_networkx(data)
#         pagerank = nx.pagerank_scipy(G)
#         sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])}
#         sampled_anchor_nodes = list(sorted_pagerank.keys())[:32]

#     return sampled_anchor_nodes

# def shortest_path_length(G, anchor_nodes, partition_length):
#     """
#     Calculate shortest path distance to every sampled anchor node and normalize by 1/distance. No path = 0
#     """
#     dists_dict = {}
#     for node in partition_length:
#         distances = []
#         for anchor_node in anchor_nodes:
#             try:
#                 distances.append(1/len(nx.shortest_path(G, source=node, target=anchor_node)))

#             except nx.NetworkXNoPath:
#                 distances.append(0)
#         dists_dict[node] = distances.copy()


#         #dists_dict[node] = nx.single_source_shortest_path_length(G, node)
#     return dists_dict

# def merge_dicts(dicts):
#     """
#     Helper function for parallel shortest path calculation. Merges dicts from jobs into one
#     """
#     result = {}
#     for dictionary in dicts:
#         result.update(dictionary)
#     return result

# def all_pairs_shortest_path_length_parallel(G, anchor_nodes, num_workers=4):
#     """
#     Distribute shortest path calculation jobs to async workers, merge dicts and return results
#     """
#     nodes = list(G.nodes)
#     pool = mp.Pool(processes=num_workers)
#     jobs = [pool.apply_async(shortest_path_length,
#             args=(G, anchor_nodes, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))])) for i in range(num_workers)]
#     output = []
#     for job in tqdm(jobs):
#         output.append(job.get())
#     dists_dict = merge_dicts(output)
#     pool.close()
#     pool.join()
#     return dists_dict

# def get_simple_distance_vector(data):
#     """
#     Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
#     """
#     distance_embedding = []
#     G = to_networkx(data)

#     dist_dict = all_pairs_shortest_path_length_parallel(G, data.anchor_nodes)
#     distance_embedding = torch.as_tensor(list(dist_dict.values()))
#     return distance_embedding

