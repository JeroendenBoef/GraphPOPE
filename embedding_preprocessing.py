import argparse
import os.path as osp
import os
import random
import numpy as np

import torch

from torch_geometric.datasets import Flickr
from torch_geometric.utils import degree

def sample_anchor_nodes(data, num_anchor_nodes, sampling_method):
    """
    Returns num_anchor_nodes amount of sampled anchor nodes based upon the sampling_method provided
    """
    if sampling_method == 'pagerank':
        G = to_networkx(data)
        pagerank = nx.pagerank_scipy(G)
        sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])}
        sampled_anchor_nodes_32 = list(sorted_pagerank.keys())[:32]
        sampled_anchor_nodes_64 = list(sorted_pagerank.keys())[:64]
        sampled_anchor_nodes_128 = list(sorted_pagerank.keys())[:128]
        sampled_anchor_nodes_256 = list(sorted_pagerank.keys())[:256]

    if sampling_method == 'betweenness_centrality':
        G = to_networkx(data)
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_betweenness_centrality = {k: v for k, v in sorted(betweenness_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes_32 = list(sorted_betweenness_centrality.keys())[:32]
        sampled_anchor_nodes_64 = list(sorted_betweenness_centrality.keys())[:64]
        sampled_anchor_nodes_128 = list(sorted_betweenness_centrality.keys())[:128]
        sampled_anchor_nodes_256 = list(sorted_betweenness_centrality.keys())[:256]

    if sampling_method == 'degree_centrality':
        G = to_networkx(data)
        degree_centrality = nx.degree_centrality(G)
        sorted_degree_centrality = {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes_32 = list(sorted_degree_centrality.keys())[:32]
        sampled_anchor_nodes_64 = list(sorted_degree_centrality.keys())[:64]
        sampled_anchor_nodes_128 = list(sorted_degree_centrality.keys())[:128]
        sampled_anchor_nodes_256 = list(sorted_degree_centrality.keys())[:256]

    if sampling_method == 'eigenvector_centrality':
        G = to_networkx(data)
        eigenvector_centrality = nx.eigenvector_centrality(G)
        sorted_eigenvector_centrality = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes_32 = list(sorted_eigenvector_centrality.keys())[:32]
        sampled_anchor_nodes_64 = list(sorted_eigenvector_centrality.keys())[:64]
        sampled_anchor_nodes_128 = list(sorted_eigenvector_centrality.keys())[:128]
        sampled_anchor_nodes_256 = list(sorted_eigenvector_centrality.keys())[:256]

    if sampling_method == 'closeness_centrality':
        G = to_networkx(data)
        closeness_centrality = nx.closeness_centrality(G)
        sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes_32 = list(sorted_closeness_centrality.keys())[:32]
        sampled_anchor_nodes_64 = list(sorted_closeness_centrality.keys())[:64]
        sampled_anchor_nodes_128 = list(sorted_closeness_centrality.keys())[:128]
        sampled_anchor_nodes_256 = list(sorted_closeness_centrality.keys())[:256]

    if sampling_method == 'clustering_coefficient':
        G = to_networkx(data)
        clustering_coefficient = nx.clustering(G)
        sorted_clustering_coefficient = {k: v for k, v in sorted(clustering_coefficient.items(), key=lambda item: item[1])}
        sampled_anchor_nodes_32 = list(sorted_clustering_coefficient.keys())[:32]
        sampled_anchor_nodes_64 = list(sorted_clustering_coefficient.keys())[:64]
        sampled_anchor_nodes_128 = list(sorted_clustering_coefficient.keys())[:128]
        sampled_anchor_nodes_256 = list(sorted_clustering_coefficient.keys())[:256]

    return sampled_anchor_nodes_32, sampled_anchor_nodes_64, sampled_anchor_nodes_128, sampled_anchor_nodes_256

def shortest_path_length(G, anchor_nodes, partition_length):
    """
    Calculate shortest path distance to every sampled anchor node and normalize by 1/distance. No path = 0
    """
    dists_dict = {}
    for node in partition_length:
        distances = []
        for anchor_node in anchor_nodes:
            try:
                distances.append(1/len(nx.shortest_path(G, source=node, target=anchor_node)))

            except nx.NetworkXNoPath:
                distances.append(0)
        dists_dict[node] = distances.copy()


        #dists_dict[node] = nx.single_source_shortest_path_length(G, node)
    return dists_dict

def merge_dicts(dicts):
    """
    Helper function for parallel shortest path calculation. Merges dicts from jobs into one
    """
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(G, anchor_nodes, num_workers=4):
    """
    Distribute shortest path calculation jobs to async workers, merge dicts and return results
    """
    try:
        nodes = list(G.nodes)
        pool = mp.Pool(processes=num_workers)
        jobs = [pool.apply_async(shortest_path_length,
                args=(G, anchor_nodes, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))])) for i in range(num_workers)]
        output = []
        for job in tqdm(jobs):
            output.append(job.get())
        dists_dict = merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict
    
    except KeyboardInterrupt:
        print('terminating workers...')
        pool.terminate()
        pool.join()
        print('workers terminated!')
        sys.exit(1)

def get_simple_distance_vector(data, anchor_nodes):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []
    G = to_networkx(data)

    dist_dict = all_pairs_shortest_path_length_parallel(G, anchor_nodes)
    distance_embedding = torch.as_tensor(list(dist_dict.values()))
    return distance_embedding


def process_and_save_static_embedding(data, num_anchor_nodes, sampling_method):
    print('sampling anchor nodes')
    data.anchor_nodes_32, data.anchor_nodes_64, data.anchor_nodes_128, data.anchor_nodes_256 = sample_anchor_nodes(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)
    print('deriving shortest paths to anchor nodes')
    for i in [data.anchor_nodes_32, data.anchor_nodes_64, data.anchor_nodes_128, data.anchor_nodes_256]:
        save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{i}.pt')
        embedding_matrix = get_simple_distance_vector(data=data, i)
        embedding_tensor = torch.as_tensor(embedding_matrix)
        torch.save(embedding_tensor, save_path)
        print(f'saved embedding as embedding_{sampling_method}_{num_anchor_nodes}')





path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

sampling_methods_list = ['pagerank', 'degree_centrality', 'eigenvector_centrality', 'closeness_centrality', 'clustering_coefficient', 'betweenness_centrality']
for sampling_method in sampling_methods_list:
    process_and_save_static_embedding(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)



# num_anchor_nodes = 256
# sampling_method = 'pagerank'
# for run in range(5, 20):

#     # ensure reproducibility
#     os.environ['PYTHONHASHSEED'] = str(run)
#     random.seed(run)
#     np.random.seed(run)
#     torch.manual_seed(run)
#     torch.cuda.manual_seed(run)
#     torch.cuda.manual_seed_all(run)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     print(f'torch seed: {run}')
    

#     process_and_save_embedding(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method, run=run)
