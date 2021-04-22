import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import multiprocessing as mp
from tqdm import tqdm
import os.path as osp
import sys

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

def get_simple_distance_vector(data):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []
    G = to_networkx(data)

    dist_dict = all_pairs_shortest_path_length_parallel(G, data.anchor_nodes)
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

def process_and_save_embedding(data, num_anchor_nodes, sampling_method, run):
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{num_anchor_nodes}_run_{run}.pt')
    print('sampling anchor nodes')
    data.anchor_nodes = sample_anchor_nodes(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)
    print('deriving shortest paths to anchor nodes')
    embedding_matrix = get_simple_distance_vector(data=data)
    embedding_tensor = torch.as_tensor(embedding_matrix)
    torch.save(embedding_tensor, save_path)
    print(f'saved embedding as embedding_{sampling_method}_{num_anchor_nodes}_run_{run}')

def load_preprocessed_embedding(data, num_anchor_nodes, sampling_method, run):
    load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{num_anchor_nodes}_run_{run}.pt')
    print('loading preprocessed embedding tensor...')
    embedding_tensor = torch.load(load_path)
    extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
    data.x = extended_features
    print('attached preprocessed embedding')
