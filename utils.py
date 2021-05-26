import numpy as np
import networkx as nx
import graph_tool.all as gt
import torch
from torch.nn import CosineSimilarity
from torch_geometric.utils import to_networkx
import multiprocessing as mp
from tqdm import tqdm
import os.path as osp
import sys

def sample_anchor_nodes(data, num_anchor_nodes, sampling_method):
    """
    Returns num_anchor_nodes amount of sampled anchor nodes based upon the sampling_method provided
    """
    if sampling_method == 'stochastic':
        node_indices = np.arange(data.num_nodes)
        sampled_anchor_nodes = np.random.choice(node_indices, num_anchor_nodes)

    if sampling_method == 'pagerank':
        G = to_networkx(data)
        pagerank = nx.pagerank_scipy(G)
        sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_pagerank.keys())[:num_anchor_nodes]

    if sampling_method == 'betweenness_centrality':
        G = to_networkx(data)
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_betweenness_centrality = {k: v for k, v in sorted(betweenness_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_betweenness_centrality.keys())[:num_anchor_nodes]

    if sampling_method == 'degree_centrality':
        G = to_networkx(data)
        degree_centrality = nx.degree_centrality(G)
        sorted_degree_centrality = {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_degree_centrality.keys())[:num_anchor_nodes]

    if sampling_method == 'eigenvector_centrality':
        G = to_networkx(data)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        sorted_eigenvector_centrality = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_eigenvector_centrality.keys())[:num_anchor_nodes]

    if sampling_method == 'closeness_centrality':
        G = to_networkx(data)
        closeness_centrality = nx.closeness_centrality(G)
        sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_closeness_centrality.keys())[:num_anchor_nodes]

    if sampling_method == 'clustering_coefficient':
        G = to_networkx(data)
        clustering_coefficient = nx.clustering(G)
        sorted_clustering_coefficient = {k: v for k, v in sorted(clustering_coefficient.items(), key=lambda item: item[1])}
        sampled_anchor_nodes = list(sorted_clustering_coefficient.keys())[:num_anchor_nodes]

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

def shortest_path_length_gt(G, anchor_nodes, partition_length):
    """
    Calculate shortest path distance to every sampled anchor node and normalize by 1/distance. No path = 0
    """
    dists_dict = {}
    for node in partition_length:
        distances = []
        for anchor_node in anchor_nodes:
            dist = gt.shortest_distance(G, source=G.vertex(node), target=G.vertex(anchor_node), max_dist=20)
            dist = 0 if distance > 0 else 1/dist
            distances.append(dist)
        dists_dict[node] = distances.copy()
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

def shortest_path_gt(G, anchor_nodes, num_workers=4):
    """
    Distribute shortest path calculation jobs to async workers, merge dicts and return results
    """
    try:
        nodes = list(G.nodes)
        pool = mp.Pool(processes=num_workers)
        jobs = [pool.apply_async(shortest_path_length_gt,
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

def get_simple_distance_vector(data, framework):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []
    G = to_networkx(data)

    if framework == 'nx':
        dist_dict = all_pairs_shortest_path_length_parallel(G, data.anchor_nodes)
    if framework == 'gt':
        dist_dict = shortest_path_gt(G, data.anchor_nodes)

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

def attach_deterministic_distance_embedding(data, num_anchor_nodes, sampling_method):
    loading_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{num_anchor_nodes}.pt')
    print('loading cached distance embeddings')
    embedding_tensor = torch.load(loading_path)
    extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
    print('feature matrix is blessed by the POPE')
    return extended_features

def process_and_save_embedding(data, num_anchor_nodes, sampling_method, run):
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{num_anchor_nodes}_run_{run}.pt')
    print('sampling anchor nodes')
    data.anchor_nodes = sample_anchor_nodes(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)
    print('deriving shortest paths to anchor nodes')
    embedding_matrix = get_simple_distance_vector(data=data)
    embedding_tensor = torch.as_tensor(embedding_matrix)
    torch.save(embedding_tensor, save_path)
    print(f'saved embedding as embedding_{sampling_method}_{num_anchor_nodes}_run_{run}')

def process_and_save_static_embedding(data, num_anchor_nodes, sampling_method):
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{num_anchor_nodes}.pt')
    print('sampling anchor nodes')
    data.anchor_nodes = sample_anchor_nodes(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)
    print('deriving shortest paths to anchor nodes')
    embedding_matrix = get_simple_distance_vector(data=data)
    embedding_tensor = torch.as_tensor(embedding_matrix)
    torch.save(embedding_tensor, save_path)
    print(f'saved embedding as embedding_{sampling_method}_{num_anchor_nodes}')

def load_preprocessed_embedding(data, num_anchor_nodes, sampling_method, run=4):
    load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'embedding_{sampling_method}_{num_anchor_nodes}_run_{run}.pt')
    print('loading preprocessed embedding tensor...')
    embedding_tensor = torch.load(load_path)
    extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
    print('attached preprocessed embedding')
    return extended_features

#similarity based feature embedding
def Cosine(data, num_anchor_nodes, seed):
    cos = CosineSimilarity(dim=0)
    anchor_nodes = sample_anchor_nodes(data, num_anchor_nodes, sampling_method='stochastic')
    anchor_features = [data.x[anchor_node] for anchor_node in anchor_nodes]
    feature_matrix = [vector for vector in data.x]
    embedding_out = [[cos(feature_vector, anchor_vector) for anchor_vector in anchor_features] for feature_vector in feature_matrix]
    embedding_out = torch.as_tensor(embedding_out)
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'cosine_{num_anchor_nodes}_seed_{seed}.pt')
    torch.save(embedding_out, save_path)
    print('Cached embedding saved!')
    return embedding_out

def Euclidean(data, num_anchor_nodes, seed):
    anchor_nodes = sample_anchor_nodes(data, num_anchor_nodes, sampling_method='stochastic')
    load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'node2vec.pt')
    node2vec_embeddings = torch.load(load_path).cpu()
    anchor_embeddings = [node2vec_embeddings[index] for index in anchor_nodes]
    embedding_out = [[1/torch.dist(base_embedding, anchor_embedding) for anchor_embedding in anchor_embeddings] for base_embedding in node2vec_embeddings]
    embedding_out = torch.as_tensor(embedding_out)
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'euclidean_{num_anchor_nodes}_seed_{seed}.pt')
    torch.save(embedding_out, save_path)
    print('Cached embedding saved!')
    return embedding_out

def attach_alternative_embedding(data, num_anchor_nodes, embedding_method, seed, caching=True):
    if embedding_method == 'cosine':
        if caching == True:
            load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'cosine_{num_anchor_nodes}_seed_{seed}.pt')
            if osp.isfile(load_path) == True:
                print('Found cached cosine embedding!')
                embedding_tensor = torch.load(load_path)
                extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
            else:
                print('No cosine embedding found, deriving cosine embedding...')
                embedding_matrix = Cosine(data, num_anchor_nodes, seed)
                embedding_tensor = torch.as_tensor(embedding_matrix)
                extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1

        else:
            print('Deriving cosine embedding...')
            embedding_matrix = Cosine(data, num_anchor_nodes, seed)
            embedding_tensor = torch.as_tensor(embedding_matrix)
            extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
        
    if embedding_method == 'euclidean':
        if caching == True:
            load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'euclidean_{num_anchor_nodes}_seed_{seed}.pt')
            if osp.isfile(load_path) == True:
                print('Found cached euclidean embedding!')
                embedding_tensor = torch.load(load_path)
                extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
            else:
                print('No euclidean embedding found, deriving euclidean embedding...')
                embedding_matrix = Euclidean(data, num_anchor_nodes, seed)
                embedding_tensor = torch.as_tensor(embedding_matrix)
                extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1

        else:
            print('Deriving euclidean embedding...')
            embedding_matrix = Euclidean(data, num_anchor_nodes, seed)
            embedding_tensor = torch.as_tensor(embedding_matrix)
            extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1

    print('feature matrix is blessed by the POPE', '\n', f'feature matrix shape: {extended_features.shape}')
    return extended_features