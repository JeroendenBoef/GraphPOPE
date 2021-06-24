import sys
import pickle
from tqdm import tqdm
import os.path as osp
import multiprocessing as mp

import numpy as np
import networkx as nx

import torch
from torch.nn import CosineSimilarity
from torch_geometric.utils import to_networkx

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances

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
        sorted_pagerank = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1])} #ascending sort
        sampled_anchor_nodes = list(sorted_pagerank.keys())[-num_anchor_nodes:] #take last n because of ascending sort

    if sampling_method == 'betweenness_centrality':
        G = to_networkx(data)
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_betweenness_centrality = {k: v for k, v in sorted(betweenness_centrality.items(), key=lambda item: item[1])} #ascending sort
        sampled_anchor_nodes = list(sorted_betweenness_centrality.keys())[-num_anchor_nodes:]#take last n because of ascending sort

    if sampling_method == 'degree_centrality':
        G = to_networkx(data)
        degree_centrality = nx.degree_centrality(G)
        sorted_degree_centrality = {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])} #ascending sort
        sampled_anchor_nodes = list(sorted_degree_centrality.keys())[-num_anchor_nodes:] #take last n because of ascending sort

    if sampling_method == 'eigenvector_centrality':
        G = to_networkx(data)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        sorted_eigenvector_centrality = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])} #ascending sort
        sampled_anchor_nodes = list(sorted_eigenvector_centrality.keys())[-num_anchor_nodes:] #take last n because of ascending sort

    if sampling_method == 'closeness_centrality':
        G = to_networkx(data)
        closeness_centrality = nx.closeness_centrality(G)
        sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])} #ascending sort
        sampled_anchor_nodes = list(sorted_closeness_centrality.keys())[-num_anchor_nodes:] #take last n because of ascending sort

    if sampling_method == 'clustering_coefficient':
        G = to_networkx(data)
        clustering_coefficient = nx.clustering(G)
        sorted_clustering_coefficient = {k: v for k, v in sorted(clustering_coefficient.items(), key=lambda item: item[1])} #ascending sort
        sampled_anchor_nodes = list(sorted_clustering_coefficient.keys())[-num_anchor_nodes:] #take last n because of ascending sort

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

def all_pairs_shortest_path_length_parallel(G, anchor_nodes, num_workers):
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

def get_geodesic_distance_vector(data, num_workers):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []
    G = to_networkx(data)

    dist_dict = all_pairs_shortest_path_length_parallel(G, data.anchor_nodes, num_workers)

    distance_embedding = torch.as_tensor(list(dist_dict.values()))
    return distance_embedding


def concat_into_features(embedding_matrix, data):
    """
    Merge features and embedding matrix, returns combined feature matrix
    """
    embedding_tensor = torch.as_tensor(embedding_matrix)
    combined = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
    return combined

def attach_distance_embedding(data, dataset, num_anchor_nodes, sampling_method, distance_function, num_workers):
    """
    Sample anchor nodes based on sampling method, returns GraphPOPE embeddings concatenated with feature matrix X
    """
    print('sampling anchor nodes...')
    data.anchor_nodes = sample_anchor_nodes(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method)
    print('deriving shortest paths to anchor nodes...')
    embedding_matrix = get_geodesic_distance_vector(data=data, num_workers=num_workers)
    extended_features = concat_into_features(embedding_matrix=embedding_matrix, data=data)
    print('feature matrix is blessed by the POPE!')
    return extended_features

def attach_node2vec(data, dataset, num_anchor_nodes, sampling_method, distance_function, num_workers):
    """
    Load cached node2vec embedding of the given graph, generates embedding space GraphPOPE embeddings which are subsequently concatenated with the feature matrix X and returned
    """
    scaler = MinMaxScaler()
    print('sampling anchor nodes...')
    loading_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{dataset}_node2vec.pt')
    node2vec_embeddings = torch.load(loading_path, map_location="cpu").detach().numpy()

    dist_map = {
        'distance': cosine_distances,
        'similarity': cosine_similarity,
        'euclidean': euclidean_distances,
    }

    cosine_function = dist_map[distance_function]
    if sampling_method == 'stochastic':
        anchor_nodes = sample_anchor_nodes(data, num_anchor_nodes, sampling_method='stochastic')
        anchor_embeddings = [node2vec_embeddings[anchor_node] for anchor_node in anchor_nodes]
    else:
        kmeans = KMeans(n_clusters=num_anchor_nodes).fit(node2vec_embeddings)
        anchor_embeddings = kmeans.cluster_centers_
        print('K means cluster anchor nodes derived!')
    

    embedding_out = cosine_function(node2vec_embeddings, anchor_embeddings)
    scaler.fit(embedding_out)
    scaled_pope = scaler.transform(embedding_out)
    extended_features = concat_into_features(embedding_matrix=scaled_pope, data=data)

    print('feature matrix is blessed by the POPE')
    return extended_features

def Graphpope(data, dataset, embedding_space, sampling_method, num_anchor_nodes, distance_function=None, num_workers=4):
    pope_map = {
        'geodesic': attach_distance_embedding,
        'node2vec': attach_node2vec,
    }
    
    pope = pope_map[embedding_space]
    enhanced_features = pope(data, dataset, num_anchor_nodes, sampling_method, distance_function, num_workers=num_workers)
    return enhanced_features