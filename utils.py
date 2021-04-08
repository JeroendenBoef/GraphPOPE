import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import multiprocessing as mp
from tqdm import tqdm

def sample_anchor_nodes(data, num_anchor_nodes=32):
    """
    Returns num_anchor_nodes amount of randomly sampled anchor nodes 
    """
    node_indices = np.arange(data.num_nodes)
    sampled_anchor_nodes = np.random.choice(node_indices, num_anchor_nodes)
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
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(G, anchor_nodes, num_workers=12):
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

def get_simple_distance_vector(data):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []
    G = to_networkx(data)

    #distances = [nx.single_source_shortest_path_length(G, i) for i in tqdm(range(data.num_nodes))] #calculate shortest path for all node connections

    dist_dict = all_pairs_shortest_path_length_parallel(G, data.anchor_nodes)
    distance_embedding = torch.as_tensor(list(dist_dict.values()))
    
    # pbar = tqdm(total=data.num_nodes)
    # pbar.set_description('Computing distance to anchor nodes')

    # for node in dist_dict.keys(): #loop through all nodes
    #     inverted_anchor_distances = []
    #     for anchor_node in data.anchor_nodes: #retrieve distance to anchor node per node
    #         distance = node.get(anchor_node, 0) #get shortest path length, 0 if not connected
    #         if distance > 0:
    #             inverted_anchor_distances.append(1/distance) #normalized distance is 1/distance, 0 if not connected
    #         else:
    #             inverted_anchor_distances.append(distance)
    #     distance_embedding.append(inverted_anchor_distances)
    #     pbar.update(1)

    return distance_embedding


def concat_into_features(embedding_matrix, data):
    """
    Merge features and embedding matrix, returns combined feature matrix
    """
    embedding_tensor = torch.as_tensor(embedding_matrix)
    combined = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
    return combined

def attach_distance_embedding(data):
    print('sampling anchor nodes')
    data.anchor_nodes = sample_anchor_nodes(data)
    print('deriving shortest paths to anchor nodes')
    embedding_matrix = get_simple_distance_vector(data)
    extended_features = concat_into_features(embedding_matrix, data)
    data.x = extended_features
    print('feature matrix is blessed by the POPE')
