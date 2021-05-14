import graph_tool.all as gt
import torch
import numpy as np
import graph_tool.all as gt
import torch
import multiprocessing as mp
from tqdm import tqdm
import os.path as osp
import sys

def shortest_path_length_gt(G, anchor_nodes, partition_length):
    """
    Calculate shortest path distance to every sampled anchor node and normalize by 1/distance. No path = 0
    """
    dists_dict = {}
    for node in partition_length:
        distances = []
        for anchor_node in anchor_nodes:
            dist = gt.shortest_distance(G, source=G.vertex(node), target=G.vertex(anchor_node), max_dist=20)
            normalized = 0 if dist >19 else 1/dist #normalize
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

def get_simple_distance_vector(G, anchor_nodes):
    """
    Calculate normalized distance vector for all nodes in given graph G. Calculation is performed using networkx shortest path length and   normalization is performed trough 1/distance.
    """
    distance_embedding = []

    dist_dict = shortest_path_gt(G, anchor_nodes)

    distance_embedding = torch.as_tensor(list(dist_dict.values()))
    return distance_embedding

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products.graphml')
G = gt.load_graph(path)
print('data loaded in!')

node_indices = np.arange(G.num_vertices())
sampled_anchor_nodes = np.random.choice(node_indices, 1000)
embedding_tensor = get_simple_distance_vector(G, sampled_anchor_nodes) 

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'products_stochastic_1000.pt')
torch.save(embedding_tensor, save_path)
print(f'saved embedding as products_stochastic_1000.pt')

# print('shortest paths derived!')
# filepath = "distances.pkl"
# with gzip.open(filepath, "wb") as f:
#     pickled = pickle.dumps(distance)
#     optimized_pickle = pickletools.optimize(pickled)
#     f.write(optimized_pickle)
# print('shortest paths saved!')

# save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'bc_ogbnproducts.pickle')
# betweenness_centrality = nx.betweenness_centrality(G)
# sorted_betweenness_centrality = {k: v for k, v in sorted(betweenness_centrality.items(), key=lambda item: item[1])}
# with open(save_path, 'wb') as handle:
#     pickle.dump(sorted_betweenness_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'saved betweenness_centrality at {save_path}!')

# save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'dc_ogbnproducts.pickle')
# degree_centrality = nx.degree_centrality(G)
# sorted_degree_centrality = {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])}
# with open(save_path, 'wb') as handle:
#     pickle.dump(sorted_degree_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'saved degree_centrality at {save_path}!')

# save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'cc_ogbnproducts.pickle')
# closeness_centrality = nx.closeness_centrality(G)
# sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
# with open(save_path, 'wb') as handle:
#     pickle.dump(sorted_closeness_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'saved closeness_centrality at {save_path}!')

# save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'ccoeff_ogbnproducts.pickle')
# clustering_coefficient = nx.clustering(G)
# sorted_clustering_coefficient = {k: v for k, v in sorted(clustering_coefficient.items(), key=lambda item: item[1])}
# with open(save_path, 'wb') as handle:
#     pickle.dump(sorted_clustering_coefficient, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'saved clustering_coefficient at {save_path}!')

# save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'ec_ogbnproducts.pickle')
# eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
# sorted_eigenvector_centrality = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
# with open(save_path, 'wb') as handle:
#     pickle.dump(sorted_eigenvector_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'saved eigenvector_centrality at {save_path}!')