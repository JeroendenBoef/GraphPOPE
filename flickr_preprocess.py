import networkx as nx
import os.path as osp
import pickle

import torch.nn.functional as F
from torch_geometric.datasets import Flickr 
from torch_geometric.utils import to_networkx

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]

G = to_networkx(data)
paths = nx.shortest_path(G)

shortest_path_lengths = {}

for source_node in paths.keys():
    shortest_path_lengths[source_node] = {}
    for target_node in paths[source_node].keys():
        length = len(paths[source_node][target_node])
        shortest_path_lengths[source_node][target_node] = length

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'flickr_path_lengths.pickle')

with open(save_path, 'wb') as handle:
    pickle.dump(shortest_path_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved shortest paths at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'flickr_dc.pickle')
degree_centrality = nx.degree_centrality(G)
sorted_degree_centrality = {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_degree_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved degree_centrality at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'flickr_cc.pickle')
closeness_centrality = nx.closeness_centrality(G)
sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_closeness_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved closeness_centrality at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'flickr_bc.pickle')
betweenness_centrality = nx.betweenness_centrality(G)
sorted_betweenness_centrality = {k: v for k, v in sorted(betweenness_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_betweenness_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved betweenness_centrality at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'flickr_clustering.pickle')
clustering_coefficient = nx.clustering(G)
sorted_clustering_coefficient = {k: v for k, v in sorted(clustering_coefficient.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_clustering_coefficient, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved clustering_coefficient at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'flickr_ec.pickle')
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
sorted_eigenvector_centrality = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_eigenvector_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved eigenvector_centrality at {save_path}!')