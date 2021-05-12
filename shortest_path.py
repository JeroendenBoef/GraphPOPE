import os
import os.path as osp
import pickle
import networkx as nx
from torch_geometric.utils import to_networkx

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import degree

dataset = PygNodePropPredDataset(name='ogbn-products')
data = dataset[0]
G = to_networkx(data)
print('data loaded in!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'sp_ogbnproducts.pickle')
paths = nx.shortest_path(G)
with open(save_path, 'wb') as handle:
    pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved shortest paths at {save_path}!')


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

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'cc_ogbnproducts.pickle')
closeness_centrality = nx.closeness_centrality(G)
sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_closeness_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved closeness_centrality at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'ccoeff_ogbnproducts.pickle')
clustering_coefficient = nx.clustering(G)
sorted_clustering_coefficient = {k: v for k, v in sorted(clustering_coefficient.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_clustering_coefficient, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved clustering_coefficient at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'ec_ogbnproducts.pickle')
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
sorted_eigenvector_centrality = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(sorted_eigenvector_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved eigenvector_centrality at {save_path}!')