import os.path as osp
import gzip, pickle, pickletools
import graph_tool.all as gt

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products.graphml')
G = gt.load_graph(path)
print('data loaded in!')

# distance = gt.shortest_distance(G)
# print('shortest paths derived!')
# filepath = "distances.pkl"
# with gzip.open(filepath, "wb") as f:
#     pickled = pickle.dumps(distance)
#     optimized_pickle = pickletools.optimize(pickled)
#     f.write(optimized_pickle)
# print('shortest paths saved!')

# save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'dc_ogbnproducts.pickle')
# degree_centrality = nx.degree_centrality(G)
# sorted_degree_centrality = {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])}
# with open(save_path, 'wb') as handle:
#     pickle.dump(sorted_degree_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'saved degree_centrality at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'cc_ogbnproducts.pickle')
closeness_centrality = gt.closeness(G)
# sorted_closeness_centrality = {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
with open(save_path, 'wb') as handle:
    pickle.dump(closeness_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved closeness_centrality at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'ccoeff_ogbnproducts.pickle')
clustering_coefficient = gt.local_clustering(G)
with open(save_path, 'wb') as handle:
    pickle.dump(clustering_coefficient, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved clustering_coefficient at {save_path}!')

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'ec_ogbnproducts.pickle')
eigenvector_centrality = gt.eigenvector(G)
with open(save_path, 'wb') as handle:
    pickle.dump(eigenvector_centrality, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f'saved eigenvector_centrality at {save_path}!')