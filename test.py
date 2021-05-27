import os.path as osp
import torch
from torch_geometric.datasets import Flickr
from utils import attach_alternative_embedding, sample_anchor_nodes
from pytorch_lightning import seed_everything
import numpy as np
from sklearn.cluster import KMeans

seed_everything(42)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
print('data loaded in!')
data.x = attach_alternative_embedding(data=data, num_anchor_nodes=256, embedding_method='euclidean', seed='42', caching=False)
#data.x = attach_alternative_embedding(data=data, num_anchor_nodes=256, embedding_method='cosine', seed='42', caching=False)
# print(data.x.shape)
# path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'node2vec.pt')
# print(osp.isfile(path))

# def Euclidean_kmeans(num_anchor_nodes, seed):
#     load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'node2vec.pt')
#     node2vec_embeddings = torch.load(load_path).cpu().detach().numpy()
#     print('node2vec embeddings loaded!')
#     kmeans = KMeans(n_clusters=num_anchor_nodes).fit(node2vec_embeddings)
#     print('K means clusters derived!')
#     anchor_embeddings = torch.as_tensor(kmeans.cluster_centers_)
#     embedding_out = [[1/torch.dist(base_embedding, anchor_embedding).cpu() for anchor_embedding in anchor_embeddings] for base_embedding in node2vec_embeddings]
#     print('Euclidean distance derived!')
#     embedding_out = torch.as_tensor(embedding_out)
#     save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'euclidean_kmeans_{num_anchor_nodes}_seed_{seed}.pt')
#     torch.save(embedding_out, save_path)
#     print('Cached embedding saved!')
#     return embedding_out

# def attach_alternative_embedding(data, num_anchor_nodes, embedding_method, seed, caching=True):       
#     if embedding_method == 'euclidean':
#         if caching == True:
#             load_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', f'euclidean_{num_anchor_nodes}_seed_{seed}.pt')
#             if osp.isfile(load_path) == True:
#                 print('Found cached euclidean embedding!')
#                 embedding_tensor = torch.load(load_path)
#                 extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1
#             else:
#                 print('No euclidean embedding found, deriving euclidean embedding...')
#                 embedding_tensor = Euclidean_kmeans(num_anchor_nodes, seed)
#                 extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1

#         else:
#             print('Deriving euclidean embedding...')
#             embedding_tensor = Euclidean_kmeans(num_anchor_nodes, seed)
#             extended_features = torch.cat((data.x, embedding_tensor), 1) #concatenate with X along dimension 1

#     print('feature matrix is blessed by the POPE', '\n', f'feature matrix shape: {extended_features.shape}')
#     return extended_features

# data.x = attach_alternative_embedding(data, 256, 'euclidean', 42, False)
# print(data.x.shape)