import os.path as osp
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Flickr 

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
print('data loaded in!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

vec = model()
embeddings = model(torch.arange(data.num_nodes, device=device))
print(embeddings.shape)

save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'processed_embeddings', 'node2vec.pt')
torch.save(embeddings, save_path)
print(f'saved node2vec embedding!')