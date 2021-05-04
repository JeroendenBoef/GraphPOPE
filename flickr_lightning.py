import argparse
import os
import os.path as osp
import random
from typing import Optional, List, NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.datasets import Flickr as PyGFlickr
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import degree, subgraph

import wandb
from logger import Logger
from utils import load_preprocessed_embedding
from samplers import GraphSAINTRandomWalkSampler

parser = argparse.ArgumentParser(description='Flickr Pytorch Lightning GraphSAINT')
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--walk_length', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--eval_steps', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--num_anchor_nodes', type=int, default=0)
parser.add_argument('--sampling_method', type=str, default='stochastic')
args = parser.parse_args()
print(args)

# ensure reproducibility
run = 0
os.environ['PYTHONHASHSEED'] = str(run)
random.seed(run)
np.random.seed(run)
torch.manual_seed(run)
torch.cuda.manual_seed(run)
torch.cuda.manual_seed_all(run)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

class Flickr(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int,
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.ToSparseTensor(remove_edge_index=False)

    @property
    def num_features(self) -> int:
        return 500#int(500 + args.num_anchor_nodes)

    @property
    def num_classes(self) -> int:
        return 7

    def prepare_data(self):
        PyGFlickr(self.data_dir, pre_transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        self.data = PyGFlickr(self.data_dir)[0]
        row, col = self.data.edge_index
        self.data.edge_weight = 1. / degree(col, self.data.num_nodes)[col]  # Norm by in-degree.
        train_index, train_weight = subgraph(self.data.train_mask, self.data.edge_index, self.data.edge_weight, relabel_nodes=True) #sample subgraph for graphsaint loader, relabel nodes
        self.train_data = Data(edge_index=train_index, edge_weight=train_weight, x=self.data.x[self.data.train_mask], y=self.data.y[self.data.train_mask])
        

    def train_dataloader(self):
        return GraphSAINTRandomWalkSampler(self.train_data, batch_size=self.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=0,
                                    num_workers=args.num_workers, worker_init_fn=seed_worker)
                                    #save_dir=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr-normalization'))

    def val_dataloader(self):
        return NeighborSampler(self.data.adj_t, node_idx=self.data.val_mask, sizes=[25, 10], 
                                transform=self.convert_batch, batch_size=self.batch_size, 
                                num_workers=args.num_workers, worker_init_fn=seed_worker, 
                                persistent_workers=True)
    
    def test_dataloader(self):
        return NeighborSampler(self.data.adj_t, node_idx=self.data.test_mask, sizes=[25, 10], 
                                transform=self.convert_batch, batch_size=self.batch_size, 
                                num_workers=args.num_workers, worker_init_fn=seed_worker, 
                                persistent_workers=True)

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            adjs_t=[adj_t for adj_t, _, _ in adjs],
        )



class SaintGCN(LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 256, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.dropout = dropout

        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x, edge_index, edge_weight=None):
        #training loop on edge index long tensors   
        if type(edge_index) != list: 
            for conv in self.convs[:-1]:
                x = conv(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index, edge_weight)
        
        #inference is on pairs of sparse tensors
        elif isinstance(edge_index[0], SparseTensor):
            for i, adj_t in enumerate(edge_index):
                x = self.convs[i]((x, x[:adj_t.size(0)]), adj_t)
                if i < len(edge_index) - 1:
                    #x = self.bns[i](x)
                    x = x.relu_()
                    #x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.edge_index)
        train_loss = F.cross_entropy(y_hat, batch.y)
        train_acc = self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        #print(len(batch))
        x, y, edge_index = batch
        y_hat = self(x, edge_index)
        #print(y_hat.softmax(dim=-1).shape, y.shape)
        val_acc = self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return val_acc

    def test_step(self, batch, batch_idx: int):
        x, y, edge_index = batch
        y_hat = self(x, edge_index)
        test_acc = self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return test_acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# def train():
#     model.train()
#     model.set_aggr('add' if args.use_normalization else 'mean')

#     total_loss = total_examples = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()

#         if args.use_normalization:
#             edge_weight = data.edge_norm * data.edge_weight
#             out = model(data.x, data.edge_index, edge_weight)
#             loss = F.nll_loss(out, data.y, reduction='none')
#             loss = (loss * data.node_norm)[data.train_mask].sum()
#         else:
#             out = model(data.x, data.edge_index)
#             loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_nodes
#         total_examples += data.num_nodes
#     return total_loss / total_examples


# @torch.no_grad()
# def test():
#     model.eval()
#     model.set_aggr('mean')

#     out = model(data.x.to(device), data.edge_index.to(device))
#     pred = out.argmax(dim=-1)
#     correct = pred.eq(data.y.to(device))

#     accs = []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         accs.append(correct[mask].sum().item() / mask.sum().item())
#     return accs

# logger = Logger(args.runs, args)
# wandb.init(project=f'GraphPOPE-{args.sampling_method}-{args.num_anchor_nodes}-nodes')
# config = wandb.config
# wandb.config.update(args) # adds all of the arguments as config variables

# for run in range(args.runs):

#     # ensure reproducibility
#     os.environ['PYTHONHASHSEED'] = str(run)
#     random.seed(run)
#     np.random.seed(run)
#     torch.manual_seed(run)
#     torch.cuda.manual_seed(run)
#     torch.cuda.manual_seed_all(run)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     print(f'torch seed: {run}')
    
#     load_preprocessed_embedding(data=data, num_anchor_nodes=args.num_anchor_nodes, sampling_method=args.sampling_method, run=run) #attach cached embedding
#     loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=args.walk_length,
#                                     num_steps=args.num_steps, sample_coverage=100,
#                                     num_workers=2, worker_init_fn=seed_worker) 
    
#     model = Net(hidden_channels=256).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     wandb.watch(model) #w&b
#     for epoch in range(1, 1 + args.epochs):
#         loss = train()
#         if epoch % args.log_steps == 0:
#             print(f'Run: {run + 1:02d}, '
#                     f'Epoch: {epoch:02d}, '
#                     f'Loss: {loss:.4f}')

#         if epoch > 9 and epoch % args.eval_steps == 0:
#             result = test()
#             logger.add_result(run, result)
#             train_acc, valid_acc, test_acc = result
#             print(f'Run: {run + 1:02d}, '
#                     f'Epoch: {epoch:02d}, '
#                     f'Train: {100 * train_acc:.2f}%, '
#                     f'Valid: {100 * valid_acc:.2f}% '
#                     f'Test: {100 * test_acc:.2f}%')
#             wandb.log({
#                 'Train': 100 * train_acc,
#                 'Valid': 100 * valid_acc,
#                 'Test': 100 * test_acc
#                 })

#     logger.add_result(run, result)
#     logger.print_statistics(run)
# logger.print_statistics()

# #NEEDED FOR datamodule - datamodule = Flickr_Dataset(path)
# #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')

def main():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
    datamodule = Flickr(path, batch_size=args.batch_size)
    model = SaintGCN(datamodule.num_features, datamodule.num_classes)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    #trainer = Trainer(gpus=[0], max_epochs=10, callbacks=[checkpoint_callback])

    # Uncomment to train on multiple GPUs:
    trainer = Trainer(gpus=2, accelerator='ddp', max_epochs=10,
                      callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=datamodule)
    trainer.test()


if __name__ == "__main__":
    main()