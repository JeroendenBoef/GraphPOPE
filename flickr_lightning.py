import argparse
import os
import os.path as osp
import random
import numpy as np
from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch.nn import ModuleList, BatchNorm1d

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree, subgraph
from torch_geometric.datasets import Flickr as PyGFlickr
from torch_geometric.data import NeighborSampler, Data, GraphSAINTRandomWalkSampler

import wandb
from utils import attach_deterministic_distance_embedding

parser = argparse.ArgumentParser(description='Flickr Pytorch Lightning GraphSAINT')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_layer_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--walk_length', type=int, default=3)
parser.add_argument('--num_steps', type=int, default=30)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=6)
#parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--num_anchor_nodes', type=int, default=0)
parser.add_argument('--sampling_method', type=str, default='baseline')
args = parser.parse_args()
print(args)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

class Flickr(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_anchor_nodes: int,
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_anchor_nodes = num_anchor_nodes
        self.transform = T.ToSparseTensor(remove_edge_index=False)

    @property
    def num_features(self) -> int:
        return int(500 + self.num_anchor_nodes)

    @property
    def num_classes(self) -> int:
        return 7

    def prepare_data(self):
        PyGFlickr(self.data_dir, pre_transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        self.data = PyGFlickr(self.data_dir)[0]
        row, col = self.data.edge_index
        self.data.edge_weight = 1. / degree(col, self.data.num_nodes)[col]  # Norm by in-degree.
        #self.data.x = attach_deterministic_distance_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
        train_index, train_weight = subgraph(self.data.train_mask, self.data.edge_index, self.data.edge_weight, relabel_nodes=True) #sample subgraph for graphsaint loader, relabel nodes
        self.train_data = Data(edge_index=train_index, edge_weight=train_weight, x=self.data.x[self.data.train_mask], y=self.data.y[self.data.train_mask])
        
    #uncomment for graphsaint
    def train_dataloader(self):
        return GraphSAINTRandomWalkSampler(self.train_data, batch_size=self.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=0,
                                    num_workers=args.num_workers, worker_init_fn=seed_worker,
                                    save_dir=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr-normalization'))
    
    #uncomment for sage
    # def train_dataloader(self):
    #     return NeighborSampler(self.data.adj_t, node_idx=self.data.train_mask, sizes=[25, 10], 
    #                             return_e_id=False, transform=self.convert_batch, 
    #                             batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers,
    #                             worker_init_fn=seed_worker, persistent_workers=True)

    def val_dataloader(self):
        return NeighborSampler(self.data.adj_t, node_idx=self.data.val_mask, sizes=[25, 10], 
                                transform=self.convert_batch, return_e_id=False,
                                batch_size=self.batch_size, num_workers=args.num_workers, 
                                worker_init_fn=seed_worker, persistent_workers=True)
    
    def test_dataloader(self):
        return NeighborSampler(self.data.adj_t, node_idx=self.data.test_mask, sizes=[25, 10], 
                                transform=self.convert_batch, return_e_id=False,
                                batch_size=self.batch_size, num_workers=args.num_workers, 
                                worker_init_fn=seed_worker, persistent_workers=True)

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            adjs_t=[adj_t for adj_t, _, _ in adjs],
        )



class SaintGCN(LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.dropout = dropout

        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.bns = ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(BatchNorm1d(hidden_channels))
        
        self.train_acc = Accuracy().clone()
        self.val_acc = Accuracy().clone()
        self.test_acc = Accuracy().clone()

    def forward(self, x, edge_index, edge_weight=None):
        #training loop on edge index long tensors   
        if type(edge_index) != list: 
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index, edge_weight)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index, edge_weight)
        
        #inference is on pairs of sparse tensors
        elif isinstance(edge_index[0], SparseTensor):
            for i, adj_t in enumerate(edge_index):
                x = self.convs[i]((x, x[:adj_t.size(0)]), adj_t)
                if i < len(edge_index) - 1:
                    x = self.bns[i](x)
                    x = x.relu_()
                    x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def training_step(self, batch, batch_idx: int):
        #uncomment for sage
        # x, y, adjs_t = batch
        # y_hat = self(x, adjs_t)
        # train_loss = F.cross_entropy(y_hat, y)
        # train_acc = self.train_acc(y_hat.softmax(dim=-1), y)
        
        #uncomment for saint
        y_hat = self(batch.x, batch.edge_index)
        train_loss = F.cross_entropy(y_hat, batch.y)
        train_acc = self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train_loss', train_loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        x, y, edge_index = batch
        y_hat = self(x, edge_index)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False,
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
        return torch.optim.Adam(self.parameters(), lr=args.lr)

def main():
    wandb_logger = WandbLogger(name=f'{args.sampling_method}-{args.num_anchor_nodes}',project='GraphPOPE-Flickr-new')
    seed_everything(42)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
    datamodule = Flickr(path, batch_size=args.batch_size, num_anchor_nodes=args.num_anchor_nodes)
    model = SaintGCN(in_channels=datamodule.num_features, out_channels=datamodule.num_classes, hidden_channels=args.hidden_layer_size, num_layers=args.num_layers)
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=2, dirpath=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'flickr_checkpoint'))
    #checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    trainer = Trainer(gpus=1, max_epochs=10, callbacks=[checkpoint_callback, pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss')],
                        gradient_clip_val=0.5, stochastic_weight_avg=True)
    # trainer = Trainer(gpus=[0], max_epochs=args.epochs, fast_dev_run=True,
    #                 callbacks=[checkpoint_callback, pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss')], logger=wandb_logger,
    #                 gradient_clip_val=0.5, stochastic_weight_avg=True)

    # Uncomment to train on multiple GPUs:
    # trainer = Trainer(gpus=2, accelerator='ddp', max_epochs=args.epochs,
    #                 callbacks=[checkpoint_callback, pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss')], logger=wandb_logger,
    #                 gradient_clip_val=0.5, stochastic_weight_avg=True)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model)


if __name__ == "__main__":
    main()