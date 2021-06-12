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
from torch.optim.lr_scheduler import ReduceLROnPlateau

#import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_geometric.datasets import Flickr as PyGFlickr
from torch_geometric.data import NeighborSampler

#from utils import attach_alternative_embedding, attach_deterministic_distance_embedding, load_preprocessed_embedding
from utils import attach_n2v, attach_deterministic_distance_embedding

parser = argparse.ArgumentParser(description='Flickr POPE GraphSAGE Pytorch Lightning')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_layer_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=2550)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--num_anchor_nodes', type=int, default=256)
parser.add_argument('--sampling_method', type=str, default='closeness_centrality')
parser.add_argument('--embedding_method', type=str, default='geodesic')
parser.add_argument('--seed', type=int, default=42)
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
        # return 500
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
        self.data.x = attach_deterministic_distance_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
        #self.data.x = attach_n2v(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
        #print(self.data)
        # self.data.x = attach_alternative_embedding(data=self.data, num_anchor_nodes=256, embedding_method='euclidean', seed=42, caching=True)
        # if args.embedding_method != 'geodesic':
        #     self.data.x = attach_alternative_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, embedding_method=args.embedding_method, seed=args.seed, caching=False)
        #elif (args.sampling_method != 'baseline') and (args.sampling_method != 'stochastic'):
            # self.data.x = attach_deterministic_distance_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
        # elif args.sampling_method == 'stochastic':
        #     self.data.x = load_preprocessed_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method, run=4)

    def train_dataloader(self):
        return NeighborSampler(self.data.adj_t, node_idx=self.data.train_mask, sizes=[25, 10], 
                                return_e_id=False, transform=self.convert_batch, 
                                batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers,
                                worker_init_fn=seed_worker, persistent_workers=True)

    def val_dataloader(self):
        return NeighborSampler(self.data.adj_t, node_idx=self.data.val_mask, sizes=[25, 10], 
                                return_e_id=False, transform=self.convert_batch, 
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

# class Cora(LightningDataModule):
#     def __init__(self, data_dir: str, batch_size: int, num_anchor_nodes: int,
#                  in_memory: bool = False):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_anchor_nodes = num_anchor_nodes
#         self.transform = T.ToSparseTensor(remove_edge_index=False)

#     @property
#     def num_features(self) -> int:
#         # return 500
#         return int(500 + self.num_anchor_nodes)

#     @property
#     def num_classes(self) -> int:
#         return 7

#     def prepare_data(self):
#         PyGFlickr(self.data_dir, pre_transform=self.transform)

#     def setup(self, stage: Optional[str] = None):
#         self.data = PyGFlickr(self.data_dir)[0]
#         row, col = self.data.edge_index
#         self.data.edge_weight = 1. / degree(col, self.data.num_nodes)[col]  # Norm by in-degree.
#         #self.data.x = attach_deterministic_distance_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
#         self.data.x = attach_n2v(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
#         #print(self.data)
#         # self.data.x = attach_alternative_embedding(data=self.data, num_anchor_nodes=256, embedding_method='euclidean', seed=42, caching=True)
#         # if args.embedding_method != 'geodesic':
#         #     self.data.x = attach_alternative_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, embedding_method=args.embedding_method, seed=args.seed, caching=False)
#         #elif (args.sampling_method != 'baseline') and (args.sampling_method != 'stochastic'):
#             # self.data.x = attach_deterministic_distance_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method)
#         # elif args.sampling_method == 'stochastic':
#         #     self.data.x = load_preprocessed_embedding(data=self.data, num_anchor_nodes=self.num_anchor_nodes, sampling_method=args.sampling_method, run=4)

#     def train_dataloader(self):
#         return NeighborSampler(self.data.adj_t, node_idx=self.data.train_mask, sizes=[25, 10], 
#                                 return_e_id=False, transform=self.convert_batch, 
#                                 batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers,
#                                 worker_init_fn=seed_worker, persistent_workers=True)

#     def val_dataloader(self):
#         return NeighborSampler(self.data.adj_t, node_idx=self.data.val_mask, sizes=[25, 10], 
#                                 return_e_id=False, transform=self.convert_batch, 
#                                 batch_size=self.batch_size, num_workers=args.num_workers, 
#                                 worker_init_fn=seed_worker, persistent_workers=True)
    
#     def test_dataloader(self):
#         return NeighborSampler(self.data.adj_t, node_idx=self.data.test_mask, sizes=[25, 10], 
#                                 transform=self.convert_batch, return_e_id=False,
#                                 batch_size=self.batch_size, num_workers=args.num_workers, 
#                                 worker_init_fn=seed_worker, persistent_workers=True)

#     def convert_batch(self, batch_size, n_id, adjs):
#         return Batch(
#             x=self.data.x[n_id],
#             y=self.data.y[n_id[:batch_size]],
#             adjs_t=[adj_t for adj_t, _, _ in adjs],
#         )

class SAGE(LightningModule):
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
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x, edge_index, edge_weight=None):
        for i, adj_t in enumerate(edge_index):
            x = self.convs[i]((x, x[:adj_t.size(0)]), adj_t)
            if i < len(edge_index) - 1:
                x = self.bns[i](x)
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def training_step(self, batch, batch_idx: int):
        x, y, adjs_t = batch
        y_hat = self(x, adjs_t)
        train_loss = F.cross_entropy(y_hat, y)
        train_acc = self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train_loss', train_loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        x, y, adjs_t = batch
        y_hat = self(x, adjs_t)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return val_acc

    def test_step(self, batch, batch_idx: int):
        x, y, adjs_t = batch
        y_hat = self(x, adjs_t)
        test_acc = self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return test_acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        lr_dict =  {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer), 
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
                'strict': True
            }
        }
        return lr_dict


def main():
    # Seed
    seed_everything(args.seed)

    # Data pipeline
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
    datamodule = Flickr(path, batch_size=args.batch_size, num_anchor_nodes=args.num_anchor_nodes)
    
    # Lightning model
    model = SAGE(in_channels=datamodule.num_features, out_channels=datamodule.num_classes, hidden_channels=args.hidden_layer_size, num_layers=args.num_layers)
    
    # Wandb logger
    wandb_logger = WandbLogger(name='cc_baseline_hparams',project='GraphPOPE-sage-flickr-newmeans')
    #wandb_logger.watch(model.net) #optional

    # Trainer callbacks
    checkpoint_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'flickr_checkpoint')
    early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.00, patience=20, verbose=False, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    # trainer = Trainer(gpus="1", max_epochs=args.epochs, logger=wandb_logger, checkpoint_callback=True, 
    #                 callbacks=[lr_monitor, early_stop_callback], gradient_clip_val=0.5, default_root_dir = checkpoint_path)


    # Uncomment for multi GPU
    trainer = Trainer(gpus=2, accelerator='ddp', max_epochs=args.epochs, logger=wandb_logger, checkpoint_callback=True, 
                    callbacks=[lr_monitor, early_stop_callback], gradient_clip_val=0.5, default_root_dir = checkpoint_path)

    trainer.fit(model, datamodule=datamodule)
    trainer.test()

if __name__ == "__main__":
    main()
