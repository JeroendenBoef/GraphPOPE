import argparse
import os
import os.path as osp
import random
from typing import Optional, List, NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.datasets import PyGFlickr
from torch_geometric.nn import GraphConv
import torch_geometric.transforms as T
from torch_geometric.utils import degree

import wandb
from logger import Logger
from utils import load_preprocessed_embedding

parser = argparse.ArgumentParser(description='Flickr Pytorch Lightning GraphSAINT')
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=3000)
parser.add_argument('--walk_length', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--eval_steps', type=int, default=2)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--num_anchor_nodes', type=int, default=128)
parser.add_argument('--sampling_method', type=str, default='stochastic')
args = parser.parse_args()
print(args)

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

class Flickr(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        #self.transform = T.ToSparseTensor()

    @property
    def num_features(self) -> int:
        return int(500 + args.num_anchor_nodes)

    @property
    def num_classes(self) -> int:
        return 7

    def prepare_data(self):
        PyGFlickr(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        self.data = PyGFlickr(self.data_dir)[0]
        row, col = self.data.edge_index
        self.data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

    #TODO: add train test split edges to match train test val mask, swap to graphsaint sampler
    
    #from reddit graphsage
    def train_dataloader(self):
        return GraphSAINTRandomWalkSampler(Data(edge_index=self.data.edge_index, edge_weight=self.data.edge_weight, x=self.data.x[self.data.train_mask], 
                                    y=self.data.y[self.data.train_mask]), batch_size=args.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=100,
                                    num_workers=2, worker_init_fn=seed_worker)

    def val_dataloader(self):
        return GraphSAINTRandomWalkSampler(Data(edge_index=self.data.edge_index, edge_weight=self.data.edge_weight, x=self.data.x[self.data.val_mask], 
                                    y=self.data.y[self.data.val_mask]), batch_size=args.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=100,
                                    num_workers=2, worker_init_fn=seed_worker)
    def test_dataloader(self):  # Test best validation model once again.
        return GraphSAINTRandomWalkSampler(Data(edge_index=self.data.edge_index, edge_weight=self.data.edge_weight, x=self.data.x[self.data.test_mask], 
                                    y=self.data.y[self.data.test_mask]), batch_size=args.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=100,
                                    num_workers=2, worker_init_fn=seed_worker)

    
    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            adjs_t=[adj_t for adj_t, _, _ in adjs],
        )

class SaintGCN(lightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 256, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.dropout = dropout

        self.convs = ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.convs.append(GraphConv(hidden_channels, hidden_channels))

        self.lns = ModuleList()
        self.lns.append(torch.nn.Linear(3 * hidden_channels, out_channels))

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x = self.convs[i]((x, x[:adj_t.size(0)]), adj_t)
            if i < len(adjs_t) - 1:
                x = self.lns[i](x)
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

#TODO: Figure out forward pass, worst case switch to graphsage format without concatenation.
#TODO: Make github issue post asking for pointers on code, how to do forward pass, how does the adjs_t function,



    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = data.x.size(1)#dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=256).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs

logger = Logger(args.runs, args)
wandb.init(project=f'GraphPOPE-{args.sampling_method}-{args.num_anchor_nodes}-nodes')
config = wandb.config
wandb.config.update(args) # adds all of the arguments as config variables

for run in range(args.runs):

    # ensure reproducibility
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

    print(f'torch seed: {run}')
    
    load_preprocessed_embedding(data=data, num_anchor_nodes=args.num_anchor_nodes, sampling_method=args.sampling_method, run=run) #attach cached embedding
    loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=100,
                                    num_workers=2, worker_init_fn=seed_worker) 
    
    model = Net(hidden_channels=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb.watch(model) #w&b
    for epoch in range(1, 1 + args.epochs):
        loss = train()
        if epoch % args.log_steps == 0:
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}')

        if epoch > 9 and epoch % args.eval_steps == 0:
            result = test()
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
            wandb.log({
                'Train': 100 * train_acc,
                'Valid': 100 * valid_acc,
                'Test': 100 * test_acc
                })

    logger.add_result(run, result)
    logger.print_statistics(run)
logger.print_statistics()

#NEEDED FOR datamodule - datamodule = Flickr_Dataset(path)
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
