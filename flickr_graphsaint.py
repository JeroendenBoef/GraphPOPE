import argparse
import os.path as osp
import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree

from utils import attach_distance_embedding
from logger import Logger
import wandb

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

parser = argparse.ArgumentParser(description='Flickr (GraphSAINT)')
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=1000)
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

# attach_distance_embedding(data, num_anchor_nodes=128, use_cache=False)

# loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=args.walk_length,
#                                      num_steps=args.num_steps, sample_coverage=100,
#                                      num_workers=4) 

class Net(torch.nn.Module):
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

    print(f'torch seed: {run}')
    

    attach_distance_embedding(data, num_anchor_nodes=args.num_anchor_nodes, sampling_method=args.sampling_method, use_cache=False)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=args.walk_length,
                                    num_steps=args.num_steps, sample_coverage=100,
                                    num_workers=4, worker_init_fn=seed_worker) 
    
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

    # except RuntimeError:
    #     continue


# for epoch in range(1, 51):
#     loss = train()
#     accs = test()
#     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
#           f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')