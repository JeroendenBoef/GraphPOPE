import argparse
import os.path as osp
import os
import random
import numpy as np

import torch

from torch_geometric.datasets import Flickr
from torch_geometric.utils import degree

from utils import process_and_save_embedding

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

num_anchors_list = [32, 64, 128, 256, 512]
sampling_methods_list = ['stochastic', 'pagerank']
for sampling_method in sampling_methods_list:
    for num_anchor_nodes in num_anchors_list:
        for run in range(20):

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
            

            process_and_save_embedding(data=data, num_anchor_nodes=num_anchor_nodes, sampling_method=sampling_method, run=run)
