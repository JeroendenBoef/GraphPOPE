import numpy as np

def sample_anchor_nodes(data, num_anchor_nodes=32):
    """
    Returns num_anchor_nodes amount of randomly sampled anchor nodes 
    """
    node_indices = np.arange(data.num_nodes)
    sampled_anchor_nodes = np.random.choice(node_indices, num_anchor_nodes)
    return sampled_anchor_nodes