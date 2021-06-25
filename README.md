# GraphPOPE - Retaining Structural Graph Information Using Position-aware Node Embeddings

This preprocessing framework embeds topological information into the feature matrix through the generation of relative distance embeddings. By sampling *anchor nodes* from a given graph, identification points are determined. Normalized relative distance embeddings are then generated for all pairings of nodes and anchor nodes. These embeddings serve as a skeleton of the graph and identify which neighborhood a node belongs to. Intuitively, GraphPOPE embeddings can be interpreted as node2vec neighborhood embeddings for the whole graph, whereas node2vec generates second-order random walks neighborhood embeddings for individual nodes. This makes GraphPOPE applicable to Multi Layer Perceptrons and local pooling models such as Graph Convolutional Networks alike, as topological information beyond the scope of a convolutional kernel is provided. This repository provides a baseline GraphSAGE architecture implemented in Pytorch Lightning and Pytorch Geometric and GraphPOPE-enhanced iterations with experiment pipelines.

We provide 2 streams of implementations: GraphPOPE-geodesic and an embedding space approximation, GraphPOPE-node2vec. Our geodesic implementation supports stochastic anchor node sampling or biased sampling assisted by node centralities (closeness-, degree-, eigenvector- centrality, pagerank, clustering coefficient). GraphPOPE-node2vec supports stochastic and K-means clustering-assisted sampling, as well as euclidean, and cosine distance and cosine similarity for distance metrics. 

This is a personal repository for my Information Studies - Data Science track master thesis titled: "GraphPOPE: Retaining Structural Graph Information Using Position-aware Node Embeddings".

# Results
Node property prediction experiment results on the Flickr and Pubmed benchmarking datasets. Displayed results are averages over 20 runs in a range of fixed seeds to ensure reproducability. Highest accuracy values are denoted in **bold**.

| Name                      | Flickr        | Pubmed    |
| --------------------------|:-------------:| ---------:|
| Betweenness centrality    | 53.93         | 89.14     |
| Closeness centrality      | 52.55         | 89.28     |
| Degree centrality         | 52.92         | 89.32     |
| Clustering coefficient    | 52.63         | 89.40     |
| Eigenvector centrality    | 52.48         | 89.29     |
| PageRank                  | **52.94**     | 89.05     |
| Stochastic                | 52.75         | **89.55** |
| node2vec-cdist            | 51.70         | 89.52     |
| node2vec-euclidean        | 51.68         | 89.52     |
| Baseline GraphSAGE        | 51.78         | 89.51     |

# Dependencies
- torch 1.8.0 (CUDA 11.1)
- torch-geometric 1.7.0
- pytorch-lightning 1.3.3
- scikit-learn 0.24.2
- wandb 0.10.31
- numpy 1.18.1

# Data
Flickr and PubMed data will be downloaded to the provided directory if they are not found. Download the raw and processed files and store them in a /data dir within the GraphPOPE dir. Embedding space versions of GraphPOPE require a node2vec embedding of the given dataset. Use `generate_node2vec_embedding.py` to generate these for Flickr & Pubmed to enable GraphPOPE-node2vec.

# Experiments
Datasets and LightningModules are combined in main, GraphPOPE embedding generation helper functions can be found in `utils.py`. Perform a 256 anchor nodes, stochastically sampled GraphPOPE-geodesic experiment on Flickr with:
```
python main.py \
--dataset "flickr" \
--embedding_space "geodesic" \
--sampling_method "stochastic" \
--num_anchor_nodes 256 \
--distance_function None \
--num_workers 6 \
--dropout 0.5 \
--lr 0.001 \
--num_layers 3 \
--batch_size 1550 \
--epochs 300 \
--seed 42 \
--wandb_logging False \
--n_gpus 1
```

Argument options of `main.py`:
```
--dataset               dataset for cached node2vec embeddings of the graph {'flickr', 'pubmed'} (default: 'flickr')
--embedding_space       space for distance calculation {'geodesic', 'node2vec'} (default: 'geodesic')
--sampling_method       approach to anchor node sampling, geodesic: {'stochastic', 'closeness_centrality', 'degree_centrality', 'eigenvector_centrality',
                        'pagerank', 'clustering_coefficient'}, node2vec: {'stochastic', 'kmeans'} (default: 'stochastic')
--num_anchor_nodes      amount of anchor nodes to use, 0 anchor nodes results in baseline model (default: 256)
--distance_function     distance function for GraphPOPE-node2vec {'distance', 'similarity', 'euclidean'} (default: None)
--num_workers           amount of workers for sssp geodesic distance calculation (default: 6)
--dropout               dropout rate (default: 0.5)
--lr                    learning rate (default: 0.001)
--num_layers            number of convolutional layers used in the model (default: 3)
--batch_size            batch size for training (default: 1550)
--epochs                number of max epochs to train for (default: 300) 
--seed                  global seed for reproducability (default: 42)
--wandb_logging         enable logging through weights & biases (default: False)
--n_gpus 1              number of gpus to train on (default: 1)

```
