#!/usr/bin/env bash
# for param in "closeness_centrality" "clustering_coefficient" "degree_centrality" "eigenvector_centrality" "pagerank"
# do
#     for num in 32 64 128 256
#     do
#         #echo $param $num
#         python popesage.py --sampling_method $param --num_anchor_nodes $num
#     done;
#     #echo $param
#     #python flickr_lightning.py --sampling_method $param
# done;

for param in 32 64 128 256
do
    python popesage.py --sampling_method "stochastic" --num_anchor_nodes $param
done;