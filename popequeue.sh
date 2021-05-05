#!/usr/bin/env bash
for param in "pagerank" "degree_centrality"
do
    #echo $param
    python flickr_lightning.py --sampling_method $param
done;

# for param in 32 64 128 256
# do
#     python flickr_graphsaint.py --sampling_method "pagerank" --num_anchor_nodes $param
# done;