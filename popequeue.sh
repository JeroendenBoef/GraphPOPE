#!/usr/bin/env bash
for param in "closeness_centrality" "eigenvector_centrality"
do
    for num in 32 64 128 256
    do
        #echo $param $num
        python flickr_lightning.py --sampling_method $param --num_anchor_nodes $num
    done;
    #echo $param
    #python flickr_lightning.py --sampling_method $param
done;

# for param in 32 64 128 256
# do
#     python flickr_graphsaint.py --sampling_method "pagerank" --num_anchor_nodes $param
# done;