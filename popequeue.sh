#!/usr/bin/env bash
for param in 64 256 512
do
    python flickr_graphsaint.py --num_anchor_nodes $param
done;

for param in 32 64 256 512
do
    python flickr_graphsaint.py --sampling_method "pagerank" --num_anchor_nodes $param
done;