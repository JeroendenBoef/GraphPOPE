#!/usr/bin/env bash
python sampling_benchmarking.py

for param in 32 64 128 256 512
do
    python flickr_graphsaint.py --num_anchor_nodes $param
done;

for param in 32 64 128 256 512
do
    python flickr_graphsaint.py --sampling_method "pagerank" --num_anchor_nodes $param
done;