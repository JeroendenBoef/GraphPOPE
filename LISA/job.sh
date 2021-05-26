#!/bin/bash
#SBATCH -p gpu_shared_jupyter
#SBATCH -t 00:25:00
# Run me with  sbatch --gpus-per-node=1 conda-job.sh

# Set environment variables
echo "Setting environment variables..."
export RUN_NAME="graphpope-products"
export _ROOT_=$HOME"/new_root"

# Environment variables for graphpope
export graphpope_CACHE_DIR="$TMPDIR"/denboef_scratch/home/.cache/torch
export graphpope_DATA_DIR="$TMPDIR"/denboef_scratch/home/.cache/torch/graphpope/data
export graphpope_SAVE_DIR="$TMPDIR"/denboef_scratch/save

# Load modules
echo "Loading modules..."
module load 2020
module load TensorFlow/2.3.1-fosscuda-2020a-Python-3.8.2
module load Anaconda3

# Check Nvidia on compute node
echo "Checking CUDA..."
nvcc -V
nvidia-smi

# Move data to scratch
echo "Moving data to scratch..."
mkdir -p "$TMPDIR"/denboef_scratch/save
cp -r $_ROOT_/home "$TMPDIR"/denboef_scratch/
chmod -R a+rwX "$TMPDIR"/denboef_scratch
cd "$TMPDIR"/denboef_scratch/home/.cache/torch/graphpope/data/datasets
unzip -qo mmimdb.zip && unzip -qo mmimdb_self_super.zip

# FINE-TUNING! For pretraining use no_labels.jsonl
mv mmimdb_self_super/labels.jsonl mmimdb/defaults/annotations/train.jsonl
cd $HOME

# Prepare output folder
echo "Preparing output folder..."
mkdir job_saves/$RUN_NAME
chmod -R a+rwX job_saves/$RUN_NAME

# Setup conda environment
echo "Setting up conda environment..."
cd $_ROOT_/opt/conda/bin && source activate
conda activate $_ROOT_/opt/conda/envs/graphpope
python3 -m pip install -e "$TMPDIR"/denboef_scratch/home/tschneider/pretraining-vl/graphpope

# No matter what happens, we copy the save folder to our login node
trap 'cp -r $TMPDIR/denboef_scratch/save $HOME/$RUN_NAME' EXIT

# Run
graphpope_run config=projects/visual_bert/configs/mmimdb/defaults.yaml \
  run_type=train_val \
  dataset=mmimdb \
  model=visual_bert \
  training.fp16=True \
  training.tensorboard=True \
  training.batch_size=4