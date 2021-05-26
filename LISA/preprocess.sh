#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jyromeh1@gmail.com
#SBATCH --constraint=avx512
#SBATCH --mem=90G

# Set environment variables
echo "Setting environment variables..."
export RUN_NAME="products_sp_all"
export _ROOT_=$HOME"/new_root"

# Load modules
echo "Loading modules..."
module load 2020
module load Anaconda3

# Move data to scratch
echo "Moving data to scratch..."
mkdir -p "$TMPDIR"/denboef_scratch/save
cp -r $_ROOT_/home "$TMPDIR"/denboef_scratch/
chmod -R a+rwX "$TMPDIR"/denboef_scratch
cd "$TMPDIR"/denboef_scratch/

# Prepare output folder
echo "Preparing output folder..."
mkdir embeddings
chmod -R a+rwX embeddings

# Setup conda environment
echo "Setting up conda environment..."
cd $_ROOT_/opt/conda/bin && source activate
conda activate $_ROOT_/opt/conda/envs/base
#python3 -m pip install -e "$TMPDIR"/denboef_scratch/

# No matter what happens, we copy the save folder to our login node
trap 'cp -r $TMPDIR/denboef_scratch/save $HOME/embeddings' EXIT

python $HOME/bfs.py
cp -r $TMPDIR/denboef_scratch/save $HOME/embeddings