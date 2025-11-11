#!/bin/bash -l
#PBS -N covis2
#PBS -l select=1:ncpus=10:ngpus=1:mem=8GB
#PBS -l walltime=10:00:00
#PBS -j oe


# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate env11

# Explicitly set the starting directory
cd /home/n11373598/work/covis_graph

# Run your python script

python train_agg_ori.py checkpoints/membership_node2vec10.npy desc_dino_node2vec10 128 50
python train_agg_ori.py checkpoints/membership_node2vec20.npy desc_dino_node2vec20 128 50
python train_agg_ori.py checkpoints/membership_node2vec50.npy desc_dino_node2vec50 128 50
python train_agg_ori.py checkpoints/membership_node2vec100.npy desc_dino_node2vec100 128 50
