#!/bin/bash -l
#PBS -N train_node2vec
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
conda activate env4

cd /home/n11373598/work/glace_experiment

scr-train node2vec --data --data datasets/robotcar --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2

