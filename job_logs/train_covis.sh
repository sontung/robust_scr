#!/bin/bash -l
#PBS -N covis
#PBS -l select=1:ncpus=10:ngpus=1:mem=8GB
#PBS -l walltime=10:00:00
#PBS -j oe


# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate env11

# Explicitly set the starting directory
cd /home/n11373598/work/covis_graph

# Run your python script


python train_agg_in_pairs.py
