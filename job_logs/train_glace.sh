#!/bin/bash -l
#PBS -N depth_init
#PBS -l select=1:ncpus=20:ngpus=1:mem=16GB
#PBS -l walltime=20:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.


mamba activate scrstudio
cd /home/n11373598/work/glace_experiment

python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_dino_dual.npy  --feat_name_test checkpoints/desc_dino_dual_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2

