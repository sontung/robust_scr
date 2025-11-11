#!/bin/bash -l
#PBS -N aachen_tune
#PBS -l select=1:ncpus=10:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=40:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate scrstudio

SCENE_PATH=../scrstudio_exp/data/data/aachen

cd /home/n11373598/work/glace_experiment

python train_agg_aachen.py $SCENE_PATH 0 desc_tune1 64 10000 0 16 1e-4 9e-10 1 0

python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_tune1.npy  --feat_name_test checkpoints/desc_tune1_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --grad_acc 2

python train_agg_aachen.py $SCENE_PATH 0 desc_tune1 64 10000 0 16 3e-4 9e-8 1 0

python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_tune1.npy  --feat_name_test checkpoints/desc_tune1_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --grad_acc 2

python train_agg_aachen.py $SCENE_PATH 0 desc_tune1 64 10000 1 16 5e-4 9e-6 1 0

python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_tune1.npy  --feat_name_test checkpoints/desc_tune1_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --grad_acc 2

