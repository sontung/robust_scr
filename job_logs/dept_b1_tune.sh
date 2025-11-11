#!/bin/bash -l
#PBS -N dept_b1_tune
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

#set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate scrstudio


cd /home/n11373598/work/glace_experiment
SCENE_PATH=../scrstudio_exp/data/dept/B1

python train_agg.py $SCENE_PATH 0 desc_b1_tune1 64 10000 0 16 5e-3 9e-10 1 0

python -u ace_trainer_with_dept.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_b1_tune1.npy  --feat_name_test checkpoints/desc_b1_tune1_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py $SCENE_PATH 0 desc_b1_tune1 64 10000 0 16 4e-3 9e-10 1 0

python -u ace_trainer_with_dept.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_b1_tune1.npy  --feat_name_test checkpoints/desc_b1_tune1_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py $SCENE_PATH 0 desc_b1_tune1 64 10000 0 16 2e-3 9e-10 1 0

python -u ace_trainer_with_dept.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_b1_tune1.npy  --feat_name_test checkpoints/desc_b1_tune1_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py $SCENE_PATH 0 desc_b1_tune1 64 10000 0 16 1e-3 9e-10 1 0

python -u ace_trainer_with_dept.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_b1_tune1.npy  --feat_name_test checkpoints/desc_b1_tune1_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py $SCENE_PATH 0 desc_b1_tune1 64 10000 0 16 3e-3 9e-10 1 0

python -u ace_trainer_with_dept.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_b1_tune1.npy  --feat_name_test checkpoints/desc_b1_tune1_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1


