#!/bin/bash -l
#PBS -N dept_all_4f
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=40:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate scrstudio


cd /home/n11373598/work/glace_experiment
SCENE_PATH=../scrstudio_exp/data/dept/4F

python train_agg.py $SCENE_PATH 0 desc_4f 128 10000 1 16 1e-5 9e-10 0 2

python -u ace_trainer_with_dept.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_4f.npy  --feat_name_test checkpoints/desc_4f_test.npy  --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1 --output_map_file head_4f.pth

