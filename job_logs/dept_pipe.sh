#!/bin/bash -l
#PBS -N gcl_dept_lr
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=40:00:00
#PBS -j oe

#set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate scrstudio


cd /home/n11373598/work/glace_experiment

python train_agg.py ../scrstudio_exp/data/dept/1F 0 desc_dept2 64 10000 1 16 3e-5 9e-10 0 2

python -u ace_trainer_with_dept.py  --scene ../scrstudio_exp/data/dept/1F --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_dept2.npy  --feat_name_test checkpoints/desc_dept2_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py ../scrstudio_exp/data/dept/1F 0 desc_dept2 64 10000 1 16 2e-5 9e-10 0 2

python -u ace_trainer_with_dept.py  --scene ../scrstudio_exp/data/dept/1F --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_dept2.npy  --feat_name_test checkpoints/desc_dept2_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py ../scrstudio_exp/data/dept/1F 0 desc_dept2 64 10000 1 16 4e-5 9e-10 0 2

python -u ace_trainer_with_dept.py  --scene ../scrstudio_exp/data/dept/1F --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_dept2.npy  --feat_name_test checkpoints/desc_dept2_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1

python train_agg.py ../scrstudio_exp/data/dept/1F 0 desc_dept2 64 10000 1 16 1e-5 9e-10 0 2

python -u ace_trainer_with_dept.py  --scene ../scrstudio_exp/data/dept/1F --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_dept2.npy  --feat_name_test checkpoints/desc_dept2_test.npy  --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --depth_init 1
