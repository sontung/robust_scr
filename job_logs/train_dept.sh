#!/bin/bash -l
#PBS -N dept_long_big_d
#PBS -l select=1:ncpus=20:ngpus=1:mem=100GB:gpu_id=H100
#PBS -l walltime=20:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.


mamba activate scrstudio
cd /home/n11373598/work/glace_experiment

python -u ace_trainer_with_dept.py  --scene ../scrstudio_exp/data/dept/1F --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name ../scrstudio_exp/data/dept/1F/desc_node2vec.npy  --feat_name_test ../scrstudio_exp/data/dept/1F/desc_node2vec_val.npy --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 0 --grad_acc 2 --membership ../scrstudio_exp/data/dept/1F/train/pose_overlap.npz --depth_init 1

