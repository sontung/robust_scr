#!/bin/bash -l
#PBS -N glace2
#PBS -l select=1:ncpus=1:ngpus=1:mem=8GB
#PBS -l walltime=48:00:00
#PBS -j oe



# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate env11

# Explicitly set the starting directory
cd /home/n11373598/work/glace_experiment

# Run your python script


python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_dino_ml_1easy.npy  --feat_name_test checkpoints/desc_dino_ml_1easy_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2


python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_dino_ml_1easy.npy  --feat_name_test checkpoints/desc_dino_ml_1easy_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2

python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_dino_node2vec100.npy  --feat_name_test checkpoints/desc_dino_node2vec100_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2
