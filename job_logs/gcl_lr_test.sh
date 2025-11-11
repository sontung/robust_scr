#!/bin/bash -l
#PBS -N gcl_blocks_small
#PBS -l select=1:ncpus=10:ngpus=1:mem=20GB
#PBS -l walltime=48:00:00
#PBS -j oe

#set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate scrstudio

cd /home/n11373598/work/covis_graph
#python train_agg_in_pairs.py 0 desc_small0 64 10000 1 16 3e-5 9e-10 0 2
#python train_agg_in_pairs.py 0 desc_small1 64 10000 1 16 3e-5 9e-10 0 3
#python train_agg_in_pairs.py 0 desc_small2 64 10000 1 16 3e-5 9e-10 0 4
#python train_agg_in_pairs.py 0 desc_small3 64 10000 1 16 3e-5 9e-10 0 5


cd /home/n11373598/work/glace_experiment

python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_small0.npy  --feat_name_test checkpoints/desc_small0_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2
python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_small1.npy  --feat_name_test checkpoints/desc_small1_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2
python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_small2.npy  --feat_name_test checkpoints/desc_small2_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2
python -u ace_trainer_with_loftr2.py  --scene ../scrstudio/data/aachen --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_small3.npy  --feat_name_test checkpoints/desc_small3_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2

