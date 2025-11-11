#!/bin/bash -l
#PBS -N aachen_main_mem
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=20:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
conda activate scrstudio

SCENE_PATH=../scrstudio_exp/data/data/aachen

cd /home/n11373598/work/glace_experiment
python train_agg_aachen.py $SCENE_PATH 0 desc_aachen 128 10000 0 16 1e-5 9e-10 0 0

python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen.npy  --feat_name_test checkpoints/desc_aachen_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2 --output_map_file head_main.pth

#python -u trainer_with_aachen.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_dino.npy  --feat_name_test checkpoints/desc_dino_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2 --output_map_file head_aachen.pth
#python -u ace_trainer_with_aachen_fast.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_dino.npy  --feat_name_test checkpoints/desc_dino_test.npy   --training_buffer_size 32000000 --max_iterations 100000 --batch_size 40960 --graph_aug 1 --map_scale_factor 1 --use_half 1 --focus_tune 1 --grad_acc 2 --output_map_file head_aachen.pth
