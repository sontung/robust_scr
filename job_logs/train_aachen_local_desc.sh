#!/bin/bash -l
#PBS -N aachen_sp
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=10:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

SCENE_PATH=../scrstudio_exp/data/data/aachen

cd /home/n11373598/work/glace_experiment

#/home/n11373598/.pixi/bin/pixi run python train_agg_aachen.py $SCENE_PATH 0 desc_aachen0 64 10000 0 16 3e-3 9e-10 1 0
#/home/n11373598/.pixi/bin/pixi run python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --local_desc d2 --pca_path "proc/pcad2_128.pth" --reuse_buffer 0 --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen0.npy  --feat_name_test checkpoints/desc_aachen0_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth
#/home/n11373598/.pixi/bin/pixi run python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --local_desc sp --pca_path "proc/pcasuperpoint_128.pth" --reuse_buffer 0 --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen0.npy  --feat_name_test checkpoints/desc_aachen0_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth
/home/n11373598/.pixi/bin/pixi run python -u ace_trainer_with_loftr2.py  --scene $SCENE_PATH --local_desc aliked --reuse_buffer 0 --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen0.npy  --feat_name_test checkpoints/desc_aachen0_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth
