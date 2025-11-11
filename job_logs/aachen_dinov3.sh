#!/bin/bash -l
#PBS -N aachen_dinov3
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error


SCENE_PATH=../scrstudio_exp/data/data/aachen
cd /home/n11373598/work/glace_experiment

/home/n11373598/.pixi/bin/pixi run python train_agg_aachen.py $SCENE_PATH 0 desc_aachen_dinov3 64 10000 0 16 3e-5 9e-10 1 0 --dino_arch dinov3 

/home/n11373598/.pixi/bin/pixi run python ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen_dinov3.npy  --feat_name_test checkpoints/desc_aachen_dinov3_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth

/home/n11373598/.pixi/bin/pixi run python train_agg_aachen.py $SCENE_PATH 0 desc_aachen_dinov3 64 10000 0 16 3e-7 9e-10 1 0 --dino_arch dinov3 

/home/n11373598/.pixi/bin/pixi run python ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen_dinov3.npy  --feat_name_test checkpoints/desc_aachen_dinov3_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth

/home/n11373598/.pixi/bin/pixi run python train_agg_aachen.py $SCENE_PATH 0 desc_aachen_dinov3 64 10000 0 16 3e-9 9e-10 1 0 --dino_arch dinov3 

/home/n11373598/.pixi/bin/pixi run python ace_trainer_with_loftr2.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen_dinov3.npy  --feat_name_test checkpoints/desc_aachen_dinov3_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth
