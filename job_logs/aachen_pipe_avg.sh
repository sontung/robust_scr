#!/bin/bash
#SBATCH --job-name=scr_exp_avg
#SBATCH --partition=main
#SBATCH --nodelist=worker-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%3j_%x.out
#SBATCH --error=%3j_%x.err

SCENE_PATH=/mnt/data/sftp/data/tungns30/aachen10

cd /home/tungns30/robust_scr

#/home/tungns30/.pixi/bin/pixi run scr-overlap-score --data $SCENE_PATH/train --max_depth 50
#/home/tungns30/.pixi/bin/pixi run scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data $SCENE_PATH
#/home/tungns30/.pixi/bin/pixi run python train_agg_aachen.py $SCENE_PATH 0 desc_aachen0 64 10000 0 16 1e-3 9e-10 1 0 --margin_t 0.9

/home/tungns30/.pixi/bin/pixi run python trainer_with_aachen_avg.py  --scene $SCENE_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 12 --feat_name checkpoints/desc_aachen0.npy  --feat_name_test checkpoints/desc_aachen0_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 1 --output_map_file head_main.pth
