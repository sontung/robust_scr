#!/bin/bash -l
#PBS -N aachen11
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error


SCENE_PATH=datasets/aachen11
DATA_PATH=../descriptor-disambiguation/datasets/aachen_v1.1

cd /home/n11373598/work/glace_experiment

#/home/n11373598/.pixi/bin/pixi run python train_agg_aachen11.py $SCENE_PATH $DATA_PATH 0 desc_aachenv11 64 10000 0 16 3e-3 9e-10 1 0 || {
#  echo "Python crashed!"
#  exit 1
#}

/home/n11373598/.pixi/bin/pixi run python -u ace_trainer_with_aachen11.py --db_dir $DATA_PATH  --scene $SCENE_PATH --depth_target 5 --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_aachenv11.npy  --feat_name_test checkpoints/desc_aachenv11_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --focus_tune 1 --reuse_buffer 1
/home/n11373598/.pixi/bin/pixi run python -u ace_trainer_with_aachen11.py --db_dir $DATA_PATH  --scene $SCENE_PATH --depth_target 12 --global_feat 1 --use_aug 1 --iter_output 1000000 --feat_name checkpoints/desc_aachenv11.npy  --feat_name_test checkpoints/desc_aachenv11_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --focus_tune 1 --reuse_buffer 1

