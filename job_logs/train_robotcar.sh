#!/bin/bash -l
#PBS -N robotcar
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error


SCENE_PATH=datasets/robotcar
DATA_PATH=../descriptor-disambiguation/datasets/robotcar

cd /home/n11373598/work/glace_experiment

#/home/n11373598/.pixi/bin/pixi run python train_agg_robotcar.py $SCENE_PATH $DATA_PATH 0 desc_robotcar 64 10000 0 16 3e-3 9e-10 1 0 || {
#  echo "Python crashed!"
#  exit 1
#}

/home/n11373598/.pixi/bin/pixi run python -u ace_trainer_with_robotcar.py  --scene $SCENE_PATH --db_dir $DATA_PATH --global_feat 1 --use_aug 1 --iter_output 1000000 --depth_target 5 --feat_name checkpoints/desc_robotcar.npy  --feat_name_test checkpoints/desc_robotcar_test.npy   --training_buffer_size 128000000 --max_iterations 100000 --batch_size 320000 --graph_aug 1 --use_half 1 --focus_tune 0 --output_map_file head_main.pth || {
  echo "Python crashed!"
  exit 1
}
