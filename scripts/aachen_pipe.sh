#!/bin/bash
#SBATCH --job-name=best
#SBATCH --partition=main
#SBATCH --nodelist=worker-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%3j_%x.out
#SBATCH --error=%3j_%x.err

SCENE_PATH=/mnt/data/sftp/data/tungns30/aachen10
PIX="/home/tungns30/.pixi/bin/pixi"
MANIFEST="/home/tungns30/robust_scr/pixi.toml"

cd /home/tungns30/work/next_gen_scr

CMD="$PIX run --manifest-path $MANIFEST python"

#$CMD train_agg_aachen.py \
#  ds_path=/mnt/data/sftp/data/tungns30/aachen10 \
#  batch_size=128 \
#  model_name='desc_b${batch_size}_lr${lr}_m${margin_t}' \
#  lr=1e-3 \
#  margin_t=0.9 \

$CMD trainer_with_aachen.py \
  scene=$SCENE_PATH \
  feat_name=checkpoints/desc_b128_lr0.003_m0.5.npy \
  feat_name_test=checkpoints/desc_b128_lr0.003_m0.5_test.npy \
  focus_tune=True \
  wandb.mode=online \
  +git_hash=$COMMIT_ID \
  reuse_buffer=False \
  output_map_file=test.pth