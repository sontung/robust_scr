#!/bin/bash -l
#PBS -N features
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=40:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

cd /home/n11373598/work/desc_clasisfier

/home/n11373598/.pixi/bin/pixi run python train_w_nn.py  || {
  echo "Python crashed!"
  exit 1
}