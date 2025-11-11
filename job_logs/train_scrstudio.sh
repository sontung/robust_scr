#!/bin/bash -l
#PBS -N scr_studio
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

DATA_PATH=../glace_experiment/datasets/aachen11

cd /home/n11373598/work/scrstudio_me

/home/n11373598/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt || {
  echo "Python crashed!"
  exit 1
}
/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq || {
  echo "Python crashed!"
  exit 1
}
/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test || {
  echo "Python crashed!"
  exit 1
}
/home/n11373598/.pixi/bin/pixi run scr-eval --load-config outputs/aachen11/scrfacto/fixed/config.yml --split test