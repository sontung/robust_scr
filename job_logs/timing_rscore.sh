#!/bin/bash -l
#PBS -N timing_r
#PBS -l select=1:ncpus=20:ngpus=1:mem=50GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

cd /home/n11373598/work/scrstudio_me
DATA_PATH=../scrstudio_exp/data/data/aachen

/home/n11373598/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt || {
  echo "Python crashed!"
  exit 1
}
/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq
/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test
/home/n11373598/.pixi/bin/pixi run scr-eval --load-config outputs/aachen/scrfacto/fixed/config.yml --split test


#DATA_PATH=../scrstudio_exp/data/dept/4F
#/home/n11373598/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt || {
#  echo "Python crashed!"
#  exit 1
#}
#/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq
#/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test
#/home/n11373598/.pixi/bin/pixi run scr-eval --load-config outputs/4F/scrfacto/fixed/config.yml --split test

#DATA_PATH=../scrstudio_exp/data/dept/B1
#/home/n11373598/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt || {
#  echo "Python crashed!"
#  exit 1
#}
#/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq
#/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test
#/home/n11373598/.pixi/bin/pixi run scr-eval --load-config outputs/B1/scrfacto/fixed/config.yml --split test

DATA_PATH=../scrstudio_exp/data/dept/1F
#/home/n11373598/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt || {
#  echo "Python crashed!"
#  exit 1
#}
##/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq
##/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test
#/home/n11373598/.pixi/bin/pixi run scr-eval --load-config outputs/1F/scrfacto/fixed/config.yml --split test