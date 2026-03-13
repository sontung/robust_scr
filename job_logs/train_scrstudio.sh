#!/bin/bash
#SBATCH --job-name=scr_studio
#SBATCH --partition=main
#SBATCH --nodelist=worker-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%3j_%x.out
#SBATCH --error=%3j_%x.err

DATA_PATH=/mnt/data/sftp/data/tungns30/aachen10

cd /home/tungns30/robust_scr

#/home/tungns30/.pixi/bin/pixi run scr-train node2vec --data $DATA_PATH --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2
/home/tungns30/.pixi/bin/pixi run scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data $DATA_PATH
cp outputs/aachen10/node2vec/fixed/scrstudio_models/head.pt /mnt/data/sftp/data/tungns30/aachen10/train/pose_n2c.pt
/home/tungns30/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt || {
  echo "Python crashed!"
  exit 1
}
#/home/tungns30/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq || {
#  echo "Python crashed!"
#  exit 1
#}
#/home/tungns30/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test || {
#  echo "Python crashed!"
#  exit 1
#}
#/home/tungns30/.pixi/bin/pixi run scr-eval --load-config outputs/aachen10/scrfacto/fixed/config.yml --split test

#/home/tungns30/.pixi/bin/pixi run scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data $DATA_PATH
#/home/tungns30/.pixi/bin/pixi run scr-overlap-score --data $DATA_PATH/train --max_depth 50
#/home/tungns30/.pixi/bin/pixi run scr-train node2vec --data $DATA_PATH --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2
#/home/tungns30/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train
#/home/tungns30/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test