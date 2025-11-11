#!/bin/bash -l
#PBS -N robotcar
#PBS -l select=1:ncpus=20:ngpus=1:mem=100GB:gpu_id=H100
#PBS -l walltime=48:00:00
#PBS -j oe

set -e  # Exit on error

DATA_PATH=../glace_experiment/datasets/robotcar
cd /home/n11373598/work/glace_experiment
/home/n11373598/.pixi/bin/pixi run python prepare_dataset.py

cd /home/n11373598/work/scrstudio_me

#/home/n11373598/.pixi/bin/pixi run scr-train scrfacto --data $DATA_PATH --pipeline.datamanager.train_dataset.feat_name pose_n2c.pt
#/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train --pq
#/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test
#/home/n11373598/.pixi/bin/pixi run scr-eval --load-config outputs/aachen11/scrfacto/fixed/config.yml --split test


/home/n11373598/.pixi/bin/pixi run scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data $DATA_PATH
/home/n11373598/.pixi/bin/pixi run scr-overlap-score --data $DATA_PATH/train --max_depth 50
/home/n11373598/.pixi/bin/pixi run scr-train node2vec --data $DATA_PATH --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2
/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/train
/home/n11373598/.pixi/bin/pixi run scr-retrieval-feat --data $DATA_PATH/test