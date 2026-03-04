#!/bin/bash
#SBATCH --job-name=node2vec
#SBATCH --partition=main
#SBATCH --nodelist=worker-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%3j_%x.out
#SBATCH --error=%3j_%x.err

cd /home/tungns30/robust_scr
/home/tungns30/.pixi/bin/pixi run scr-train node2vec --data --data /mnt/data/sftp/data/tungns30/aachen10 --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2

