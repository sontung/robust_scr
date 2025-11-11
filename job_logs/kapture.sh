#!/bin/bash -l
#PBS -N kapture
#PBS -l select=1:ncpus=5:ngpus=1:mem=50:qlist=qvpr
#PBS -l walltime=20:00:00
#PBS -j oe

set -e  # Exit on error

# Note: gputype line is optional! Delete if any gpu is fine.

# Activate your conda environment (should already be created and all packages installed)
mamba activate scrstudio


cd /home/n11373598/work/kapture-localization/pipeline/examples
./run_hyundai_dept_store.sh
