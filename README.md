
# Installation
### If you are using [Pixi](https://pixi.sh/latest/installation/):
```shell
cd env_pixi
pixi install
pixi shell # activate the environment
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

### Otherwise, plain Conda:
```shell
conda create -n env2 python=3.10 pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.4 cuml=25.02 -c pytorch  -c rapidsai -c conda-forge -c nvidia
conda activate env2
conda install -c conda-forge pytorch-metric-learning h5py pykdtree poselib scikit-image scikit-learn python-lmdb

pip install --upgrade pip
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install opencv-python

pip install faiss-gpu nanopq kornia
pip install open3d
```

# Datasets
### Install scrstudio to download and process datasets:
```shell
git clone --recursive https://github.com/cvg/scrstudio.git
cd scrstudio
pip install --upgrade pip setuptools
pip install -e .
```
### Hyundai department store:
```shell
scr-download-data naver --capture-name dept_1F
scr-overlap-score --data ../scrstudio_exp/data/dept/4F/train --max_depth 8
scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data ../scrstudio_exp/data/dept/4F
scr-retrieval-feat --data ../scrstudio_exp/data/dept/4F/train
scr-retrieval-feat --data ../scrstudio_exp/data/dept/4F/val
scr-retrieval-feat --data ../scrstudio_exp/data/dept/4F/test
``` 

### Aachen day/night:
PGT poses for Aachen day/night test set can be downloaded from [here](https://drive.google.com/file/d/1Qj2vvcv68_EnniW4QChmx3bf5RaQKW99/view?usp=sharing) (useful for validating without submitting to benchmark website).

```shell
scr-download-data aachen
scr-overlap-score --data ../scrstudio_exp/data/aachen/train --max_depth 50
scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data ../scrstudio_exp/data/aachen
``` 

### Oxford RobotCar:
```shell
scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data /home/n11373598/hpc-home/work/glace_experiment/datasets/robotcar
scr-overlap-score --data /home/n11373598/hpc-home/work/glace_experiment/datasets/robotcar/train --max_depth 50
scr-train node2vec --data /home/n11373598/hpc-home/work/glace_experiment/datasets/robotcar --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2
scr-retrieval-feat --data /home/n11373598/hpc-home/work/glace_experiment/datasets/robotcar/train
scr-retrieval-feat --data /home/n11373598/hpc-home/work/glace_experiment/datasets/robotcar/test
```

### Aachen v1.1:
```shell
scr-encoding-pca dedode --encoder.detector L --encoder.descriptor B --n_components 128 --data /home/n11373598/hpc-home/work/glace_experiment/datasets/aachen11
scr-overlap-score --data /home/n11373598/hpc-home/work/glace_experiment/datasets/aachen11/train --max_depth 50
scr-train node2vec --data /home/n11373598/hpc-home/work/glace_experiment/datasets/aachen11 --pipeline.model.graph pose_overlap.npz --pipeline.model.edge_threshold 0.2
scr-retrieval-feat --data /home/n11373598/hpc-home/work/glace_experiment/datasets/aachen11/train
scr-retrieval-feat --data /home/n11373598/hpc-home/work/glace_experiment/datasets/aachen11/test
```