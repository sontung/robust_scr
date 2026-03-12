from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from dataset import CamLocDatasetConfig
from networks import get_model


def load():
    state_dict = torch.load("checkpoints/mlp.pt")
    node2vec_ori = torch.load("checkpoints/node2vec.pt")

    desc_node2vec = node2vec_ori["model.embedding.weight"]

    train_config = CamLocDatasetConfig(
        data=Path("/home/vr/work/datasets/aachen/aachen"),
        split="train",
    )

    raw_ds = train_config.setup()
    kmeans = KMeans(n_clusters=50, random_state=0).fit(
        raw_ds.pose_values[:, :3, 3].astype(np.float32))
    model = get_model(in_channels=384, head_channels=1280, metadata={"cluster_centers": torch.from_numpy(kmeans.cluster_centers_)})

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k[len("backbone."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    print()



if __name__ == '__main__':
    load()
