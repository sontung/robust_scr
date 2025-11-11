import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm, trange

from utils import set_seed, whitenapply, pcawhitenlearn
from dataset import CamLocDatasetConfig
from salad_model import FullModel, FullModel_DINOV3
from train_agg import SampleDataset, process, ContrastiveLoss, test

DEBUG_MODE = 0


class SampleDatasetAachen(SampleDataset):
    def __init__(
        self,
        root_dir,
        pose_graph_file,
        image_dir,
        batch_size=256,
        nb_iterations=1000,
        hard_mining=True,
        train=False,
    ):
        super().__init__(
            root_dir,
            pose_graph_file,
            image_dir,
            batch_size,
            nb_iterations,
            hard_mining,
            train,
        )

    def read_all_images(self):
        mat = torch.zeros((len(self.rgb_files), 3, 224, 224), dtype=torch.float32)
        for idx in trange(len(self.rgb_files), desc="Reading all images"):
            frame_path = str(self.rgb_files[idx])
            image_ori = Image.open(f"{self.ori_ds_dir}/{frame_path}")
            image = self.image_transform(image_ori)
            mat[idx] = image
        return mat

    @staticmethod
    def setup_dataset(root_dir):
        train_config = CamLocDatasetConfig(
            data=root_dir / "..",
            split=str(root_dir).split("/")[-1],
            feat_name="nothing",
            num_decoder_clusters=50,
            loading_depth=False,
        )
        ds = train_config.setup()
        return ds

    def get_pretrained_gl_desc(self, root_dir, salad_db_dir, ds):
        if not salad_db_dir.exists():
            model = FullModel(pretrained=True)
            model.cuda()
            model.eval()
            self.salad_db_desc = process(ds, model, Path(self.ori_ds_dir))
            np.save(salad_db_dir, self.salad_db_desc)
        else:
            self.salad_db_desc = np.load(salad_db_dir)
        self.salad_db_desc = torch.tensor(self.salad_db_desc)


def train_loop(
    train_data, optimizer, scheduler, model, loss, init_step, nb_accumulation_steps=1
):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False
    )
    data_iter = iter(train_loader)
    dim = None
    pbar = tqdm(range(train_data.nb_iterations))
    while True:
        try:
            optimizer.zero_grad()
            batch = next(data_iter)
            pairs = batch[0].squeeze(0).cuda()
            scores = batch[1].squeeze(0).cuda()
            images = batch[2].squeeze().cuda()
            embs = model(images)
            if dim is None:
                dim = embs.shape[1]
            embs_paired = embs[pairs]
            error = loss(embs_paired[:, 0], embs_paired[:, 1], scores)
            null_losses = torch.sum(error == 0).item() / len(error)
            error = torch.mean(error) / nb_accumulation_steps
            error.backward()
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{error.item():.3f}",
                batch_acc=f"{null_losses:.3f}",
                epoch=f"{init_step}",
            )
            optimizer.step()
            scheduler.step()
            init_step += 1
        except StopIteration:
            break
    pbar.close()
    return dim


def main_loop():
    parser = argparse.ArgumentParser(description="Enter a directory path.")
    parser.add_argument("ds_path", type=str, help="Path to the directory")
    parser.add_argument("debug_mode", type=int, help="Path to the directory")
    parser.add_argument("model_name", type=str, help="Path to the directory")
    parser.add_argument("batch_size", type=int, help="Path to the directory")
    parser.add_argument("nb_iter", type=int, help="Path to the directory")
    parser.add_argument("hard_mining", type=int, help="Path to the directory")
    parser.add_argument("nb_clusters", type=int, help="Path to the directory")
    parser.add_argument("lr", type=float, help="Path to the directory")
    parser.add_argument("weight_decay", type=float, help="Path to the directory")
    parser.add_argument("cosine", type=int, help="Path to the directory")
    parser.add_argument("nb_trainable_blocks", type=int, help="Path to the directory")
    parser.add_argument(
        "--loss",
        default="original",
        type=str,
    )
    parser.add_argument(
        "--dino_arch",
        default="dinov2_vitb14",
        type=str,
    )
    parser.add_argument(
        "--margin_t",
        default=0.5,
        type=float,
    )

    args = parser.parse_args()
    ds_path = Path(args.ds_path)
    model_name = args.model_name
    batch_size = args.batch_size
    nb_iter = args.nb_iter
    debug_mode = args.debug_mode
    hard_mining = bool(args.hard_mining)
    nb_clusters = args.nb_clusters
    learning_rate = args.lr
    weight_decay = args.weight_decay
    dino_arch = args.dino_arch
    nb_trainable_blocks = args.nb_trainable_blocks
    cosine = args.cosine
    margin_t = args.margin_t
    graph_file = str(ds_path / "train/pose_overlap.npz")

    loss = ContrastiveLoss(margin_t, binary=False, cosine=cosine)
    print(f"Using loss with margin {margin_t}")

    loss.cuda()
    if debug_mode == 1:
        nb_iter = 10
        batch_size = 64
        ds_path = Path("../scrstudio/data/aachen")
        global DEBUG_MODE
        DEBUG_MODE = 1

    if "dinov3" in dino_arch:
        model = FullModel_DINOV3(nb_clusters=nb_clusters)
    else:
        model = FullModel(
            nb_clusters=nb_clusters,
            dino_arch=dino_arch,
            trainable_blocks=nb_trainable_blocks,
        )
    model.cuda()
    model.train()

    nb_accumulation_steps = 1
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    print(f"Using lr {learning_rate} and weight decay {weight_decay}")
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.2,
        total_iters=nb_iter // nb_accumulation_steps,
    )
    init_step = 0
    optimizer.zero_grad()
    train_data = SampleDatasetAachen(
        ds_path / "train",
        graph_file,
        image_dir=ds_path / "train" / "../images_upright",
        batch_size=batch_size,
        nb_iterations=nb_iter,
        hard_mining=bool(hard_mining),
        train=True,
    )
    dim = train_loop(train_data, optimizer, scheduler, model, loss, init_step)

    model.eval()
    train_data = SampleDatasetAachen(
        ds_path / "train",
        graph_file,
        image_dir=ds_path / "train" / "../images_upright",
        batch_size=batch_size,
        nb_iterations=nb_iter,
    )
    mat1 = test(train_data, model, dim)

    train_data = SampleDatasetAachen(
        ds_path / "test",
        graph_file,
        batch_size=batch_size,
        nb_iterations=nb_iter,
        image_dir=ds_path / "train" / "../images_upright",
    )
    mat2 = test(train_data, model, dim)

    if dim > 256:
        if dim > 1000:
            from cuml import PCA

            pca_torch = PCA(n_components=256, copy=False)
            pca_torch.fit(mat1)
            mat1 = pca_torch.transform(mat1)
            mat2 = pca_torch.transform(mat2)
        else:
            m, p = pcawhitenlearn(mat1.T)
            mat1 = whitenapply(mat1.T, m, p, dimensions=256)
            mat2 = whitenapply(mat2.T, m, p, dimensions=256)
            mat1 = mat1.T
            mat2 = mat2.T

    np.save(f"checkpoints/{model_name}.npy", mat1)
    np.save(f"checkpoints/{model_name}_test.npy", mat2)


if __name__ == "__main__":
    # pca()
    set_seed(1000)
    main_loop()
