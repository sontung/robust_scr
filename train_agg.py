import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import scipy
import torch
import torchvision
from PIL import Image
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm, trange

from utils import set_seed, whitenapply, pcawhitenlearn
from dataset import CamLocDatasetConfig
from salad_model import FullModel

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
DEBUG_MODE = 0


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.5, binary=False, cosine=False):
        print("Using Contrastive Loss")
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.binary = binary
        self.cosine = cosine
        if self.cosine:
            self.distance = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            print("Using cosine distance")
        else:
            print("Using dot product distance")
            self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, out0, out1, label):
        if self.binary:
            label[label > 0] = 1
        gt = label.float()
        if self.cosine:
            dist = 1 - self.distance(out0, out1).float().squeeze()
        else:
            dist = self.distance(out0, out1).float().squeeze()
        loss = gt * 0.5 * torch.pow(dist, 2) + (1 - gt) * 0.5 * torch.pow(
            torch.clamp(self.margin - dist, min=0.0), 2
        )
        return loss


def normalize(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)


def process(ds2, model, image_root_dir):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    mat = np.zeros((len(ds2.rgb_files), 8448))
    with torch.no_grad():
        for idx, name in enumerate(tqdm(ds2.rgb_files)):
            image = Image.open(image_root_dir / name).convert("RGB")
            image = transform(image)
            image_descriptor = model(image.unsqueeze(0).cuda())
            image_descriptor = image_descriptor.squeeze().cpu().numpy()
            mat[idx] = image_descriptor
    return mat


def extract(random_ints, pairs, scores):
    pos_pairs = pairs[:, random_ints]
    if scores is None:
        return pos_pairs, torch.zeros((pos_pairs.size(1),), dtype=torch.float32)
    pos_scores = torch.tensor(scores[random_ints])
    return pos_pairs, pos_scores


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        pose_graph_file,
        image_dir=None,
        batch_size=256,
        nb_iterations=1000,
        hard_mining=True,
        train=False,
    ):
        self.hard_negative_pairs = None
        self.soft_negative_prob = None
        self.positive_prob = None
        self.prob = None
        self.salad_db_desc = None
        self.num_pairs = None
        self.negative_prob = None
        graph = scipy.sparse.load_npz(pose_graph_file).tocoo()
        print(f"Loaded graph file at {pose_graph_file}")
        self.edge_index = torch.tensor(
            np.stack([graph.row, graph.col]), dtype=torch.long
        )
        self.data = graph.data

        self.positive_pairs = self.edge_index[:, self.data > 0.5]
        self.positive_scores = self.data[self.data > 0.5]
        self.soft_negative_pairs = self.edge_index[
            :, (self.data >= 0.25) & (self.data <= 0.5)
        ]
        self.soft_negative_scores = self.data[(self.data >= 0.25) & (self.data <= 0.5)]
        self.hard_mining = hard_mining
        self.batch_size = batch_size

        self.all_indices = torch.arange(np.max(graph.row) + 1)
        self.csr_arr = graph.tocsr()
        self.nb_iterations = nb_iterations

        ds = self.setup_dataset(root_dir)
        if image_dir is not None:
            self.ori_ds_dir = str(image_dir)
        else:
            self.ori_ds_dir = str(root_dir / "../../..")

        self.rgb_files = ds.rgb_files

        salad_db_dir = root_dir / "../desc_salad_db.npy"
        self.get_pretrained_gl_desc(root_dir, salad_db_dir, ds)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )
        if train:
            self.compute_hard_mining_probs()
            self.all_images = self.read_all_images()

    def compute_hard_mining_probs(self):
        self.prob = self.compute_dist_matrix()
        self.positive_prob = self.prob[self.data > 0.5]
        self.soft_negative_prob = self.prob[(self.data >= 0.25) & (self.data <= 0.5)]

        self.num_pairs = len(self.all_indices) * len(self.all_indices)
        self.negative_prob, self.hard_negative_pairs = self.pre_mine_hard_negatives()
        self.negative_prob = self.negative_prob / self.negative_prob.sum()

    def pre_mine_hard_negatives(self):

        desc = self.salad_db_desc.numpy().astype(np.float32)
        cpu_index = faiss.IndexFlatL2(desc.shape[1])
        cpu_index.add(desc)
        distances, indices = cpu_index.search(desc, 21)

        first_col = indices[:, 0:1]  # shape (4328, 1)
        rest_cols = indices[:, 1:]  # shape (4328, 20)
        first_col_repeated = np.repeat(
            first_col, rest_cols.shape[1], axis=1
        )  # (4328, 20)
        pairs = np.stack([first_col_repeated, rest_cols], axis=2)  # (4328, 20, 2)
        pairs = pairs.reshape(-1, 2)
        paired_distances = distances[:, 1:].reshape(-1)

        scores = [self.csr_arr[pair[0], pair[1]] for pair in pairs]
        scores = np.array(scores)
        mask = scores < 0.2
        pairs = pairs[mask]
        paired_distances = paired_distances[mask]
        all_distances = torch.tensor(paired_distances, dtype=torch.float32)
        all_pairs = torch.tensor(pairs, dtype=torch.long)
        epsilon = 1e-8  # to avoid division by zero

        # Invert distances to make small distances correspond to high scores
        inv_scores = 1.0 / (all_distances + epsilon)
        return inv_scores, all_pairs.T

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
            self.salad_db_desc = process(ds, model, root_dir / "../../..")
            np.save(salad_db_dir, self.salad_db_desc)
        else:
            self.salad_db_desc = np.load(salad_db_dir)
        self.salad_db_desc = torch.tensor(self.salad_db_desc)

    def compute_dist_matrix(self):
        results = []
        B = 4096
        for i in range(0, self.edge_index.shape[1], B):
            idx0 = self.edge_index[0, i : i + B]
            idx1 = self.edge_index[1, i : i + B]
            desc0 = self.salad_db_desc[idx0]
            desc1 = self.salad_db_desc[idx1]
            sq_norm = ((desc0 - desc1) ** 2).sum(dim=1)
            results.append(sq_norm)
        distance_matrix = torch.cat(results, dim=0)
        return distance_matrix

    def __len__(self):
        return self.nb_iterations

    def sample_random_pair(self):

        # Sample a random index
        rand_idx = torch.randint(0, self.num_pairs, (1,)).item()

        # Compute indices in list_a and list_b
        i = rand_idx // len(self.all_indices)
        j = rand_idx % len(self.all_indices)

        pair = (self.all_indices[i].item(), self.all_indices[j].item())
        return pair

    def read_all_images(self):
        mat = torch.zeros((len(self.rgb_files), 3, 224, 224), dtype=torch.float32)
        for idx in trange(len(self.rgb_files), desc="Reading all images"):
            frame_path = str(self.rgb_files[idx])
            image_ori = Image.open(f"{self.ori_ds_dir}/{frame_path}")
            image = self.image_transform(image_ori)
            mat[idx] = image
        return mat

    def read_image(self, indices):
        imgs = []
        for idx2 in indices:
            image = self.all_images[idx2]
            imgs.append(image)
        return torch.stack(imgs)

    def compute_dist(self, i0, i1):
        dist = self.salad_db_desc[i0] - self.salad_db_desc[i1]
        return torch.dot(dist, dist)

    def sample_hard_pairs(self):
        nb_positives = self.batch_size // 3

        pos_pairs, pos_scores = extract(
            torch.multinomial(self.positive_prob, nb_positives),
            self.positive_pairs,
            self.positive_scores,
        )

        soft_neg_pairs, soft_neg_scores = extract(
            torch.multinomial(self.soft_negative_prob, nb_positives),
            self.soft_negative_pairs,
            self.soft_negative_scores,
        )

        nb_negatives = self.batch_size - nb_positives * 2
        neg_pairs, neg_scores = extract(
            torch.multinomial(self.negative_prob, nb_negatives),
            self.hard_negative_pairs,
            None,
        )

        batch = torch.cat(
            [pos_pairs, neg_pairs, soft_neg_pairs], dim=1
        )  # shape: (batch_size, 2)
        all_scores = torch.cat([pos_scores, neg_scores, soft_neg_scores], dim=0)
        return batch, all_scores

    def batch_sampling1(self):
        batch, all_scores = [], []
        while len(batch) < self.batch_size // 2:
            i0, i1 = self.sample_random_pair()
            if i0 == i1:
                continue
            if self.csr_arr[i0, i1] == 0.0:
                batch.append([i0, i1])
                all_scores.append(0.0)

        pos_pairs, pos_scores = extract(
            torch.randint(0, self.positive_pairs.size(1), (self.batch_size // 2,)),
            self.positive_pairs,
            self.positive_scores,
        )

        neg_pairs, neg_scores = extract(
            torch.randint(0, self.soft_negative_pairs.size(1), (self.batch_size // 2,)),
            self.soft_negative_pairs,
            self.soft_negative_scores,
        )

        batch.extend(pos_pairs.T)
        batch.extend(neg_pairs.T)
        all_scores.extend(pos_scores)
        all_scores.extend(neg_scores)

        batch = torch.tensor(batch)
        all_scores = torch.tensor(all_scores)
        return batch, all_scores

    def __getitem__(self, idx):
        batch, all_scores = self.batch_sampling1()
        img_indices = torch.unique(batch)

        sorted_indices, sorted_pos = img_indices.sort()
        mapping = torch.full(
            (sorted_indices.max().item() + 1,),
            -1,
            dtype=torch.long,
            device=img_indices.device,
        )
        mapping[sorted_indices] = torch.arange(
            len(sorted_indices), device=img_indices.device
        )
        remapped_pairs = mapping[batch]
        images = self.read_image(img_indices)
        return remapped_pairs, all_scores, images, img_indices


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
        "--dino_arch",
        default="dinov2_vitb14",
        type=str,
    )
    global DEBUG_MODE
    args = parser.parse_args()
    ds_path = Path(args.ds_path)
    model_name = args.model_name
    batch_size = args.batch_size
    nb_iter = args.nb_iter
    DEBUG_MODE = args.debug_mode
    hard_mining = args.hard_mining
    nb_clusters = args.nb_clusters
    learning_rate = args.lr
    weight_decay = args.weight_decay
    dino_arch = args.dino_arch
    cosine = args.cosine
    nb_trainable_blocks = args.nb_trainable_blocks

    loss = ContrastiveLoss(0.5, binary=False, cosine=True)

    loss.cuda()
    whiten = True
    if DEBUG_MODE == 1:
        nb_iter = 10
        batch_size = 64
        ds_path = Path("/home/n11373598/hpc-home/work/scrstudio_exp/data/dept/1F")

    graph_file = str(ds_path / "train/pose_overlap.npz")
    train_data = SampleDataset(
        ds_path / "train",
        graph_file,
        batch_size=batch_size,
        nb_iterations=nb_iter,
        hard_mining=bool(hard_mining),
        train=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=4
    )

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
    print(
        f"Using optimizer {optimizer} with lr {learning_rate} and weight decay {weight_decay}"
    )
    print(f"Batch size={batch_size}")
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.2,
        total_iters=nb_iter // nb_accumulation_steps,
    )
    init_step = 0
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    pbar = tqdm(range(nb_iter))
    dim = None
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
    model.eval()
    train_data = SampleDataset(
        ds_path / "train", graph_file, batch_size=batch_size, nb_iterations=nb_iter
    )
    mat1 = test(train_data, model, dim)

    train_data = SampleDataset(
        ds_path / "test", graph_file, batch_size=batch_size, nb_iterations=nb_iter
    )
    mat2 = test(train_data, model, dim)

    train_data = SampleDataset(
        ds_path / "val", graph_file, batch_size=batch_size, nb_iterations=nb_iter
    )
    mat3 = test(train_data, model, dim)

    if whiten and dim > 256:
        if dim > 1000:
            from cuml import PCA

            pca_torch = PCA(n_components=256, copy=False)
            pca_torch.fit(mat1)
            print(f"Explained variance: {pca_torch.explained_variance_ratio_.sum()}")
            mat1 = pca_torch.transform(mat1)
            mat2 = pca_torch.transform(mat2)
            mat3 = pca_torch.transform(mat3)
        else:
            m, p = pcawhitenlearn(mat1.T)
            mat1 = whitenapply(mat1.T, m, p, dimensions=256)
            mat2 = whitenapply(mat2.T, m, p, dimensions=256)
            mat3 = whitenapply(mat3.T, m, p, dimensions=256)
            mat1 = mat1.T
            mat2 = mat2.T
            mat3 = mat3.T

    print(f"Obtained global descriptors of shape {mat1.shape} and {mat2.shape}")
    np.save(f"checkpoints/{model_name}.npy", mat1)
    np.save(f"checkpoints/{model_name}_test.npy", mat2)
    np.save(f"checkpoints/{model_name}_val.npy", mat3)
    print(f"Saved descriptors to checkpoints/{model_name}.npy")


def test(train_data, model, dim):
    mat = np.zeros((len(train_data.rgb_files), dim))
    start = time.time()
    with torch.no_grad():
        for idx in trange(len(train_data.rgb_files), desc="Inferring descriptors"):
            frame_path = str(train_data.rgb_files[idx])

            image_ori = Image.open(f"{train_data.ori_ds_dir}/{frame_path}")
            image = train_data.image_transform(image_ori)
            image = image.unsqueeze(0).cuda()
            emb = model(image)
            mat[idx] = emb.cpu().numpy()
    end = time.time()
    print(
        f"Time taken: {end - start:.2f} seconds for {len(train_data.rgb_files)} images"
    )
    return mat


if __name__ == "__main__":
    # pca()
    set_seed(1000)
    main_loop()
