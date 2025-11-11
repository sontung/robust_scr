import functools
import gc
import logging
import os
import pickle
import time
from collections import defaultdict, Counter
from multiprocessing import Pool

import cv2
import matplotlib
import torch
from pykdtree.kdtree import KDTree
from scipy.sparse import load_npz
from torch import autocast
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
from tqdm import trange

from base_trainer import BaseTrainer
from rscore_loss import get_losses
from utils import (
    get_options,
    pose_estimate,
    CSRGraph,
    run_salad_model, read_nvm_file,
)
from config_classes import BatchRandomSamplerConfig, PQKNN
from dataset import get_dataset
from encoder import get_encoder
from networks import (
    PositionRefinerConfig,
    ResBlockConfig,
    PositionEncoderConfig,
    PositionDecoderConfig,
    InputBlockConfig,
    BlockListConfig,
)

matplotlib.use("Agg")  # use non-interactive backend
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import CamLocDatasetConfig

_logger = logging.getLogger(__name__)


class TrainerACE(BaseTrainer):
    def __init__(
        self,
        options_,
        h_c=768,
        weight_file="/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_b1.pth",
    ):
        super().__init__(options_)
        self.feature_dim = self.encoder.out_channels
        self.global_feat_dim = self.dataset.global_feat_dim
        head_channels = self.feature_dim + self.global_feat_dim
        self.buffer_size_dim = self.feature_dim

        self.h_c = h_c
        self.head = self.create_head_network(head_channels)

        state_dict = torch.load(
            weight_file,
            weights_only=True,
        )

        # Strip 'backbone.' prefix from keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("backbone.", "")  # remove prefix
            new_state_dict[new_k] = v

        self.head.load_state_dict(new_state_dict, strict=False)

        print(f"Loaded pretrained weights for head.")
        self.head.eval()
        self.head.cuda()

        self.ds_dir = str(self.options.scene)
        pose_graph_dir = self.options.scene / "train/pose_overlap.npz"
        print(f"Load membership from {pose_graph_dir}")
        covis_score = load_npz(pose_graph_dir)
        covis_score.data[covis_score.data < 0.2] = 0
        covis_score.setdiag(1)
        covis_score.eliminate_zeros()
        covis_score = covis_score.tocsr()
        self.covis_graph = CSRGraph.from_csr_array(covis_score, self.device)
        self.loss_functions = get_losses(self.options.max_iterations, db_std=3)
        self.gradient_accumulation_steps = self.options.grad_acc

        sampler_config = BatchRandomSamplerConfig(batch_size=self.options.batch_size)
        self.batch_sampler = iter(
            sampler_config.setup(
                dataset_size=self.options.training_buffer_size,
                generator=self.training_generator,
            )
        )

        train_config = CamLocDatasetConfig(
            data=self.options.scene,
            split="train",
        )

        raw_ds = train_config.setup()
        run_salad_model(raw_ds, self.options.scene / "images_upright")
        run_salad_model(self.test_dataset, self.options.scene / "images_upright")

        save_path = "pid_xyz.pkl"

        if os.path.exists(save_path):
            print(f"[INFO] Loading existing {save_path}...")
            with open(save_path, "rb") as f:
                data = pickle.load(f)
        else:
            (
                self.xyz_arr,
                self.image2points,
                self.image2name,
                self.image2info,
                self.image2uvs,
                self.image2pose,
            ) = read_nvm_file(self.options.scene / "aachen_cvpr2018_db.nvm")
            import open3d as o3d
            # Filter with Open3D
            pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz_arr))
            cl, inlier_ind = pc1.remove_radius_outlier(
                nb_points=16, radius=5, print_progress=True
            )

            xyz = self.xyz_arr[inlier_ind]

            # Map pid -> images
            pid2images = {}
            inlier_set = set(inlier_ind)
            for i, pts in tqdm(self.image2points.items()):
                for p in pts:
                    if p in inlier_set:
                        pid2images.setdefault(p, []).append(i)

            # Save
            data = {
                "pid": inlier_ind,
                "xyz": xyz,
                "pid2images": pid2images,
                "image2name": self.image2name
            }
            with open(save_path, "wb") as f:
                pickle.dump(data, f)

            print(f"[INFO] Saved to {save_path}")
        self.pid_list = np.array(data["pid"])
        self.xyz = data["xyz"]
        self.pid2images = data["pid2images"]
        self.image2name = data["image2name"]

    def create_training_buffer(self):
        buffer_dir = f"checkpoints/training_buffer_{self.options.training_buffer_size}/{str(self.options.scene).split('/')[-1]}"
        os.makedirs(buffer_dir, exist_ok=True)

        required_keys = [
            "features",
            "target_px",
            "gt_poses_inv",
            "gt_poses",
            "intrinsics",
            "intrinsics_inv",
            "img_idx",
            "sample_idx",
            "depths",
        ]

        buffer_exists = all(
            os.path.exists(os.path.join(buffer_dir, f"{k}.pt")) for k in required_keys
        )

        if buffer_exists:
            print(f"Loading training buffer from {buffer_dir} (to {self.device})...")
            self.training_buffer = {
                k: torch.load(
                    os.path.join(buffer_dir, f"{k}.pt"), map_location="cpu"
                ).to(self.device)
                for k in required_keys
            }
            print("Training buffer loaded successfully.")
            return
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        batch_sampler = sampler.RandomSampler(
            self.dataset, generator=self.batch_generator
        )

        training_dataloader = DataLoader(
            dataset=self.dataset,
            sampler=batch_sampler,
            batch_size=1,
            drop_last=False,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            timeout=120 if self.num_data_loader_workers > 0 else 0,
        )

        assert self.global_feats.shape[0] == len(self.dataset.rgb_files)

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            "features": torch.empty(
                (self.options.training_buffer_size, self.buffer_size_dim),
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=self.device,
            ),
            "target_px": torch.empty(
                (self.options.training_buffer_size, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses_inv": torch.empty(
                (self.options.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses": torch.empty(
                (self.options.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics": torch.empty(
                (self.options.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics_inv": torch.empty(
                (self.options.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "img_idx": torch.empty(
                (self.options.training_buffer_size,),
                dtype=torch.int32,
                device=self.device,
            ),
            "sample_idx": torch.empty(
                (self.options.training_buffer_size,),
                dtype=torch.int32,
                device=self.device,
            ),
            "depths": torch.empty(
                (self.options.training_buffer_size,),
                dtype=torch.float32,
                device=self.device,
            ),
        }

        # Iterate until the training buffer is full.
        buffer_idx = 0
        pbar = tqdm(total=self.options.training_buffer_size, desc="Filling buffer")
        example_idx = 0

        while buffer_idx < self.options.training_buffer_size:
            for batch in training_dataloader:
                if len(batch) == 10:
                    (
                        image_B1HW,
                        image_ori,
                        image_mask_B1HW,
                        gt_pose_B44,
                        gt_pose_inv_B44,
                        intrinsics_B33,
                        intrinsics_inv_B33,
                        global_feat,
                        idx,
                        filename,
                    ) = batch.values()
                    depths = None
                else:
                    (
                        image_B1HW,
                        image_ori,
                        image_mask_B1HW,
                        gt_pose_B44,
                        gt_pose_inv_B44,
                        intrinsics_B33,
                        intrinsics_inv_B33,
                        depths,
                        global_feat,
                        idx,
                        filename,
                    ) = batch.values()
                    depths = depths.to(self.device, non_blocking=True)

                image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                gt_pose_B44 = gt_pose_B44.to(self.device, non_blocking=True)
                intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                intrinsics_inv_B33 = intrinsics_inv_B33.to(
                    self.device, non_blocking=True
                )

                image_B1HW = image_B1HW.to(self.device, non_blocking=True)

                with torch.no_grad():
                    with autocast(enabled=self.options.use_half, device_type="cuda"):
                        encoder_output = self.encoder.keypoint_features(
                            {"image": image_B1HW, "mask": image_mask_B1HW},
                            n=self.options.samples_per_image,
                            generator=self.sampling_generator,
                        )

                target_pix = encoder_output["keypoints"]
                feat = encoder_output["descriptors"]
                nb_features = target_pix.shape[0]

                image_mask_B1HW = image_mask_B1HW.bool()

                # If the current mask has no valid pixels, continue.
                if image_mask_B1HW.sum() == 0:
                    continue

                # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                gt_pose_inv = gt_pose_inv_B44[:, :3]
                gt_pose_inv = (
                    gt_pose_inv.unsqueeze(1)
                    .expand(1, nb_features, 3, 4)
                    .reshape(-1, 3, 4)
                )

                gt_pose = gt_pose_B44[:, :3]
                gt_pose = (
                    gt_pose.unsqueeze(1).expand(1, nb_features, 3, 4).reshape(-1, 3, 4)
                )

                # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                intrinsics = (
                    intrinsics_B33.unsqueeze(1)
                    .expand(1, nb_features, 3, 3)
                    .reshape(-1, 3, 3)
                )
                intrinsics_inv = (
                    intrinsics_inv_B33.unsqueeze(1)
                    .expand(1, nb_features, 3, 3)
                    .reshape(-1, 3, 3)
                )

                batch_data = {
                    "features": feat.cuda(),
                    "target_px": target_pix.cuda(),
                    "gt_poses_inv": gt_pose_inv,
                    "gt_poses": gt_pose,
                    "intrinsics": intrinsics,
                    "intrinsics_inv": intrinsics_inv,
                }

                image_mask_N1 = torch.ones(nb_features).cuda()
                # if depths is not None:
                #     kp = target_pix.int()
                #     gt_depths = depths[0, kp[:, 1], kp[:, 0]]
                #     image_mask_N1 *= (gt_depths > 0).float()

                if image_mask_N1.sum() == 0:
                    continue

                # Over-sample according to image mask.
                features_to_select = min(
                    self.options.samples_per_image * 1, nb_features
                )
                features_to_select = min(
                    features_to_select,
                    self.options.training_buffer_size - buffer_idx,
                )

                # Sample indices uniformly, with replacement.
                sample_idxs = torch.multinomial(
                    image_mask_N1.view(-1),
                    features_to_select,
                    replacement=True,
                    generator=self.sampling_generator,
                )

                # Select the data to put in the buffer.
                for k in batch_data:
                    batch_data[k] = batch_data[k][sample_idxs]

                # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                buffer_offset = buffer_idx + features_to_select
                for k in batch_data:
                    self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                img_idx = idx.item()
                self.training_buffer["img_idx"][buffer_idx:buffer_offset] = img_idx
                self.training_buffer["sample_idx"][
                    buffer_idx:buffer_offset
                ] = example_idx
                example_idx += 1
                kp = batch_data["target_px"].int()
                if depths is not None:
                    gt_depths = depths[0, kp[:, 1], kp[:, 0]]
                    self.training_buffer["depths"][buffer_idx:buffer_offset] = gt_depths
                assert img_idx < self.global_feats.shape[0]
                buffer_idx = buffer_offset
                pbar.update(features_to_select)

                if buffer_idx >= self.options.training_buffer_size:
                    break
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB

        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Reserved memory : {reserved:.2f} GB")

        example_indices = self.training_buffer["sample_idx"]
        nb_unique_examples = torch.max(example_indices).item() + 1
        light_buffer = {
            "gt_poses_inv": torch.zeros(
                (nb_unique_examples, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses": torch.zeros(
                (nb_unique_examples, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics": torch.zeros(
                (nb_unique_examples, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics_inv": torch.zeros(
                (nb_unique_examples, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
        }
        for example_idx in range(nb_unique_examples):
            mask = example_indices == example_idx
            for k in light_buffer:
                light_buffer[k][example_idx] = self.training_buffer[k][mask][0]
        for k in light_buffer:
            del self.training_buffer[k]
            gc.collect()
            torch.cuda.empty_cache()
            self.training_buffer[k] = light_buffer[k]
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB

        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Reserved memory : {reserved:.2f} GB")
        print(f"Saving training buffer to {buffer_dir} (one file per key on CPU)...")
        for k, v in self.training_buffer.items():
            torch.save(v.cpu(), os.path.join(buffer_dir, f"{k}.pt"))
        print("All buffer keys saved successfully.")
        pbar.close()

    def get_next_batch(self):
        random_batch_indices = next(self.batch_sampler)
        features_batch = torch.empty(
            (
                self.options.batch_size,
                self.feature_dim + self.global_feat_dim,
            ),
            dtype=self.training_buffer["features"].dtype,
            device=self.device,
        )

        img_idx = self.training_buffer["img_idx"][random_batch_indices].long()
        if self.options.graph_aug:
            num_neighbors = img_idx.shape[0] // 2
            img_idx[:num_neighbors] = self.covis_graph.sample_neighbors(
                img_idx[:num_neighbors], self.gn_generator
            )

        assert (
            torch.max(random_batch_indices).item()
            < self.training_buffer["features"].shape[0]
        )
        assert (
            torch.max(img_idx).item() < self.global_feats.shape[0]
        ), f"{torch.max(img_idx).item()} >= {self.global_feats.shape[0]}"

        features_batch[:, : self.global_feat_dim] = self.global_feats[img_idx]

        features_batch[:, self.global_feat_dim :] = self.training_buffer["features"][
            random_batch_indices
        ]
        dict_ = {"features": features_batch.contiguous()}
        dict_.update(
            {
                k: self.training_buffer[k][random_batch_indices].contiguous()
                for k in [
                    "target_px",
                    "img_idx",
                    "depths",
                ]
            }
        )
        random_batch_indices = self.training_buffer["sample_idx"][random_batch_indices]
        dict_.update(
            {
                k: self.training_buffer[k][random_batch_indices].contiguous()
                for k in [
                    "gt_poses",
                    "gt_poses_inv",
                    "intrinsics",
                    "intrinsics_inv",
                ]
            }
        )
        return dict_

    def run_epoch(self, pbar_, max_iterations=None):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Iterate with mini batches.
        self.optimizer.zero_grad()
        assert self.global_feats.shape[0] == len(self.dataset.rgb_files)

        for _ in range(self.gradient_accumulation_steps):
            dict_ = self.get_next_batch()
            loss_val, mean_re, ma1, mi1 = self.training_step(dict_)
            pbar_.set_postfix(
                loss=f"{loss_val*self.gradient_accumulation_steps:.1f}",
                re=f"{mean_re:.1f} {ma1:.1f} {mi1:.1f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.5f}",
            )

        self.iteration += 1
        old_optimizer_step = self.get_step_count()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        curr_step = self.get_step_count()
        if old_optimizer_step < curr_step < self.scheduler.total_steps:
            self.scheduler.step()

        pbar_.update(1)

    def training_step(
        self,
        dict_,
    ):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """

        (
            features_bC,
            target_px_b2,
            img_idx_b1,
            gt_depths_b1,
            gt_poses_b34,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
        ) = dict_.values()

        B = target_px_b2.shape[0]
        uv_h = torch.cat(
            [target_px_b2, torch.ones(B, 1, device=target_px_b2.device)], dim=1
        )  # (B, 3)
        cam_dirs = torch.bmm(invKs_b33, uv_h.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        cam_points = cam_dirs * gt_depths_b1.unsqueeze(1)  # (B, 3)
        R_inv = gt_poses_b34[:, :, :3]  # (B, 3, 3)
        t_inv = gt_poses_b34[:, :, 3]  # (B, 3)

        xyz_world = (
            torch.bmm(R_inv, cam_points.unsqueeze(-1)).squeeze(-1) + t_inv
        )  # (B, 3)
        xyz_world[(gt_depths_b1 == 0).flatten()] = 0

        if not self.options.graph_aug:
            features_bC = self.diffuse_feature(features_bC)

        with autocast(enabled=self.options.use_half, device_type="cuda"):
            outputs = self.head({"features": features_bC})
        nan_mask = torch.all(torch.bitwise_not(torch.isnan(outputs["sc"])), 1)

        outputs["target_px"] = target_px_b2[nan_mask]
        outputs["gt_poses_inv"] = gt_inv_poses_b34[nan_mask]
        outputs["intrinsics"] = Ks_b33[nan_mask]
        outputs["intrinsics_inv"] = invKs_b33[nan_mask]
        outputs["sc"] = outputs["sc"][nan_mask]
        outputs["sc0"] = outputs["sc0"][nan_mask]
        outputs["step"] = self.iteration
        outputs["gt_coords"] = xyz_world

        if torch.mean(nan_mask.float()) < 1:
            raise ValueError("NaN found in the output")
        for loss_function in self.loss_functions:
            outputs = loss_function(outputs)

        metrics_dict = outputs["metrics"]
        loss_dict = {
            "loss": metrics_dict["loss"],
        }
        with autocast(enabled=self.options.use_half, device_type="cuda"):
            loss = (
                functools.reduce(torch.add, loss_dict.values())
                / self.gradient_accumulation_steps
            )
        self.scaler.scale(loss).backward()

        return (
            loss.item(),
            metrics_dict["median_rep_error"].item(),
            outputs["sc"].max().item(),
            outputs["sc"].min().item(),
        )

    def get_gl_descriptors(self):
        if self.options.debug_mode:
            print("Using netvlad")
            # method = "mixvpr"
            # db_desc = np.load(f"checkpoints/{method}_train.npy")
            # test_desc = np.load(f"checkpoints/{method}_test.npy")
            # db_desc = np.load(self.dataset.config.data / "train/netvlad_feats.npy")
            # test_desc = np.load(self.dataset.config.data / "test/netvlad_feats.npy")
            db_desc = np.load(self.dataset.config.data / "train/desc_salad.npy")
            test_desc = np.load(self.dataset.config.data / "test/desc_salad.npy")
        else:
            if self.options.use_salad:
                db_desc = np.load(self.dataset.config.data / "train/desc_salad.npy")
                test_desc = np.load(self.dataset.config.data / "test/desc_salad.npy")
            else:
                db_desc = np.load(self.dataset.config.data / "train/netvlad_feats.npy")
                test_desc = np.load(self.dataset.config.data / "test/netvlad_feats.npy")

        db_desc = db_desc.astype(np.float32)
        test_desc = test_desc.astype(np.float32)
        emb = torch.tensor(self.dataset.global_feats).cuda()
        return emb, db_desc, test_desc, None

    def test_model(self):
        self.clear_training_buffer()
        network = self.head
        network.eval()
        self.n_neighbors = 10
        emb, db_desc, test_desc, val_desc = self.get_gl_descriptors()

        from nanopq import PQ

        pq = PQ(M=256, verbose=False).fit(db_desc, seed=self.base_seed)
        codes = pq.encode(db_desc)
        knn = PQKNN(pq, codes, n_neighbors=self.n_neighbors)

        res = self.test_loop(
            DataLoader(self.test_dataset, shuffle=False, num_workers=1),
            knn,
            emb,
            test_desc,
            write_gt_poses=False,
            scene="outdoor",
        )
        return res

    def create_dataset(self):
        self.encoder = get_encoder(
            pca_path=str(self.options.scene / self.options.pca_path),
            model_type=self.options.local_desc,
        )
        self.encoder.cuda()
        self.encoder.eval()
        return get_dataset(
            self.encoder,
            root_ds=self.options.scene,
            feat_name_train=self.options.feat_name,
            feat_name_test=self.options.feat_name_test,
        )

    def create_head_network(self, in_channels):
        mat2 = self.dataset.metadata["cluster_centers"]
        backbone_config = BlockListConfig(
            blocks=[
                (InputBlockConfig(), 1),
                (ResBlockConfig(), 3),
                (
                    PositionDecoderConfig(
                        output_name="sc0",
                    ),
                    1,
                ),
                (
                    PositionEncoderConfig(
                        input_name="sc0", period=2048, num_freqs=16, max_freq_exp=12
                    ),
                    1,
                ),
                (ResBlockConfig(), 2),
                (
                    PositionRefinerConfig(
                        base_name="sc0",
                        output_name="sc",
                    ),
                    1,
                ),
            ]
        )
        head = backbone_config.setup(
            in_channels=in_channels,
            head_channels=self.h_c,
            mlp_ratio=2.0,
            metadata={"cluster_centers": mat2},
        )
        print(f"Created with {in_channels} input channels")
        return head

    def test_loop(
        self,
        testset_loader,
        knn,
        emb,
        val_desc,
        write_gt_poses=False,
        scene="indoor",
        no_db_desc=False,
    ):
        pool = Pool(10)
        device = "cuda"
        ransac_opt = {
            "max_reproj_error": 10,
            "max_iterations": 10000,
            "seed": self.base_seed,
        }
        _logger = logging.getLogger(__name__)
        self.clear_training_buffer()

        metrics = defaultdict(list)
        poses = defaultdict(list)
        pool_results = []
        C_global = self.global_feat_dim
        start = time.time()

        tree = KDTree(self.xyz)
        all_images_retrieved = {}
        with torch.no_grad():
            for batch in tqdm(testset_loader, desc="Testing"):
                image, idx = batch["image"].to(device, non_blocking=True), batch["idx"]
                if no_db_desc:
                    global_feat = batch["global_feat"].to(device, non_blocking=True)
                else:
                    indices = knn.kneighbors(val_desc[idx])
                    global_feat = emb[indices].to(device, non_blocking=True).squeeze(0)
                if len(global_feat.shape) == 1:
                    global_feat = global_feat.unsqueeze(0)
                with autocast(enabled=True, device_type="cuda"):

                    encoder_output = self.encoder.keypoint_features(
                        {"image": image}, n=0
                    )
                    keypoints = encoder_output["keypoints"]
                    descriptors = encoder_output["descriptors"]
                    N, C_local = descriptors.shape
                    gl_feat = torch.empty(
                        (self.n_neighbors, N, C_global + C_local), device=device
                    )
                    gl_feat[:, :, :C_global] = global_feat.unsqueeze(1).expand(
                        -1, N, -1
                    )
                    gl_feat[:, :, C_global:] = descriptors.unsqueeze(0).expand(
                        self.n_neighbors, -1, -1
                    )

                    scene_coords = self.head(
                        {"features": gl_feat.reshape(self.n_neighbors * N, -1)}
                    )["sc"]

                keypoints = keypoints.float().cpu()

                scene_coords = (
                    scene_coords.float().cpu().numpy().reshape(self.n_neighbors, N, 3)
                )

                dis, ind = tree.query(scene_coords.reshape(-1, 3))
                mask = dis < 5
                pid_retrieved = self.pid_list[ind[mask]]
                images_retrieved = []
                for p in pid_retrieved:
                    images_retrieved.extend([self.image2name[g] for g in self.pid2images[p]])

                gt_pose, intrinsics, frame_name = (
                    batch["pose"][0].numpy(),
                    batch["intrinsics"][0].numpy(),
                    batch["filename"][0],
                )
                keypoints_np = keypoints.numpy()

                camera = {
                    "model": "PINHOLE",
                    "width": image.shape[3],
                    "height": image.shape[2],
                    "params": intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]],
                }
                knn_results = []
                for neighbor_idx in range(self.n_neighbors):
                    knn_results.append(
                        pool.apply_async(
                            pose_estimate,
                            args=(
                                keypoints_np,
                                scene_coords[neighbor_idx],
                                camera,
                                ransac_opt,
                                gt_pose,
                            ),
                        )
                    )

                pool_results.append((frame_name, knn_results))
                all_images_retrieved[frame_name] = images_retrieved

        final_results = []
        for frame_name, knn_results in tqdm(pool_results):
            knn_results = [res.get() for res in knn_results]
            result = max(knn_results, key=lambda x: x["num_inliers"])
            result["places"] = all_images_retrieved[frame_name]
            result["xyz"] = result["xyz"][result["inliers"]]
            final_results.append([frame_name, result])
            for key in ("pose_q", "pose_t"):
                poses[key].append(result[key])
            for key in ("t_err", "r_err", "inlier_ratio"):
                metrics[key].append(result[key])
        end = time.time()
        _logger.info(
            f"Time: {end - start:.1f}s for {len(testset_loader.dataset)} images"
        )
        acc_thresh = {
            "outdoor": ((5, 10), (0.5, 5), (0.25, 2)),
            "indoor": ((1, 5), (0.25, 2), (0.1, 1)),
        }

        if write_gt_poses:
            self.write_poses_to_file(final_results)

        if testset_loader.dataset.gt_pose_avail:
            for t, r in acc_thresh[scene]:
                acc = (np.array(metrics["t_err"]) < t) & (
                    np.array(metrics["r_err"]) < r
                )
                _logger.info(f"Accuracy: {t}m/{r}deg: {acc.mean() * 100:.1f}%")
            median_rErr = np.median(metrics["r_err"])
            median_tErr = np.median(metrics["t_err"]) * 100
            _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
            _logger.info(f"Mean Inliers: {np.mean(metrics['inlier_ratio']):.2f}")
        pool.close()
        pool.join()
        return final_results


def top_k_common(lst, k):
    return Counter(lst).most_common(k)


def get_data(res_, res_1):
    names = []
    err1 = []
    err2 = []
    pose1 = []
    pose2 = []
    places1 = []
    places2 = []
    map_data = {}
    count = 0
    for (n1, d1), (n2, d2) in zip(res_, res_1):
        assert n1 == n2
        names.append(n1)
        err1.append(d1["t_err"])
        err2.append(d2["t_err"])
        pose1.append(d1["position"])
        pose2.append(d2["position"])
        places1.append(top_k_common(d1["places"], 10))
        places2.append(top_k_common(d2["places"], 10))
        map_data[count] = [[d1["position"], d1["position_gt"], d1["xyz"]],
                           [d2["position"], d2["position_gt"], d2["xyz"]]]
        count += 1
    return names, err1, err2, pose1, pose2, places1, places2, map_data


def pose_retrieval(options_, pose1_, pose2_, places1, places2, mask, topk=5):
    val_config = CamLocDatasetConfig(
        data=options_.scene,
        split="train",
        feat_name="nothing",
    )
    val_dataset = val_config.setup()

    val_config = CamLocDatasetConfig(
        data=options_.scene,
        split="test",
        feat_name="nothing",
    )
    test_dataset = val_config.setup()

    db_poses = val_dataset.pose_values[:, :3, 3]
    from pykdtree.kdtree import KDTree

    # tree = KDTree(db_poses)
    # d1, i1 = tree.query(pose1_, topk)
    # d2, i2 = tree.query(pose2_, topk)
    db_names = np.array(val_dataset.rgb_files)
    name2id = {n: i for i, n in enumerate(db_names)}

    test_gt_poses = test_dataset.pose_values[:, :3, 3][mask]

    all_d1 = []
    all_d2 = []
    i1 = []
    i2 = []
    for du1 in range(places1.shape[0]):
        pl1 = [name2id[j_] for j_ in places1[du1][:, 0]]
        pl2 = [name2id[j_] for j_ in places2[du1][:, 0]]
        d1 = test_gt_poses[du1] - db_poses[pl1]
        d2 = test_gt_poses[du1] - db_poses[pl2]
        d1 = np.mean(np.abs(d1), axis=1)[:topk]
        d2 = np.mean(np.abs(d2), axis=1)[:topk]
        all_d1.append(d1)
        all_d2.append(d2)
        i1.append(pl1[:topk])
        i2.append(pl2[:topk])
    all_d1 = np.array(all_d1)
    all_d2 = np.array(all_d2)
    i1 = np.array(i1)
    i2 = np.array(i2)

    # d1 = np.mean(np.abs(test_gt_poses[:, None, :] - db_poses[i1]), axis=2)
    # d2 = np.mean(np.abs(test_gt_poses[:, None, :] - db_poses[i2]), axis=2)
    # intrinsics = np.eye(3)
    # intrinsics[0, 0] = 738
    # intrinsics[1, 1] = 738
    # intrinsics[0, 2] = 427  # 427
    # intrinsics[1, 2] = 240
    # import open3d as o3d
    #
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # for idx in tqdm(range(0, len(val_dataset), 5)):
    #     pose_mat = val_dataset.pose_values[idx]
    #     pose_mat = np.linalg.inv(pose_mat)
    #     cam = o3d.geometry.LineSet.create_camera_visualization(
    #         427 * 2, 240 * 2, intrinsics, pose_mat, scale=1
    #     )
    #     cam.paint_uniform_color((1, 0, 0))
    #     vis.add_geometry(cam)
    #     # break
    #
    # for idx in tqdm(range(0, len(test_dataset), 1)):
    #     pose_mat = test_dataset.pose_values[idx]
    #     pose_mat = np.linalg.inv(pose_mat)
    #     cam = o3d.geometry.LineSet.create_camera_visualization(
    #         427 * 2, 240 * 2, intrinsics, pose_mat, scale=1
    #     )
    #     cam.paint_uniform_color((0, 1, 0))
    #     vis.add_geometry(cam)
    # vis.run()
    # vis.destroy_window()

    return db_names, all_d1, all_d2, i1, i2


def load_images(
    image_list,
    target_size=None,
    f_dir="/home/n11373598/work/scrstudio/data/aachen/images_upright",
):
    images = []
    for fname in image_list:
        path = f"{f_dir}/{fname}"
        img = Image.open(path).convert("RGB")
        if target_size:
            img = img.resize(target_size, Image.BILINEAR)
        images.append(np.array(img))
    return images


def overlay_errors(images, errors, color_mask, label="Query"):
    images_out = images.copy()
    color_mask = color_mask.astype(int)
    fontsize = 15
    for i in range(len(errors)):
        img = images[i]
        h, w, _ = img.shape
        dpi = 100
        fig_w = w / dpi
        extra = 0.5 if i == 0 else 0.0  # Only extra space for first column
        fig, ax = plt.subplots(figsize=(fig_w + extra, h / dpi), dpi=dpi)
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0.2 if i == 0 else 0, right=1, bottom=0, top=1)

        if i == 0:
            if not NO_TEXT:
                # Vertical label on the left
                ax.text(
                    0,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=fontsize,
                    rotation="vertical",
                    # color=color,
                    # weight="bold",
                )
        err_text = f"{errors[i]:.1f}m"
        err_color = (0, 1, 0) if color_mask[i] == 1 else (1, 0, 0)

        ax.text(
            0.01,
            0.99,
            err_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            color=err_color,
            backgroundcolor="white",
            weight="bold",
        )

        fig.canvas.draw()
        w2, h2 = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h2, w2, 4)
        rgb = buf[:, :, [1, 2, 3]]  # drop alpha

        plt.close(fig)
        images_out[i] = rgb
    return images_out


def stack_images_horizontally(images, spacing=5, bg_color=255):
    h, w, c = images[0].shape
    spacer = np.ones((h, spacing, c), dtype=np.uint8) * bg_color
    row = images[0]
    for img in images[1:]:
        row = np.hstack([row, spacer, img])
    return row


def stack_image_rows(map_img, n0, n1, err0, err1, resize_to=None, spacing=5):
    imgs0 = load_images(n0, resize_to)
    imgs1 = load_images(n1, resize_to)

    # Overlay errors on last 5 images
    row0 = overlay_errors(imgs0[1:], err0, err0 < err1, label="R-Score")  # red
    row1 = overlay_errors(imgs1[1:], err1, err1 < err0, label="Ours")  # green

    # Remove query image from rows before stacking
    row0 = stack_images_horizontally(row0, spacing=10)  # skip query
    row1 = stack_images_horizontally(row1, spacing=10)  # skip query
    # Stack vertically with spacer
    h_spacer = np.ones((10, row0.shape[1], 3), dtype=np.uint8) * 255
    retrievals = np.vstack([row0, h_spacer, row1])

    # Combine with resized query image
    query_im = load_images([n0[0]])[0]  # The query image with vertical label
    target_h = retrievals.shape[0]
    target_w = int(target_h * 1.5)
    # print(target_w, target_h)
    map_img = np.array(
        Image.fromarray(map_img).resize((target_w, target_h), Image.BILINEAR)
    )

    # print(target_w, target_h)
    query_resized = np.array(
        Image.fromarray(query_im).resize((target_w, target_h), Image.BILINEAR)
    )

    # Now hstack
    final = np.hstack([map_img, query_resized, retrievals])
    return final


def visualize(path0, path1, errors0, errors1, names, query_names, maps, spacing=5):
    names = np.array(names)
    all_rows = []
    # selected = [2, 7, 16, 17, 15]
    selected = [0, 1, 2]
    # selected = np.arange(path0.shape[0])

    path0 = path0[selected]
    path1 = path1[selected]
    maps = maps[selected]
    for i in trange(path0.shape[0]):
        q_name = query_names[i]
        n0 = names[path0[i]]
        n1 = names[path1[i]]

        n0 = np.concatenate(([q_name], n0))
        n1 = np.concatenate(([q_name], n1))

        err0 = errors0[i]
        err1 = errors1[i]
        output_image = stack_image_rows(
            maps[i], n0, n1, err0, err1, spacing=spacing, resize_to=(200, 150)
        )
        all_rows.append(output_image)
    all_rows = np.array(all_rows)
    final_image = all_rows[0]
    for img in all_rows[1:]:
        v_spacer = np.ones((20, final_image.shape[1], 3), dtype=np.uint8) * 255
        try:
            final_image = np.vstack([final_image, v_spacer, img])
        except ValueError as e:
            continue

    dpi = 300
    h, w, _ = final_image.shape
    fig_w, fig_h = w / 100, h / 100

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.1), dpi=dpi)  # Add vertical space
    ax.imshow(final_image, extent=[0, w, 0, h])  # show image with real pixel coords
    ax.axis("off")

    # Add top-k labels centered above each column
    col_width = 200 + 10
    y_offset = h + 0  # 20 pixels above the top of the image

    if not NO_TEXT:
        for i in range(path0.shape[1]):
            x_center = i * col_width + 530*2
            ax.text(
                x_center,
                y_offset,
                f"top-{i + 1}",
                ha="left",
                va="bottom",
                fontsize=15,
                # weight="bold",
                color="black",
            )

    if not NO_TEXT:
        fig.savefig(
            "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/pose_retrievals.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )
        fig.savefig(
            "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/pose_retrievals.png",
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )
    else:
        fig.savefig(
            "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/retrievals2.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0.05,
        )

    plt.close(fig)


def main():
    options = get_options()

    res_path = "res.pkl"
    if os.path.exists(res_path):
        with open(res_path, "rb") as f:
            res = pickle.load(f)
        print(f"Loaded from {res_path}")
    else:
        options.feat_name = "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_margin_t2.npy"
        trainer = TrainerACE(
            options,
            768,
            "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_aachen.pth",
        )
        res = trainer.test_model()
        with open(res_path, "wb") as f:
            pickle.dump(res, f)
        print(f"Saved new result to {res_path}")

    # Second run
    res2_path = "res2.pkl"
    if os.path.exists(res2_path):
        with open(res2_path, "rb") as f:
            res2 = pickle.load(f)
        print(f"Loaded from {res2_path}")
    else:
        options.feat_name = "/home/n11373598/hpc-home/work/scrstudio_exp/data/data/aachen/train/pose_n2c.pt"
        trainer2 = TrainerACE(
            options,
            1280,
            "/home/n11373598/hpc-home/work/scrstudio_me/outputs/aachen/scrfacto/fixed/scrstudio_models/head.pt",
        )
        res2 = trainer2.test_model()
        with open(res2_path, "wb") as f:
            pickle.dump(res2, f)
        print(f"Saved new result to {res2_path}")

    names, err1, err2, pose1, pose2, places1, places2, map_data = get_data(res, res2)
    all_indices = np.arange(len(names))
    gt_err1 = np.array(err1)
    gt_err2 = np.array(err2)
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)
    places1 = np.array(places1)
    places2 = np.array(places2)
    print(np.median(err1), np.median(err2))
    # diff = err2 - err1
    mask = (gt_err1 < gt_err2) & (gt_err1 < 5) & (gt_err2 > 5)
    print(np.sum(mask))
    # mask = gt_err1 < gt_err2
    names = np.array(names)
    names = names[mask]
    pose1 = pose1[mask]
    pose2 = pose2[mask]
    places1 = places1[mask]
    places2 = places2[mask]
    gt_err1 = gt_err1[mask]
    gt_err2 = gt_err2[mask]
    all_indices = all_indices[mask]

    (db_names, dis1, dis2, ind1, ind2) = pose_retrieval(
        options, pose1, pose2, places1, places2, mask, topk=3)
    err1 = np.mean(dis1, 1)
    err2 = np.mean(dis2, 1)
    diff = err2 - err1
    mask = (err1 < err2)

    indices = np.argsort(diff[mask])[::-1][:10]

    names = names[mask][indices]
    ind1 = ind1[mask][indices]
    ind2 = ind2[mask][indices]
    dis1 = dis1[mask][indices]
    dis2 = dis2[mask][indices]
    gt_err1 = gt_err1[mask][indices]
    gt_err2 = gt_err2[mask][indices]
    all_indices = all_indices[mask][indices]

    maps = []
    first = True
    for index in all_indices:
        (p1, p_gt1, xyz1), (p2, p_gt2, xyz2) = map_data[index]
        im = plot_points(p1, p2, p_gt1, xyz1, xyz2, dpi=100, width=465, height=310,
                         first=index==441)
        first = False
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        maps.append(im)
    maps = np.array(maps)
    print(dis2)
    print(dis1)
    print(gt_err2)
    print(gt_err1)
    visualize(ind2, ind1, dis2, dis1, db_names, names, maps)


def plot_points(p1, p2, p_gt1, xyz1, xyz2, width=1600, height=1200, dpi=100, first=False):
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)

    # Plot points
    x, y, _ = zip(*xyz1)
    ax.scatter(x, y, color="green", marker=".", label="Our predictions")

    x, y, _ = zip(*xyz2)
    ax.scatter(x, y, color="red", marker=".", label="R-Score's predictions")

    x, y, _ = zip(*p1.reshape(1, -1))
    ax.scatter(x, y, color="green", marker="x", label="Our pose", s=200)

    x, y, _ = zip(*p2.reshape(1, -1))
    ax.scatter(x, y, color="red", marker="x", label="R-Score's pose", s=200)

    x, y, _ = zip(*p_gt1.reshape(1, -1))
    ax.scatter(x, y, color="cyan", marker="*", label="GT pose", s=400)

    # Formatting
    if first:
        ax.legend(loc='upper left', fontsize=9)
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.05)
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout(pad=0)

    # Render to numpy array
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(height, width, 4)

    plt.close(fig)
    return img


if __name__ == "__main__":
    NO_TEXT = False
    main()
