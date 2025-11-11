import functools
import gc
import logging
import os
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import cv2
import faiss
import h5py
import numpy as np
import poselib
import torch
import torch.optim as optim
from cuml import PCA
from pykdtree.kdtree import KDTree
from scipy.sparse import load_npz
from torch import autocast
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm, trange

from base_trainer import BaseTrainer
from rscore_loss import get_losses
from utils import (
    read_nvm_file,
    get_options,
    pose_estimate,
    CSRGraph,
    read_intrinsic,
    stack_images_horizontally,
    run_salad_model,
)
from config_classes import BatchRandomSamplerConfig, PQKNN
from dataset import get_dataset, CamLocDatasetConfig, ImageAugmentConfig
from encoder import get_encoder
from networks import (
    PositionRefinerConfig,
    ResBlockConfig,
    PositionEncoderConfig,
    PositionDecoderConfig,
    InputBlockConfig,
    BlockListConfig,
)

_logger = logging.getLogger(__name__)


class TrainerACE(BaseTrainer):
    def __init__(self, options_):
        super().__init__(options_)
        self.feature_dim = self.encoder.out_channels
        self.global_feat_dim = self.dataset.global_feat_dim
        if self.options.global_feat:
            head_channels = self.feature_dim + self.global_feat_dim
        else:
            head_channels = self.feature_dim
        self.buffer_size_dim = self.feature_dim
        self.head = self.create_head_network(head_channels)

        if self.options.test_mode:
            state_dict = torch.load(
                "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_b1.pth",
                weights_only=True,
            )

            self.head.load_state_dict(state_dict)

            print(f"Loaded pretrained weights for head.")
            self.head.eval()
        else:
            self.head.train()

        self.head.to(self.device)
        self.optimizer = optim.AdamW(
            self.head.parameters(),
            lr=self.options.learning_rate_max,
            weight_decay=1e-2,
        )

        # Setup learning rate scheduler.
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            total_steps=self.options.max_iterations,
            cycle_momentum=False,
            pct_start=0.04,
        )

        print(f"Using {self.options.learning_rate_max} as learning rate")

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
        run_salad_model(raw_ds, self.options.scene / "../..")
        run_salad_model(self.val_dataset, self.options.scene / "../..")
        run_salad_model(self.test_dataset, self.options.scene / "../..")

        if self.options.test_mode:
            # self.debug_model()
            self.test_model()
            sys.exit()

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

        self.test_loop(
            DataLoader(self.val_dataset, shuffle=False, num_workers=1),
            knn,
            emb,
            val_desc,
            write_gt_poses=False,
        )
        self.test_loop(
            DataLoader(self.test_dataset, shuffle=False, num_workers=1),
            knn,
            emb,
            test_desc,
            write_gt_poses=True,
        )
        return

    def create_dataset(self):
        self.encoder = get_encoder(
            pca_path=str(self.options.scene / "proc/pcad3LB_128.pth"),
        )
        self.encoder.cuda()
        self.encoder.eval()

        train_config = CamLocDatasetConfig(
            data=self.options.scene,
            augment=ImageAugmentConfig(),
            split="train",
            feat_name=self.options.feat_name,
            num_decoder_clusters=50,
            loading_depth=self.options.depth_init,
        )
        val_config = CamLocDatasetConfig(
            data=self.options.scene,
            split="val",
            feat_name="nothing",
        )
        test_config = CamLocDatasetConfig(
            data=self.options.scene,
            split="test",
            feat_name="nothing",
        )

        ds = train_config.setup(preprocess=self.encoder.preprocess)
        ds2 = test_config.setup(preprocess=self.encoder.preprocess)
        self.val_dataset = val_config.setup(preprocess=self.encoder.preprocess)

        return ds, ds2

    def write_poses_to_file(self, poses):
        now = datetime.now()
        os.makedirs("results", exist_ok=True)

        # Format it as a string
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        scene_id = str(options.scene).split("/")[-1]
        pose_log_file = f"results/res-dept-{scene_id}-{date_time_str}.txt"

        pose_log = open(pose_log_file, "w", 1)

        for frame_name, data in poses:
            q = data["pose_q"]
            t = data["pose_t"]
            frame_name = frame_name.split("records_data/")[-1]

            pose_log.write(
                f"{frame_name} "
                f"{q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]}\n"
            )
        pose_log.close()
        print(f"Saved poses to {pose_log_file}")

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
        if str(self.options.scene).split("/")[-1] == "4F":
            h_c = 768
        else:
            h_c = 1280
        head = backbone_config.setup(
            in_channels=in_channels,
            head_channels=h_c,
            mlp_ratio=2.0,
            metadata={"cluster_centers": mat2},
        )
        print(f"Created with {in_channels} input channels")
        return head


if __name__ == "__main__":
    options = get_options()
    debug = options.debug_mode == 1
    if debug:
        options.training_buffer_size = 50000
        options.batch_size = 512
        options.max_iterations = 100
        trainer = TrainerACE(options)
        trainer.train()
        trainer.test_model()
        sys.exit()

    trainer = TrainerACE(options)
    trainer.train()
    trainer.test_model()
