import functools
import gc
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf, DictConfig
from pykdtree.kdtree import KDTree
from scipy.sparse import load_npz
from torch import autocast
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
import wandb

import utils
from base_trainer import BaseTrainer
from networks import get_model
from rscore_loss import get_losses
from utils import (
    read_nvm_file,
    get_options,
    CSRGraph,
    run_salad_model,
)
from config_classes import BatchRandomSamplerConfig, PQKNN
from dataset import get_dataset, CamLocDatasetConfig
from encoder import get_encoder

_logger = logging.getLogger(__name__)


class TrainerAachen(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        super().__init__(cfg)
        # self.dump_dir = "/mnt/data/sftp/data/tungns30/aachen10_dump_folder"
        self.dump_dir = cfg.dump_dir

        self.feature_dim = self.encoder.out_channels
        self.global_feat_dim = self.dataset.global_feat_dim
        if self.cfg.global_feat:
            head_channels = self.feature_dim + self.global_feat_dim
        else:
            head_channels = self.feature_dim
        self.buffer_size_dim = self.feature_dim
        self.head = self.create_head_network(head_channels)

        if self.cfg.test_mode:

            self.head.load_state_dict(
                torch.load(
                    "checkpoints/best_model.pth",
                    # "checkpoints/head_main.pth",
                    weights_only=True,
                )
            )
            print(f"Loaded pretrained weights for head.")
            self.head.eval()
        else:
            self.head.train()

        self.head.to(self.device)
        self.optimizer = optim.AdamW(
            self.head.parameters(),
            lr=self.cfg.learning_rate_max,
            weight_decay=1e-2,
        )

        # Setup learning rate scheduler.
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.learning_rate_max,
            total_steps=self.cfg.max_iterations,
            cycle_momentum=False,
            pct_start=0.04,
        )

        print(f"Using {self.cfg.learning_rate_max} as learning rate")

        self.ds_dir = str(self.cfg.scene)
        pose_graph_dir = self.cfg.scene / "train/pose_overlap.npz"
        print(f"Load membership from {pose_graph_dir}")
        covis_score = load_npz(pose_graph_dir)
        covis_score.data[covis_score.data < 0.2] = 0
        covis_score.setdiag(1)
        covis_score.eliminate_zeros()
        covis_score = covis_score.tocsr()
        self.covis_graph = CSRGraph.from_csr_array(covis_score, self.device)
        self.loss_functions = get_losses(self.cfg.max_iterations)
        self.gradient_accumulation_steps = self.cfg.grad_acc

        sampler_config = BatchRandomSamplerConfig(batch_size=self.cfg.batch_size)
        self.batch_sampler = iter(
            sampler_config.setup(
                dataset_size=self.cfg.training_buffer_size,
                generator=self.training_generator,
            )
        )

        train_config = CamLocDatasetConfig(
            data=self.cfg.scene,
            split="train",
        )

        raw_ds = train_config.setup()
        run_salad_model(raw_ds, self.cfg.scene / "images_upright")
        run_salad_model(self.test_dataset, self.cfg.scene / "images_upright")
        self.prev_re = None
        if self.cfg.test_mode:
            self.test_model_train_set()
            self.test_model()
            sys.exit()

        if self.cfg.focus_tune:
            try:
                (
                    self.xyz_arr,
                    self.image2points,
                    self.image2name,
                    self.image2info,
                    self.image2uvs,
                    self.image2pose,
                ) = read_nvm_file(self.cfg.scene / "aachen_cvpr2018_db.nvm")
                self.name2id = {v: k for k, v in self.image2name.items()}
            except FileNotFoundError:
                print(
                    f"Cannot find nvm model at {self.cfg.scene / 'aachen_cvpr2018_db.nvm'})"
                )
                sys.exit()

    def create_head_network(self, in_channels):
        mat2 = self.dataset.metadata["cluster_centers"]
        head = get_model(in_channels, {"cluster_centers": mat2}, head_channels=1280)
        print(f"Created with {in_channels} input channels")
        return head

    def retrieve_gt_xyz(
        self,
        image_ori_B1HW,
        frame_path,
        uv_grid_arr,
        gt_pose_inv_B44,
        intrinsics_B33,
        radius=5,
    ):
        # Get image dimensions from the actual tensor shape
        _, H, W, _ = image_ori_B1HW.shape

        image_id = self.name2id[frame_path[0]]
        pid_list = self.image2points[image_id]
        xyz = self.xyz_arr[pid_list]  # Assuming xyz_arr is a numpy array

        # Project XYZ to Camera Coords
        # Using @ for matrix multiplication is cleaner
        xyz_homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))]).T
        cam_coords = gt_pose_inv_B44[0, :3].cpu().numpy() @ xyz_homo

        # Project to Screen Space
        uv_homo = intrinsics_B33[0].cpu().numpy() @ cam_coords
        z = np.clip(uv_homo[2], 0.1, None)
        uv = uv_homo[:2] / z
        uv = uv.T

        # Filter points actually landing within the image frame
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        if np.sum(in_frame) < 10:
            return torch.zeros(uv_grid_arr.shape[0], device="cuda"), None, None

        # Match sampled keypoints to projected map points
        tree = KDTree(uv[in_frame])
        distances, indices = tree.query(uv_grid_arr)

        # Create mask based on distance threshold
        mask = distances < radius

        # Extract depth and 3D coords for the matched points
        # Note: we must index into the 'in_frame' subset of xyz
        valid_xyz = xyz[in_frame]
        valid_cam_coords = cam_coords[:, in_frame]

        xyz_gt = valid_xyz[indices]
        depths = valid_cam_coords[2, indices]

        return torch.from_numpy(mask.astype(np.int32)).cuda(), depths, xyz_gt

    def create_training_buffer(self):
        buffer_dir = f"{self.dump_dir}/training_buffer_{self.cfg.training_buffer_size}/{str(self.cfg.scene).split('/')[-1]}"
        os.makedirs(buffer_dir, exist_ok=True)

        required_keys = [
            "features",
            "target_px",
            "gt_poses_inv",
            "gt_poses",
            "intrinsics",
            "intrinsics_inv",
            "img_idx",
            "xyz_gt",
            "sample_idx",
        ]

        # Check if all saved files exist
        buffer_exists = all(
            os.path.exists(os.path.join(buffer_dir, f"{k}.pt")) for k in required_keys
        )

        if buffer_exists and self.cfg.reuse_buffer:
            print(
                f"Loading training buffer from {buffer_dir} (moving to {self.device})..."
            )
            self.training_buffer = {
                k: torch.load(
                    os.path.join(buffer_dir, f"{k}.pt"),
                    map_location="cpu",
                    weights_only=True,
                ).to(self.device)
                for k in required_keys
            }
            print("Training buffer loaded successfully.")
            return

        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            "features": torch.empty(
                (self.cfg.training_buffer_size, self.buffer_size_dim),
                dtype=(torch.float32, torch.float16)[self.cfg.use_half],
                device=self.device,
            ),
            "target_px": torch.empty(
                (self.cfg.training_buffer_size, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses_inv": torch.empty(
                (self.cfg.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses": torch.empty(
                (self.cfg.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics": torch.empty(
                (self.cfg.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics_inv": torch.empty(
                (self.cfg.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "img_idx": torch.empty(
                (self.cfg.training_buffer_size,),
                dtype=torch.int32,
                device=self.device,
            ),
            "xyz_gt": torch.empty(
                (self.cfg.training_buffer_size, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "sample_idx": torch.empty(
                (self.cfg.training_buffer_size,),
                dtype=torch.int32,
                device=self.device,
            ),
        }

        # Iterate until the training buffer is full.
        buffer_idx = 0
        dataset_passes = 0
        example_idx = 0
        pbar = tqdm(total=self.cfg.training_buffer_size, desc="Filling buffer")

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
            timeout=120,
        )

        while buffer_idx < self.cfg.training_buffer_size:
            dataset_passes += 1
            for batch in training_dataloader:
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
                image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                gt_pose_B44 = gt_pose_B44.to(self.device, non_blocking=True)
                intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                intrinsics_inv_B33 = intrinsics_inv_B33.to(
                    self.device, non_blocking=True
                )
                image_B1HW = image_B1HW.to(self.device, non_blocking=True)

                with torch.no_grad():
                    with autocast(
                        enabled=self.cfg.use_half,
                        dtype=torch.bfloat16,
                        device_type="cuda",
                    ):
                        encoder_output = self.encoder.keypoint_features(
                            {"image": image_B1HW, "mask": image_mask_B1HW},
                            n=self.cfg.samples_per_image,
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
                xyz_gt = None
                if self.cfg.focus_tune:
                    mask, gt_depths, xyz_gt = self.retrieve_gt_xyz(
                        image_ori,
                        filename,
                        target_pix.cpu().numpy(),
                        gt_pose_inv_B44,
                        intrinsics_B33,
                    )
                    image_mask_N1 *= mask
                if image_mask_N1.sum() == 0:
                    continue

                # Over-sample according to image mask.
                features_to_select = min(self.cfg.samples_per_image * 1, nb_features)
                features_to_select = min(
                    features_to_select,
                    self.cfg.training_buffer_size - buffer_idx,
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

                if xyz_gt is not None:
                    xyz_gt = torch.from_numpy(xyz_gt).cuda()
                    self.training_buffer["xyz_gt"][buffer_idx:buffer_offset] = xyz_gt[
                        sample_idxs
                    ]

                buffer_idx = buffer_offset
                pbar.update(features_to_select)
                if buffer_idx >= self.cfg.training_buffer_size:
                    break
        pbar.close()
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
        if self.cfg.reuse_buffer:
            print(
                f"Saving training buffer to {buffer_dir} (one file per key on CPU)..."
            )
            for k, v in self.training_buffer.items():
                torch.save(v.cpu(), os.path.join(buffer_dir, f"{k}.pt"))
            print("All buffer keys saved successfully on CPU.")

    def get_next_batch(self):
        random_batch_indices = next(self.batch_sampler)
        features_batch = torch.empty(
            (
                self.cfg.batch_size,
                self.feature_dim + self.global_feat_dim,
            ),
            dtype=self.training_buffer["features"].dtype,
            device=self.device,
        )

        img_idx = self.training_buffer["img_idx"][random_batch_indices].long()
        if self.cfg.graph_aug:
            num_neighbors = img_idx.shape[0] // 2
            img_idx[:num_neighbors] = self.covis_graph.sample_neighbors(
                img_idx[:num_neighbors], self.gn_generator
            )

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
                    "xyz_gt",
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

    def handle_re_spike(self, error_val, batch_data, iteration):
        print(f"\n⚠️ SPIKE DETECTED: RE = {error_val:.2f} at iteration {iteration}")

        # 1. Save the model state immediately
        spike_dir = Path(self.dump_dir) / "spikes" / f"iter_{iteration}"
        spike_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.head.state_dict(), spike_dir / "head_spike.pth")

        # 2. Save the exact batch that caused the spike for offline debugging
        # We move to CPU to avoid keeping GPU memory tied up
        batch_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch_data.items()
        }
        torch.save(batch_cpu, spike_dir / "batch_data.pt")

        # 3. Log an artifact or alert to WandB
        wandb.log(
            {
                "event/spike_detected": 1,
                "event/spike_re_value": error_val,
                "iteration": iteration,
            }
        )

        _logger.warning(f"Saved spike diagnostic data to {spike_dir}")
        sys.exit()

    def run_epoch(self, pbar_, max_iterations=None):
        torch.backends.cudnn.benchmark = True
        self.optimizer.zero_grad()

        epoch_re = 0
        valid_batches = 0
        for _ in range(self.gradient_accumulation_steps):
            dict_ = self.get_next_batch()
            loss_tensor, mean_re, ma1, mi1 = self.training_step(
                dict_, skip_backward=False
            )
            epoch_re += mean_re
            valid_batches += 1

            pbar_.set_postfix(
                loss=f"{loss_tensor.detach() * self.gradient_accumulation_steps:.1f}",
                re=f"{epoch_re / valid_batches:.1f}",
            )

        old_optimizer_step = self.get_step_count()
        self.optimizer.step()

        curr_step = self.get_step_count()
        if old_optimizer_step < curr_step < self.scheduler.total_steps:
            self.scheduler.step()

        wandb.log(
            {
                "iteration": self.iteration,
                "train/median_repro_error": epoch_re / valid_batches,
            }
        )

        self.iteration += 1
        self.optimizer.zero_grad()
        pbar_.update(1)

    def training_step(self, dict_, skip_backward=False, safe_version=False):
        """
        Run one iteration of training, computing metrics.
        Returns loss tensor and scalar metrics.
        """

        (
            features_bC,
            target_px_b2,
            img_idx_b1,
            xyz_gt_b3,
            gt_poses_b34,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
        ) = dict_.values()

        if not self.cfg.graph_aug:
            features_bC = self.diffuse_feature(features_bC)

        with autocast(
            enabled=self.cfg.use_half, dtype=torch.bfloat16, device_type="cuda"
        ):
            outputs = self.head({"features": features_bC})

        if self.iteration % 500 == 0:
            nan_mask = torch.all(torch.bitwise_not(torch.isnan(outputs["sc"])), 1)
            if not nan_mask.all():
                wandb.finish()
                raise ValueError("NaN found in the output")

        outputs.update(
            {
                "target_px": target_px_b2,
                "gt_poses_inv": gt_inv_poses_b34,
                "intrinsics": Ks_b33,
                "intrinsics_inv": invKs_b33,
                "sc": outputs["sc"],
                "sc0": outputs["sc0"],
                "step": self.iteration,
                "gt_coords": xyz_gt_b3,
            }
        )

        for loss_function in self.loss_functions:
            if safe_version:
                outputs = loss_function.forward_safe_version(outputs)
            else:
                outputs = loss_function(outputs)
                # outputs = loss_function.forward_fast(outputs)

        metrics_dict = outputs["metrics"]

        # We keep the loss as a tensor if we need to call .backward() on it later
        loss = metrics_dict["loss"] / self.gradient_accumulation_steps

        if not skip_backward:
            loss.backward()
            # self.scaler.scale(loss).backward()

        return (
            loss,  # Tensor
            metrics_dict["median_rep_error"],
            outputs["sc"].max(),
            outputs["sc"].min(),
        )

    def get_gl_descriptors(self):
        if self.cfg.debug_mode:
            db_desc = np.load(self.dataset.config.data / "train/desc_salad.npy")
            test_desc = np.load(self.dataset.config.data / "test/desc_salad.npy")
        else:
            if self.cfg.use_salad:
                db_desc = np.load(self.dataset.config.data / "train/desc_salad.npy")
                test_desc = np.load(self.dataset.config.data / "test/desc_salad.npy")
            else:
                db_desc = np.load(self.dataset.config.data / "train/netvlad_feats.npy")
                test_desc = np.load(self.dataset.config.data / "test/netvlad_feats.npy")

        db_desc = db_desc.astype(np.float32)
        test_desc = test_desc.astype(np.float32)
        emb = torch.tensor(self.dataset.global_feats).cuda()
        return emb, db_desc, test_desc, None

    def create_dataset(self):
        self.encoder = get_encoder(
            pca_path=str(self.cfg.scene / self.cfg.pca_path),
            model_type=self.cfg.local_desc,
        )
        self.encoder.cuda()
        self.encoder.eval()
        return get_dataset(
            self.encoder,
            root_ds=self.cfg.scene,
            feat_name_train=self.cfg.feat_name,
            feat_name_test=self.cfg.feat_name_test,
        )

    def test_model(self):
        with torch.no_grad():
            for name, param in self.head.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data = param.data.half()
        self.clear_training_buffer()
        network = self.head
        network.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        query_stems = [du.split("/")[-1] for du in self.test_dataset.rgb_files]
        utils.read_csv_to_dict("checkpoints/retrieval_global.csv")
        dict_ = utils.read_csv_to_dict("checkpoints/retrieval_reranked.csv")
        self.n_neighbors = 10
        top_k_retrievals = [dict_[du][: self.n_neighbors] for du in query_stems]
        stem2id = {
            k2.split("/")[-1].split(".")[0]: k1
            for k1, k2 in enumerate(self.dataset.rgb_files)
        }
        top_k_retrievals = [
            [stem2id.get(du2.split(".")[0], 0) for du2 in du] for du in top_k_retrievals
        ]

        emb, db_desc, test_desc, val_desc = self.get_gl_descriptors()

        from nanopq import PQ

        pq = PQ(M=256, verbose=False).fit(db_desc, seed=self.base_seed)
        codes = pq.encode(db_desc)
        knn = PQKNN(pq, codes, n_neighbors=self.n_neighbors)

        self.test_loop(
            DataLoader(self.test_dataset, shuffle=False, num_workers=1),
            knn,
            emb,
            test_desc,
            write_gt_poses=True,
            scene="outdoor",
            all_indices=top_k_retrievals,
            # no_db_desc=True,
        )

    def test_model_train_set(self):
        train_config = CamLocDatasetConfig(
            data=self.cfg.scene,
            split="train",
            feat_name=self.cfg.feat_name,
            num_decoder_clusters=50,
        )

        ds = train_config.setup(preprocess=self.encoder.preprocess)
        device = "cuda"
        ransac_opt = {
            "max_reproj_error": 10,
            "max_iterations": 10000,
            "seed": self.base_seed,
        }
        _logger = logging.getLogger(__name__)
        self.clear_training_buffer()

        pool_results = []
        C_global = self.global_feat_dim

        testset_loader = DataLoader(ds, shuffle=False, num_workers=1)
        final_results = []

        metrics = defaultdict(list)
        poses = defaultdict(list)
        count = 0
        with torch.no_grad():
            for batch in tqdm(testset_loader, desc="Testing"):
                image, idx = batch["image"].to(device, non_blocking=True), batch["idx"]
                global_feat = batch["global_feat"].to(device, non_blocking=True)
                if len(global_feat.shape) == 1:
                    global_feat = global_feat.unsqueeze(0)
                with autocast(enabled=self.cfg.use_half, device_type="cuda"):
                    encoder_output = self.encoder.keypoint_features(
                        {"image": image}, n=0
                    )
                    keypoints = encoder_output["keypoints"]
                    descriptors = encoder_output["descriptors"]
                    N, C_local = descriptors.shape
                    gl_feat = torch.empty(
                        (N, C_global + C_local), device=device
                    )
                    gl_feat[:, :C_global] = global_feat.expand(keypoints.shape[0], -1)
                    gl_feat[:, C_global:] = descriptors

                    scene_coords = self.head(
                        {"features": gl_feat}
                    )["sc"]

                result = self.pose_estimation_helper(image, ransac_opt, batch, keypoints, scene_coords)

                gt_pose, intrinsics, frame_name = (
                    batch["pose"][0].numpy(),
                    batch["intrinsics"][0].numpy(),
                    batch["filename"][0],
                )
                final_results.append([frame_name, result])
                for key in ("pose_q", "pose_t"):
                    poses[key].append(result[key])
                for key in ("t_err", "r_err", "inlier_ratio"):
                    metrics[key].append(result[key])

        final_results = []
        for frame_name, knn_results in tqdm(pool_results):
            knn_results = [res.get() for res in knn_results]
            result = max(knn_results, key=lambda x: x["num_inliers"])
            final_results.append([frame_name, result])
            for key in ("pose_q", "pose_t"):
                poses[key].append(result[key])
            for key in ("t_err", "r_err", "inlier_ratio"):
                metrics[key].append(result[key])
        acc_thresh = {
            "outdoor": ((5, 10), (0.5, 5), (0.25, 2)),
            "indoor": ((1, 5), (0.25, 2), (0.1, 1)),
        }

        # 1. Prepare a dictionary for W&B logging
        test_metrics_to_log = {}

        for t, r in acc_thresh["outdoor"]:
            acc = (np.array(metrics["t_err"]) < t) & (
                    np.array(metrics["r_err"]) < r
            )
            acc_percent = acc.mean() * 100
            _logger.info(f"Accuracy: {t}m/{r}deg: {acc_percent:.1f}%")

            # Log each threshold specifically
            test_metrics_to_log[f"test/acc_{t}m_{r}deg"] = acc_percent

        median_rErr = np.median(metrics["r_err"])
        median_tErr = np.median(metrics["t_err"]) * 100
        mean_inliers = np.mean(metrics["inlier_ratio"])

        _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
        _logger.info(f"Mean Inliers: {mean_inliers:.2f}")

        output_h5_path = Path("outputs/model_outputs.h5")

        with h5py.File(output_h5_path, "w") as f:
            # Create groups to organize data nicely
            metrics_group = f.create_group("metrics")
            for key, val in metrics.items():
                metrics_group.create_dataset(key, data=np.array(val))

            poses_group = f.create_group("poses")
            for key, val in poses.items():
                poses_group.create_dataset(key, data=np.array(val))

            # Handle frame-specific string mappings and complex results
            results_group = f.create_group("final_results")
            for frame_name, res in final_results:
                # Create a subgroup for each frame using its name
                # HDF5 doesn't like complex types, so flat dictionaries work best
                frame_group = results_group.create_group(str(frame_name))
                for k, v in res.items():
                    if isinstance(v, (np.ndarray, list)):
                        frame_group.create_dataset(k, data=np.array(v))
                    elif isinstance(v, (int, float, str)):
                        frame_group.attrs[k] = v  # store scalars as attributes

        _logger.info(f"Successfully saved outputs to {output_h5_path}")



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Handle Debug Mode Overrides
    if cfg.debug_mode == 1:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["training_buffer_size"] = 5000
        cfg["batch_size"] = 512
        cfg["max_iterations"] = 100
        cfg = OmegaConf.create(cfg)

    # Initialize W&B
    # It automatically captures the Hydra config
    wandb_mode = "disabled" if cfg.debug_mode == 1 else cfg.wandb.mode
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        dir=cfg.wandb.dir,
        mode=wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = TrainerAachen(cfg)

    if not cfg.test_mode:
        trainer.train()
    trainer.test_model()

    if cfg.debug_mode == 1:
        wandb.finish()
        sys.exit()

    # Finish the W&B run
    wandb.finish()


def main_debug():
    options = get_options()
    debug = options.debug_mode == 1
    if debug:
        options.training_buffer_size = 5000
        options.batch_size = 512
        options.max_iterations = 100
        trainer = TrainerAachen(options)
        trainer.dump_dir = "checkpoints"
        trainer.cfg.focus_tune = True
        trainer.train()
        trainer.test_model()
        sys.exit()


if __name__ == "__main__":
    # main_debug()
    main()
