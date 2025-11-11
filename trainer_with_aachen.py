import functools
import gc
import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from pykdtree.kdtree import KDTree
from scipy.sparse import load_npz
from torch import autocast
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from base_trainer import BaseTrainer
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

            self.head.load_state_dict(
                torch.load(
                    # "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_loftr_best.pth",
                    "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_main.pth",
                    # "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_loftr_node2vec_gt.pth",
                    # "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_loftr_fm.pth",
                    # "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_loftr2.pth",
                    # "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_radial.pth"
                    # "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/head_loftr.pth",
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
        self.loss_functions = get_losses(self.options.max_iterations)
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

        if self.options.test_mode:
            # self.debug_model()
            self.test_model()
            sys.exit()
        if self.options.focus_tune:
            try:
                (
                    self.xyz_arr,
                    self.image2points,
                    self.image2name,
                    self.image2info,
                    self.image2uvs,
                    self.image2pose,
                ) = read_nvm_file(self.options.scene / "aachen_cvpr2018_db.nvm")
                self.name2id = {v: k for k, v in self.image2name.items()}
            except FileNotFoundError:
                pass

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
            "xyz_gt",
            "sample_idx",
        ]

        # Check if all saved files exist
        buffer_exists = all(
            os.path.exists(os.path.join(buffer_dir, f"{k}.pt")) for k in required_keys
        )

        if buffer_exists and self.options.reuse_buffer:
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
            "xyz_gt": torch.empty(
                (self.options.training_buffer_size, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "sample_idx": torch.empty(
                (self.options.training_buffer_size,),
                dtype=torch.int32,
                device=self.device,
            ),
        }

        # Iterate until the training buffer is full.
        buffer_idx = 0
        dataset_passes = 0
        example_idx = 0
        pbar = tqdm(total=self.options.training_buffer_size, desc="Filling buffer")

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

        while buffer_idx < self.options.training_buffer_size:
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
                xyz_gt = None
                if self.options.focus_tune:
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

                if xyz_gt is not None:
                    xyz_gt = torch.from_numpy(xyz_gt).cuda()
                    self.training_buffer["xyz_gt"][buffer_idx:buffer_offset] = xyz_gt[
                        sample_idxs
                    ]

                buffer_idx = buffer_offset
                pbar.update(features_to_select)
                if buffer_idx >= self.options.training_buffer_size:
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
        if self.options.reuse_buffer:
            print(
                f"Saving training buffer to {buffer_dir} (one file per key on CPU)..."
            )
            for k, v in self.training_buffer.items():
                torch.save(v.cpu(), os.path.join(buffer_dir, f"{k}.pt"))
            print("All buffer keys saved successfully on CPU.")

    def retrieve_gt_xyz(
        self,
        image_ori_B1HW,
        frame_path,
        uv_grid_arr,
        gt_pose_inv_B44,
        intrinsics_B33,
        radius=5,
    ):
        batch_idx = 0

        # image_ori = image_ori_B1HW[0].cpu().numpy().astype(np.uint8)
        image_id_from_map = self.name2id[frame_path[0]]
        pid_list = self.image2points[image_id_from_map]
        xyz = np.take(self.xyz_arr, pid_list, axis=0)
        xyzt = np.hstack([xyz, np.ones((xyz.shape[0], 1))]).T
        gt_inv_pose_34 = gt_pose_inv_B44[batch_idx, :3]
        cam_coords = np.matmul(gt_inv_pose_34.cpu().numpy(), xyzt)

        uv = np.matmul(intrinsics_B33[batch_idx].cpu().numpy(), cam_coords)
        uv[2] = np.clip(uv[2], 0.1, None)  # Set minimum value to 0.1
        uv = uv[0:2] / uv[2]
        uv = uv.T

        # for u, v in uv:
        #     cv2.circle(image_ori, (int(u), int(v)), 2, (0, 255, 0), -1)
        # image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        #
        # uv_gt = np.array(self.image2uvs[image_id_from_map])
        # image2 = cv2.imread(f"/home/n11373598/work/glace/datasets/aachen_source/images_upright/{frame_path[0]}")
        # for u, v in uv_gt:
        #     cv2.circle(image2, (int(u), int(v)), 2, (0, 255, 0), -1)
        #
        # image3 = stack_images_horizontally(image_ori, image2)
        # cv2.imwrite(f"debug/{image_id_from_map}.png", image3)

        b1, b2 = np.max(uv_grid_arr, 0)
        oob_mask1 = np.bitwise_and(0 <= uv[:, 0], uv[:, 0] < b1)
        oob_mask2 = np.bitwise_and(0 <= uv[:, 1], uv[:, 1] < b2)
        oob_mask = np.bitwise_and(oob_mask1, oob_mask2)

        depths = None
        xyz_gt = None
        if np.sum(oob_mask) < 10:
            mask = np.zeros(uv_grid_arr.shape[0])
        else:

            tree = KDTree(uv[oob_mask])
            dis, ind = tree.query(uv_grid_arr)
            mask = dis < radius
            depths = cam_coords[2, ind]
            xyz_gt = xyz[ind]

        mask = mask.astype(int)
        mask = torch.from_numpy(mask).cuda()

        return mask, depths, xyz_gt

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

    def run_epoch(self, pbar_, max_iterations=None):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Iterate with mini batches.
        self.optimizer.zero_grad()
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
            xyz_gt_b3,
            gt_poses_b34,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
        ) = dict_.values()

        # B = target_px_b2.shape[0]
        # uv_h = torch.cat([target_px_b2, torch.ones(B, 1, device=target_px_b2.device)], dim=1)  # (B, 3)
        # cam_dirs = torch.bmm(invKs_b33, uv_h.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        # cam_points = cam_dirs * gt_depths_b1.unsqueeze(1)  # (B, 3)
        # R_inv = gt_poses_b34[:, :, :3]  # (B, 3, 3)
        # t_inv = gt_poses_b34[:, :, 3]  # (B, 3)
        #
        # xyz_world = torch.bmm(R_inv, cam_points.unsqueeze(-1)).squeeze(-1) + t_inv  # (B, 3)

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
        outputs["gt_coords"] = xyz_gt_b3

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

        self.n_neighbors = 10
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
            # no_db_desc=True,
        )


if __name__ == "__main__":
    options = get_options()
    debug = options.debug_mode == 1
    if debug:
        options.training_buffer_size = 5000
        options.batch_size = 512
        options.max_iterations = 100
        trainer = TrainerACE(options)
        trainer.train()
        trainer.test_model()
        sys.exit()

    trainer = TrainerACE(options)
    trainer.train()
    trainer.test_model()
