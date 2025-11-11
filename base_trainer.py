import gc
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed, pose_estimate
from config_classes import PQKNN
from networks import get_model

_logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, options):

        self.options = options
        device_id = 0
        self.device = torch.device("cuda", device_id)
        torch.cuda.set_device(device_id)

        # Setup randomness for reproducibility.
        self.base_seed = 42 + device_id
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        # Generator for global feature noise
        self.gn_generator = torch.Generator(device=self.device)
        self.gn_generator.manual_seed(self.base_seed + 24601)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 3
        self.encoder = None
        self.val_dataset = None
        self.dataset, self.test_dataset = self.create_dataset()

        self.global_feat_dim = self.dataset.global_feat_dim
        self.global_feats = torch.tensor(
            self.dataset.global_feats,
            dtype=(torch.float32, torch.float16)[self.options.use_half],
            device=self.device,
        )

        torch.backends.cudnn.benchmark = True

        # Gradient scaler in case we train with half precision.
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.options.use_half)
        except AttributeError:
            self.scaler = GradScaler(enabled=self.options.use_half)

        # Compute total number of iterations.
        self.iterations = self.options.max_iterations
        self.iterations_output = (
            self.options.iter_output
        )  # print loss every n iterations

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        self.head = None

    def create_head_network(self, in_channels):
        mat2 = self.dataset.metadata["cluster_centers"]
        head = get_model(in_channels, {"cluster_centers": mat2})
        print(f"Created with {in_channels} input channels")
        return head

    def create_dataset(self):
        return None, None

    def clear_training_buffer(self):
        if self.training_buffer is not None:
            values_list = list(self.training_buffer.keys())
            for k in values_list:
                del self.training_buffer[k]
            gc.collect()
            torch.cuda.empty_cache()

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        max_iterations = self.options.max_iterations
        self.create_training_buffer()
        pbar = tqdm(total=self.iterations, desc="Training")

        while self.iteration < self.options.max_iterations:
            self.run_epoch(pbar, max_iterations=max_iterations)
        self.clear_training_buffer()
        self.save_model()

    def create_training_buffer(self):
        pass

    def run_epoch(self, pbar_, max_iterations=None):
        pass

    def training_step(self, dict_):
        pass

    def get_step_count(self):
        try:
            step_count = self.optimizer.state[
                self.optimizer.param_groups[0]["params"][0]
            ]["step"]

        except KeyError:
            step_count = 0
        return int(step_count)

    def save_model(self):
        with torch.no_grad():
            for name, param in self.head.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data = param.data.half()
        torch.save(
            self.head.state_dict(), f"checkpoints/{self.options.output_map_file}"
        )
        return

    def write_poses_to_file(self, poses):
        now = datetime.now()
        os.makedirs("results", exist_ok=True)

        # Format it as a string
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        pose_log_file = f"results/res-aachen-{date_time_str}.txt"

        pose_log = open(pose_log_file, "w", 1)

        for frame_name, data in poses:
            q = data["pose_q"]
            t = data["pose_t"]
            frame_name = frame_name.split("/")[-1]

            pose_log.write(
                f"{frame_name} "
                f"{q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]}\n"
            )
        pose_log.close()
        print(f"Saved poses to {pose_log_file}")

    def diffuse_feature(self, features_bC):
        features_bC[:, : self.global_feat_dim] += torch.empty_like(
            features_bC[:, : self.global_feat_dim]
        ).normal_(mean=0, std=self.options.feat_noise_std, generator=self.gn_generator)
        features_bC[:, : self.global_feat_dim] = torch.nn.functional.normalize(
            features_bC[:, : self.global_feat_dim], dim=1
        )
        return features_bC

    def get_gl_descriptors(self):

        db_desc = np.load(self.dataset.config.data / "train/desc_salad.npy")
        test_desc = np.load(self.dataset.config.data / "test/desc_salad.npy")
        val_desc = np.load(self.dataset.config.data / "val/desc_salad.npy")

        db_desc = db_desc.astype(np.float32)
        test_desc = test_desc.astype(np.float32)
        val_desc = val_desc.astype(np.float32)
        emb = torch.tensor(self.dataset.global_feats).cuda()
        return emb, db_desc, test_desc, val_desc

    def test_loop(
        self,
        testset_loader,
        knn,
        emb,
        val_desc,
        write_gt_poses=False,
        scene="indoor",
        no_db_desc=False,
        max_samples=-1
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

        assert len(testset_loader.dataset) == val_desc.shape[0]
        count = 0
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
                count += 1
                if 0 < max_samples <= count:
                    break

        final_results = []
        for frame_name, knn_results in tqdm(pool_results):
            knn_results = [res.get() for res in knn_results]
            result = max(knn_results, key=lambda x: x["num_inliers"])
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

    def test_model(self):
        with torch.no_grad():
            for name, param in self.head.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data = param.data.half()
        self.clear_training_buffer()
        network = self.head
        network.eval()

        self.n_neighbors = 10
        emb, db_desc, test_desc, val_desc = self.get_gl_descriptors()

        from nanopq import PQ

        pq = PQ(M=256, verbose=False).fit(db_desc, seed=self.base_seed)
        codes = pq.encode(db_desc)
        knn = PQKNN(pq, codes, n_neighbors=self.n_neighbors)
        if val_desc is not None:
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
