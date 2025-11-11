# Copyright © Niantic, Inc. 2022.
import math
from torch import autocast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Type

from config_classes import InstantiateConfig, CoordLossConfig


def weighted_gm(rep_errs, weight, reduction="mean"):
    err_square = 9 * (rep_errs / weight) ** 2
    loss = weight * (err_square / (err_square + 4))
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")


def weighted_tanh(repro_errs, weight, reduction="mean"):
    loss = weight * torch.tanh(repro_errs / weight)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")


def get_schedule_weight(iteration, total_iterations, schedule):
    if schedule == "constant":
        return 1
    schedule_weight = iteration / total_iterations
    schedule_weight = min(schedule_weight, 1)
    if schedule == "linear":
        return schedule_weight
    if schedule == "circle":
        return 1 - math.sqrt(1 - schedule_weight**2)
    if schedule == "cosine":
        return (1 + math.cos(math.pi * schedule_weight)) / 2
    raise ValueError(f"Unknown schedule type: {schedule}")


@dataclass
class RobustLossConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: RobustLoss)
    total_iterations: Optional[int] = None
    soft_clamp: int = 50
    soft_clamp_min: int = 1
    robust_type: str = "tanh"
    schedule: str = "circle"


@dataclass
class ReproLossConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: ReproLoss)
    robust_loss: RobustLossConfig = field(default_factory=lambda: RobustLossConfig())
    depth_min: float = 0.1
    depth_max: float = 1000.0
    depth_target: float = 10.0
    db_std: float = 1.0
    input_name: str = "sc"
    output_name: str = ""
    hard_clamp: int = 1000
    invalid_weight: float = 1.0
    eval_inlier_threshold: float = 10
    depth_bias_adjust: bool = False
    coord_loss: Optional[CoordLossConfig] = None
    loss_weight: float = 1.0
    scale_factor: float = 1.0


class ReproLoss(nn.Module):
    def __init__(self, config: ReproLossConfig, total_iterations, **kwargs):
        super().__init__()
        self.config = config
        self.robust_loss: RobustLoss = config.robust_loss.setup(
            total_iterations=total_iterations, **kwargs
        )
        if config.coord_loss:
            self.closs_steps = int(total_iterations * config.coord_loss.step_ratio)

    def forward(self, batch):

        step = batch["step"]
        # disable autocast
        with autocast(enabled=False, device_type="cuda"):
            scene_coords = batch[self.config.input_name].float()
            camera_coords = torch.matmul(
                batch["gt_poses_inv"],
                F.pad(scene_coords, (0, 1), value=1).unsqueeze(-1),
            )

            target_px: Tensor = batch["target_px"]

            repro_error = torch.matmul(batch["intrinsics"], camera_coords).squeeze(-1)
            repro_error[:, 2].clamp_(min=self.config.depth_min)

            repro_error = (repro_error[:, :2] / repro_error[:, 2:] - target_px).norm(
                dim=-1
            )

            depth = camera_coords[:, 2].squeeze(-1)
            invalid = (
                (depth < self.config.depth_min)
                | (repro_error > self.config.hard_clamp)
                | (depth > self.config.depth_max)
            )
            valid = ~invalid

            repro_error = repro_error[valid]
            if self.config.depth_bias_adjust:
                depth_square = depth[valid] ** 2
                loss_valid = self.robust_loss(
                    torch.sqrt(
                        depth_square / (depth_square + (self.config.db_std) ** 2)
                    )
                    * repro_error,
                    step,
                )
            else:
                loss_valid = self.robust_loss(repro_error, step)

            batch_size = scene_coords.shape[0]
            loss_invalid = (
                self.config.depth_target
                * torch.matmul(
                    batch["intrinsics_inv"][invalid],
                    F.pad(target_px[invalid], (0, 1), value=1).unsqueeze(-1),
                )
                - camera_coords[invalid]
            ).norm(dim=1).sum() / batch_size

            fraction_valid = valid.float().mean(dtype=torch.float32)
            loss_valid = loss_valid * fraction_valid
            loss = loss_valid + loss_invalid * self.config.invalid_weight

            metrics = batch.get("metrics", {"loss": 0})

            metrics[f"loss_valid{self.config.output_name}"] = loss_valid
            metrics[f"loss_invalid{self.config.output_name}"] = loss_invalid
            metrics[f"valid_fraction{self.config.output_name}"] = fraction_valid
            metrics[f"median_rep_error{self.config.output_name}"] = torch.median(
                repro_error
            )
            metrics[f"median_depth{self.config.output_name}"] = torch.median(depth)
            if self.config.eval_inlier_threshold > 0:
                inlier = repro_error < self.config.eval_inlier_threshold
                metrics[f"mean_inlier_rep_error{self.config.output_name}"] = (
                    repro_error[inlier].mean()
                )
                metrics[f"inlier_fraction{self.config.output_name}"] = (
                    inlier.float().mean(dtype=torch.float32) * fraction_valid
                )

            if self.config.coord_loss:
                closs_weight = self.config.coord_loss.max_weight * get_schedule_weight(
                    step, self.closs_steps, "cosine"
                )

                if "gt_coords" in batch:
                    gt_coords = batch["gt_coords"].float()
                    gt_coords_mask = (gt_coords != 0).all(dim=1)
                    scene_coords2 = batch[self.config.coord_loss.target_name].float()
                    gt_coord_dist1 = torch.norm(
                        gt_coords[gt_coords_mask] - scene_coords[gt_coords_mask], dim=-1
                    )
                    gt_coord_dist2 = torch.norm(
                        gt_coords[gt_coords_mask] - scene_coords2[gt_coords_mask],
                        dim=-1,
                    )
                    gt_coord_dist = gt_coord_dist1 + gt_coord_dist2
                else:
                    gt_coords = batch[self.config.coord_loss.target_name].float()
                    gt_coords_mask = (gt_coords != 0).all(dim=1)
                    gt_coord_dist = torch.norm(
                        gt_coords[gt_coords_mask] - scene_coords[gt_coords_mask], dim=-1
                    )
                coord_loss = (
                    gt_coord_dist[
                        gt_coord_dist > self.config.coord_loss.tolerance
                    ].sum()
                    / batch_size
                )
                loss += coord_loss * closs_weight
                metrics[f"coord_loss{self.config.output_name}"] = coord_loss
                metrics[f"closs_weight{self.config.output_name}"] = closs_weight

        metrics["loss"] += loss * self.config.loss_weight
        batch["metrics"] = metrics

        return batch


class RobustLoss:
    def __init__(
        self, config: RobustLossConfig, total_iterations: int = 100000, **kwargs
    ):
        self.config = config
        self.total_iterations = (
            config.total_iterations
            if config.total_iterations is not None
            else total_iterations
        )
        self.soft_clamp = config.soft_clamp
        self.soft_clamp_min = config.soft_clamp_min
        self.schedule = config.schedule
        self.robust_type = config.robust_type

    def __call__(self, loss, step):
        schedule_weight = get_schedule_weight(
            step, self.total_iterations, self.schedule
        )
        loss_weight = (
            1 - schedule_weight
        ) * self.soft_clamp + schedule_weight * self.soft_clamp_min
        if self.robust_type == "tanh":
            return weighted_tanh(loss, loss_weight)
        elif self.robust_type == "gm":
            return weighted_gm(loss, loss_weight)
        else:
            raise ValueError(f"Unknown robust loss type: {self.robust_type}")


def get_losses(total_iterations, db_std=8):
    losses = [
        ReproLossConfig(
            input_name="sc",
            output_name="",
            robust_loss=RobustLossConfig(
                soft_clamp=25,
                soft_clamp_min=1,
            ),
            coord_loss=CoordLossConfig(
                target_name="sc0",
            ),
        ),
        ReproLossConfig(
            robust_loss=RobustLossConfig(
                soft_clamp=50, soft_clamp_min=1, robust_type="gm"
            ),
            depth_bias_adjust=True,
            input_name="sc0",
            output_name="0",
            db_std=db_std,
            depth_target=10,
        ),
    ]
    all_fn = []
    for config in losses:
        loss_fn = config.setup(total_iterations=total_iterations)
        all_fn.append(loss_fn)
    return all_fn
