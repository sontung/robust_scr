# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Xudong Jiang (ETH Zurich)
import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from kornia.feature.dedode.dedode_models import get_descriptor, get_detector
from kornia.feature.dedode.utils import sample_keypoints
from kornia.geometry.conversions import denormalize_pixel_coordinates
from torch import nn
# from lightglue import ALIKED
from config_classes import InstantiateConfig
# from d2net_all import _D2Net, process_multiscale
from dataset import PreprocessConfig
# from superpoint_all import SuperPoint


@dataclass
class EncoderConfig(InstantiateConfig):
    """Configuration for Encoder instantiation"""

    _target: Type = field(default_factory=lambda: Encoder)
    """target class to instantiate"""


@dataclass
class DedodeEncoderConfig(EncoderConfig):
    _target: Type = field(default_factory=lambda: DedodeEncoder)

    detector: str = "L"

    descriptor: str = "B"

    k: int = 5000


urls = {
    "detector": {
        "L-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
        "Lv2-upright": "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
        "L-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_C4.pth",
        "L-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_SO2.pth",
    },
    "descriptor": {
        "B-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
        "B-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth",
        "B-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth",
        "G-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
        "G-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_descriptor_setting_C.pth",
    },
}


class Encoder(nn.Module):
    OUTPUT_SUBSAMPLE = 0
    sparse = False
    out_channels = 0

    def __init__(
        self,
        config: EncoderConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.preprocess = PreprocessConfig()

    def keypoint_features(self, data, n=0, generator=None):

        features = self.forward(data)["features"]
        B, C, H, W = features.shape
        assert B == 1, "Only batch size 1 is supported"
        s = self.OUTPUT_SUBSAMPLE
        dtype, device = features.dtype, features.device
        target_px = torch.empty(H, W, 2, dtype=dtype, device=device)
        target_px[..., 0].copy_(
            torch.arange(s * 0.5, W * s, s, dtype=dtype, device=device)
        )
        target_px[..., 1].copy_(
            torch.arange(s * 0.5, H * s, s, dtype=dtype, device=device).unsqueeze(-1)
        )

        features = features[0].flatten(1).transpose(0, 1)
        target_px = target_px.flatten(0, 1)

        if "mask" in data:
            mask = TF.resize(
                data["mask"], [H, W], interpolation=TF.InterpolationMode.NEAREST
            )
            mask = mask.bool().flatten()
            features = features[mask]
            target_px = target_px[mask]

        if n > 0:
            idx = torch.randperm(
                features.size(0), generator=generator, device=features.device
            )[:n]
            features = features[idx]
            target_px = target_px[idx]

        return {
            "keypoints": target_px,
            "descriptors": features,
        }

    def decode(self, descriptors):
        return descriptors


class DedodeEncoder(Encoder):
    """
    Dedode encoder, used to extract features from the input images.
    """

    out_channels = 256
    sparse = True

    def __init__(self, config: DedodeEncoderConfig, **kwargs):
        print("Using Dedode encoder")
        super().__init__(config)
        self.preprocess = PreprocessConfig(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            grayscale=False,
            use_half=True,
            size_multiple=14 if config.descriptor[0] == "G" else 8,
        )

        self.OUTPUT_SUBSAMPLE = 14 if config.descriptor[0] == "G" else 8
        self.detector = get_detector(config.detector[0])
        self.descriptor = get_descriptor(config.descriptor[0])

        self.detector.load_state_dict(
            torch.hub.load_state_dict_from_url(
                urls["detector"][f"{config.detector}-upright"], map_location="cpu"
            )
        )
        self.descriptor.load_state_dict(
            torch.hub.load_state_dict_from_url(
                urls["descriptor"][f"{config.descriptor}-upright"], map_location="cpu"
            )
        )

        self.k = config.k

        self.eval()
        self.detector.compile()
        self.descriptor.compile()

    def keypoint_features(self, data, n=0, generator=None):
        images = data["image"]
        assert not self.training

        B, C, H, W = images.shape
        logits = self.detector.forward(images)
        scoremap = logits.reshape(B, H * W).softmax(dim=-1).reshape(B, H, W)
        if "mask" in data:
            mask = TF.resize(
                data["mask"], [H, W], interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(1)
            scoremap = scoremap * mask

        keypoints, scores = sample_keypoints(scoremap, num_samples=self.k)

        if n > 0:
            idx = torch.randperm(
                keypoints.size(1), generator=generator, device=keypoints.device
            )[:n]
            keypoints = keypoints[:, idx]
            scores = scores[:, idx]

        descriptions = self.descriptor.forward(images)
        descriptions = F.grid_sample(
            descriptions.float(),
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].transpose(-2, -1)

        return {
            "keypoints": denormalize_pixel_coordinates(keypoints, H, W)[0],
            "keypoint_scores": scores[0],
            "descriptors": descriptions[0],
        }

    def forward(self, data, det=False):
        raise NotImplementedError


class AlikedEncoder(Encoder):
    """
    Dedode encoder, used to extract features from the input images.
    """

    out_channels = 128
    sparse = True
    required_inputs = ["image"]

    def __init__(self, config: DedodeEncoderConfig, **kwargs):
        print("Using ALIKED encoder")
        super().__init__(config)
        self.preprocess = PreprocessConfig(
            mean=0,
            std=1,
            grayscale=False,
            use_half=False,
            size_multiple=1,
        )

        self.conf = {
            "model_name": "aliked-n16",
            "max_num_keypoints": -1,
            "detection_threshold": 0.2,
            "nms_radius": 2,
        }
        self.encoder = ALIKED(**self.conf).eval().to("cuda")

    def keypoint_features(self, data, n=0, generator=None):
        out = self.encoder(data)
        return {
            "keypoints": out["keypoints"][0],
            "descriptors": out["descriptors"][0],
        }

    def forward(self, data, det=False):
        raise NotImplementedError


class SuperPointEncoder(Encoder):
    """
    Dedode encoder, used to extract features from the input images.
    """

    out_channels = 256
    sparse = True
    required_inputs = ["image"]

    def __init__(self, config: DedodeEncoderConfig, **kwargs):
        print("Using Superpoint encoder")
        super().__init__(config)
        self.preprocess = PreprocessConfig(
            mean=0,
            std=1,
            grayscale=True,
            use_half=True,
            size_multiple=1,
        )

        self.conf = {}
        self.encoder = SuperPoint(self.conf).eval().to("cuda")

    def keypoint_features(self, data, n=0, generator=None):
        out = self.encoder(data)
        return {
            "keypoints": out["keypoints"][0],
            "scores": out["scores"][0],
            "descriptors": out["descriptors"][0].T,
        }

    def forward(self, data, det=False):
        raise NotImplementedError


class D2Encoder(Encoder):
    """
    Dedode encoder, used to extract features from the input images.
    """

    out_channels = 512
    sparse = True
    d2net_path = Path("third_party/d2net")
    default_conf = {
        "model_name": "d2_tf.pth",
        "checkpoint_dir": d2net_path / "models",
        "use_relu": True,
        "multiscale": False,
    }
    required_inputs = ["image"]

    def __init__(self, config: DedodeEncoderConfig, **kwargs):
        print("Using D2 encoder")
        super().__init__(config)
        self.preprocess = PreprocessConfig(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            grayscale=False,
            use_half=True,
            size_multiple=1,
        )

        conf = {**self.default_conf, **kwargs}
        self.conf = conf
        # self.OUTPUT_SUBSAMPLE = 14 if config.descriptor[0] == "G" else 8
        model_file = conf["checkpoint_dir"] / conf["model_name"]
        if not model_file.exists():
            model_file.parent.mkdir(exist_ok=True, parents=True)
            cmd = [
                "wget",
                "https://dusmanu.com/files/d2-net/" + conf["model_name"],
                "-O",
                str(model_file),
            ]
            subprocess.run(cmd, check=True)

        self.net = _D2Net(
            model_file=model_file, use_relu=conf["use_relu"], use_cuda=False
        )

    def keypoint_features(self, data, n=0, generator=None):
        image = data["image"]
        image = image.flip(1)  # RGB -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = image * 255 - norm.view(1, 3, 1, 1)  # caffe normalization

        if self.conf["multiscale"]:
            keypoints, scores, descriptors = process_multiscale(image, self.net)
        else:
            keypoints, scores, descriptors = process_multiscale(
                image, self.net, scales=[1]
            )
        keypoints = keypoints[:, [1, 0]]  # (x, y) and remove the scale

        return {
            "keypoints": torch.from_numpy(keypoints).cuda(),
            "scores": torch.from_numpy(scores).cuda(),
            "descriptors": torch.from_numpy(descriptors).cuda(),
        }

    def forward(self, data, det=False):
        raise NotImplementedError


@dataclass
class PCAEncoderConfig(EncoderConfig):

    _target: Type = field(default_factory=lambda: PCAEncoder)

    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    pca_path: str = "pca.pth"


class PCAEncoder(Encoder):
    OUTPUT_SUBSAMPLE = 8
    out_channels = 128

    def __init__(self, config, data_path=None, **kwargs):
        super().__init__(config)
        self.encoder = config.encoder.setup(data_path=data_path, **kwargs)
        self.preprocess = self.encoder.preprocess

        # config.pca_path = "/home/n11373598/hpc-home/work/scrstudio_exp/data/data/aachen/proc/pcad3LB_32.pth"
        try:
            state_dict = torch.load(
                config.pca_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            state_dict = torch.load(config.pca_path, map_location="cpu")
        self.out_channels = state_dict["weight"].shape[0]

        if self.encoder.sparse:
            self.linear = nn.Linear(
                self.encoder.out_channels,
                self.out_channels,
                bias="bias" in state_dict.keys(),
            )
            state_dict["weight"] = state_dict["weight"].view(
                self.out_channels, self.encoder.out_channels
            )
            self.linear.load_state_dict(state_dict)
            self.linear.eval()
        else:
            self.conv = nn.Conv2d(
                self.encoder.out_channels,
                self.out_channels,
                1,
                bias="bias" in state_dict.keys(),
            )
            self.conv.load_state_dict(state_dict)
            self.conv.eval()

    def keypoint_features(self, data, n=0, generator=None):
        if self.encoder.sparse:
            ret = self.encoder.keypoint_features(data, n, generator)
            ret["descriptors"] = self.linear(ret["descriptors"])
            return ret
        else:
            return super().keypoint_features(data, n, generator)

    def forward(self, data, det=False):

        ret = self.encoder(data)
        ret["features"] = self.conv(ret["features"])

        return ret


def get_encoder(pca_path="checkpoints/pcad3LB_128.pth", model_type="dedode"):
    print(f"Loading pca path at {pca_path}")
    if model_type == "d2":
        config = PCAEncoderConfig(
            encoder=D2EncoderConfig(),
            pca_path=pca_path,
        )
    elif model_type == "dedode":
        config = PCAEncoderConfig(
            encoder=DedodeEncoderConfig(
                detector="L",
                descriptor="B",
                k=5000,
            ),
            pca_path=pca_path,
        )
    elif model_type == "sp":
        config = PCAEncoderConfig(
            encoder=SuperPointEncoderConfig(),
            pca_path=pca_path,
        )
    elif model_type == "aliked":
        config = AlikedEncoderConfig()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    model = config.setup()
    return model


if __name__ == "__main__":
    model = get_encoder()
    print(model)
