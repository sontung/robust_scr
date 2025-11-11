# BSD 3-Clause License
import math
import os
import pickle
import random
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Type
from typing import Tuple, Union

import cv2
import h5py
import imageio.v3 as iio
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage import color
from skimage.transform import resize, rotate
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm, trange

from config_classes import InstantiateConfig, PrintableConfig


@dataclass
class ImageAugmentConfig(PrintableConfig):
    aug_rotation: int = 15
    """Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees."""
    aug_scale_min: float = 2 / 3
    """Lower limit of image scale factor for uniform sampling"""
    aug_scale_max: float = 3 / 2
    """Upper limit of image scale factor for uniform sampling"""
    aug_black_white: float = 0.1
    """Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]"""
    aug_color: float = 0.3
    """Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]"""


@dataclass
class PreprocessConfig(InstantiateConfig):
    """Configuration for preprocessing data"""

    _target: Type = field(default_factory=lambda: Preprocess)
    mean: Optional[Union[float, Tuple[float, float, float]]] = (0.485, 0.456, 0.406)
    """mean value for normalization"""

    std: Optional[Union[float, Tuple[float, float, float]]] = (0.229, 0.224, 0.225)
    """standard deviation value for normalization"""

    grayscale: bool = False
    """whether to convert image to grayscale"""

    use_half: bool = True
    """whether to use half precision"""

    size_multiple: int = 8
    """size multiple for input image"""


class Preprocess:
    def __init__(
        self,
        config: PreprocessConfig,
        augment: Optional[ImageAugmentConfig] = None,
        smaller_size=480,
    ):
        self.config = config
        self.augment = augment
        self.smaller_size = smaller_size
        image_transform = []
        if config.grayscale:
            image_transform.append(transforms.Grayscale())

        if self.augment:
            image_transform.append(
                transforms.ColorJitter(
                    brightness=self.augment.aug_black_white,
                    contrast=self.augment.aug_black_white,
                )
            )
        image_transform.append(transforms.ToTensor())
        if config.mean is not None and config.std is not None:
            image_transform.append(
                transforms.Normalize(mean=config.mean, std=config.std)
            )
        self.image_transform = transforms.Compose(image_transform)

    @staticmethod
    def _resize_image(image, size):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, size)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode="constant"):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def round_to_multiple(self, x):
        return math.ceil(x / self.config.size_multiple) * self.config.size_multiple

    def __call__(self, data):

        image = data["image"]
        focal_length, center_point = data["calib"]
        pose = data["pose"]
        depth = data.get("depth", None)
        if depth is not None:
            if len(depth.shape) == 1:
                height, width, _ = image.shape
                depth = depth.reshape((height // 3, width // 3))

        scale_factor = (
            random.uniform(self.augment.aug_scale_min, self.augment.aug_scale_max)
            if self.augment
            else 1
        )
        smaller_size = self.round_to_multiple(int(self.smaller_size * scale_factor))
        old_wh = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float32)
        smaller_idx = torch.argmin(old_wh)
        new_wh = torch.empty(2, dtype=torch.float32)
        new_wh[smaller_idx] = smaller_size
        new_wh[1 - smaller_idx] = self.round_to_multiple(
            int(smaller_size * old_wh[1 - smaller_idx] / old_wh[smaller_idx])
        )

        focal_length *= new_wh / old_wh
        if center_point is not None:
            center_point *= new_wh / old_wh

        new_hw = new_wh.int().tolist()[1::-1]
        image = self._resize_image(image, new_hw)
        image_mask = torch.ones((1, *new_hw))

        if depth is not None:
            depth = resize(depth, new_hw, order=0, anti_aliasing=False)
        image_ori = np.copy(np.array(image))
        image = self.image_transform(image)

        if self.augment:
            angle = random.uniform(
                -self.augment.aug_rotation, self.augment.aug_rotation
            )
            image = self._rotate_image(image, angle, 1, "reflect")
            image_mask = self._rotate_image(image_mask, angle, order=1, mode="constant")
            image_ori = rotate(
                image_ori, angle, order=1, mode="reflect", preserve_range=True
            )
            if depth is not None:
                depth = rotate(depth, angle, order=0, mode="constant")

            # Rotate ground truth camera pose as well.
            angle = angle * math.pi / 180.0
            pose_rot = torch.eye(4)
            pose_rot[:2, :2] = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            pose = torch.matmul(pose, pose_rot)

        if self.config.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        intrinsics = torch.eye(3)
        intrinsics[[0, 1], [0, 1]] = focal_length
        if center_point is not None:
            intrinsics[[0, 1], [2, 2]] = center_point
        else:
            intrinsics[[0, 1], [2, 2]] = new_wh / 2

        pose_inv = pose.inverse()
        intrinsics_inv = intrinsics.inverse()

        data = {
            "image": image,
            "image_ori": image_ori,
            "mask": image_mask,
            "pose": pose,
            "pose_inv": pose_inv,
            "intrinsics": intrinsics,
            "intrinsics_inv": intrinsics_inv,
        }
        if depth is not None:
            data["depth"] = depth

        return data


@dataclass
class ReaderConfig(InstantiateConfig):
    """Config for image reader."""

    _target: Type = field(default_factory=lambda: Reader)
    img_type: Literal["images", "depths", "bytes"] = "images"
    data: str = "rgb"


class Reader(nn.Module):
    """Base class for image reader."""

    file_list: list
    index: Optional[list] = None

    def __init__(self, config: ReaderConfig, root: Path = Path(""), file_list=None):
        self.config = config
        self.path = root / config.data
        self.populate()
        self.index = (
            [self.file_list.index(x) if x in self.file_list else -1 for x in file_list]
            if file_list
            else None
        )
        if config.img_type == "images":
            self.decoder = iio.imread
        # elif config.img_type == "depths":
        #     self.decoder = depth_decoder
        elif config.img_type == "bytes":
            self.decoder = lambda x: x

    @abstractmethod
    def populate(self):
        """Populate the reader with a list of files."""

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.file_list)

    def read(self, index: int):
        raise NotImplementedError

    def __getitem__(self, index: int):
        if self.index is not None:
            index = self.index[index]
        if index == -1:
            return None
        return self.read(index)


@lru_cache(maxsize=3)
def open_lmdb(lmdb_path, readonly=True):
    lmdb_path = str(lmdb_path)
    if readonly:
        return lmdb.open(
            lmdb_path,
            subdir=os.path.isdir(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=1,
        )
    else:
        return lmdb.open(
            lmdb_path,
            subdir=os.path.isdir(lmdb_path),
            map_size=2**32,
            readonly=False,
            meminit=False,
            map_async=True,
            max_dbs=1,
        )


@dataclass
class LMDBReaderConfig(ReaderConfig):
    """Config for LMDB reader."""

    _target: Type = field(default_factory=lambda: LMDBReader)
    data: str = "rgb_lmdb"
    db_name: Optional[str] = None


class LMDBReader(Reader):
    config: LMDBReaderConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def populate(self):
        with open(self.path / "file_list.txt", "r") as f:
            self.file_list = f.read().splitlines()
        self.db = None

    def lazy_populate(self):
        self.env = open_lmdb(self.path, readonly=True)
        db_name = self.config.db_name if self.config.db_name else self.config.img_type
        self.db = self.env.open_db(db_name.encode("ascii"), integerkey=True)

    def read(self, index: int):
        if self.db is None:
            self.lazy_populate()
        with self.env.begin(write=False) as txn:
            image_bytes = txn.get(key=int(index).to_bytes(4, sys.byteorder), db=self.db)
        return self.decoder(image_bytes)


@dataclass
class CamLocDatasetConfig(InstantiateConfig):
    """Config for CamLocDataset."""

    _target: Type = field(default_factory=lambda: CamLocDataset)
    data: Optional[Path] = None
    split: str = "train"
    depth: Optional[ReaderConfig] = None
    rgb: ReaderConfig = field(default_factory=lambda: LMDBReaderConfig())
    augment: Optional[ImageAugmentConfig] = None
    num_decoder_clusters: int = 0
    feat_name: str = "features.npy"
    smaller_size: int = 480
    loading_depth: bool = False


class CamLocDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    rgb_reader: Reader

    def __init__(
        self,
        config: CamLocDatasetConfig,
        preprocess: PreprocessConfig = PreprocessConfig(
            mean=None, std=None, grayscale=False, use_half=False, size_multiple=1
        ),
    ):
        self.config = config
        self.preprocess = preprocess.setup(
            augment=config.augment, smaller_size=config.smaller_size
        )

        self.metadata = {}
        assert self.config.data is not None, "Data folder must be set"
        root = self.config.data / self.config.split
        if not root.exists():
            self.rgb_files = []
            return
        self.rgb_reader = self.config.rgb.setup(root=root)
        self.rgb_files = self.rgb_reader.file_list

        self.depth_reader = None

        if self.config.split == "train" and self.config.loading_depth:
            self.depth_file_dir = root / "depth_lmdb"
            depth_dir = Path(self.depth_file_dir)
            files = list(
                depth_dir.glob("*")
            )  # all files and folders directly under depth_file_dir
            num_files = len(files)

            if self.depth_file_dir.exists() and num_files > 0:
                print(f"Loading depths from {self.depth_file_dir}")
            else:
                dim = None
                env = lmdb.open(
                    str(self.depth_file_dir), map_size=10**12
                )  # adjust map_size as needed (e.g. 1TB here)

                for name in tqdm(self.rgb_files):
                    depth_name = name.replace(".jpg", ".depth").replace(
                        "images", "depths"
                    )
                    depth_path = self.config.data / "../.." / depth_name
                    im = np.fromfile(str(depth_path), dtype=np.float32)
                    dim = im.shape[0]
                    break

                with env.begin(write=True) as txn:
                    for idx_, name in enumerate(tqdm(self.rgb_files)):
                        depth_name = name.replace(".jpg", ".depth").replace(
                            "images", "depths"
                        )
                        depth_path = self.config.data / "../.." / depth_name
                        im = np.fromfile(str(depth_path), dtype=np.float32)
                        assert im.shape[0] == dim
                        key = f"{idx_:15d}".encode("ascii")
                        txn.put(key, im.tobytes())

            self.depth_reader = lmdb.open(
                str(self.depth_file_dir), readonly=True, lock=False
            )

        self.calibration_values = np.load(root / "calibration.npy")
        self.gt_pose_avail = True
        if not (root / "poses.npy").exists():
            self.pose_values = np.empty((len(self.rgb_files), 4, 4))
            self.pose_values[:] = np.eye(4)
            print(f"Pose file not found, using dummy eyes {self.pose_values.shape}")
            self.gt_pose_avail = False
        else:
            self.pose_values = np.load(root / "poses.npy")

        if not Path(self.config.feat_name).exists():
            self.global_feats = np.zeros((len(self.rgb_files), 256))
            if self.config.feat_name != "nothing":
                print(
                    f"Global feature {self.config.feat_name} not found\nUsing dummy zeros {self.global_feats.shape}"
                )
        elif self.config.feat_name.endswith(".npy"):
            self.global_feats = np.load(self.config.feat_name)
            print(f"Loaded global features from {self.config.feat_name}")
        elif self.config.feat_name.endswith(".pt"):
            self.global_feats = torch.load(self.config.feat_name, map_location="cpu")[
                "model.embedding.weight"
            ].numpy()
            print(f"Loaded global features from {self.config.feat_name}")
        self.global_feat_dim = self.global_feats.shape[1]

        if self.config.num_decoder_clusters > 1 and self.config.split == "train":
            cluster_file = str(root / "clusters.npy")
            if Path(cluster_file).exists():
                self.metadata["cluster_centers"] = torch.from_numpy(
                    np.load(cluster_file)
                ).float()
                print(f"Loaded cluster centers from {cluster_file}")
            else:
                kmeans = KMeans(
                    n_clusters=self.config.num_decoder_clusters,
                    random_state=0,
                    n_init=10,
                ).fit(self.pose_values[:, :3, 3].astype(np.float32))
                cluster_centers = kmeans.cluster_centers_
                self.metadata["cluster_centers"] = torch.from_numpy(
                    cluster_centers
                ).float()
                print(
                    f"Computed cluster centers from {self.pose_values.shape}, saving to {cluster_file}"
                )
                np.save(cluster_file, cluster_centers)
        else:
            self.metadata["cluster_centers"] = torch.from_numpy(
                self.pose_values[:, :3, 3].mean(0, keepdims=True)
            ).float()

    def _load_image(self, idx):
        image = self.rgb_reader[idx]
        if image is not None and len(image.shape) < 3:
            image = color.gray2rgb(image)

        return image

    def _load_depth(self, idx):
        assert self.depth_reader is not None, "Depth reader is not set"
        with self.depth_reader.begin() as txn:
            key = f"{idx:15d}".encode("ascii")
            raw = txn.get(key)
            depth = np.frombuffer(raw, dtype=np.float32)
            depth = depth.reshape(-1)
        return depth

    def _load_pose(self, idx):
        return torch.from_numpy(self.pose_values[idx]).float()

    def _load_calib(self, idx):
        k = self.calibration_values[idx]
        if k.size == 1:
            focal_length = torch.tensor([k, k], dtype=torch.float32)
            center_point = None
        elif k.shape == (3, 3):
            focal_length = torch.tensor(k[[0, 1], [0, 1]], dtype=torch.float32)
            center_point = torch.tensor(k[[0, 1], 2], dtype=torch.float32)
        else:
            raise Exception(
                "Calibration file must contain either a 3x3 camera \
                intrinsics matrix or a single float giving the focal length \
                of the camera."
            )
        return focal_length, center_point

    def __getitem__(self, idx):
        data = {
            "image": self._load_image(idx),
            "calib": self._load_calib(idx),
            "pose": self._load_pose(idx),
            "idx": idx,
        }
        if self.depth_reader is not None:
            data["depth"] = self._load_depth(idx)

        data = self.preprocess(data)
        data.update(
            {
                "global_feat": torch.from_numpy(self.global_feats[idx]).float(),
                "idx": idx,
                "filename": str(self.rgb_files[idx]),
            }
        )

        return data

    def __len__(self):
        return len(self.rgb_files)


def get_dataset(
    encoder,
    root_ds=Path("/home/n11373598/work/scrstudio/data/aachen"),
    feat_name_train="checkpoints/desc_node2vec.npy",
    feat_name_test="checkpoints/desc_node2vec_test.npy",
):
    train_config = CamLocDatasetConfig(
        data=root_ds,
        augment=ImageAugmentConfig(),
        split="train",
        feat_name=feat_name_train,
        num_decoder_clusters=50,
    )

    test_config = CamLocDatasetConfig(
        data=root_ds,
        split="test",
        feat_name=feat_name_test,
    )

    ds = train_config.setup(preprocess=encoder.preprocess)
    ds2 = test_config.setup(preprocess=encoder.preprocess)
    return ds, ds2


if __name__ == "__main__":
    train_config = CamLocDatasetConfig(
        data=Path("/home/n11373598/hpc-home/work/scrstudio_exp/data/dept/1F"),
        split="train",
        feat_name="nothing",
        num_decoder_clusters=50,
        loading_depth=True,
    )
    ds = train_config.setup()
    for i in trange(50):
        ds[i]
