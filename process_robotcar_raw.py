import argparse
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from io import BytesIO
import imagesize
import quaternion

from ace_util import read_nvm_file
from dataset import LMDBReaderConfig


CONDITIONS = [
    "dawn",
    "dusk",
    "night",
    "night-rain",
    "overcast-summer",
    "overcast-winter",
    "rain",
    "snow",
    "sun",
]


def read_train_poses(a_file, cl=False):
    with open(a_file) as file:
        lines = [line.rstrip() for line in file]
    if cl:
        lines = lines[4:]
    name2mat = {}
    for line in lines:
        img_name, *matrix = line.split(" ")
        if len(matrix) == 16:
            matrix = np.array(matrix, float).reshape(4, 4)
        name2mat[img_name] = matrix
    return name2mat


class RobotCarDataset(Dataset):
    images_dir_str: str

    def __init__(self, ds_dir="datasets/robotcar", train=True, evaluate=False):
        self.ds_type = "robotcar"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/3D-models/all-merged/all.nvm"
        self.images_dir = Path(f"{self.ds_dir}/images")
        # self.test_file1 = f"{ds_dir}/robotcar_v2_train.txt"
        self.test_file2 = f"{ds_dir}/robotcar_v2_test.txt"
        self.ds_dir_path = Path(self.ds_dir)
        self.images_dir_str = str(self.images_dir)
        self.train = train
        self.evaluate = evaluate
        if evaluate:
            assert not self.train

        if self.train:
            (
                self.xyz_arr,
                self.image2points,
                self.image2name,
                self.image2info,
                self.image2uvs,
                self.image2pose,
            ) = read_nvm_file(self.sfm_model_dir)
            self.name2image = {v: k for k, v in self.image2name.items()}
            self.img_ids = list(self.image2name.keys())

        else:
            self.ts2cond = {}
            for condition in CONDITIONS:
                all_image_names = list(Path.glob(self.images_dir, f"{condition}/*/*"))

                for name in all_image_names:
                    time_stamp = str(name).split("/")[-1].split(".")[0]
                    self.ts2cond.setdefault(time_stamp, []).append(condition)
            for ts in self.ts2cond:
                assert len(self.ts2cond[ts]) == 3

            self.name2mat = read_train_poses(self.test_file2)
            self.img_ids = list(self.name2mat.keys())

        return

    def _process_id_to_name(self, img_id):
        name = self.image2name[img_id].split("./")[-1]
        name2 = str(self.images_dir / name).replace(".png", ".jpg")
        return name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]
            image_name = self._process_id_to_name(img_id)

        else:
            name0 = self.img_ids[idx]

            time_stamp = str(name0).split("/")[-1].split(".")[0]
            cond = self.ts2cond[time_stamp][0]
            name1 = f"{cond}/{name0}"
            if ".png" in name1:
                name1 = name1.replace(".png", ".jpg")

            image_name = str(self.images_dir / name1)
        return image_name

    def __getitem__(self, idx):

        return self._get_single_item(idx)


def folder2lmdb(image_folder, lmdb_path):
    image_paths = glob(os.path.join(image_folder, '**', '*.*'), recursive=True)
    image_paths = [f for f in image_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_paths.sort()
    lmdb_path = str(lmdb_path)
    if not os.path.isdir(lmdb_path):
        os.makedirs(lmdb_path)
    else:
        shutil.rmtree(lmdb_path)
        print('lmdb path exist, remove it')
        os.makedirs(lmdb_path)
    with open(os.path.join(lmdb_path, 'file_list.txt'), 'w') as f:
        for image_path in image_paths:
            # remove image_folder from image_path
            image_path = os.path.relpath(image_path, image_folder)
            f.write(image_path+'\n')
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                         map_size=2**36, readonly=False,
                         meminit=False, map_async=True,max_dbs=1)
    db=env.open_db(b'images',integerkey=True)
    with env.begin(write=True, db=db) as txn:
        for i, image_path in enumerate(tqdm(image_paths)):
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            txn.put(key=i.to_bytes(4,sys.byteorder), value=image_bytes,append=True)
    env.close()


def parse_nvm_intrinsics(image_folder, name_id, ds):
    lines = []
    dict_ = {
        "left": [400, 500.107605, 511.461426],
        "rear": [400, 508.222931, 498.187378],
        "right": [400, 502.503754, 490.259033],
    }
    for name in tqdm(ds):
        # h, w = imagesize.get(name)
        name = str(name).replace(image_folder + "/", "")
        cam_id = name.split("/")[1]
        f, cx, cy = dict_[cam_id]
        out_line = f"{name} SIMPLE_RADIAL {1024} {1024} {f} {cx} {cy} {0.0}"
        lines.append(out_line)
    with open(f"intrinsics_{name_id}.txt", "w") as f:
        f.write("\n".join(lines))
    return


def run_command(cmd: str, verbose=False):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    return out


def read_lines_from_file(filepath):
    """Read all lines from a text file and return them as a list."""
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip().split(" ")[0] for line in lines]  # Remove trailing newlines


def lmdb_image_shapes(path):
    path=Path(path)
    reader=LMDBReaderConfig(img_type='bytes',db_name="images").setup(root=path)
    image_shapes = []
    for i in tqdm(range(len(reader))):
        bio=BytesIO(reader[i])
        w,h=imagesize.get(bio)
        image_shapes.append((h,w))
    image_shapes=np.stack(image_shapes)
    np.save(path/'image_shapes.npy',image_shapes)

def parse_args():
    parser = argparse.ArgumentParser(description="Process directories")
    parser.add_argument(
        "--main_dir",
        type=str,
        required=True,
        help="Path to the main directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to the save/output directory"
    )
    return parser.parse_args()

def main(ds_main_dir, save_dir):
    conditions = [
        "dawn",
        "dusk",
        "night",
        "night-rain",
        "overcast-summer",
        "overcast-winter",
        "overcast-reference",
        "rain",
        "snow",
        "sun",
    ]
    root = save_dir
    root.mkdir(parents=True, exist_ok=True)

    for c in conditions:
        for l in ["left", "rear", "right"]:
            (
                Path(
                    f"{save_dir}/train/rgb"
                )
                / c
                / l
            ).mkdir(parents=True, exist_ok=True)
            (
                Path(
                    f"{save_dir}/test/rgb"
                )
                / c
                / l
            ).mkdir(parents=True, exist_ok=True)

    # ds_main_dir = "../descriptor-disambiguation/datasets/robotcar"
    # ds_main_dir = (
    #     "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/robotcar"
    # )
    image_folder = f"{ds_main_dir}/images"
    recon_file = f"{ds_main_dir}/3D-models/all-merged/all.nvm"
    test_file = f"{ds_main_dir}/robotcar_v2_test.txt"

    parse_nvm_intrinsics(image_folder, "train",
                         RobotCarDataset(ds_dir=ds_main_dir, train=True))
    parse_nvm_intrinsics(image_folder,
                         "test",
                         RobotCarDataset(ds_dir=ds_main_dir, train=False))

    test_dir = Path(f"{root}/test")
    train_dir = Path(f"{root}/train")
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    cmd = f"colmap image_undistorter_standalone --input_file intrinsics_train.txt --image_path {image_folder} --output_path {root}/train/rgb"
    run_command(cmd, verbose=True)

    cmd = f"colmap image_undistorter_standalone --input_file intrinsics_test.txt --image_path {image_folder} --output_path {root}/test/rgb"
    run_command(cmd, verbose=True)

    # create folders (colmap will not create them)
    folder2lmdb(test_dir / "rgb", test_dir / "rgb_lmdb")

    # if not os.path.exists(test_dir / "rgb_lmdb"):
    #     folder2lmdb(test_dir / "rgb", test_dir / "rgb_lmdb")
    if not os.path.exists(train_dir / "rgb_lmdb"):
        folder2lmdb(train_dir / "rgb", train_dir / "rgb_lmdb")
    if not os.path.exists(train_dir / "image_shapes.npy"):
        lmdb_image_shapes(train_dir)

    ts2cond = {}
    for condition in conditions:
        all_image_names = list(Path.glob(Path(image_folder), f"{condition}/*/*"))

        for name in all_image_names:
            time_stamp = str(name).split("/")[-1]
            ts2cond.setdefault(time_stamp, []).append(condition)
    test_paths = [
        f"{image_folder}/{ts2cond[f_.replace('png', 'jpg').split('/')[-1]][0]}/{f_.replace('png', 'jpg')}"
        for f_ in read_lines_from_file(test_file)
    ]
    test_paths.sort()
    calibrations = np.ones(len(test_paths), dtype=np.float32) * 400
    np.save(test_dir / "calibration.npy", calibrations)

    with open(recon_file, "r") as f:
        reconstruction = f.readlines()

    num_cams = int(reconstruction[2])
    camera_list = [x.split() for x in reconstruction[3 : 3 + num_cams]]
    camera_list.sort(key=lambda x: x[0])

    train_paths = [
        f"{image_folder}/{c_[0][2:].replace('png', 'jpg')}" for c_ in camera_list
    ]
    calibrations = np.ones(len(train_paths), dtype=np.float32) * 400
    np.save(train_dir / "calibration.npy", calibrations)
    all_poses = []
    for cam_idx, camera in enumerate(camera_list):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.quaternion(*[float(r) for r in camera[2:6]]).inverse()
        )
        pose[:3, 3] = [float(r) for r in camera[6:9]]
        all_poses.append(pose)
    all_poses = np.stack(all_poses)
    np.save(train_dir / "poses.npy", all_poses)


if __name__ == "__main__":
    args = parse_args()
    print("Main dir:", args.main_dir)
    print("Save dir:", args.save_dir)
    main(args.main_dir, args.save_dir)
