import argparse
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm
from io import BytesIO
import imagesize
import quaternion

from dataset import LMDBReaderConfig


def lmdb_image_shapes(path):
    path = Path(path)
    reader = LMDBReaderConfig(img_type="bytes", db_name="images").setup(root=path)
    image_shapes = []
    for i in tqdm(range(len(reader))):
        bio = BytesIO(reader[i])
        w, h = imagesize.get(bio)
        image_shapes.append((h, w))
    image_shapes = np.stack(image_shapes)
    np.save(path / "image_shapes.npy", image_shapes)


def folder2lmdb(image_folder, lmdb_path):
    image_paths = glob(os.path.join(image_folder, "**", "*.*"), recursive=True)
    image_paths = [
        f for f in image_paths if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    image_paths.sort()
    lmdb_path = str(lmdb_path)
    if not os.path.isdir(lmdb_path):
        os.makedirs(lmdb_path)
    else:
        shutil.rmtree(lmdb_path)
        print("lmdb path exist, remove it")
        os.makedirs(lmdb_path)
    with open(os.path.join(lmdb_path, "file_list.txt"), "w") as f:
        for image_path in image_paths:
            # remove image_folder from image_path
            image_path = os.path.relpath(image_path, image_folder)
            f.write(image_path + "\n")
    env = lmdb.open(
        lmdb_path,
        subdir=os.path.isdir(lmdb_path),
        map_size=2**36,
        readonly=False,
        meminit=False,
        map_async=True,
        max_dbs=1,
    )
    db = env.open_db(b"images", integerkey=True)
    with env.begin(write=True, db=db) as txn:
        for i, image_path in enumerate(tqdm(image_paths)):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            txn.put(key=i.to_bytes(4, sys.byteorder), value=image_bytes, append=True)
    env.close()
    return image_paths


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
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)

    # ds_main_dir = (
    #     "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/aachen_v1.1"
    # )
    image_folder = f"{ds_main_dir}/images_upright"
    recon_file = f"{ds_main_dir}/3D-models/aachen_v_1_1/aachen_v_1_1.nvm"

    db_file = f"{ds_main_dir}/3D-models/aachen_v_1_1/database_intrinsics_v1_1.txt"
    day_file = f"{ds_main_dir}/queries/day_time_queries_with_intrinsics.txt"
    night_file = f"{ds_main_dir}/queries/night_time_queries_with_intrinsics.txt"

    # create folders (colmap will not create them)
    test_dir = Path(f"{root}/test")
    train_dir = Path(f"{root}/train")
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    fn_f = []
    for file in (day_file, night_file):
        all_subdirs = set()
        with open(file, "r") as f:
            for line in f:
                line = line.split()
                fn_f.append((line[0], line[4]))
                all_subdirs.add((test_dir / "rgb" / line[0]).parent)
        for subdir in all_subdirs:
            os.makedirs(subdir, exist_ok=True)
        cmd = f"colmap image_undistorter_standalone --input_file {file} --image_path {image_folder} --output_path {root}/test/rgb"
        print(cmd)
        run_command(cmd, verbose=True)
    if not os.path.exists(test_dir / "rgb_lmdb"):
        folder2lmdb(test_dir / "rgb", test_dir / "rgb_lmdb")
    fn_f.sort(key=lambda x: x[0])
    calibrations = np.array([float(f) for _, f in fn_f])
    np.save(test_dir / "calibration.npy", calibrations)

    fn_f = []
    all_subdirs = set()
    with open(db_file, "r") as f:
        for line in f:
            line = line.split()
            fn_f.append((line[0], line[4]))
            all_subdirs.add((train_dir / "rgb" / line[0]).parent)

    if not os.path.exists(train_dir / "rgb"):
        for subdir in all_subdirs:
            os.makedirs(subdir, exist_ok=True)
        cmd = f"colmap image_undistorter_standalone --input_file {db_file} --image_path {image_folder} --output_path {root}/train/rgb"
        print(cmd)
        run_command(cmd, verbose=True)
    if not os.path.exists(train_dir / "rgb_lmdb"):
        folder2lmdb(train_dir / "rgb", train_dir / "rgb_lmdb")
    if not os.path.exists(train_dir / "image_shapes.npy"):
        lmdb_image_shapes(train_dir)
    fn_f.sort(key=lambda x: x[0])
    calibrations = np.array([float(f) for _, f in fn_f])
    np.save(train_dir / "calibration.npy", calibrations)

    with open(recon_file, "r") as f:
        reconstruction = f.readlines()

    num_cams = int(reconstruction[2])
    camera_list = [x.split() for x in reconstruction[3 : 3 + num_cams]]
    camera_list.sort(key=lambda x: x[0])

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
