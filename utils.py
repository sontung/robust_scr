import argparse
import csv
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path
from types import SimpleNamespace

import poselib
import scipy
import skimage
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation as R

import PIL.Image
import cv2
import numpy as np
import torch
import torchvision.transforms as tvf
import torchvision
import faiss
from PIL import Image
from tqdm import tqdm

from tqdm import trange

from salad_model import FullModel


@dataclass
class CSRGraph:
    indices: torch.Tensor
    indptr: torch.Tensor
    rowsizes: torch.Tensor

    @staticmethod
    def from_csr_array(csr_array: scipy.sparse.csr_array, device: torch.device):
        return CSRGraph(
            torch.tensor(
                np.ascontiguousarray(csr_array.indices),
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(
                np.ascontiguousarray(csr_array.indptr), dtype=torch.int32, device=device
            ),
            torch.tensor(np.diff(csr_array.indptr), dtype=torch.int32, device=device),
        )

    def sample_neighbors(self, start: torch.Tensor, generator: torch.Generator):
        offset = (
            torch.randint(
                0,
                torch.iinfo(torch.int32).max,
                (start.shape[0],),
                device=start.device,
                generator=generator,
            )
            % self.rowsizes[start]
        )
        return self.indices[self.indptr[start] + offset]


def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing="ij")
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output


def read_nvm_file(file_name, return_rgb=False):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    nb_cameras = int(lines[2])
    image2info = {}
    image2pose = {}
    image2name = {}
    for i in range(nb_cameras):
        cam_info = lines[3 + i]
        if "\t" not in cam_info:
            img_name, *info = cam_info.split(" ")
            focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = map(float, info)
        else:
            img_name, info = cam_info.split("\t")
            focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = map(float, info.split(" "))
        image2name[i] = img_name
        image2info[i] = [focal, radial]
        image2pose[i] = [qw, qx, qy, qz, tx, ty, tz]
    nb_points = int(lines[4 + nb_cameras])
    image2points = defaultdict(list)
    image2uvs = defaultdict(list)
    xyz_arr = np.zeros((nb_points, 3), np.float64)
    rgb_arr = np.zeros((nb_points, 3), np.uint8)
    start_idx = 5 + nb_cameras
    for j in trange(nb_points, desc="Reading 3D points"):
        point_info = lines[start_idx + j].split()
        features_info = point_info[7:]
        nb_features = int(point_info[6])

        x, y, z = float(point_info[0]), float(point_info[1]), float(point_info[2])
        r, g, b = float(point_info[3]), float(point_info[4]), float(point_info[5])
        rgb_arr[j] = (r, g, b)
        xyz_arr[j] = (x, y, z)

        for k in range(nb_features):
            base = k * 4
            image_id = int(features_info[base])
            image2points[image_id].append(j)
            if return_rgb:
                continue
            u = float(features_info[base + 2])
            v = float(features_info[base + 3])
            image2uvs[image_id].append([u, v])

    if return_rgb:
        return xyz_arr, image2points, image2name, rgb_arr
    else:
        return xyz_arr, image2points, image2name, image2info, image2uvs, image2pose


ratios_resolutions = {
    224: {1.0: [224, 224]},
    512: {
        4 / 3: [512, 384],
        32 / 21: [512, 336],
        16 / 9: [512, 288],
        2 / 1: [512, 256],
        16 / 5: [512, 160],
    },
}


def get_resize_function(maxdim, patch_size, H, W, is_mask=False):
    if [max(H, W), min(H, W)] in ratios_resolutions[maxdim].values():
        return lambda x: x, np.eye(3), np.eye(3)
    else:
        target_HW = get_HW_resolution(H, W, maxdim=maxdim, patchsize=patch_size)

        ratio = W / H
        target_ratio = target_HW[1] / target_HW[0]
        to_orig_crop = np.eye(3)
        to_rescaled_crop = np.eye(3)
        if abs(ratio - target_ratio) < np.finfo(np.float32).eps:
            crop_W = W
            crop_H = H
        elif ratio - target_ratio < 0:
            crop_W = W
            crop_H = int(W / target_ratio)
            to_orig_crop[1, 2] = (H - crop_H) / 2.0
            to_rescaled_crop[1, 2] = -(H - crop_H) / 2.0
        else:
            crop_W = int(H * target_ratio)
            crop_H = H
            to_orig_crop[0, 2] = (W - crop_W) / 2.0
            to_rescaled_crop[0, 2] = -(W - crop_W) / 2.0

        crop_op = tvf.CenterCrop([crop_H, crop_W])

        if is_mask:
            resize_op = tvf.Resize(
                size=target_HW, interpolation=tvf.InterpolationMode.NEAREST_EXACT
            )
        else:
            resize_op = tvf.Resize(size=target_HW)
        to_orig_resize = np.array(
            [[crop_W / target_HW[1], 0, 0], [0, crop_H / target_HW[0], 0], [0, 0, 1]]
        )
        to_rescaled_resize = np.array(
            [[target_HW[1] / crop_W, 0, 0], [0, target_HW[0] / crop_H, 0], [0, 0, 1]]
        )

        op = tvf.Compose([crop_op, resize_op])

        return op, to_rescaled_resize @ to_rescaled_crop, to_orig_crop @ to_orig_resize


def get_HW_resolution(H, W, maxdim, patchsize=16):
    assert (
        maxdim in ratios_resolutions
    ), "Error, maxdim can only be 224 or 512 for now. Other maxdims not implemented yet."
    ratios_resolutions_maxdim = ratios_resolutions[maxdim]
    mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
    islandscape = W >= H
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions_maxdim[selkey]
    # check patchsize and make sure output resolution is a multiple of patchsize
    if isinstance(patchsize, tuple):
        assert (
            len(patchsize) == 2
            and isinstance(patchsize[0], int)
            and isinstance(patchsize[1], int)
        ), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
        patchsize = patchsize[0]
    assert max(res) == maxdim
    assert min(res) in mindims
    return res[::-1] if islandscape else res  # return HW


def read_image_by_hloc(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image_by_hloc(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


def read_csv_to_dict(filepath):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        key_col = reader.fieldnames[0]
        result = {}
        for row in reader:
            key = row[key_col]
            result[key] = [v for k, v in row.items() if k != key_col]
        return result


def read_and_preprocess(name, conf, image=None):
    if image is None:
        image = read_image_by_hloc(name, conf.grayscale)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    scale = 1

    if conf.resize_max and (conf.resize_force or max(size) > conf.resize_max):
        scale = conf.resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = resize_image_by_hloc(image, size_new, conf.interpolation)

    if conf.grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0
    return image, scale


def quat_trans_to_matrix_scipy(quat, trans):
    # quat: [q_w, q_x, q_y, q_z]
    # trans: [t_x, t_y, t_z]

    # Create the rotation matrix from the quaternion
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    rotation = rotation.as_matrix()  # 3x3 rotation matrix

    # Create the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = trans

    return T


def matrix_to_quaternion_and_translation(matrix):
    # Ensure the input is 4x4
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix."

    # Extract the rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Extract the translation components
    translation = matrix[:3, 3]
    t_x, t_y, t_z = translation

    # Convert to quaternion using scipy
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()  # Returns [x, y, z, w]

    # Rearrange quaternion to [w, x, y, z]
    q_x, q_y, q_z, q_w = quaternion
    q1 = np.array([q_w, q_x, q_y, q_z, t_x, t_y, t_z])

    return q1


def read_kp_and_desc(name, features_h5):
    img_id = "/".join(name.split("/")[-2:])
    try:
        grp = features_h5[img_id]
    except KeyError:
        grp = features_h5[name]

    pred = {k: np.array(v) for k, v in grp.items()}
    if "scale" not in pred:
        scale = 1
    else:
        scale = pred["scale"]
    keypoints = (pred["keypoints"] + 0.5) / scale - 0.5

    if "descriptors" in pred:
        descriptors = pred["descriptors"].T
    else:
        descriptors = None
    return keypoints, descriptors


def write_to_h5_file(fd, name, dict_, process_name=True):
    if process_name:
        name = "/".join(name.split("/")[-2:])
    try:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        for k, v in dict_.items():
            grp.create_dataset(k, data=v)
    except OSError as error:
        if "No space left on device" in error.args[0]:
            print("No space left")
            del grp, fd[name]
        raise error
    return name


def transform_kp(kp, image_resize, angle, scale_factor):
    kp = kp * scale_factor

    h = image_resize.size(1)
    w = image_resize.size(2)

    translate = {"x": 0, "y": 0}

    shear = {"x": -0.0, "y": -0.0}
    scale = {"x": 1.0, "y": 1.0}

    rotate = -angle
    shift_x = w / 2 - 0.5
    shift_y = h / 2 - 0.5

    matrix_to_topleft = skimage.transform.SimilarityTransform(
        translation=[-shift_x, -shift_y]
    )
    matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
    matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
    matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
    matrix_transforms = skimage.transform.AffineTransform(
        scale=(scale["x"], scale["y"]),
        translation=(translate["x"], translate["y"]),
        rotation=np.deg2rad(rotate),
        shear=np.deg2rad(shear["x"]),
    )
    matrix_to_center = skimage.transform.SimilarityTransform(
        translation=[shift_x, shift_y]
    )
    matrix = (
        matrix_to_topleft
        + matrix_shear_y_rot
        + matrix_shear_y
        + matrix_shear_y_rot_inv
        + matrix_transforms
        + matrix_to_center
    )

    kp2 = np.copy(kp)
    # kp2[:, [1, 0]] = kp2[:, [0, 1]]
    kp2 = np.expand_dims(kp2, 0)
    kp2 = cv2.transform(kp2, matrix.params[:2]).squeeze()

    return kp2


def quaternion_to_matrix(pose_q):
    qw, qx, qy, qz, tx, ty, tz = pose_q
    # Normalize the quaternion to ensure it's valid
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Rotation matrix from quaternion
    r = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    return r, np.array([tx, ty, tz])


def project_3d_to_2d_matrix(cam_points, fx, cx, cy, k1):
    # Convert points_3d to a (k, 3) numpy array
    fy = fx
    points_cam = cam_points.cpu().numpy().T

    # Extract X_c, Y_c, Z_c
    X_c, Y_c, Z_c = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Perspective projection (normalized coordinates)
    x_n = X_c / Z_c
    y_n = Y_c / Z_c

    # Radial distortion
    r2 = x_n**2 + y_n**2  # Shape (k,)
    distortion = 1 + k1 * r2  # Shape (k,)
    x_d = x_n * distortion  # Shape (k,)
    y_d = y_n * distortion  # Shape (k,)

    # Convert to pixel coordinates
    u = fx * x_d + cx  # Shape (k,)
    v = fy * y_d + cy  # Shape (k,)

    # Stack u and v to form the (k, 2) output
    uv = np.vstack((u, v)).T  # Shape (k, 2)
    return uv


def read_intrinsic(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    name2params = {}
    for line in lines:
        img_name, cam_type, w, h, f, cx, cy, k = line.split(" ")
        f, cx, cy, k = map(float, [f, cx, cy, k])
        w, h = map(int, [w, h])
        name2params[img_name] = [cam_type, w, h, f, cx, cy, k]
    return name2params


def to_csr_format(membership):
    num_nodes = len(membership)

    # Create node-to-group matrix (shape: num_nodes x num_groups)
    num_groups = membership.max() + 1
    rows = np.arange(num_nodes)
    cols = membership
    data = np.ones(num_nodes)
    node_to_group = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_groups))

    # Co-membership: dot product of node-group matrix with its transpose
    # (i, j) will be non-zero if node i and j share a group
    co_membership = (node_to_group @ node_to_group.T).astype(bool).astype(int)
    return co_membership


def pose_estimate(keypoints, scene_coords, camera, ransac_opt, gt_pose):
    pose, info = poselib.estimate_absolute_pose(
        keypoints, scene_coords, camera, ransac_opt, {}
    )
    pred_pose = np.eye(4)
    R_pred = pose.R.T
    pred_pose[:3, :3] = R_pred
    pred_pose[:3, 3] = -R_pred @ pose.t

    return {
        "num_inliers": info["num_inliers"],
        "inlier_ratio": info["inlier_ratio"],
        "t_err": np.linalg.norm(gt_pose[:3, 3] - pred_pose[:3, 3]),
        "r_err": np.linalg.norm(cv2.Rodrigues(gt_pose[:3, :3] @ pose.R)[0])
        * 180
        / math.pi,
        "pose_q": pose.q,
        "pose_t": pose.t,
        "position": pred_pose[:3, 3],
        "position_gt": gt_pose[:3, 3],
        "xyz": scene_coords,
        "inliers": info["inliers"],
    }


def get_options():
    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Fast training of a scene coordinate regression network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scene",
        type=Path,
        default="../ace/datasets/Cambridge_GreatCourt",
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "--db_dir",
        type=Path,
        default="",
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "--pca_path",
        type=Path,
        default="proc/pcad3LB_128.pth",
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "--local_desc",
        type=str,
        default="dedode",
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "--output_map_file",
        type=Path,
        default="head.pth",
        help="target file for the trained network",
    )

    parser.add_argument(
        "--global_feat", type=_strtobool, default=False, help="Use global feature."
    )

    parser.add_argument(
        "--test_mode", type=_strtobool, default=False, help="Use global feature."
    )

    parser.add_argument(
        "--focus_tune", type=_strtobool, default=False, help="Use global feature."
    )

    parser.add_argument(
        "--normalize_inputs", type=_strtobool, default=False, help="Use global feature."
    )

    parser.add_argument(
        "--reuse_buffer", type=_strtobool, default=True, help="Use global feature."
    )

    parser.add_argument(
        "--use_salad", type=_strtobool, default=True, help="Use global feature."
    )

    parser.add_argument(
        "--graph_aug", type=_strtobool, default=True, help="Use global feature."
    )

    parser.add_argument(
        "--feat_name",
        type=str,
        default="checkpoints/desc_dino.npy",
        help="global feature name.",
    )

    parser.add_argument(
        "--membership",
        type=str,
        default="checkpoints/pose_overlap.npz",
        help="global feature name.",
    )

    parser.add_argument(
        "--feat_name_test",
        type=str,
        default="checkpoints/desc_dino_test.npy",
        help="global feature name.",
    )

    parser.add_argument(
        "--feat_noise_std", type=float, default=0.1, help="global feature noise std."
    )

    parser.add_argument(
        "--num_decoder_clusters", type=int, default=1, help="number of decoder clusters"
    )

    parser.add_argument(
        "--head_channels",
        type=int,
        default=768,
        help="depth of the regression head, defines the map size",
    )

    parser.add_argument(
        "--grad_acc",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mlp_ratio", type=float, default=1.0, help="mlp ratio for res blocks"
    )

    parser.add_argument(
        "--lambda_w",
        type=float,
        default=0.5,
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "--encoder_path",
        type=Path,
        default=Path(__file__).parent / "ace_encoder_pretrained.pt",
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--num_head_blocks",
        type=int,
        default=1,
        help="depth of the regression head, defines the map size",
    )

    parser.add_argument(
        "--learning_rate_min",
        type=float,
        default=0.0005,
        help="lowest learning rate of 1 cycle scheduler",
    )

    parser.add_argument(
        "--learning_rate_max",
        type=float,
        default=0.003,
        help="highest learning rate of 1 cycle scheduler",
    )

    parser.add_argument(
        "--iter_output",
        type=int,
        default=100000,
        help="maximum number of iterations for the training loop",
    )

    parser.add_argument(
        "--training_buffer_size",
        type=int,
        default=32000000,
        help="number of patches in the training buffer",
    )

    parser.add_argument(
        "--samples_per_image",
        type=int,
        default=1024,
        help="number of patches drawn from each image when creating the buffer",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=40960,
        help="number of patches for each parameter update (has to be a multiple of 512)",
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100000,
        help="maximum number of iterations for the training loop",
    )

    parser.add_argument(
        "--repro_loss_hard_clamp",
        type=int,
        default=1000,
        help="hard clamping threshold for the reprojection losses",
    )

    parser.add_argument(
        "--repro_loss_soft_clamp",
        type=int,
        default=50,
        help="soft clamping threshold for the reprojection losses",
    )

    parser.add_argument(
        "--repro_loss_soft_clamp_min",
        type=int,
        default=1,
        help="minimum value of the soft clamping threshold when using a schedule",
    )

    parser.add_argument(
        "--use_half", type=_strtobool, default=True, help="train with half precision"
    )

    parser.add_argument(
        "--use_homogeneous",
        type=_strtobool,
        default=True,
        help="train with half precision",
    )

    parser.add_argument(
        "--use_aug", type=_strtobool, default=True, help="Use any augmentation."
    )

    parser.add_argument(
        "--depth_init", type=_strtobool, default=False, help="Use any augmentation."
    )

    parser.add_argument(
        "--aug_rotation", type=int, default=15, help="max inplane rotation angle"
    )

    parser.add_argument("--aug_scale", type=float, default=1.5, help="max scale factor")

    parser.add_argument(
        "--image_resolution", type=int, default=480, help="base image resolution"
    )

    parser.add_argument(
        "--repro_loss_type",
        type=str,
        default="dyntanh",
        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
        help="Loss function on the reprojection error. Dyn varies the soft clamping threshold",
    )

    parser.add_argument(
        "--repro_loss_schedule",
        type=str,
        default="circle",
        choices=["circle", "linear"],
        help="How to decrease the softclamp threshold during training, circle is slower first",
    )

    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.1,
        help="enforce minimum depth of network predictions",
    )

    parser.add_argument(
        "--depth_target",
        type=float,
        default=10,
        help="default depth to regularize training",
    )

    parser.add_argument(
        "--depth_max",
        type=float,
        default=1000,
        help="enforce maximum depth of network predictions",
    )

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="split the training sequence in this number of clusters. disabled by default",
    )

    parser.add_argument(
        "--cluster_idx",
        type=int,
        default=None,
        help="train on images part of this cluster. required only if --num_clusters is set.",
    )

    parser.add_argument(
        "--debug_mode",
        type=int,
        default=0,
    )

    options_ = parser.parse_args()
    return options_


def _strtobool(x):
    return bool(strtobool(x))


def read_embeddings(path="checkpoints/node2vec_embeddings.txt"):
    with open(path, "r") as file:
        lines = file.readlines()
    nb_vectors, dim = map(int, lines[0].strip().split(" "))
    vectors = [line.strip() for line in lines][1:]
    indices = [int(vector.split(" ")[0]) for vector in vectors]
    nb_vectors = max(indices) + 1
    embeddings = np.random.rand(nb_vectors, dim)

    for vector in vectors:
        vector = map(float, vector.split(" "))
        ind, *numbers = vector
        ind = int(ind)
        embeddings[ind] = numbers

    return embeddings


def stack_images_horizontally(img1, img2):
    """
    Resize and stack two images horizontally.

    Parameters:
    - img1, img2: Input images as NumPy arrays.

    Returns:
    - stacked_img: A single image with img1 and img2 stacked horizontally.
    """
    # Get the height of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Determine the common height (use the larger one or pick one)
    common_height = max(h1, h2)

    # Resize both images to have the same height
    def resize_to_height(img, target_height):
        h, w = img.shape[:2]
        scale = target_height / h
        new_width = int(w * scale)
        return cv2.resize(img, (new_width, target_height))

    img1_resized = resize_to_height(img1, common_height)
    img2_resized = resize_to_height(img2, common_height)

    # Stack horizontally
    stacked_img = np.hstack((img1_resized, img2_resized))

    return stacked_img


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def whitenapply(X, m, P, dimensions=None):
    if not dimensions:
        dimensions = P.shape[0]

    X = np.dot(P[:dimensions, :], X - m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return X.copy(order="C")


def pcawhitenlearn(X):
    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc, Xc.T)
    Xcov = (Xcov + Xcov.T) / (2 * N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)
    if P.dtype == "complex128":
        P = np.real(P)
        print("Warning: complex numbers in eigenvec and eigenvals")

    return m, P


def run_salad_model(ds2, image_root_dir):
    checkpoint_dir = ds2.config.data / ds2.config.split / "desc_salad.npy"
    if checkpoint_dir.exists():
        return
    model = FullModel(pretrained=True)
    model.cuda()
    model.eval()
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    mat = np.zeros((len(ds2.rgb_files), 8448))
    with torch.no_grad():
        for idx, name in enumerate(tqdm(ds2.rgb_files)):
            image = Image.open(image_root_dir / name).convert("RGB")
            image = transform(image)
            image_descriptor = model(image.unsqueeze(0).cuda())
            image_descriptor = image_descriptor.squeeze().cpu().numpy()
            mat[idx] = image_descriptor
    np.save(checkpoint_dir, mat)


def inspect_gl_descriptors(salad_train, salad_test, ds, ds2):
    train_vecs = np.ascontiguousarray(salad_train, dtype=np.float32)
    test_vecs = np.ascontiguousarray(salad_test, dtype=np.float32)

    dim = train_vecs.shape[1]

    # --- Build index ---
    # Use L2 distance; swap for faiss.IndexFlatIP for cosine (after L2-norm)
    index = faiss.IndexFlatL2(dim)
    index.add(train_vecs)

    distances, indices = index.search(test_vecs, 1)
    indices = indices.flatten()
    pos0 = ds.pose_values[:, :3, 3]
    pos1 = ds2.pose_values[:, :3, 3]
    err = np.mean(np.abs(pos0[indices] - pos1), 1)
    r1 = np.mean(err < 1) * 100
    r5 = np.mean(err < 5) * 100
    print(f"Global descriptors")
    print(f"  Recall @ 1m  : {r1:.2f}%")
    print(f"  Recall @ 5m  : {r5:.2f}%")


def draw_feature_matches_cv2(uv0, uv1, name0, name1, out_path):
    """
    Draw feature matches using **only OpenCV** and save the output image.

    uv0: (N,2) keypoints in image 0
    uv1: (N,2) keypoints in image 1
    name0, name1: image paths
    out_path: where to save the visualization
    """

    # Load images
    img0 = cv2.imread(name0)
    img1 = cv2.imread(name1)

    if img0 is None:
        raise ValueError(f"Cannot load image: {name0}")
    if img1 is None:
        raise ValueError(f"Cannot load image: {name1}")

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Canvas to place images side-by-side
    h = max(h0, h1)
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)

    canvas[:h0, :w0] = img0
    canvas[:h1, w0 : w0 + w1] = img1

    # Shift uv1 x-coordinates to match the concatenated image
    uv1_shifted = uv1.copy()
    uv1_shifted[:, 0] += w0

    # Draw matches
    for p0, p1 in zip(uv0, uv1_shifted):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        p0 = tuple(map(int, p0))
        p1 = tuple(map(int, p1))

        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

    # Save output file
    cv2.imwrite(out_path, canvas)


def draw_feature_matches_cv2_w_images(uv0, uv1, img0, img1, out_path):

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Canvas to place images side-by-side
    h = max(h0, h1)
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)

    canvas[:h0, :w0] = img0
    canvas[:h1, w0 : w0 + w1] = img1

    # Shift uv1 x-coordinates to match the concatenated image
    uv1_shifted = uv1.copy()
    uv1_shifted[:, 0] += w0

    # Draw matches
    for p0, p1 in zip(uv0, uv1_shifted):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        p0 = tuple(map(int, p0))
        p1 = tuple(map(int, p1))

        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

    # Save output file
    cv2.imwrite(out_path, canvas)


def check_npy_determinism(path, array, atol=1e-6):
    path = Path(path)

    if not path.exists():
        print(f"[NEW] {path} does not exist yet.")
        return

    try:
        existing = np.load(path)

        if existing.shape != array.shape:
            print(f"[DIFF SHAPE] {path}: {existing.shape} vs {array.shape}")
            return

        if np.allclose(existing, array, atol=atol):
            print(f"[DETERMINISTIC] {path} matches existing file.")
        else:
            max_diff = np.max(np.abs(existing - array))
            print(f"[NON-DETERMINISTIC] {path} differs (max diff: {max_diff:.3e})")

    except Exception as e:
        print(f"[ERROR] Could not compare {path}: {e}")


def get_patch_centres(image_ori, patch_size=14, proc_w=224, proc_h=224):

    ORIG_H, ORIG_W = image_ori.shape[:2]
    assert ORIG_H > 10 and ORIG_W > 10
    num_ph = proc_h // patch_size
    num_pw = proc_w // patch_size
    grid_y, grid_x = np.meshgrid(np.arange(num_ph), np.arange(num_pw), indexing="ij")

    cx = (grid_x.flatten() * patch_size + patch_size / 2) * (ORIG_W / proc_w)
    cy = (grid_y.flatten() * patch_size + patch_size / 2) * (ORIG_H / proc_h)

    return np.stack([cx, cy], axis=1).astype(np.float32)  # [P, 2]


def patch_mask_overlap(
    pixel_xy_topleft, masks, patch_size_x, patch_size_y, iou_thresh=0.3
):
    """
    pixel_xy_topleft : float32 [P, 2]  patch top-left corners (x, y)
    masks            : bool [N, H, W]
    Returns          : bool [P], float32 [P] max overlap ratio per patch
    """
    H, W = masks.shape[1], masks.shape[2]
    px0 = np.clip(pixel_xy_topleft[:, 0].astype(int), 0, W - 1)
    py0 = np.clip(pixel_xy_topleft[:, 1].astype(int), 0, H - 1)
    px1 = np.clip((pixel_xy_topleft[:, 0] + patch_size_x).astype(int), 0, W)
    py1 = np.clip((pixel_xy_topleft[:, 1] + patch_size_y).astype(int), 0, H)

    patch_area = patch_size_x * patch_size_y
    max_overlap = np.zeros(len(pixel_xy_topleft), dtype=np.float32)

    for mask in masks:
        for i in range(len(pixel_xy_topleft)):
            patch_region = mask[py0[i] : py1[i], px0[i] : px1[i]]
            inter = patch_region.sum()
            if inter == 0:
                continue
            max_overlap[i] = max(max_overlap[i], inter / patch_area)

    return max_overlap >= iou_thresh, max_overlap


def process_sam3_masks(masks, dino_patch_size=14, proc_w=224, proc_h=224):
    N, H_m, W_m = masks.shape  # native mask resolution

    mask_areas = np.sum(masks, axis=(1, 2))
    sorted_m_indices = np.argsort(mask_areas)

    processed_pixels = np.zeros((H_m, W_m), dtype=bool)
    observations = []
    m_indices = []
    for m_idx in sorted_m_indices:
        current_mask = masks[m_idx] > 0
        unique_mask_region = current_mask & ~processed_pixels

        original_pixel_count = mask_areas[m_idx]
        if original_pixel_count == 0:
            continue

        unique_pixel_count = np.sum(unique_mask_region)
        if (unique_pixel_count / original_pixel_count) < 0.5:
            continue

        processed_pixels |= current_mask

        ORIG_H, ORIG_W = unique_mask_region.shape[:2]
        patch_size_x = dino_patch_size * (ORIG_W / proc_w)
        patch_size_y = dino_patch_size * (ORIG_H / proc_h)

        # get_patch_centres returns centres — shift to top-left for bbox IoU
        pixel_xy = get_patch_centres(
            unique_mask_region, patch_size=dino_patch_size, proc_w=proc_w, proc_h=proc_h
        )
        pixel_xy_topleft = pixel_xy - np.array([patch_size_x / 2, patch_size_y / 2])

        inside, overlap_scores = patch_mask_overlap(
            pixel_xy_topleft,
            unique_mask_region[None],
            patch_size_x,
            patch_size_y,
            iou_thresh=0.2,
        )

        # obs = {
        #     "mask": unique_mask_region.astype(float)*255,
        #     # "chosen_patches": inside.astype(int).tolist(),
        # }
        observations.append(unique_mask_region.astype(float) * 255)
        m_indices.append(m_idx)

    return observations, m_indices


def find_dino_patch_coords(
    unique_mask_region, dino_patch_size=14, proc_w=224, proc_h=224
):
    ORIG_H, ORIG_W = unique_mask_region.shape[:2]
    patch_size_x = dino_patch_size * (ORIG_W / proc_w)
    patch_size_y = dino_patch_size * (ORIG_H / proc_h)

    # get_patch_centres returns centres — shift to top-left for bbox IoU
    pixel_xy = get_patch_centres(
        unique_mask_region, patch_size=dino_patch_size, proc_w=proc_w, proc_h=proc_h
    )
    pixel_xy_topleft = pixel_xy - np.array([patch_size_x / 2, patch_size_y / 2])

    inside, overlap_scores = patch_mask_overlap(
        pixel_xy_topleft,
        unique_mask_region[None],
        patch_size_x,
        patch_size_y,
        iou_thresh=0.2,
    )
    return inside, pixel_xy_topleft[inside]


def visualize_stem_masks(
    image, masks, patches, dino_patch_size=14, proc_w=224, proc_h=224
):
    """
    Visualizes masks with unique random colors and patch selections on a given input image.

    Args:
        image (np.ndarray): The input image in BGR format [H, W, 3].
        masks (np.ndarray): Boolean or uint8 mask array [N, H, W].
        patches (list or np.ndarray): Bounding patch arrays or coordinates.
        dino_patch_size (int): Base patch size dimension. Defaults to 14.
        proc_w (int): Processing width baseline used for scaling. Defaults to 224.
        proc_h (int): Processing height baseline used for scaling. Defaults to 224.

    Returns:
        np.ndarray: The visualized image with mask overlays and patch bounding boxes.
    """
    # Create a deep copy to prevent mutating the original input array in place
    vis = image.copy()
    ORIG_H, ORIG_W = vis.shape[:2]

    # Calculate dynamic patch scaling ratios
    patch_size_x = dino_patch_size * (ORIG_W / proc_w)
    patch_size_y = dino_patch_size * (ORIG_H / proc_h)

    # 1. Draw masks as a semi-transparent overlay with random colors
    if masks is not None and len(masks) > 0:
        overlay = vis.copy()

        for mask in masks:
            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue

            # Generate a random vibrant BGR color (avoiding completely dark colors)
            color_array = np.random.randint(50, 256, size=3, dtype=np.uint8)
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))

            # Blend the random color onto the overlay canvas copy
            overlay[mask_bool] = (vis[mask_bool] * 0.4 + color_array * 0.6).astype(
                np.uint8
            )

            # Draw contours using the matching color for crisp structural boundaries
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        vis = overlay

    # 2. Draw selected bounding patch boxes
    if patches is not None:
        for pixel_xy in patches:
            for cx, cy in pixel_xy.astype(int):
                px0, py0 = cx, cy
                px1 = int(cx + patch_size_x)
                py1 = int(cy + patch_size_y)
                # Clip values internally to safely handle edge boundaries near borders
                px1 = min(px1, ORIG_W)
                py1 = min(py1, ORIG_H)
                cv2.rectangle(vis, (px0, py0), (px1, py1), (220, 0, 0), 1)

    return vis





def visualize_topk_retrievals(
    obs_i, list_obs_j, path_i, list_path_j, output_dir, name="", write_img=False,
):
    """Draws a query image and its top-k retrievals side-by-side with mask overlays.

    Stacked right-to-left from query: [ Query Image | Retrieval 1 | Retrieval 2 | ... ]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the main query image
    img_i = cv2.imread(path_i)
    if img_i is None:
        print(f"⚠️ Failed to load query image: {path_i}")
        return

    # Ensure inputs are lists for iteration
    if not isinstance(list_path_j, list):
        list_path_j = [list_path_j]
    if not isinstance(list_obs_j, list):
        list_obs_j = [list_obs_j]

    DINO_GRID = 16
    h_target = 600  # Standardized height for stacking

    def overlay_dino_bbox(img, obs, color, thickness=5):
        H, W = img.shape[:2]
        cell_h = H / DINO_GRID
        cell_w = W / DINO_GRID

        # Reshape the 1D list of 256 bits back into its true 16x16 spatial matrix layout
        chosen_mask_2d = np.array(obs["chosen_patches"]).reshape(
            DINO_GRID, DINO_GRID
        )

        # Find coordinates where patches are active
        grid_rows, grid_cols = np.where(chosen_mask_2d > 0)
        num_active_patches = len(grid_rows)

        # Draw a tight bounding box around the active patches if any exist
        if num_active_patches > 0:
            min_r, max_r = np.min(grid_rows), np.max(grid_rows)
            min_c, max_c = np.min(grid_cols), np.max(grid_cols)

            # Map back to continuous absolute image pixel space smoothly
            x0 = int(round(min_c * cell_w))
            y0 = int(round(min_r * cell_h))
            x1 = int(round((max_c + 1) * cell_w))
            y1 = int(round((max_r + 1) * cell_h))

            # Draw the bounding box outline
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

        return img

    def overlay_dino_patches(img, obs, patch_color, grid_color=(60, 60, 60)):
        H, W = img.shape[:2]
        cell_h = H / DINO_GRID
        cell_w = W / DINO_GRID

        # Patch Overlay Layer
        patch_overlay = img.copy()
        chosen_mask_2d = np.array(obs["chosen_patches"]).reshape(
            DINO_GRID, DINO_GRID
        )
        grid_rows, grid_cols = np.where(chosen_mask_2d > 0)
        num_active_patches = len(grid_rows)

        for pr, pc in zip(grid_rows, grid_cols):
            x0 = int(round(pc * cell_w))
            y0 = int(round(pr * cell_h))
            x1 = int(round((pc + 1) * cell_w))
            y1 = int(round((pr + 1) * cell_h))
            cv2.rectangle(
                patch_overlay, (x0, y0), (x1, y1), patch_color, cv2.FILLED
            )

        # Alpha blend
        img = cv2.addWeighted(img, 0.75, patch_overlay, 0.25, 0)

        return img

    # Start our canvas list with the Query Image on the far left
    tiles_to_stack = []

    # 2. Process the query image (img_i)
    img_i = overlay_dino_patches(img_i, obs_i, patch_color=(0, 220, 0))
    img_i = overlay_dino_bbox(img_i, obs_i, (0, 0, 255))
    w_i = int(img_i.shape[1] * (h_target / img_i.shape[0]))
    tiles_to_stack.append(cv2.resize(img_i, (w_i, h_target)))

    # 3. Process and append all retrieval images (img_j) to the right
    for idx, (path_j, obs_j) in enumerate(zip(list_path_j, list_obs_j)):
        img_j = cv2.imread(path_j)
        if img_j is None:
            print(f"⚠️ Failed to load retrieval image at index {idx}: {path_j}")
            continue

        img_j = overlay_dino_patches(img_j, obs_j, patch_color=(0, 100, 255))
        img_j = overlay_dino_bbox(img_j, obs_j, (0, 0, 255))

        # Resize and append
        w_j = int(img_j.shape[1] * (h_target / img_j.shape[0]))
        tiles_to_stack.append(cv2.resize(img_j, (w_j, h_target)))

    # 4. Horizontally stack all processed canvases
    if len(tiles_to_stack) > 1:
        final_tile = np.hstack(tiles_to_stack)
        out_name = f"{name}.jpg"
        if write_img:
            cv2.imwrite(os.path.join(output_dir, out_name), final_tile)
        return final_tile
    else:
        print("⚠️ No valid images to stack.")
        return None

if __name__ == "__main__":
    (xyz_arr, image2points, image2name, rgb_arr) = read_nvm_file(
        "/home/n11373598/work/scrstudio/data/aachen/aachen_cvpr2018_db.nvm",
        return_rgb=True,
    )

    (xyz_arr2, image2points2, image2name, rgb_arr) = read_nvm_file(
        "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/aachen_v1.1/3D-models/aachen_v_1_1/aachen_v_1_1.nvm",
        return_rgb=True,
    )
    (xyz_arr2.shape[0] - xyz_arr.shape[0]) / xyz_arr.shape[0]
    print()
