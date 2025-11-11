import argparse
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
        default=1.0,
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


if __name__ == "__main__":
    (xyz_arr, image2points,
     image2name, rgb_arr) = read_nvm_file(
        "/home/n11373598/work/scrstudio/data/aachen/aachen_cvpr2018_db.nvm",
        return_rgb=True,
    )

    (xyz_arr2, image2points2,
     image2name, rgb_arr) = read_nvm_file(
        "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/aachen_v1.1/3D-models/aachen_v_1_1/aachen_v_1_1.nvm",
        return_rgb=True,
    )
    (xyz_arr2.shape[0]-xyz_arr.shape[0])/xyz_arr.shape[0]
    print()