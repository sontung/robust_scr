import itertools
from itertools import combinations
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
from collections import defaultdict

import scipy
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.transform import Rotation
from tqdm import trange, tqdm


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

    return xyz_arr, image2points, image2name, image2info, image2uvs, image2pose, rgb_arr


def return_pose_mat_no_inv(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    return pose_4x4


class Frustum:
    def __init__(self, c2w, intrinsics, image_shapes):
        images_wh = image_shapes[:, 1::-1]
        if intrinsics.ndim == 1:  # M
            fxy = intrinsics[:, None]
            cxy = images_wh / 2
        else:  # M x 3 x 3
            fxy = intrinsics[:, [0, 1], [0, 1]]
            cxy = intrinsics[:, [0, 1], [2, 2]]
        self.min_xy = -cxy / fxy
        self.max_xy = (images_wh - cxy) / fxy
        self.c2w = c2w

    def __len__(self):
        return len(self.c2w)


class ComputeOverlap:
    """Download a dataset"""

    data: Path = Path("data/")
    seed: int = 0
    num_samples: int = 1024
    min_depth: float = 0.1
    max_depth: float = 8

    def main(self, c2w, intrinsics, image_shapes, max_depth):
        self.max_depth = max_depth
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(self.seed)
        frustum = Frustum(c2w, intrinsics, image_shapes)
        src_frustum = dst_frustum = frustum
        pose_covis = torch.zeros((len(src_frustum), len(dst_frustum)), device=device)
        c2w = torch.tensor(src_frustum.c2w, dtype=torch.float32).cuda()
        w2c = torch.inverse(torch.tensor(dst_frustum.c2w, dtype=torch.float32).cuda())
        depth_samples = (
            torch.empty(self.num_samples, dtype=torch.float32)
            .cuda()
            .uniform_(self.min_depth, self.max_depth, generator=generator)
        )
        dst_frustum_min_xy = torch.tensor(
            dst_frustum.min_xy, dtype=torch.float32
        ).cuda()
        dst_frustum_max_xy = torch.tensor(
            dst_frustum.max_xy, dtype=torch.float32
        ).cuda()
        for i in tqdm(range(len(src_frustum))):
            w_samples = (
                torch.empty(self.num_samples, dtype=torch.float32)
                .cuda()
                .uniform_(
                    src_frustum.min_xy[i, 0],
                    src_frustum.max_xy[i, 0],
                    generator=generator,
                )
            )
            h_samples = (
                torch.empty(self.num_samples, dtype=torch.float32)
                .cuda()
                .uniform_(
                    src_frustum.min_xy[i, 1],
                    src_frustum.max_xy[i, 1],
                    generator=generator,
                )
            )
            src_coords = torch.stack([w_samples, h_samples, depth_samples], dim=1)
            reproj_mat = w2c @ c2w[i]
            src_dir = reproj_mat[:, :3, :3] @ src_coords.T
            dst_coords = src_dir + reproj_mat[:, :3, 3:4]
            dst_depths = dst_coords[:, 2]
            dst_xy = dst_coords[:, :2, :] / dst_coords[:, 2:]
            score = (torch.cosine_similarity(src_dir, dst_coords, dim=1) + 1) * 0.5
            mask = (dst_depths > self.min_depth) & (dst_depths < self.max_depth)
            mask &= (dst_xy >= dst_frustum_min_xy[..., None]).all(dim=1) & (
                dst_xy < dst_frustum_max_xy[..., None]
            ).all(dim=1)
            mask = score * mask
            pose_covis[i] = mask.sum(dim=1) / self.num_samples
        pose_covis = pose_covis.cpu().numpy()
        np.fill_diagonal(pose_covis, 0)
        pose_covis_sym = 2 * pose_covis * pose_covis.T / (pose_covis + pose_covis.T)
        np.fill_diagonal(pose_covis_sym, 0)
        pose_covis_sym = np.nan_to_num(pose_covis_sym)
        coo_covis = scipy.sparse.coo_array(pose_covis_sym)
        return coo_covis


def find_graph(image2points, pose_values, image2pose, image2name):
    image2points = {img: set(pts) for img, pts in image2points.items()}

    # Step 1: Build point->images mapping
    point2images = defaultdict(list)
    for img, points in tqdm(list(image2points.items())):
        for p in points:
            point2images[p].append(img)

    all_pairs = set()
    for p, imgs in tqdm(point2images.items()):
        pairs = list(itertools.combinations(imgs, 2))
        for u, v in pairs:
            if (u, v) in all_pairs or (v, u) in all_pairs:
                continue
            pid1 = image2points[u]
            pid2 = image2points[v]
            common_points = pid1.intersection(pid2)
            if len(common_points) > 10:
                all_pairs.add((u, v))

    n_images = len(image2points)

    mat = np.zeros((n_images, n_images), dtype=np.uint8)
    for idx, idx2 in all_pairs:
        mat[idx, idx2] = 1
    A = csr_matrix(mat, dtype=np.uint8)

    # for img1 in tqdm(covis_graph):
    #     p1 = pose_values[img1][:3, 3]
    #     pid1 = image2points[img1]
    #     for img2, _ in covis_graph[img1]:
    #         p2 = pose_values[img2][:3, 3]
    #         dist = np.linalg.norm(p1 - p2)
    #         if dist > 100:
    #             tqdm.write(f"Large distance {dist} between {img1} and {img2}")
    #             pid2 = image2points[img2]
    #             common_points = pid1.intersection(pid2)
    #             if len(common_points) == 0:
    #                 print("Zero common points!")

    draw_mst_graph(A.tocoo(), pose_values, "overlap.pdf")


def print_graph_stats_coo(A: coo_matrix):
    n_nodes = A.shape[0]
    # Count edges: if symmetric, divide by 2 to avoid double-counting
    symmetric = (A != A.T).nnz == 0
    n_edges = A.nnz // 2 if symmetric else A.nnz

    # Degree = count of edges per node
    degrees = np.bincount(A.row, minlength=n_nodes)

    print("Graph stats:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Avg degree: {degrees.mean():.2f}")
    print(f"  Min degree: {degrees.min()}")
    print(f"  Max degree: {degrees.max()}")
    print(f"  Density: {n_edges / (n_nodes * (n_nodes - 1) / 2):.6f}")

    # Degree histogram
    hist, bins = np.histogram(degrees, bins=10)
    print("  Degree histogram:")
    for left, right, h in zip(bins[:-1], bins[1:], hist):
        print(f"    {int(left)}–{int(right)}: {h}")


def draw_mst_graph(sparse_graph, pose_values, name="MST"):
    print_graph_stats_coo(sparse_graph)
    G = nx.Graph()
    idx1 = sparse_graph.row
    idx2 = sparse_graph.col
    weights = sparse_graph.data
    mask = weights > 0.2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Use negative weights so MST keeps strongest edges
    for i, j, w in zip(idx1, idx2, weights):
        if i != j:
            G.add_edge(i, j, weight=-w)

    # Step 2: Compute MST
    mst = nx.minimum_spanning_tree(G)

    # Step 3: Prepare edges for Open3D
    all_points = pose_values[:, :3, -1]  # extract camera translations
    lines = np.array(list(mst.edges()))

    # Step 4: Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(all_points))  # all black

    # Step 5: Create Open3D LineSet for MST edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        [[0, 0, 1] for _ in range(lines.shape[0])]
    )  # blue edges

    # Step 6: Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.get_render_option().point_size = 10.0
    vis.run()
    vis.destroy_window()


def draw_graph(graph, pose_values, name):
    idx1 = graph.row
    idx2 = graph.col
    values = graph.data
    mask = values > 0.2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    all_points = pose_values[:, :3, -1]

    num_edges = idx1.shape[0]
    lines = np.vstack([idx1, idx2]).T  # edges from pos1[i] -> pos2[i]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Optional: color the nodes
    colors = np.zeros_like(all_points)
    # colors[:num_edges] = [1, 0, 0]  # pos1 in red
    # colors[num_edges:] = [0, 1, 0]  # pos2 in green
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create Open3D LineSet for edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        [[0, 0, 1] for _ in range(num_edges)]
    )  # blue edges

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 10.0

    vis.add_geometry(line_set)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def main():
    pose_values = np.load(
        "/home/n11373598/hpc-home/work/scrstudio_exp/data/data/aachen/train/poses.npy"
    )

    xyz_arr, image2points, image2name, image2info, _, image2pose, rgb_arr = (
        read_nvm_file(
            "/home/n11373598/work/scrstudio/data/aachen/aachen_cvpr2018_db.nvm",
            # "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/robotcar/3D-models/all-merged/all.nvm",
            # "/home/n11373598/work/descriptor-disambiguation/datasets/cambridge/StMarysChurch/reconstruction.nvm",
            return_rgb=True,
        )
    )

    pose_values_sfm = np.zeros_like(pose_values)
    for img_id, pose in image2pose.items():
        pose_q = pose[:4]
        pose_t = pose[4:]
        pose_mat = return_pose_mat_no_inv(pose_q, pose_t)
        pose_values_sfm[img_id] = pose_mat
    find_graph(image2points, pose_values_sfm, image2pose, image2name)
    draw_mst_graph(
        scipy.sparse.load_npz(
            "/home/n11373598/hpc-home/work/scrstudio_exp/data/data/aachen/train/pose_overlap.npz"
        ),
        pose_values,
        "overlap.pdf",
    )
    intrinsics = np.eye(3)

    intrinsics[0, 0] = 738
    intrinsics[1, 1] = 738
    intrinsics[0, 2] = 427  # 427
    intrinsics[1, 2] = 240
    name2id = {v: k for k, v in image2name.items()}

    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
    cl, inlier_ind = pc1.remove_radius_outlier(
        nb_points=16, radius=5, print_progress=True
    )
    cl.colors = o3d.utility.Vector3dVector(rgb_arr[inlier_ind] / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 2.0

    vis.add_geometry(cl)

    for idx in tqdm(range(0, len(pose_values), 5)):
        pose_mat = pose_values[idx]
        pose_mat = np.linalg.inv(pose_mat)
        cam = o3d.geometry.LineSet.create_camera_visualization(
            427 * 2, 240 * 2, intrinsics, pose_mat, scale=5
        )
        cam.paint_uniform_color([1, 0, 0])
        vis.add_geometry(cam)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == "__main__":
    main()
