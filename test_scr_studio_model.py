import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import faiss
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_classes import PQKNN
from dataset import CamLocDatasetConfig
from encoder import get_encoder
from networks import get_model
from utils import pose_estimate
from datetime import datetime


def load():
    salad_train = np.load("checkpoints/salad_train.npy")
    salad_test = np.load("checkpoints/salad_test.npy")
    node2vec_ori = torch.load("checkpoints/node2vec.pt")
    desc_node2vec = node2vec_ori["model.embedding.weight"]

    train_vecs = np.ascontiguousarray(salad_train, dtype=np.float32)
    test_vecs = np.ascontiguousarray(salad_test, dtype=np.float32)

    dim = train_vecs.shape[1]

    # --- Build index ---
    # Use L2 distance; swap for faiss.IndexFlatIP for cosine (after L2-norm)
    index = faiss.IndexFlatL2(dim)
    index.add(train_vecs)
    print(f"Index built: {index.ntotal} train vectors of dim {dim}")

    distances, indices = index.search(test_vecs, 1)
    test_node2vec = desc_node2vec[indices.flatten()]
    state_dict = torch.load("checkpoints/mlp.pt")


    centers = np.load("checkpoints/scene_centers.npy")
    model = get_model(in_channels=128, head_channels=1280,
                      metadata={"cluster_centers": torch.from_numpy(centers)})

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k[len("backbone."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    return test_node2vec, model, desc_node2vec, salad_train, salad_test



def main():
    gl_test, model, emb, salad_train, salad_test = load()
    model.eval()
    model.cuda()
    train_config = CamLocDatasetConfig(
        data=Path("/home/vr/work/datasets/aachen10"),
        split="test",
        feat_name="checkpoints/salad_test.npy"
    )
    raw_ds = train_config.setup()

    encoder = get_encoder(
        pca_path="checkpoints/pca.pth",
    )
    encoder.cuda()
    encoder.eval()
    
    from nanopq import PQ

    pq = PQ(M=256, verbose=False).fit(salad_train.astype(np.float32), seed=0)
    codes = pq.encode(salad_train.astype(np.float32))
    knn = PQKNN(pq, codes, n_neighbors=10)

    test_loop(
        model,
        encoder,
        DataLoader(raw_ds, shuffle=False, num_workers=1),
        knn,
        emb.cuda(),
        salad_test,
        write_gt_poses=True,
        scene="outdoor",
        # no_db_desc=True,
    )

    return

def test_loop(
    head,
    encoder,
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
        "seed": 10,
    }
    _logger = logging.getLogger(__name__)

    metrics = defaultdict(list)
    poses = defaultdict(list)
    pool_results = []
    C_global = 256
    n_neighbors = 10
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

                encoder_output = encoder.keypoint_features(
                    {"image": image}, n=0
                )
                keypoints = encoder_output["keypoints"]
                descriptors = encoder_output["descriptors"]
                N, C_local = descriptors.shape
                gl_feat = torch.empty(
                    (n_neighbors, N, C_global + C_local), device=device
                )
                gl_feat[:, :, :C_global] = global_feat.unsqueeze(1).expand(
                    -1, N, -1
                )
                gl_feat[:, :, C_global:] = descriptors.unsqueeze(0).expand(
                    n_neighbors, -1, -1
                )

                scene_coords = head(
                    {"features": gl_feat.reshape(n_neighbors * N, -1)}
                )["sc"]

            keypoints = keypoints.float().cpu()

            scene_coords = (
                scene_coords.float().cpu().numpy().reshape(n_neighbors, N, 3)
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
            for neighbor_idx in range(n_neighbors):
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
    print(
        f"Time: {end - start:.1f}s for {len(testset_loader.dataset)} images"
    )
    acc_thresh = {
        "outdoor": ((5, 10), (0.5, 5), (0.25, 2)),
        "indoor": ((1, 5), (0.25, 2), (0.1, 1)),
    }

    if write_gt_poses:
        write_poses_to_file(final_results)

    if testset_loader.dataset.gt_pose_avail:
        for t, r in acc_thresh[scene]:
            acc = (np.array(metrics["t_err"]) < t) & (
                np.array(metrics["r_err"]) < r
            )
            print(f"Accuracy: {t}m/{r}deg: {acc.mean() * 100:.1f}%")
        median_rErr = np.median(metrics["r_err"])
        median_tErr = np.median(metrics["t_err"]) * 100
        print(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
        print(f"Mean Inliers: {np.mean(metrics['inlier_ratio']):.2f}")
    pool.close()
    pool.join()


def write_poses_to_file(poses):
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

if __name__ == '__main__':
    main()
