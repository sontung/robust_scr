import os
import pickle
from pathlib import Path
import matplotlib
import torch

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from dataset import CamLocDatasetConfig
from paper.visualize_retrieval import retrieve


def get_aachen():
    train_config = CamLocDatasetConfig(
        data=Path("/home/n11373598/work/scrstudio/data/aachen"),
        split="train",
    )
    ds = train_config.setup()

    # Load and prepare descriptors
    node2vec_desc = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_node2vec.npy"
    ).astype(np.float32)
    desc2 = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_aachen0.npy"
    ).astype(np.float32)
    node2vec_desc = np.ascontiguousarray(node2vec_desc)
    desc2 = np.ascontiguousarray(desc2)
    salad_desc = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_salad_db.npy"
    ).astype(np.float32)
    salad_desc = np.ascontiguousarray(salad_desc)
    netvlad_desc = np.load(
        "/home/n11373598/hpc-home/work/scrstudio_exp/data/data/aachen/train/netvlad_feats.npy"
    ).astype(np.float32)
    netvlad_desc = np.ascontiguousarray(netvlad_desc)
    all_descs = {
        "R-Score": node2vec_desc,
        "Ours": desc2,
        # "SALAD": salad_desc,
        # "NetVLAD": netvlad_desc,
    }
    return "Aachen day/night", ds, all_descs


def get_dept_1f():
    train_config = CamLocDatasetConfig(
        data=Path("/home/n11373598/hpc-home/work/scrstudio_exp/data/dept/1F"),
        split="train",
    )
    ds = train_config.setup()

    # Load and prepare descriptors
    node2vec_desc = (
        torch.load(
            "/home/n11373598/hpc-home/work/glace_experiment/outputs/1F/node2vec/2025-07-07_141353/scrstudio_models/head.pt",
            weights_only=True,
        )["model.embedding.weight"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    desc2 = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_1f.npy"
    ).astype(np.float32)
    node2vec_desc = np.ascontiguousarray(node2vec_desc)
    desc2 = np.ascontiguousarray(desc2)
    all_descs = {
        "R-Score": node2vec_desc,
        "Ours": desc2,
        # "salad": salad_desc,
    }
    return "Dept. 1F", ds, all_descs


def get_dept_4f():
    train_config = CamLocDatasetConfig(
        data=Path("/home/n11373598/hpc-home/work/scrstudio_exp/data/dept/4F"),
        split="train",
    )
    ds = train_config.setup()

    # Load and prepare descriptors
    node2vec_desc = (
        torch.load(
            "/home/n11373598/hpc-home/work/glace_experiment/outputs/4F/node2vec/2025-07-07_140555/scrstudio_models/head.pt",
            weights_only=True,
        )["model.embedding.weight"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    desc2 = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_4f.npy"
    ).astype(np.float32)
    node2vec_desc = np.ascontiguousarray(node2vec_desc)
    desc2 = np.ascontiguousarray(desc2)
    all_descs = {
        "R-Score": node2vec_desc,
        "Ours": desc2,
        # "salad": salad_desc,
    }
    return "Dept. 4F", ds, all_descs


def get_dept_b1():
    train_config = CamLocDatasetConfig(
        data=Path("/home/n11373598/hpc-home/work/scrstudio_exp/data/dept/B1"),
        split="train",
    )
    ds = train_config.setup()

    # Load and prepare descriptors
    node2vec_desc = (
        torch.load(
            "/home/n11373598/hpc-home/work/glace_experiment/outputs/B1/node2vec/2025-07-07_142213/scrstudio_models/head.pt",
            weights_only=True,
        )["model.embedding.weight"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    desc2 = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_b1.npy"
    ).astype(np.float32)
    node2vec_desc = np.ascontiguousarray(node2vec_desc)
    desc2 = np.ascontiguousarray(desc2)
    all_descs = {
        "R-Score": node2vec_desc,
        "Ours": desc2,
    }
    return "Dept. B1", ds, all_descs


def compute_errors(poses, indices):
    errors = []
    for i in range(indices.shape[0]):
        pose = poses[indices[i, 1:]][:, :3, 3]
        pose = pose - poses[indices[i, 0]][:3, 3]
        error = np.linalg.norm(pose, axis=1)
        errors.append(error)
    return np.array(errors)


def main():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 18,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    datasets = [get_aachen(), get_dept_1f(), get_dept_4f(), get_dept_b1()]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(20, 5))

    markers = {
        "R-Score": ("o", "tab:blue"),
        "Ours": ("^", "tab:green"),
        "SALAD": ("s", "tab:orange"),
        "NetVLAD": ("D", "tab:purple"),
    }
    max_top_k = 15
    top_ks = list(range(1, max_top_k + 1))

    for i in range(len(datasets)):
        ds_name, ds, all_descs = datasets[i]
        ds_id = str(ds.config.data).split("/")[-1]

        pkl_path = f"{ds_id}_errors.pkl"
        if os.path.exists(pkl_path):
            # Reload existing data
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                errors = data["errors"]
                errors_gt = data["errors_gt"]
            print(f"Loaded errors from {pkl_path}")

        else:
            errors_gt = []
            errors = {}
            for top_k in tqdm(top_ks, desc=f"Iteration {i + 1}"):
                for key, desc in all_descs.items():
                    _, i0 = retrieve(desc, top_k=top_k + 1)
                    e0 = compute_errors(ds.pose_values, i0)
                    errors.setdefault(key, []).append(np.median(e0[:, -1]))
                e3 = compute_errors(
                    ds.pose_values,
                    retrieve(
                        np.ascontiguousarray(ds.pose_values[:, :3, 3]).astype(
                            np.float32
                        ),
                        top_k + 1,
                    )[1],
                )
                errors_gt.append(np.median(e3[:, -1]))
            with open(pkl_path, "wb") as f:
                pickle.dump({"errors": errors, "errors_gt": errors_gt}, f)
            print(f"Saved errors to {pkl_path}")

        ax = axes[i]
        for key, error in errors.items():
            marker, color = markers[key]
            ax.plot(top_ks, error, marker=marker, label=key, color=color)

        ax.plot(top_ks, errors_gt, marker="x", label="Ground Truth", color="tab:red")

        ax.set_xlabel("top-k")
        ax.set_xticks(list(range(1, max_top_k + 1, 2)))  # From 1 to 19, step 2

        ax.set_title(ds_name)
        if i == 0:
            ax.set_ylabel("Median Translation Error (m)")
            legend_handles, legend_labels = [], []
            for key, error in errors.items():
                marker, color = markers[key]
                (line,) = ax.plot([], [], marker=marker, label=key, color=color)
                legend_handles.append(line)
                legend_labels.append(key)
            (line_gt,) = ax.plot(
                [], [], marker="x", label="Ground Truth", color="tab:red"
            )
            legend_handles.append(line_gt)
            legend_labels.append("Ground Truth")

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=25,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for legend
    fig.savefig(
        "/home/n11373598/Documents/wacv_subm25/figures/drawings/top_k.pdf",
        format="pdf",
        dpi=800,
        # bbox_inches="tight"
    )

    return


if __name__ == "__main__":
    main()
