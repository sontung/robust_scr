import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import trange


def compute_pr(mat_desc, label_threshold=0.2, sample_size=1_000_000, seed=42):
    print(mat_desc.dtype)
    np.random.seed(seed)
    mat_desc = mat_desc.astype(np.float32)

    # === Load covisibility graph ===
    pose_graph_dir = (
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/pose_overlap.npz"
    )
    covis_score = load_npz(pose_graph_dir)
    covis_score.setdiag(1)
    covis_score = covis_score.tocsr()

    N = covis_score.shape[0]

    # Generate all i<j pairs
    rows_all, cols_all = np.triu_indices(N, k=1)
    scores_all = np.array(covis_score[rows_all, cols_all]).flatten()

    # Create labels
    labels = (scores_all >= label_threshold).astype(int)

    # Sample positives and negatives
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    pos_sample = np.random.choice(
        pos_indices, size=min(sample_size, len(pos_indices)), replace=False
    )
    neg_sample = np.random.choice(
        neg_indices, size=min(sample_size, len(neg_indices)), replace=False
    )

    keep_indices = np.concatenate([pos_sample, neg_sample])
    rows_sample = rows_all[keep_indices]
    cols_sample = cols_all[keep_indices]
    labels_sample = labels[keep_indices]

    distances = compute_distances_in_batches(
        mat_desc, rows_sample, cols_sample, batch_size=50000
    )

    # Compute PR curve
    precision, recall, _ = precision_recall_curve(labels_sample, -distances)
    ap = average_precision_score(labels_sample, -distances)

    return precision, recall, ap


def compute_distances_in_batches(mat_desc, rows, cols, batch_size=50000):
    """
    Compute L2 distances between descriptor pairs (rows[i], cols[i]) in batches.
    """
    distances = np.empty(len(rows), dtype=np.float32)

    for start in trange(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        feat_i = mat_desc[rows[start:end]]
        feat_j = mat_desc[cols[start:end]]
        distances[start:end] = np.linalg.norm(feat_i - feat_j, axis=1)

    return distances


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 18,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    mat_aachen = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_aachen.npy"
    )
    mat_node2vec = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_node2vec.npy"
    )

    # Compute PR for each descriptor
    pr_aachen = compute_pr(mat_aachen)
    pr_node2vec = compute_pr(mat_node2vec)
    pr_salad = compute_pr(
        np.load(
            "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_salad_db.npy"
        )
    )  # Placeholder for another descriptor if needed
    pr_boq = compute_pr(
        np.load("/home/n11373598/work/glace/checkpoints/boq_train.npy")
    )  # Placeholder for another descriptor if needed
    pr_mixvpr = compute_pr(
        np.load("/home/n11373598/work/glace/checkpoints/mixvpr_train.npy")
    )  # Placeholder for another descriptor if needed
    pr_megaloc = compute_pr(
        np.load("/home/n11373598/work/glace/checkpoints/megaloc_train.npy")
    )  # Placeholder for another descriptor if needed
    pr_eigenplaces = compute_pr(
        np.load("/home/n11373598/work/glace/checkpoints/eigenplaces_train.npy")
    )  # Placeholder for another descriptor if needed

    # Plot all PR curves on the same figure
    plt.figure(figsize=(7, 6))
    plt.plot(pr_aachen[1], pr_aachen[0], label=f"Ours (AP={pr_aachen[2]:.2f})")
    plt.plot(
        pr_node2vec[1], pr_node2vec[0], label=f"Node2Vec (AP={pr_node2vec[2]:.2f})"
    )
    plt.plot(
        pr_salad[1], pr_salad[0], label=f"SALAD (AP={pr_salad[2]:.2f})", linewidth=2
    )
    plt.plot(pr_boq[1], pr_boq[0], label=f"BoQ (AP={pr_boq[2]:.2f})", linewidth=2)
    plt.plot(
        pr_mixvpr[1], pr_mixvpr[0], label=f"MixVPR (AP={pr_mixvpr[2]:.2f})", linewidth=2
    )
    plt.plot(
        pr_megaloc[1],
        pr_megaloc[0],
        label=f"MegaLoc (AP={pr_megaloc[2]:.2f})",
        linewidth=2,
    )
    plt.plot(
        pr_eigenplaces[1],
        pr_eigenplaces[0],
        label=f"EigenPlaces (AP={pr_eigenplaces[2]:.2f})",
        linewidth=2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("pr_curve.pdf", format="pdf", dpi=800, bbox_inches="tight")
