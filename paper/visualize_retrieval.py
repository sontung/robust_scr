from pathlib import Path
import matplotlib
from scipy.sparse import load_npz
from tqdm import trange

matplotlib.use("Agg")  # use non-interactive backend
import cv2
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

from dataset import CamLocDatasetConfig

NO_TEXT = False


def retrieve(desc_, top_k=6):
    index = faiss.IndexFlatL2(desc_.shape[1])
    index.add(desc_)

    distances, indices = index.search(desc_, top_k)

    return distances, indices


def compute_errors(poses, indices):
    errors = []
    for i in range(indices.shape[0]):
        pose = poses[indices[i, 1:]][:, :3, 3]
        pose = pose - poses[indices[i, 0]][:3, 3]
        error = np.linalg.norm(pose, axis=1)
        errors.append(error)
    return np.array(errors)


def main():
    train_config = CamLocDatasetConfig(
        data=Path("/home/n11373598/work/scrstudio/data/aachen"),
        split="train",
    )
    ds = train_config.setup()

    node2vec_desc = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_node2vec.npy"
    )
    desc2 = np.load(
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/desc_aachen0.npy"
    )
    pose_graph_dir = (
        "/home/n11373598/hpc-home/work/glace_experiment/checkpoints/pose_overlap.npz"
    )
    covis_score = load_npz(pose_graph_dir)
    covis_score.setdiag(1)
    covis_score = covis_score.tocsr()

    node2vec_desc = node2vec_desc.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    node2vec_desc = np.ascontiguousarray(node2vec_desc)
    desc2 = np.ascontiguousarray(desc2)
    _, i0 = retrieve(node2vec_desc)
    _, i1 = retrieve(desc2)
    s0 = compute_overlapping_scores(covis_score, i0)
    s1 = compute_overlapping_scores(covis_score, i1)
    e0 = compute_errors(ds.pose_values, i0)
    e1 = compute_errors(ds.pose_values, i1)
    print(f"Node2Vec: {e0.mean()} +- {e0.std()}")
    print(f"ours: {e1.mean()} +- {e1.std()}")

    # s_e0 = s0.mean(1)+e0.mean(1)
    # indices2 = np.argsort(s_e0)[::-1][:500]
    #
    # e0 = e0[indices2]
    # e1 = e1[indices2]
    # s0 = s0[indices2]
    # s1 = s1[indices2]

    diff = np.abs(e0.mean(1) - e1.mean(1))
    mask = e0.mean(1) > e1.mean(1)
    mask2 = np.median(s0, 1) < 10
    mask3 = np.median(s1, 1) < 10
    mask = mask & mask2 & mask3
    print(f"Number of selected samples: {np.sum(mask)}")

    if np.sum(mask) == 0:
        print()
    indices = np.arange(len(diff))
    indices = indices[mask]
    diff = diff[mask]

    k = min(50, len(diff) - 1)
    topk_indices = np.argpartition(-diff, k)[:k]
    topk_indices = topk_indices[np.argsort(-diff[topk_indices])]
    path_indices = indices[topk_indices]
    diff = np.abs(e0.mean(1) - e1.mean(1))
    print(diff[path_indices])
    print(e0.mean(1)[path_indices])
    print(e1.mean(1)[path_indices])

    visualize(
        i0[path_indices],
        i1[path_indices],
        e0[path_indices],
        e1[path_indices],
        s0[path_indices],
        s1[path_indices],
        ds.rgb_files,
    )
    return


def compute_overlapping_scores(covis_score, indices):
    scores = []
    for i in range(indices.shape[0]):
        score = covis_score[indices[i, 0], indices[i, 1:]].toarray().flatten()
        scores.append(score)
    return np.array(scores) * 100


def load_images(
    image_list,
    target_size=None,
    f_dir="/home/n11373598/work/scrstudio/data/aachen/images_upright",
):
    images = []
    for fname in image_list:
        path = f"{f_dir}/{fname}"
        img = Image.open(path).convert("RGB")
        if target_size:
            img = img.resize(target_size, Image.BILINEAR)
        images.append(np.array(img))
    return images


def overlay_errors(images, errors, color_mask, scores, score_mask, label="Query"):
    images_out = images.copy()
    color_mask = color_mask.astype(int)
    fontsize = 15
    for i in range(len(errors)):
        img = images[i]
        h, w, _ = img.shape
        dpi = 100
        fig_w = w / dpi
        extra = 0.5 if i == 0 else 0.0  # Only extra space for first column
        fig, ax = plt.subplots(figsize=(fig_w + extra, h / dpi), dpi=dpi)
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0.2 if i == 0 else 0, right=1, bottom=0, top=1)

        if i == 0:
            if not NO_TEXT:
                # Vertical label on the left
                ax.text(
                    0,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=fontsize,
                    rotation="vertical",
                    # color=color,
                    # weight="bold",
                )
        err_text = f"{errors[i]:.1f}m"
        err_color = (0, 1, 0) if color_mask[i] == 1 else (1, 0, 0)

        sc_text = f"{scores[i]:.1f}%"
        sc_color = (0, 1, 0) if score_mask[i] == 1 else (1, 0, 0)

        ax.text(
            0.01,
            0.99,
            err_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            color=err_color,
            backgroundcolor="white",
            weight="bold",
        )

        ax.text(
            0.99,
            0.99,  # top-right
            sc_text,
            transform=ax.transAxes,
            va="top",
            ha="right",  # right align text
            fontsize=fontsize,
            color=sc_color,
            backgroundcolor="white",
            weight="bold",
        )

        fig.canvas.draw()
        w2, h2 = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h2, w2, 4)
        rgb = buf[:, :, [1, 2, 3]]  # drop alpha

        plt.close(fig)
        images_out[i] = rgb
    return images_out


def stack_images_horizontally(images, spacing=5, bg_color=255):
    h, w, c = images[0].shape
    spacer = np.ones((h, spacing, c), dtype=np.uint8) * bg_color
    row = images[0]
    for img in images[1:]:
        row = np.hstack([row, spacer, img])
    return row


def stack_image_rows(n0, n1, err0, err1, sc0, sc1, resize_to=None, spacing=5):
    imgs0 = load_images(n0, resize_to)
    imgs1 = load_images(n1, resize_to)

    # Overlay errors on last 5 images
    row0 = overlay_errors(
        imgs0[1:], err0, err0 < err1, sc0, sc0 >= sc1, label="R-Score"
    )  # red
    row1 = overlay_errors(
        imgs1[1:], err1, err1 < err0, sc1, sc1 >= sc0, label="Ours"
    )  # green

    # Remove query image from rows before stacking
    row0 = stack_images_horizontally(row0, spacing=10)  # skip query
    row1 = stack_images_horizontally(row1, spacing=10)  # skip query
    # Stack vertically with spacer
    h_spacer = np.ones((10, row0.shape[1], 3), dtype=np.uint8) * 255
    retrievals = np.vstack([row0, h_spacer, row1])

    # Combine with resized query image
    query_im = load_images([n0[0]])[0]  # The query image with vertical label
    target_h = retrievals.shape[0]
    target_w = int(target_h * 1.5)
    # print(target_w, target_h)
    query_resized = np.array(
        Image.fromarray(query_im).resize((target_w, target_h), Image.BILINEAR)
    )

    # Now hstack
    final = np.hstack([query_resized, retrievals])
    return final


def visualize(path0, path1, errors0, errors1, scores0, scores1, names, spacing=5):
    names = np.array(names)
    all_rows = []
    # selected = [2, 7, 16, 17, 15]
    selected = [0, 1, 2]
    # selected = np.arange(path0.shape[0])

    path0 = path0[selected]
    path1 = path1[selected]
    for i in trange(path0.shape[0]):
        n0 = names[path0[i]]
        n1 = names[path1[i]]
        err0 = errors0[i]
        err1 = errors1[i]
        score0 = scores0[i]
        score1 = scores1[i]
        output_image = stack_image_rows(
            n0, n1, err0, err1, score0, score1, spacing=spacing, resize_to=(200, 150)
        )
        all_rows.append(output_image)

    # Stack all rows
    # for idx, img in enumerate(all_rows):
    #     cv2.imwrite(f"../debug/{idx}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    all_rows = np.array(all_rows)
    final_image = all_rows[0]
    for img in all_rows[1:]:
        v_spacer = np.ones((20, final_image.shape[1], 3), dtype=np.uint8) * 255
        try:
            final_image = np.vstack([final_image, v_spacer, img])
        except ValueError as e:
            continue

    dpi = 300
    h, w, _ = final_image.shape
    fig_w, fig_h = w / 100, h / 100

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.1), dpi=dpi)  # Add vertical space
    ax.imshow(final_image, extent=[0, w, 0, h])  # show image with real pixel coords
    ax.axis("off")

    # Add top-k labels centered above each column
    col_width = 200 + 10
    y_offset = h + 0  # 20 pixels above the top of the image

    if not NO_TEXT:
        for i in range(5):
            x_center = i * col_width + 590
            ax.text(
                x_center,
                y_offset,
                f"top-{i + 1}",
                ha="left",
                va="bottom",
                fontsize=15,
                # weight="bold",
                color="black",
            )

    if not NO_TEXT:
        fig.savefig(
            # "/home/n11373598/Documents/wacv_subm25/figures/drawings/retrievals.pdf",
            "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/retrievals.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )

        fig.savefig(
            "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/retrievals.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0.05,
        )
    else:
        fig.savefig(
            "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/retrievals2.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0.05,
        )

    plt.close(fig)


if __name__ == "__main__":
    main()
