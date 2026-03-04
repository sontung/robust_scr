import torch
import numpy as np


def main():
    m1 = torch.load("checkpoints/head.pt")
    m1 = m1["model.embedding.weight"].cpu().numpy()
    m2 = np.load("checkpoints/desc_aachen0.npy")
    print(f"global: mean={m1.mean():.3f}, std={m1.std():.3f}, "
          f"min={m1.min():.3f}, max={m1.max():.3f}")

    print(f"global: mean={m2.mean():.3f}, std={m2.std():.3f}, "
          f"min={m2.min():.3f}, max={m2.max():.3f}")
    l1 = torch.load("checkpoints/features.pt", map_location="cpu")[:1000000]
    print(f"global: mean={l1.mean():.3f}, std={l1.std():.3f}, "
          f"min={l1.min():.3f}, max={l1.max():.3f}")
    local_mean = l1.mean()
    local_std = l1.std()
    local_min = l1.min()
    local_max = l1.max()
    l2 = (l1 - local_mean) / (local_std + 1e-6)

    l3 = 2 * (l1 - local_min) / (local_max - local_min) - 1
    print(f"global: mean={l2.mean():.3f}, std={l2.std():.3f}, "
          f"min={l2.min():.3f}, max={l2.max():.3f}")
    print(f"global: mean={l3.mean():.3f}, std={l3.std():.3f}, "
          f"min={l3.min():.3f}, max={l3.max():.3f}")
    print()


if __name__ == '__main__':
    main()