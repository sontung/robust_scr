import os
import h5py
import torch
from tqdm import tqdm
from lightglue import SuperPoint
from lightglue.utils import load_image

# Input and output paths
img_dir = (
    "/home/n11373598/hpc-home/work/scrstudio_exp/data/data/aachen/images_upright/db"
)
out_path = "superpoint_features.h5"

# Initialize SuperPoint extractor
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()

# Collect image paths
img_paths = [
    os.path.join(img_dir, f)
    for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Create HDF5 file
with h5py.File(out_path, "w") as hfile:
    for img_path in tqdm(img_paths, desc="Extracting SuperPoint features"):
        # Load image
        image = load_image(img_path).cuda()  # (3,H,W) in [0,1]

        # Extract features
        with torch.no_grad():
            feats = extractor.extract(image)

        # Convert to numpy
        keypoints = feats["keypoints"].cpu().numpy()  # (N,2)
        descriptors = feats["descriptors"].cpu().numpy()  # (N,D)

        # Use relative path inside the HDF5 file
        rel_path = os.path.relpath(img_path, img_dir)

        # Store features
        grp = hfile.create_group(rel_path)
        grp.create_dataset("keypoints", data=keypoints, compression="gzip")
        grp.create_dataset("descriptors", data=descriptors, compression="gzip")

print(f"✅ Features saved to {out_path}")
