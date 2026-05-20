#!/usr/bin/env python3
"""
TECHNIQUE 1 — Reinhard Stain Normalization
===========================================
Problem it solves:
    Different hospitals and microscopes produce images with different
    color distributions (staining protocols). A model trained on PBC
    (pinkish-purple staining) fails on Raabin (darker purple staining)
    because it treats color as a discriminative feature.

Solution:
    Reinhard normalization transforms every image's LAB color distribution
    to match a reference training domain image. This makes all images
    "look like" they came from the same microscope before feeding to model.

Math:
    For each LAB channel i:
        normalized = (pixel - src_mean_i) / src_std_i
        output     = normalized * target_std_i + target_mean_i

Impact observed:
    Eosinophils F1: 0.10 → 0.97 (biggest beneficiary)
    Overall accuracy: 85% → 91%+ after combining with other techniques
"""

import cv2
import numpy as np
import glob
import os


# ============================================================
# CORE FUNCTION
# ============================================================
def normalize_stain_reinhard(img_bgr, target_mean, target_std):
    """
    Normalize staining of img_bgr to match target_mean and target_std.

    Args:
        img_bgr     : input image in BGR format (uint8)
        target_mean : LAB mean of training domain [L, A, B]
        target_std  : LAB std  of training domain [L, A, B]

    Returns:
        normalized image in BGR format (uint8)
    """
    img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = img_lab.reshape(-1, 3).mean(axis=0)
    src_std  = img_lab.reshape(-1, 3).std(axis=0)

    for i in range(3):
        # Step 1 — remove source color style
        img_lab[:, :, i] = (img_lab[:, :, i] - src_mean[i]) / (src_std[i] + 1e-6)
        # Step 2 — inject training domain color style
        img_lab[:, :, i] =  img_lab[:, :, i] * target_std[i] + target_mean[i]

    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)


# ============================================================
# COMPUTE TARGET STATS FROM TRAINING DOMAIN
# Use average across 20 images per class for stability
# ============================================================
def compute_target_stats(dataset_train_path, n_per_class=20):
    """
    Compute robust TARGET_MEAN and TARGET_STD from training dataset.
    Uses average across multiple images to avoid single-image bias.

    Args:
        dataset_train_path : path to merged_dataset/train/
        n_per_class        : number of images per class to sample

    Returns:
        TARGET_MEAN : shape (3,) — LAB mean
        TARGET_STD  : shape (3,) — LAB std
    """
    all_means, all_stds = [], []

    for directory_path in glob.glob(os.path.join(dataset_train_path, "*")):
        for img_path in glob.glob(
                os.path.join(directory_path, "*.jpg"))[:n_per_class]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            all_means.append(lab.reshape(-1, 3).mean(axis=0))
            all_stds.append(lab.reshape(-1, 3).std(axis=0))

    TARGET_MEAN = np.array(all_means).mean(axis=0)
    TARGET_STD  = np.array(all_stds).mean(axis=0)
    return TARGET_MEAN, TARGET_STD


# ============================================================
# HARDCODED STATS — computed from merged_dataset/train
# Use these directly to avoid recomputing every run
# ============================================================
TARGET_MEAN = np.array([181.25313, 142.2399,  131.06912])
TARGET_STD  = np.array([ 40.716705,  9.863444,  13.54429])


def normalize_stain(img_bgr):
    """Wrapper — normalizes any image to training domain style."""
    return normalize_stain_reinhard(img_bgr, TARGET_MEAN, TARGET_STD)


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test on a Raabin image
    raabin_path = "/home/admin92/patent/rabbin dataset/Train/Neutrophils/"
    img_path    = glob.glob(os.path.join(raabin_path, "*.jpg"))[0]

    original   = cv2.imread(img_path)
    normalized = normalize_stain(original)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(original,   cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Raabin image")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    axes[1].set_title("After Reinhard normalization\n(matched to PBC style)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    print("TARGET_MEAN:", TARGET_MEAN)
    print("TARGET_STD :", TARGET_STD)
