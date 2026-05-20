#!/usr/bin/env python3
"""
TECHNIQUE 2 — Multi-Source Dataset Merging
===========================================
Problem it solves:
    A model trained on one dataset (PBC only) learns that dataset's
    specific visual patterns. It cannot generalize to new microscopes
    because it has never seen visual diversity during training.

Solution:
    Combine PBC + Raabin datasets into one merged training set with
    stratified 80/20 split. This forces the model to learn features
    that work across both staining styles simultaneously.

Impact observed:
    Training on PBC only    → 84.89% on Raabin test
    Training on PBC + Raabin → 91-99% on merged test set
    Key: model sees both domains during training

Dataset statistics:
    PBC    : 9,390 images (5 classes)
    Raabin : 14,514 images (train + test combined)
    Merged : 23,904 images total
"""

import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


# ============================================================
# CLASS NAME STANDARDIZATION
# Raabin test folder has inconsistent naming vs PBC
# ============================================================
NAME_MAP = {
    'Basophils'  : 'Basophils',
    'Eosinophils': 'Eosinophils',
    'Esinophils' : 'Eosinophils',   # typo in Raabin test
    'Neutrophils': 'Neutrophils',
    'Monocytes'  : 'Monocytes',
    'Monocyte'   : 'Monocytes',     # singular in Raabin test
    'Lymphocytes': 'Lymphocytes',
    'Lymphocyte' : 'Lymphocytes',   # singular in Raabin test
}


# ============================================================
# STEP 1 — COLLECT ALL IMAGE PATHS + NORMALIZED LABELS
# ============================================================
def collect_all_paths(sources):
    """
    Collect image paths and standardized labels from multiple dataset sources.

    Args:
        sources: list of glob patterns pointing to class folders

    Returns:
        all_paths  : numpy array of image file paths
        all_labels : numpy array of standardized class names
    """
    all_paths, all_labels = [], []

    for source_pattern in sources:
        for directory_path in glob.glob(source_pattern):
            raw_label = directory_path.split("/")[-1]
            label     = NAME_MAP.get(raw_label, raw_label)

            for img_path in glob.glob(
                    os.path.join(directory_path, "*.jpg")):
                all_paths.append(img_path)
                all_labels.append(label)

    return np.array(all_paths), np.array(all_labels)


# ============================================================
# STEP 2 — STRATIFIED 80/20 SPLIT
# ============================================================
def create_stratified_split(all_paths, all_labels,
                             test_size=0.20, random_state=42):
    """
    Create stratified train/test split ensuring class balance in both sets.

    Args:
        all_paths  : image file paths
        all_labels : class labels
        test_size  : fraction for test set
        random_state: reproducibility seed

    Returns:
        train_paths, test_paths, train_labels, test_labels
    """
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels
    )
    return train_paths, test_paths, train_labels, test_labels


# ============================================================
# STEP 3 — COPY INTO ORGANIZED FOLDER STRUCTURE
# ============================================================
def save_merged_dataset(train_paths, test_paths,
                         train_labels, test_labels,
                         output_dir):
    """
    Copy images into class-organized train/test folder structure.

    Output structure:
        output_dir/
            train/
                Basophils/
                Eosinophils/
                ...
            test/
                Basophils/
                ...
    """
    for split, paths, labels in [
            ("train", train_paths, train_labels),
            ("test",  test_paths,  test_labels)]:

        for path, label in zip(paths, labels):
            dest_dir = os.path.join(output_dir, split, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(path, dest_dir)

    print(f"Merged dataset saved to: {output_dir}")


# ============================================================
# MAIN — run to create merged dataset
# ============================================================
if __name__ == "__main__":

    SOURCES = [
        "/home/admin92/patent/PBC_dataset_normal_DIB_224/*",
        "/home/admin92/patent/rabbin dataset/Train/*",
        "/home/admin92/patent/rabbin dataset/TestA/*",
    ]
    OUTPUT_DIR = "/home/admin92/patent/merged_dataset"

    print("Collecting image paths from all sources...")
    all_paths, all_labels = collect_all_paths(SOURCES)
    print(f"Total images found: {len(all_paths)}")
    print("\nClass distribution (before split):")
    for cls, count in Counter(all_labels).items():
        print(f"  {cls:15s}: {count}")

    print("\nCreating stratified 80/20 split...")
    train_paths, test_paths, train_labels, test_labels = \
        create_stratified_split(all_paths, all_labels)

    print(f"\nTrain: {len(train_paths)} | Test: {len(test_paths)}")
    print("\nTrain distribution:")
    for cls, count in Counter(train_labels).items():
        print(f"  {cls:15s}: {count}")
    print("\nTest distribution:")
    for cls, count in Counter(test_labels).items():
        print(f"  {cls:15s}: {count}")

    print("\nSaving merged dataset...")
    save_merged_dataset(train_paths, test_paths,
                         train_labels, test_labels,
                         OUTPUT_DIR)
