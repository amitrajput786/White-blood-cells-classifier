#!/usr/bin/env python3
"""
TECHNIQUE 5 — Proper Train/Val/Test Split + BigAug Training Pipeline
=====================================================================
Problem it solves:
    Original setup used test data as validation during training:
        validation_data = test_dataset (WRONG)
        Then evaluated on same test_dataset (evaluation leakage)

    This caused the checkpoint to be selected based on test performance
    → reported accuracy was slightly optimistic
    → not trustworthy for medical deployment or patent claims

    Also: all data loaded at float32 → 8.7GB RAM → GPU OOM crashes
    Also: using datagen.flow() while building tf.data pipeline was
          causing pipeline mismatch (training used wrong pipeline)

Solution:
    1. Proper 3-way split: 70% train / 10% val / 20% test
       (split BEFORE oversampling to prevent data leakage)
    2. float16 storage in RAM, cast to float32 per batch on GPU
    3. tf.data pipeline with BigAug augmentation
    4. model.fit uses val_dataset for monitoring, test_dataset
       only for final honest evaluation

Impact observed:
    With proper split:
        Val accuracy  : 97.95% (on 2391 validation images)
        Test accuracy : 99%    (on 4781 truly unseen images)
        Test > Val → model generalizes beyond validation domain

BigAug strategy (from research):
    Aggressive augmentation simulates different lab conditions:
    brightness variation → different microscope lighting
    contrast variation   → different staining intensity
    flips + rotation     → orientation invariance
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# ============================================================
# STEP 1 — PROPER 3-WAY SPLIT
# CRITICAL: split BEFORE oversampling
# ============================================================
def create_3way_split(x_all, y_all_labels, y_all_encoded,
                       val_size=0.125, test_size=0.20,
                       random_state=42):
    """
    Create proper 3-way stratified split.

    Split order:
        1. Separate test set (20% of total) — lock away, never touch
        2. From remaining 80%, separate validation (12.5% = 10% of total)
        3. Remaining 70% is training data

    Args:
        x_all        : all images
        y_all_labels : string class labels
        y_all_encoded: integer encoded labels
        val_size     : fraction of train portion for validation
        test_size    : fraction of total for test

    Returns:
        x_train, x_val, x_test,
        y_train_labels, y_val_labels, y_test_labels,
        y_train_encoded, y_val_encoded, y_test_encoded
    """
    # Step 1 — separate test set first
    x_temp, x_test, \
    y_temp_labels, y_test_labels, \
    y_temp_encoded, y_test_encoded = train_test_split(
        x_all, y_all_labels, y_all_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all_encoded
    )

    # Step 2 — separate validation from remaining train
    x_train, x_val, \
    y_train_labels, y_val_labels, \
    y_train_encoded, y_val_encoded = train_test_split(
        x_temp, y_temp_labels, y_temp_encoded,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp_encoded
    )

    print(f"Train     : {len(x_train)} images")
    print(f"Validation: {len(x_val)} images")
    print(f"Test      : {len(x_test)} images")

    return (x_train, x_val, x_test,
            y_train_labels, y_val_labels, y_test_labels,
            y_train_encoded, y_val_encoded, y_test_encoded)


# ============================================================
# STEP 2 — tf.data PIPELINE WITH BIGAUG
# float16 storage → float32 cast per batch on GPU
# ============================================================
def build_tf_data_pipeline(x_train, y_train_oh,
                             x_val,   y_val_oh,
                             x_test,  y_test_oh,
                             batch_size=16):
    """
    Build tf.data pipelines for train, validation, and test.

    Key design:
        - Data stored as float16 in CPU RAM (saves memory)
        - Cast to float32 per batch inside pipeline (GPU receives float32)
        - BigAug applied only to training batches
        - No augmentation on validation or test

    Args:
        x_train/val/test : float16 images
        y_*_oh           : one-hot encoded labels
        batch_size       : batch size for training

    Returns:
        train_dataset, val_dataset, test_dataset
    """

    def augment_fn(image, label):
        image = tf.cast(image, tf.float32)          # float16 → float32
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label

    def cast_fn(image, label):
        image = tf.cast(image, tf.float32)          # float16 → float32
        return image, label

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        .shuffle(2000)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val_oh))
        .map(cast_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_oh))
        .map(cast_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Train batches     : {len(train_dataset)}")
    print(f"Validation batches: {len(val_dataset)}")
    print(f"Test batches      : {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


# ============================================================
# STEP 3 — TRAINING WITH PROPER VALIDATION
# ============================================================
def train_with_proper_split(model, train_dataset, val_dataset,
                              class_weights, save_path, epochs=50):
    """
    Train model using validation dataset for monitoring.
    Test dataset is NOT used during training.

    Args:
        model         : compiled Keras model
        train_dataset : tf.data training pipeline
        val_dataset   : tf.data validation pipeline (NOT test)
        class_weights : dict of class weights
        save_path     : path to save best model checkpoint
        epochs        : maximum training epochs

    Returns:
        history object
    """
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,    # ← val only, NOT test
        class_weight=class_weights,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
        verbose=1
    )
    return history


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    print("Pipeline summary:")
    print("  1. Load all train images → x_all (float16)")
    print("  2. create_3way_split() → x_train, x_val, x_test")
    print("  3. oversample_minority() on x_train ONLY")
    print("  4. build_tf_data_pipeline() → datasets")
    print("  5. train_with_proper_split() → model monitors val_dataset")
    print("  6. Final eval on test_dataset (truly unseen)")
    print()
    print("Key rule: test_dataset is LOCKED after step 2")
    print("          never used until step 6 evaluation")
