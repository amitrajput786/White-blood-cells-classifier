#!/usr/bin/env python3
"""
TECHNIQUE 4 — 3-Zone Selective Fine-Tuning of MobileNetV2 Backbone
===================================================================
Problem it solves:
    Standard approach: base_model.trainable = False (fully frozen)
    Result: model stuck at 20% accuracy — backbone features are
    ImageNet-specific (cats, cars, planes) and irrelevant for WBC.

    Simple fine_tune_at=N approach: unfreezes everything after layer N
    Result: 91-94% — better but layers after block_13_expand are
    never actually used (they are not in the computation path) so
    they receive no gradients and never update.

Solution — 3-zone selective unfreezing:
    Instead of one continuous unfrozen region, unfreeze 3 targeted
    windows — one just before each feature extraction point.
    This ensures ALL three feature levels (block_3, block_6, block_13)
    adapt to WBC-specific patterns simultaneously.

MobileNetV2 layer positions (verified):
    block_3_expand  @ Layer 26   → low-level features (edges, shapes)
    block_6_expand  @ Layer 53   → mid-level features (textures)
    block_13_expand @ Layer 116  → high-level features (morphology)
    Total layers    : 155

Zone configuration (best result):
    Zone 1: layers 11-26  (15 layers before block_3_expand)
    Zone 2: layers 38-53  (15 layers before block_6_expand)
    Zone 3: layers 90-116 (26 layers before block_13_expand)

Impact observed:
    Fully frozen                    : 20% accuracy (stuck)
    fine_tune_at=100 (simple)       : 92% accuracy
    fine_tune_at=110 (simple)       : 94% accuracy
    3-zone strategy                 : 99% accuracy on merged test
    Basophils F1 journey:
        frozen       → 0.17
        fine_tune=100 → 0.56
        fine_tune=110 → 0.89
        3-zone        → 0.98
"""

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input


# ============================================================
# 3-ZONE SELECTIVE UNFREEZING
# ============================================================
def apply_3zone_finetuning(base_model,
                            zone1=(11, 27),
                            zone2=(38, 54),
                            zone3=(90, 117)):
    """
    Apply 3-zone selective fine-tuning to MobileNetV2 backbone.

    Unfreezes targeted windows of layers just before each feature
    extraction point (block_3_expand, block_6_expand, block_13_expand).
    All other layers remain frozen to preserve stable ImageNet features.

    Args:
        base_model : MobileNetV2 base model
        zone1      : (start, end) layer indices for Zone 1
        zone2      : (start, end) layer indices for Zone 2
        zone3      : (start, end) layer indices for Zone 3

    Returns:
        base_model with selective layers unfrozen
    """
    # First freeze ALL layers
    for layer in base_model.layers:
        layer.trainable = False

    # Zone 1 — just before block_3_expand (low-level WBC features)
    for layer in base_model.layers[zone1[0]:zone1[1]]:
        layer.trainable = True

    # Zone 2 — just before block_6_expand (mid-level WBC features)
    for layer in base_model.layers[zone2[0]:zone2[1]]:
        layer.trainable = True

    # Zone 3 — just before block_13_expand (high-level WBC morphology)
    for layer in base_model.layers[zone3[0]:zone3[1]]:
        layer.trainable = True

    # Summary
    frozen   = sum(1 for l in base_model.layers if not l.trainable)
    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"Frozen  : {frozen} layers")
    print(f"Unfrozen: {unfrozen} layers")
    print("\nTrainable layers:")
    for i, layer in enumerate(base_model.layers):
        if layer.trainable:
            print(f"  Layer {i:4d} | {layer.name}")

    return base_model


# ============================================================
# ALTERNATIVE — Simple fine_tune_at (for comparison)
# ============================================================
def apply_simple_finetuning(base_model, fine_tune_at=110):
    """
    Simple approach: freeze layers before fine_tune_at, unfreeze after.
    Less effective than 3-zone because layers after block_13_expand
    are never used and receive no gradients.

    Args:
        base_model   : MobileNetV2 base model
        fine_tune_at : layer index boundary

    Returns:
        base_model with layers fine_tune_at+ unfrozen
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    frozen   = sum(1 for l in base_model.layers if not l.trainable)
    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"Simple fine_tune_at={fine_tune_at}: "
          f"Frozen={frozen}, Unfrozen={unfrozen}")
    return base_model


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":

    input_tensor = Input(shape=(128, 128, 3))
    base_model   = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )

    print("=" * 50)
    print("3-Zone Fine-tuning (RECOMMENDED)")
    print("=" * 50)
    base_model = apply_3zone_finetuning(base_model)

    total_trainable = sum(
        tf.size(w).numpy() for w in base_model.trainable_weights
    )
    print(f"\nTotal trainable params in backbone: {total_trainable:,}")

    print("\n" + "=" * 50)
    print("Why 3-zone is better than simple fine_tune_at:")
    print("=" * 50)
    print("Simple fine_tune_at=110:")
    print("  feature1 (layer 26) FROZEN → never adapts to WBC")
    print("  feature2 (layer 53) FROZEN → never adapts to WBC")
    print("  feature3 (layer 116) TRAINABLE → only this adapts")
    print("\n3-zone strategy:")
    print("  feature1 (layer 26) TRAINABLE → adapts to WBC edges/shapes")
    print("  feature2 (layer 53) TRAINABLE → adapts to WBC textures")
    print("  feature3 (layer 116) TRAINABLE → adapts to WBC morphology")
    print("  All 3 scales learn WBC-specific patterns simultaneously")
