# WBC Generalization Project
## Techniques Applied to Solve Domain Shift in White Blood Cell Classification

---

## Problem Statement

A deep learning architecture for WBC classification achieved **99.57% accuracy** on the
PBC dataset but dropped to **84.89%** when tested on the Raabin dataset — same 5 classes,
different microscope, different staining protocol.

Eosinophils F1 collapsed from **1.00 → 0.10**.
Basophils F1 collapsed from **1.00 → 0.15**.

This is the **domain shift problem** — a model that cannot generalize across
imaging conditions cannot be trusted for real clinical deployment.

---

## Architecture

Custom hybrid model combining:
- **MobileNetV2** backbone (pretrained on ImageNet)
- **SKBlock** (Selective Kernel Block) — channel attention with multi-scale receptive fields
- **MSFF** (Multi-Scale Feature Fusion) — fuses features from 3 backbone levels
- **CAB** (Context Attention Block) — spatial attention with dilated depthwise convolutions
- Features extracted from: `block_3_expand`, `block_6_expand`, `block_13_expand`

---

## Techniques Applied (5 files)

| File | Technique | Problem Solved | Impact |
|------|-----------|---------------|--------|
| `technique1_stain_normalization.py` | Reinhard stain normalization | Color domain shift between microscopes | Eosinophils 0.10 → 0.97 |
| `technique2_multi_source_merging.py` | PBC + Raabin combined training | Single-domain overfitting | Overall 85% → 91%+ |
| `technique3_oversampling_class_weights.py` | Minority oversampling + class weights | Class imbalance (Basophils 7x less than Neutrophils) | Basophils 0.15 → 0.98 |
| `technique4_3zone_finetuning.py` | 3-zone selective backbone unfreezing | Frozen backbone = wrong ImageNet features | Accuracy 20% → 99% |
| `technique5_proper_split_bigaug_pipeline.py` | Proper 3-way split + BigAug pipeline | Evaluation leakage + RAM OOM + pipeline mismatch | Honest 99% test accuracy |

---

## Results Summary (see hit_and_trial_results.csv for full table)

| Stage | Strategy | Overall Accuracy | Basophils F1 | Eosinophils F1 |
|-------|----------|-----------------|--------------|----------------|
| Baseline (PBC only) | Fully frozen | 99.57% (PBC) / 84.89% (Raabin) | 1.00 / 0.15 | 1.00 / 0.10 |
| After stain norm + merging | Fully frozen | 91% | 0.17 | 0.97 |
| After fine_tune_at=110 | Simple unfreezing | 94% | 0.89 | 0.89 |
| After fine_tune_at=90 + new kernels | Simple unfreezing | 95% | 0.69 | 0.96 |
| **wbc20 — PRODUCTION MODEL** | **3-zone + proper split** | **99%** | **0.98** | **0.99** |

---

## Key Learnings

1. High single-dataset accuracy is NOT a deployment readiness indicator
2. Stain normalization is the single most impactful preprocessing step for cross-domain generalization
3. 3-zone selective fine-tuning outperforms simple fine_tune_at because all 3 feature
   extraction points adapt simultaneously
4. Proper train/val/test split is essential for honest medical AI evaluation
5. Architecture changes have diminishing returns once preprocessing is correct —
   the remaining generalization gap requires more domain diversity in training data

---

## Production Model: wbc20 (production_wbcf20.keras)

Configuration:
- SK branch 1: (3x3) + (3x3) standard convolution
- SK branch 2: (5x5)dilated_rate=3 + (5x5)dilated_rate=5
- CAB: DepthwiseConv(5, dilation=3) + DepthwiseConv(7, dilation=5)
- 3-zone fine-tuning: zones [11:27], [38:54], [90:117]
- Stain normalization: TARGET_MEAN=[181.25, 142.23, 131.06], TARGET_STD=[40.71, 9.86, 13.54]
- Oversampling: Basophils + Monocytes → 3000 images each
- Proper 3-way split: 70% train / 10% val / 20% test

Test results on completely unseen test set (4781 images):
- Basophils F1   : 0.98
- Eosinophils F1 : 0.99
- Lymphocytes F1 : 0.98
- Monocytes F1   : 0.96
- Neutrophils F1 : 1.00
- Overall accuracy: 99%

---

## Next Steps

- [ ] Apply CLAHE preprocessing to further improve cross-domain robustness
- [ ] Add BCCD and Kaggle blood cell datasets for more domain diversity
- [ ] Implement continual learning to adapt to new hospital data without forgetting
- [ ] Deploy GradCAM visualizations on Hugging Face Space for interpretability
- [ ] Target 75%+ accuracy on completely unseen clinical datasets
