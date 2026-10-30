# Verification of the deep-research report (30 Oct 2026)

This file checks the claims in the research report against the papers before they are used in TL;DRs or in the proposal to Prof. Mitra's group.

**How the checks were done.** arxiv.org and nature.com are blocked by this cloud session's network settings. The checks therefore used two other routes:

- Paper abstracts and indexed text from the Hugging Face papers index.
- Web-search snippets of the papers' own text.

**Status labels**

- **Verified**: seen in the paper's abstract or indexed text.
- **Corrected**: the report differs from the source.
- **Check in the paper**: could not be confirmed from here.

Confirm every number in the PDF before quoting it.

## The TL;DR ladder (8 papers; write at least 6)

| # | Paper | Status of the key claims |
|---|---|---|
| 1 | Tsutsui, Su & Wen, *Benchmarking White Blood Cell Classification Under Domain Shift*, ICASSP 2023 (arXiv 2303.01777) | **Verified:** title, authors, venue. **Verified:** train on Raabin-WBC and test on LISC; ResNet drops to around 23%, attributed to batch normalization. **Verified:** repaired ResNets reach about 74%, matching VGG. **Verified:** Test-B rejected because of missing WBC types and heavy imbalance. **Verified:** code at github.com/apple2373/wbc-benchmark. **Check in the paper:** the exact figures 98.53 → 23.11, 73.07, 74.44 and basophil recall 30.91. |
| 2 | Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization* (arXiv 2007.01434) | **Verified:** DG algorithms "without a model selection strategy should be regarded as incomplete". **Verified:** 7 datasets, 9 algorithms, 3 selection criteria. **Verified:** carefully implemented ERM is state of the art across all datasets. **Check in the paper:** venue (ICLR 2021). |
| 3 | Umer, Gruber et al., *Imbalanced Domain Generalization for Robust Single Cell Classification in Hematological Cytomorphology*, ICLR 2023 workshop (arXiv 2303.07771) | **Verified:** title, workshop venue; three hospitals; imbalance, missing classes and domain shift; best macro-F1 among compared methods. **Check in the paper:** 13 classes, ERM 0.40 vs 0.77, alignment methods 0.35–0.46. |
| 4 | Tellez et al., Medical Image Analysis 58:101544, 2019 (arXiv 1902.06543) | **Verified:** stain colour augmentation was crucial whatever normalization was used. **Verified:** HED ≈ HSV augmentation; light ≈ strong, slight edge for light. **Corrected:** "best = no normalization + HED-light". Summaries differ, and one reports network-based normalization plus augmentation as best, so read the final table. **Check in the paper:** "normalizers did not beat grayscale". |
| 5 | *Stain-aware Domain Alignment for Imbalance Blood Cell Classification* (SADA, arXiv 2412.02976) | **Verified:** leave-one-out evaluation. **Verified:** average macro-F1 68.6, which is 3.9 points above BoDA (second best). **Verified:** LDWBC 68.2 and Raabin-WBC 61.9; external test 85.1 vs CORAL 83.3; code is public. **Corrected:** the dataset list. A snippet of the ERM table lists Acevedo-20, LDWBC, Raabin-WBC and Zheng-18, not SYSU3H. **Check in the paper:** ERM average 63.2. |
| 6 | Jahanifar et al., *Domain Generalization in Computational Pathology: Survey and Guidelines* (arXiv 2310.19656) | **Verified:** 28 algorithms on MIDOG22 and 1,260 experiments (3×3×5×28). **Verified:** StainAug F1 76.0 vs ERM 74.7; RSC 75.7, EQRM 75.2; EQRM best accuracy 80.2. **Check in the paper:** author list and the guideline wording. |
| 7 | Sharma & Tapadiya, *Can You Trust Frozen Hematology Foundation Models under Acquisition Shift?* (arXiv 2608.25148, Aug 2026) | **Verified:** 15 frozen encoders on 4 public datasets; DinoBloom-L 1st in-domain → 10th of 15 on MLL23. **Verified:** ECE 0.004 → 0.35. **Verified:** label-free adaptation fails under realistic WBC class-prior shift; the training-free Class-Balanced Re-standardization helps. **Verified:** MLL23 is DinoBloom's internal cohort. **Check in the paper:** "34–72% macro-F1 loss". |
| 8 | Xu, …, Haeffele, *Adaptive Stain Normalization for Cross-Domain Medical Histology* (BeerLaNet), MICCAI 2025 (arXiv 2510.06592) | **Corrected:** the title (BeerLaNet is the method's name). **Verified:** NMF plus algorithmic unrolling, a plug-in for any backbone. **Verified:** public pathology data plus an *internal* malaria smear set; code at github.com/xutianyue/BeerLaNet. **Verified:** a MICCAI review notes it is task-specific, has no ablation on the number of colour components, and shows no consistent advantage. |

## Context claims

- **WBCBench 2026 (arXiv 2604.10797): Verified.**
  - 13 classes, patient-level splits, single-site images, synthetic noise/blur/illumination shift, macro-F1 ranking.
  - The dataset is on Hugging Face as `Xin-Tian/wbcbench2026` (about 101,716 rows, gated access).
- **MIDOG 2025 Track 2:** participant papers consistently give a final-phase baseline of 0.8274 balanced accuracy and a winner at 0.908. The official report (arXiv 2606.07368) was not readable from here, so these items remain **Check in the paper**:
  - the κ values (0.68 vs 0.48)
  - the number of domains
  - the CC BY-NC licence
- **arXiv 2608.10657** is Zamora et al., *Retrieval-Augmented Vision Foundation Models for Robust Leukemia Cell Classification across Multiple Microscopy Datasets*. Its held-out-dataset protocol is **Verified**. "ALL recall 0.0000" is **Check in the paper**.
- **Castro, Walker & Glocker 2020:** the paper and its shift types are real. The quotes "cannot be corrected without strong parametric assumptions" and "no guarantees for extrapolating…" are **Check in the paper**.
- **Yoon et al. survey:** the current title is *Domain Generalization for Medical Image Analysis: A Review* (v3, Dec 2024). It categorizes methods as data-, feature-, model- and analysis-level.
- **Prof. Mitra's six scope categories** (single-task domain-specific … domain-agnostic) were **not found** in the Yoon or Niu surveys. Ask the group which source they use.

## Research direction: the red-cell anchor

- No paper was found that uses red blood cells as an in-image stain reference for RGB leukocyte classification.
- Two pieces of **prior art** must be cited before claiming any novelty:
  - A spectral-imaging patent ("Spectral bio-imaging data for cell classification using internal reference", US 6690817) treats red cells as an intrinsically invariant stain reference.
  - A 2024 *Scientific Data* paper on 1 million segmented red cells notes that histopathology normalizers do not transfer well to smears, and that red-cell colour is the stain colour closest to red.
- Novelty is therefore **not established**. A full search is still needed, and so is a word with a patent attorney before preprinting the idea.

## To complete the open checks

Allow `arxiv.org`, `ar5iv.labs.arxiv.org` and `www.nature.com` in the cloud environment's network access settings (the environment menu in the session title bar, then Edit), then rerun the checks. Steps: https://code.claude.com/docs/en/cloud-environments#network-access
