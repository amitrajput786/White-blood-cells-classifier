# Jewellery AI: try-on image generator

This folder supports a website where local jewellers list their products. For each product it generates a
**"model wearing this jewellery"** photo from:

1. a photo of the jewellery product, and
2. a photo of a model who isn't wearing jewellery.

It uses [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) and runs in batches on Kaggle, so it doesn't hit the 40-minute limit of the HF Space.

## Files
- `qwen_jewellery_tryon_kaggle.ipynb`: the Kaggle notebook (upload it with **Kaggle → Create → Import Notebook**).

## Quick start
1. Import the notebook into Kaggle. Set **Accelerator = GPU T4 x2** and **Internet = On**.
2. **Add-ons → Secrets**: add `HF_TOKEN` (a *read* token).
3. Upload a dataset with `models/` and `jewellery/` folders, plus an optional `jobs.csv`.
4. Run cells 1–8 to test one image. Then use **Save Version → Save & Run All** for the whole batch.

## Model backends
| `BACKEND` | Weights | Download | Licence |
|---|---|---|---|
| `nf4` (default) | `ovedrive/Qwen-Image-Edit-2511-4bit` | ~17 GB | CC-BY-NC-SA: **prototype only, not commercial** |
| `gguf` | `unsloth/Qwen-Image-Edit-2511-GGUF` Q4_K_M + original text encoder (4-bit at load) | ~30 GB | Apache-2.0: OK for the business |

Speed-up: the `lightx2v/Qwen-Image-Edit-2511-Lightning` 8-step LoRA (Apache-2.0).
