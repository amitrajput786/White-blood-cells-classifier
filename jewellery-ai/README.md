# Jewellery AI: try-on image generator

This folder supports a website where local jewellers list their products. For each product it generates a
**"model wearing this jewellery"** photo from a jewellery product photo and a model photo, using
[Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) on **free Kaggle GPUs (2× T4)**.

## File
- `qwen_jewellery_tryon_kaggle.ipynb`: install, load the model, test one image, then open a page that works like the HF Space (Edit + Batch tabs, public `gradio.live` link).

## Setup that works on 2× T4 (tested)
| Part | Source | Where |
|---|---|---|
| Image model (4-bit NF4) | `ovedrive/Qwen-Image-Edit-2511-4bit` | layers 0–35 on GPU 0, 36–59 on GPU 1 |
| Text encoder (4-bit, vision part full precision) | `Qwen/Qwen-Image-Edit-2511` | GPU 1 |
| VAE | `Qwen/Qwen-Image-Edit-2511` | GPU 1 |
| Speed-up | `lightx2v/Qwen-Image-Edit-2511-Lightning` 8-step LoRA | — |

Pinned versions: `diffusers==0.39.0`, `transformers==4.57.1`, `huggingface_hub<1.0`, `gradio==5.50.0`, with `torchao` removed.

Measured speed: about **9.5 min per image** at 768×960 output with input photos processed at 768 px (8 steps).

## Known issues, already worked around in the notebook
- The 4-bit text encoder in the `ovedrive` repo uses an old weight layout and loads with empty layers, so the original text encoder is used instead.
- bitsandbytes 8-bit maths fails on T4 (`cublasLt` error), so the text encoder uses 4-bit.
- There is no fast bf16 attention on T4, so the attention op runs in fp16.
- Kaggle's diffusers sometimes doesn't detect bitsandbytes, so Cell 2 tells it explicitly.

## Licence note
`ovedrive/Qwen-Image-Edit-2511-4bit` is **CC-BY-NC-SA (non-commercial)**. Fine for building demos; switch the image model to an
Apache-2.0 quantisation (e.g. `unsloth/Qwen-Image-Edit-2511-GGUF`) before selling generated images.
