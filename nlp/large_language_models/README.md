# Large Language Models (LLMs)

Masked-language-model mechanics and the three transformer families, all through
Hugging Face: fill-mask on an encoder, translation with an encoder-decoder, text
generation with a decoder-only model, then ViT for images and a vision-language
model (BLIP) for captioning.

| | |
|---|---|
| Intranet project | [3601 — Large Language Models (LLMs)](https://intranet-dlh.hbtn.io/projects/3601) |
| Difficulty | Novice · Weight 1 |
| Author | Malek Mrabti |
| Window | Sep 21 → Oct 04 |
| Stack | `python3` 3.11, `pycodestyle` 2.14.0, `transformers`, `torch`, `tensorflow` |
| Status | Scaffolded 2026-10-12 — no task files yet |
| Resources | [`RESOURCES.md`](RESOURCES.md) |

---

## Tasks

| # | Task | Status |
|---|------|--------|
| 0 | (MLM) Load RoBERTa | not started |
| 1 | (MLM) Tokenization | not started |
| 2 | (MLM) Get Mask Indices | not started |
| 3 | (MLM) Compute Mask Logits | not started |
| 4 | (MLM) Decode Mask Predictions | not started |
| 5 | (Encoder-Only) Fill Mask | not started |
| 6 | (Encoder-Decoder) Translation | not started |
| 7 | (Decoder-Only) Text Generation | not started |
| 8 | Performing Computer Vision Tasks | not started |
| 9 | VLM for Image Captioning | not started |

---

## Requirements (from the spec)

- Ubuntu 20.04 LTS, `python3` 3.11.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with a newline.
- `README.md` at the project root is mandatory; `pycodestyle` 2.14.0 clean.
- Module, class and function docstrings everywhere.

---

## Notes for this project

- Mixes PyTorch and TensorFlow paths (the HF pipelines are demonstrated in both);
  the DistilGPT2 text-generation task is TensorFlow-side. Expect both frameworks
  in one project — check the task spec before assuming the stack.
- Model weights are downloaded at runtime and must be gitignored.
- Verification environment: `tf-venv/bin/python` (Python 3.11, TF 2.15).
