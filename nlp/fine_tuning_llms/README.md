# Fine-Tuning Large Language Models

End-to-end sentiment fine-tuning of DistilBERT on a GLUE-style dataset with the
Hugging Face `Trainer`: dataset loading, pretrained model loading, tokenization,
dynamic padding, evaluation metrics, `TrainingArguments`, train mode, inference
mode.

| | |
|---|---|
| Intranet project | [3689 — Fine-Tuning Large Language Models](https://intranet-dlh.hbtn.io/projects/3689) |
| Difficulty | Novice · Weight 1 |
| Author | Malek Mrabti |
| Window | Sep 28 → Oct 11 |
| Stack | `python3` 3.11, `pycodestyle` 2.14.0, `transformers`, `datasets`, `torch` |
| Status | Scaffolded 2026-10-12 — no task files yet |
| Resources | [`RESOURCES.md`](RESOURCES.md) |

> Manual QA review is required — request it when the project is done.

---

## Tasks

| # | Task | Status |
|---|------|--------|
| 0 | Load Dataset for Sentiment Analysis | not started |
| 1 | Load Pretrained DistilBERT for Text Classification | not started |
| 2 | Tokenize Dataset | not started |
| 3 | Dynamic Padding | not started |
| 4 | Evaluation Metrics | not started |
| 5 | Configure Training Arguments | not started |
| 6 | Training Mode | not started |
| 7 | Inference Mode | not started |

---

## Requirements (from the spec)

- Ubuntu 20.04 LTS, `python3` 3.11.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with a newline.
- `README.md` at the project root is mandatory; `pycodestyle` 2.14.0 clean.
- Module, class and function docstrings everywhere.

---

## Notes for this project

- Dynamic padding is the performance hinge: a collator that pads per batch beats
  padding the whole dataset once.
- Cached Hugging Face datasets/checkpoints belong in `.gitignore`.
- Verification environment: `tf-venv/bin/python`.
