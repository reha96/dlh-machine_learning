# Attention

The attention mechanism and the transformer, built with TensorFlow/Keras:
scaled dot-product attention, multi-head attention, and the encoder and decoder
blocks that stack into a transformer.

| | |
|---|---|
| Intranet project | [2338 — Attention](https://intranet-dlh.hbtn.io/projects/2338) |
| Track | AI Machine Learning Specialization → Main Block |
| Difficulty | Master · Weight 5 |
| Author | Alexa Orrico (Holberton School) |
| Stack | `python3` 3.9, `numpy` 1.25.2, `tensorflow` 2.15, `pycodestyle` 2.11.1 |
| Status | Scaffolded 2026-10-20 — no task implemented yet |
| Resources | [`RESOURCES.md`](RESOURCES.md) — not yet ingested |

---

## Tasks

The Tasks tab renumbers the mandatory set 0–4, but the **file names keep their
original numbering** and the task headers read 3–7. The `File:` line on each
task page is authoritative.

| Tasks tab | Task header | File | Score | Status |
|---|---|---|---|---|
| 0 | 3. Positional Encoding | `4-positional_encoding.py` | 0/3 | scaffolded |
| 1 | 4. Scaled Dot Product Attention | `5-sdp_attention.py` | 0/3 | scaffolded |
| 2 | 5. Multi Head Attention | `6-multihead_attention.py` | 0/4 | scaffolded |
| 3 | 6. Transformer Encoder Block | `7-transformer_encoder_block.py` | 0/4 | scaffolded |
| 4 | 7. Transformer Decoder Block | `8-transformer_decoder_block.py` | 0/4 | scaffolded |

Six **advanced** tasks unlock once the mandatory set is done. Their names come
from the reference forks (`Teheremiti`, `dassantoss`) and cover the parts this
project does not: `0-rnn_encoder.py`, `1-self_attention.py`,
`2-rnn_decoder.py`, `9-transformer_encoder.py`, `10-transformer_decoder.py`,
`11-transformer.py`. Confirm against the intranet once they unlock.

Each task mandates a chained import, so the order matters:

| Task | Must use |
|---|---|
| 2 | `sdp_attention = __import__('5-sdp_attention').sdp_attention` |
| 3 | `MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention` |
| 4 | `MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention` |

---

## Learning Objectives

Explain to anyone, without looking anything up:

- What is the attention mechanism? How to apply attention to RNNs?
- What is a transformer? How to create an encoder-decoder transformer model?
- What is GPT? What is BERT?
- What is self-supervised learning?
- How to use BERT for specific NLP tasks.
- What is SQuAD? What is GLUE?

---

## Requirements (from the spec)

- Allowed editors: `vi`, `vim`, `emacs`; Ubuntu 20.04 LTS, `python3` 3.9.
- Executed with `numpy` 1.25.2 and `tensorflow` 2.15.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with
  a newline and must be executable.
- `README.md` at the project root is mandatory; `pycodestyle` 2.11.1 clean.
- Module, class and function documentation everywhere.
- **`import tensorflow as tf` only** — "unless otherwise stated, you cannot
  import any module except `import tensorflow as tf`". Task 4
  (positional encoding) is the exception and says to use `import numpy as np`.

---

## Notes for this project

- This is a TF project, so verification runs in `tf-venv`
  (`tf-venv/bin/python`, Python 3.11, TensorFlow 2.15.0, keras 2.15.0,
  numpy 1.25.2), **not** `myenv`.
- Importing any task module prints TensorFlow's CUDA/cuDNN registration lines
  on a CPU-only box. That is TF's own logging, not module output — the modules
  are silent under `TF_CPP_MIN_LOG_LEVEL=3`.
- `dm` is the model depth and `h` the head count, and `dm` must be divisible by
  `h`. Each head works on `depth = dm // h` dimensions.
