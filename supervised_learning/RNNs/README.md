# RNNs

Recurrent architectures built from scratch with NumPy: the vanilla RNN cell, the
full RNN forward pass, the GRU cell, the LSTM cell, and a deep (multi-layer) RNN.

| | |
|---|---|
| Intranet project | [2337 — RNNs](https://intranet-dlh.hbtn.io/projects/2337) |
| Difficulty | Master · Weight 4 |
| Author | Alexa Orrico (Holberton School) |
| Stack | `python3` 3.9, `numpy` 1.25.2, `pycodestyle` 2.11.1 |
| Status | Scaffolded 2026-10-12 — no task files yet |
| Resources | [`RESOURCES.md`](RESOURCES.md) |

---

## Tasks

| # | Task | File | Status |
|---|------|------|--------|
| 0 | RNN Cell | `0-rnn_cell.py` | not started |
| 1 | RNN | `1-rnn.py` | not started |
| 2 | GRU Cell | `2-gru_cell.py` | not started |
| 3 | LSTM Cell | `3-lstm_cell.py` | not started |
| 4 | Deep RNN | `4-deep_rnn.py` | not started |

File names follow the repo convention (`<n>-<task_name>.py`); confirm against the
intranet task page before writing code.

---

## Learning Objectives

Explain to anyone, without looking anything up:

**General**
- What is an RNN? What is an LSTM? What is a GRU? What is a BRNN?
- What is the exploding-gradient problem, and when does it occur?
- What is the vanishing-gradient problem, and when does it occur?
- How do LSTM and GRU overcome the vanishing-gradient problem?

---

## Requirements (from the spec)

- Allowed editors: `vi`, `vim`, `emacs`.
- Ubuntu 20.04 LTS, `python3` 3.9, `numpy` 1.25.2.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with a newline.
- `README.md` at the project root is mandatory.
- `pycodestyle` 2.11.1 clean.
- Module, class and function docstrings everywhere.

---

## Notes for this project

- The project spec flags a mistake in the GRU equation of the linked RNNs
  Tutorial part 3: the state update should read `s_t = (1 - z) * s_{t-1} + z * h_t`.
  Match the reference forks, not the tutorial.
- Verification environment: this is a NumPy project, so run it in `myenv/bin/python`.
