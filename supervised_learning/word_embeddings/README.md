# Word Embeddings

Natural-language embeddings built from scratch and with gensim: bag of words,
TF-IDF, and a trained Word2Vec model, plus three advanced tasks unlocked after
the mandatory set.

| | |
|---|---|
| Intranet project | [2360 — Natural Language Processing - Word Embeddings](https://intranet-dlh.hbtn.io/projects/2360) |
| Track | AI Machine Learning Specialization → Main Block |
| Difficulty | Master · Weight 2 |
| Author | Alexa Orrico (Holberton School) |
| Stack | `python3` 3.9, `numpy` 1.25.2, `tensorflow` 2.15.0, `gensim` 4.3.3, `pycodestyle` 2.11.1 |
| Status | Mandatory tasks complete 2026-10-20 — output verified against the intranet |
| Resources | [`RESOURCES.md`](RESOURCES.md) — not yet ingested |

---

## Tasks

| # | Task | File | Score | Status |
|---|------|------|-------|--------|
| 0 | Bag Of Words | `0-bag_of_words.py` | 0/5 | implemented |
| 1 | TF-IDF | `1-tf_idf.py` | 0/5 | implemented |
| 2 | Train Word2Vec | `2-word2vec.py` | 0/6 | implemented |
| 3 | *gensim → Keras embedding* (advanced) | `3-gensim_to_keras.py` | — | locked |
| 4 | *Train fastText* (advanced) | `4-fasttext.py` | — | locked |
| 5 | *ELMo quiz* (advanced) | `5-elmo` | — | locked |

Mandatory tasks 0–2 are implemented and their output reproduces the intranet's
expected output **text-for-text** with the official `*-main.py` programs. Task 0
is pure numpy + `re`; task 1 uses `sklearn.feature_extraction.text.TfidfVectorizer`;
task 2 trains the gensim model twice (constructor pass + explicit `train`), which
is what reproduces the expected `w2v.wv["computer"]` values bit-for-bit. The three
advanced tasks stay locked until the mandatory set is done; their names come from
the reference forks (`Teheremiti`, `dassantoss`), so confirm against the intranet
once they unlock.

---

## Learning Objectives

Explain to anyone, without looking anything up:

- What is natural language processing? What is a word embedding?
- What is bag of words? What is TF-IDF?
- What is CBOW? What is a skip-gram?
- What is an n-gram? What is negative sampling?
- What is word2vec, GloVe, fastText, ELMo?

---

## Requirements (from the spec)

- Allowed editors: `vi`, `vim`, `emacs`; Ubuntu 20.04 LTS, `python3` 3.9.
- Executed with `numpy` 1.25.2 and `tensorflow` 2.15.
- `gensim==4.3.3`; Keras must report `2.15.0`.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with
  a newline and must be executable.
- `README.md` at the project root is mandatory; `pycodestyle` 2.11.1 clean.
- Module and function documentation everywhere.
- **Task 0 explicitly forbids the gensim library.**

---

## Notes for this project

- Verification runs in `tf-venv` (`tf-venv/bin/python`, Python 3.11,
  TensorFlow 2.15.0, numpy 1.25.2), **not** `myenv` — `myenv` is Python 3.14
  and cannot install TF 2.15. Task 0 needs only numpy, but the project as a
  whole needs gensim + TF.
- The task list is gated behind a 5-question quiz on the Description/Quiz tab;
  it was validated 2026-10-19, so the Tasks tab is open.
- `features` is documented as a list in the spec but is returned as a numpy
  `ndarray` — the sample output prints `['are' 'awesome' ...]` with no commas,
  which is numpy formatting. Both reference forks return `np.array(vocab)`.
- A supplied `vocab` keeps the caller's order (task 1's expected output is
  `['awesome' 'learning' ...]`, not sorted); only an auto-built vocabulary is
  sorted.
