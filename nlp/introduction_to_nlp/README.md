# Introduction to NLP

Classical text pipeline from raw documents to vectors: normalization,
tokenization, stop-word removal, filtering, lemmatization vs stemming, n-grams,
frequency distributions, word clouds, bag of words, TF-IDF, then the two
embedding families Word2Vec and FastText.

| | |
|---|---|
| Intranet project | [3603 — Introduction to NLP](https://intranet-dlh.hbtn.io/projects/3603) |
| Difficulty | Novice · Weight 1 |
| Author | Khaled Haguiga |
| Window | Sep 14 → Sep 27 |
| Stack | `python3` 3.11, `pycodestyle` 2.14.0 |
| Status | Scaffolded 2026-10-12 — no task files yet |
| Resources | [`RESOURCES.md`](RESOURCES.md) |

---

## Tasks

| # | Task | File | Status |
|---|------|------|--------|
| 0 | Basic Exploration | `0-basic_exploration.py` | not started |
| 1 | Text normalization | `1-text_normalization.py` | not started |
| 2 | Tokenization | `2-tokenization.py` | not started |
| 3 | Stopwords Removal | `3-stopwords_removal.py` | not started |
| 4 | Filtering | `4-filtering.py` | not started |
| 5 | Lemma vs Stem | `5-lemma_vs_stem.py` | not started |
| 6 | N-gram | `6-n_gram.py` | not started |
| 7 | Word Frequency Distribution | `7-word_frequency_distribution.py` | not started |
| 8 | Wordclouds | `8-wordclouds.py` | not started |
| 9 | BoW | `9-bow.py` | not started |
| 10 | TF-IDF | `10-tf_idf.py` | not started |
| 11 | Word2Vec | `11-word2vec.py` | not started |
| 12 | FastText | `12-fasttext.py` | not started |

File names follow the repo convention; confirm against the intranet task page.

---

## Requirements (from the spec)

- Ubuntu 20.04 LTS, `python3` 3.11.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with a newline.
- `README.md` at the project root is mandatory; `pycodestyle` 2.14.0 clean.
- Module, class and function docstrings everywhere.

---

## Notes for this project

- Heavy `nltk` usage: corpora (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) must be
  downloaded locally before the checker runs.
- `wordcloud` and `gensim` are needed for the later tasks; add them to `.gitignore`
  artifacts (downloaded corpora are large and must stay uncommitted).
- Verification environment: `myenv/bin/python`.
