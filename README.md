# DLH — Machine Learning

A structured curriculum building the mathematical and data engineering foundations for machine learning.

---

## Directory Structure

```
dlh-machine_learning/
├── math/                        # Mathematical foundations
│   ├── linear_algebra/          # Matrix operations: Python lists → NumPy
│   │   ├── 0-slice_me_up.py     through 14-saddle_up.py
│   │   ├── 100-slice_like_a_ninja.py through 102-squashed_like_sardines.py
│   │   └── README.md
│   ├── advanced_linear_algebra/ # Determinant, minor, cofactor, adjugate, inverse, definiteness
│   │   ├── 0-determinant.py through 5-definiteness.py, quiz.py
│   │   └── README.md
│   ├── calculus/                # Derivatives, integrals, partial derivatives, double integrals
│   │   ├── 0-sigma_is_for_sum through 17-integrate.py
│   │   └── README.md
│   ├── bayesian_prob/           # Likelihood, intersection, marginal, posterior probability
│   │   ├── 0-likelihood.py through 3-posterior.py
│   │   └── README.md
│   ├── plotting/                # Matplotlib: line, scatter, bar, frequency, PCA, gradient
│   │   ├── 0-line.py through 101-pca.py
│   │   └── README.md
│   ├── probability/             # Distributions: binomial, normal, poisson, exponential
│   │   ├── binomial.py, normal.py, poisson.py, exponential.py
│   │   └── README.md
│   ├── multivariate_prob/       # Mean vector, covariance, correlation, multivariate normal
│   │   ├── 0-mean_cov.py, 1-correlation.py, multinormal.py
│   │   └── README.md
│   └── README.md
├── pipeline/                    # Data engineering
│   ├── databases/               # SQL: creation, CRUD, joins, aggregates, triggers
│   │   ├── 0-create_database_if_missing.sql through 18-valid_email.sql
│   │   ├── hbtn_0d_tvshows.sql, hbtn_0d_tvshows_rate.sql
│   │   ├── metal_bands.sql, temperatures.sql
│   │   └── README.md
│   ├── pandas/                  # Pandas: DataFrame creation, cleaning, concat, resample, plot
│   │   ├── 0-from_numpy.py through 14-visualize.py
│   │   ├── *.csv, *.csv.zip
│   │   └── README.md
│   └── README.md
├── supervised_learning/         # Neural networks & deep learning
│   ├── classification/          # Neuron → neural network → deep NN (NumPy)
│   │   ├── 0-neuron.py through 15-neural_network.py
│   │   ├── README.md, RESOURCES.md
│   │   └── data/                # MNIST + binary datasets (gitignored)
│   ├── keras/                   # TensorFlow 2 & Keras
│   │   ├── 0-sequential.py through 10-weights.py
│   │   └── README.md, RESOURCES.md
│   └── rnn/                     # RNN, GRU, LSTM, Deep RNN (NumPy, Master)
│       ├── 0-rnn_cell.py through 4-deep_rnn.py
│       └── README.md, RESOURCES.md
├── unsupervised_learning/       # Clustering & dimensionality reduction
│   ├── clustering/              # k-means, GMM, agglomerative (NumPy)
│   │   ├── 0-initialize.py through 12-agglomerative.py
│   │   └── README.md, RESOURCES.md
│   └── dimensionality_reduction/ # PCA, t-SNE (NumPy)
│       ├── 0-pca.py through 8-tsne.py
│       └── README.md, RESOURCES.md
├── nlp/                         # Natural language processing & LLMs
│   ├── introduction_to_nlp/     # Normalization → tokens → TF-IDF → Word2Vec/FastText
│   │   └── README.md, RESOURCES.md
│   ├── large_language_models/   # MLM internals, encoder/decoder, ViT, BLIP
│   │   └── README.md, RESOURCES.md
│   ├── fine_tuning_llms/        # DistilBERT sentiment fine-tuning with Trainer
│   │   └── README.md, RESOURCES.md
│   └── langchain/               # Prompts, agents, RAG
│       └── README.md, RESOURCES.md
├── my-venv/                     # Python virtual environment
└── README.md
```

---

## Quick Reference

| Track | Module | Topics | Tasks |
|-------|--------|--------|-------|
| **Math** | [Linear Algebra](math/linear_algebra/) | Slicing, shape, transpose, element-wise ops, concat, matrix multiply, NumPy, n-D generalization | 19 |
| **Math** | [Advanced Linear Algebra](math/advanced_linear_algebra/) | Determinant, minor, cofactor, adjugate, inverse, definiteness (manual + NumPy) | 7 |
| **Math** | [Calculus](math/calculus/) | Derivatives, partial derivatives, integrals, definite/indefinite, double integrals | 17 |
| **Math** | [Bayesian Probability](math/bayesian_prob/) | Likelihood, intersection, marginal, posterior probability | 4 |
| **Math** | [Plotting](math/plotting/) | Line, scatter, bar, frequency, all-in-one, gradient descent, PCA | 9 |
| **Math** | [Probability](math/probability/) | Binomial, normal, poisson, exponential distributions | 4 |
| **Math** | [Multivariate Probability](math/multivariate_prob/) | Mean vector, covariance, correlation, multivariate normal distribution | 3 |
| **Pipeline** | [Databases](pipeline/databases/) | DDL, CRUD, WHERE, ORDER BY, GROUP BY, JOINS, aggregates, constraints, triggers | 18 (+4 schemas) |
| **Pipeline** | [Pandas](pipeline/pandas/) | DataFrame creation, rename, slice, fill, concat, hierarchy, describe, resample, visualize | 15 |
| **Supervised** | [Classification](supervised_learning/classification/) | Binary classification: neuron (logistic), NN with one hidden layer, deep NN; NumPy from scratch | 16 |
| **Supervised** | [Keras](supervised_learning/keras/) | TensorFlow 2 & Keras: Sequential, Input, optimization, training, model persistence | 11 |
| **Supervised** | [RNNs](supervised_learning/rnn/) | Recurrent cells from scratch: RNN, GRU, LSTM, deep RNN (NumPy) | 5 |
| **Unsupervised** | [Clustering](unsupervised_learning/clustering/) | k-means, variance, optimum k, Gaussian Mixture Models, EM, BIC, agglomerative | 13 |
| **Unsupervised** | [Dimensionality Reduction](unsupervised_learning/dimensionality_reduction/) | PCA, t-SNE (P/affinities, gradient descent, cost) | 9 |
| **NLP** | [Introduction to NLP](nlp/introduction_to_nlp/) | Normalization, tokenization, stemming/lemmatization, n-grams, BoW, TF-IDF, Word2Vec, FastText | 13 |
| **NLP** | [Large Language Models](nlp/large_language_models/) | Masked language modeling (RoBERTa), encoder/decoder-only usage, ViT, BLIP captioning | 10 |
| **NLP** | [Fine-Tuning LLMs](nlp/fine_tuning_llms/) | DistilBERT sentiment fine-tuning: tokenization, dynamic padding, metrics, `Trainer` | 8 |
| **NLP** | [LangChain](nlp/langchain/) | Prompt templates, tool/agent pipelines, RAG: loaders, chunking, vector store, retrieval | 16 |

---

## Learning Progression

### Math Track
1. **Python Slicing** → 2. **Manual Matrix Ops** (nested loops) → 3. **NumPy Vectorization** → 4. **N-Dimensional Generalization** → 5. **Advanced Linear Algebra** (determinant → inverse → definiteness) → 6. **Calculus** (derivatives → integrals) → 7. **Probability & Statistics** (distributions → Bayesian) → 8. **Multivariate Probability** (mean/cov → correlation → multivariate normal) → 9. **Visualization** (plotting → PCA)

### Pipeline Track
1. **Foundation** (CREATE) → 2. **CRUD** → 3. **Filtering/Sorting** → 4. **Joins** → 5. **Constraints** → 6. **Real-World Data** → 7. **Triggers** → 8. **Pandas** (DataFrame ops → cleaning → concat → resample → visualize)

### Supervised Learning Track
1. **Classification Fundamentals** (neuron → single hidden layer → deep NN, NumPy from scratch) → 2. **Keras** (high-level API) → 3. **RNNs** (recurrent, gated, deep)

### Unsupervised Learning Track
1. **Clustering** (k-means → variance → GMM/EM → agglomerative) → 2. **Dimensionality Reduction** (PCA → t-SNE)

### NLP / LLM Track
1. **Classical NLP** (normalization → tokens → TF-IDF → embeddings) → 2. **LLMs** (masked language modeling → encoder/decoder families → ViT/BLIP) → 3. **Fine-Tuning** (DistilBERT + `Trainer`) → 4. **LangChain** (prompts → agents → RAG)

---

## Setup

```bash
cd dlh-machine_learning
source my-venv/bin/activate
pip install numpy
```

Keras code (supervised_learning/keras/) runs in `tf-venv` (Python 3.11, TensorFlow 2.15).

---

## Resources

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [Python Official Documentation](https://docs.python.org/3/)
