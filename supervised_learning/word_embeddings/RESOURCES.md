# RESOURCES — Natural Language Processing - Word Embeddings (2360)

Intranet project [2360](https://intranet-dlh.hbtn.io/projects/2360) · Ingested 2026-10-20.
All summaries are paraphrased; raw transcripts live outside the repo and are never
committed. rltoken URLs are auth-gated and intentionally omitted — final public
destinations are recorded instead.

> **rltoken label/destination issues found during ingestion (2 of 26):**
> 1. **"ELMo (website)"** resolves to `github.com/allenai/allennlp-models` (a model
>    zoo repo), not an ELMo demo/website.
> 2. **"GloVe Explained"** resolves to a **truncated** Medium URL ending
>    `...glove-51`, which 404s. The real article id is `51339db89639`; the full
>    URL below works. Both forks of the expected output are unaffected.

---

## Read or watch

### An Introduction to Word Embeddings

URL: https://www.springboard.com/blog/data-science/introduction-word-embeddings/ · Date: 2017-08-28 (byline) · Status: summary

First half of a two-part beginner series (applications, written by Aaron Geelon So) arguing that word embeddings turn words into vectors so that meaning becomes geometry. It showcases the famous analogical arithmetic (royalty/gender offsets), country–capital structure, cross-language embedding correlation as a translation hint, and real deployments like music/clothing recommendations and search ranking, before pointing at successor methods (GloVe, fastText, WordRank).

- Goal: embed words by similarity/relationship so words with similar contexts land near each other in R^n
- Mikolov 2013 regularities: king − man + woman ≈ queen; country→capital; singular/plural and gender axes
- Cross-language structures correlate — a hint toward machine translation via embeddings (Mikolov 2013c)
- Uses: Spotify music recs, Stitch Fix clothing recs, RankBrain; community-specific sentiment (e.g. "soft" reads negative in sports talk, positive in cartoon talk)
- Successors named: GloVe (Stanford), fastText (Facebook), WordRank
- Fetch note: the live site 403s non-browser clients; content retrieved from the Wayback Machine

### Introduction to Word Embeddings (Word Embeddings in NLP)

URL: https://hunterheidenreich.com/posts/intro-to-word-embeddings/ · Date: 2018-08-05 (byline) · Status: summary

A structured survey of every embedding family, starting from the distributional hypothesis (words in similar contexts mean similar things) and walking one-hot counts → TF-IDF → co-occurrence matrices → neural probabilistic models → Word2Vec → GloVe → FastText/Poincaré → contextual models (ELMo, probabilistic FastText). It contrasts dense vs sparse representations, explains CBOW vs skip-gram and window-size effects, and closes with a static-vs-contextual selection guide.

- Embedding = map word → dense R^n; similar meanings cluster (grounded in Harris' distributional hypothesis)
- Sparse baselines (one-hot, counts, TF-IDF) are cheap but semantics-blind; co-occurrence matrices capture global stats at V² cost
- Word2Vec: CBOW predicts the center word from its context; skip-gram does the reverse; large windows → topical, small windows → syntactic similarity
- GloVe factorizes global word–word co-occurrence counts, blending word2vec-style analogies with corpus-wide statistics
- FastText = skip-gram + character n-grams → embeds unseen/morphologically related words; ELMo makes vectors context-dependent ("bank" differs in river vs finance)
- Trade-off: static embeddings (w2v/GloVe/fastText) for speed and pretraining coverage; contextual (ELMo/BERT/GPT) for polysemy-heavy accuracy

### Natural Language Processing | Bag Of Words Intuition

URL: https://www.youtube.com/watch?v=IKgBLTeQQL8 · Date: 2020-05-02 (YouTube upload) · Status: summary (transcript; 562s)

Krish Naik builds a bag-of-words matrix by hand on toy sentences, showing the full path from raw text to count vectors: normalization (lowercasing with care for proper nouns like "US"), vocabulary collection, feature columns, and filling counts — then reveals scikit-learn's `CountVectorizer` as the production version of the same idea.

- Normalize first: lowercase, lemmatize/stem, drop stopwords; keep exceptions like country abbreviations aside
- Features = the vocabulary (sorted); rows = sentences; cell = occurrence count
- Two variants: plain count BoW and binary BoW (presence 0/1)
- `sklearn.feature_extraction.text.CountVectorizer` reproduces the manual matrix in one call
- Honest limitation stated on the board: counts carry no semantics → sets up TF-IDF

### Natural Language Processing | TF-IDF Intuition | Text Preprocessing

URL: https://www.youtube.com/watch?v=D2V1okCEsiE · Date: 2020-05-02 (YouTube upload) · Status: summary (transcript; 507s)

The follow-up derives TF-IDF on the same toy corpus: term frequency counts how often a word appears in a sentence, inverse document frequency shrinks words that appear everywhere (log of sentences ÷ sentences containing the word), and their product gives weighted vectors where rare, locally frequent words dominate.

- TF = normalized word frequency inside one sentence
- IDF = log(total sentences / sentences containing the word); a word in every sentence gets IDF = 0
- TF-IDF entry = TF × IDF per word per sentence
- Worked case: "good" appears in all three sentences → its IDF zeroed out, exactly BoW's equal-weight weakness fixed
- Bag-of-words then TF-IDF is the canonical preprocessing ladder before embeddings

### Word Embedding – Natural Language Processing | Deep Learning

URL: https://www.youtube.com/watch?v=pO_6Jk0QtKw · Date: 2020-05-08 (YouTube upload) · Status: summary (transcript; 909s)

Krish Naik presents word embeddings as the third representation after BoW and TF-IDF, emphasizing that learned dense vectors put related words on shared axes (a gender axis for boy/girl/king/queen, an object axis for car/vehicle), then connects the concept to practice by wiring a Keras `Embedding` layer into an LSTM sentiment classifier.

- Ladder so far: BoW (counts) → TF-IDF (weighted counts) → embeddings (learned dense features)
- Related words share coordinate structure — the vectors literally encode relations like gender or category
- Nearby vectors ⇒ similar meaning; this is what semantics-in-numbers means
- Practical route in Keras: `Embedding` layer feeding an LSTM for text classification
- Names the two training families to study next: word2vec and GloVe

### Word2Vec Tutorial – The Skip-Gram Model

URL: https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/ · Date: 2016-04-19 (byline) · Status: summary

Chris McCormick's classic hands-on teardown of skip-gram: train a single-hidden-layer network on a pretext task (given a center word, predict a randomly drawn nearby word inside a window), then discard the output layer — the rows of the hidden weight matrix are the word vectors. Includes the one-hot-as-row-lookup insight and the intuition that shared contexts force similar vectors.

- Pretext task: center word in → probability over vocabulary of a randomly picked neighbor (window ≈ 5 each side)
- One-hot × weight matrix = row selection: the hidden layer is just a lookup table
- Output layer is softmax over the full vocabulary; training targets are one-hot neighbors
- Intuition: words with similar contexts must get similar vectors (synonyms, "ant"/"ants")
- Scale check: 300 features × 10k vocab = 3M weights per layer — motivates part 2's training tricks
- Note: the network learns nothing about left/right position of the neighbor

### Word2Vec Tutorial Part 2 – Negative Sampling

URL: https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/ · Date: 2017-01-11 (byline) · Status: summary

Part 2 covers the two tricks from the second word2vec paper that make training tractable and better: subsampling of frequent words (probability-of-keeping formula with `sample=0.001`) and negative sampling (each example touches only the positive output row plus a handful of negatives drawn from unigram counts raised to 3/4). It also explains phrase detection ("Boston Globe" as a single token).

- Subsampling keep-prob P(w) = (√(z(w)/0.001)+1)·0.001/z(w); only words above ~0.26% of the corpus are ever dropped
- Negative sampling updates k negatives + 1 positive instead of all ~3M output weights (~0.06% touched)
- Negatives drawn from unigram^{3/4} — lifts rare words' chance relative to plain frequency
- k = 5–20 for small corpora, 2–5 for large; the C code's 100M-entry unigram table does the sampling
- Phrases: repeated bigram passes ("New_York", then "New_York_City"); Google News phrase model reaches 3M vocab
- Both tricks improve vector quality, not just speed (paper's finding)

### GloVe Explained — Word Vectors in NLP: Global Vectors (GloVe)

URL: https://medium.com/sciforce/word-vectors-in-natural-language-processing-global-vectors-glove-51339db89639 · Date: 2018-08-14 (byline) · Status: summary

A short conceptual comparison positioning GloVe as a count-based model against word2vec's predictive one: GloVe builds a global word–word co-occurrence matrix and learns vectors by (weighted) dimensionality reduction of those counts, with a loss motivated by ratios of co-occurrence probabilities, so it inherits global statistics while keeping word2vec-style linear analogies.

- Count-based = factorize co-occurrence statistics; predictive = train a network to guess neighbors
- Co-occurrence matrix over the whole corpus is the raw material (global counts + local windows)
- Objective uses ratios of co-occurrence probabilities to encode meaning structure
- Weighted least squares keeps frequent pairs influential without letting them swamp the loss
- Ingestion note: the intranet rltoken points at a truncated URL (`...glove-51`, 404); the full id `51339db89639` above is the working destination

### FastText: Under the Hood

URL: https://medium.com/data-science/fasttext-under-the-hood-11efc57b2b3 · Date: 2018-07-07 (byline) · Status: summary

Nishan Subedi's implementation tour of fastText (Facebook's C++ library for word representations and sentence classification): how words become bags of character n-grams, how the hashing trick buckets those n-grams into a fixed table, how negative sampling/hierarchical softmax train it, and how quantization makes classification models small and CPU-fast.

- fastText does both word vectors and text classification, multiprocessing-friendly, in C++
- Word representation = word + character n-grams (marked with boundary symbols like `<` `>`)
- Hashing trick maps n-grams into a fixed-size bucket table — no giant vocabulary dictionary
- Subword composition gives meaningful vectors for out-of-vocabulary and morphologically related words
- Classifier side: bag-of-n-grams features, hierarchical softmax or negative sampling training
- Quantization compresses models dramatically with small accuracy trade-off — CPU/edge friendly

### ELMo Explained — Deep Contextualized Word Representations

URL: https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/ · Date: 2018-10 (byline: October 2018) · Status: summary

A faithful walkthrough of the ELMo paper (Peters et al., AI2): a 2-layer bidirectional LSTM language model with a residual connection and a character-CNN + highway input encoder, from which a task-specific weighted sum of layer states is frozen and concatenated to downstream models. The post highlights layer specialization (lower = syntax, higher = semantics), the SOTA sweep across six benchmarks, and the sample-efficiency gains.

- Input pipeline: character embeddings → conv + max-pool → 2-layer highway network → biLSTMs (residual between layers)
- Char-level input handles OOV and morphology; convolutions add n-gram features
- ELMo vector = γ·(s0·x + s1·h1 + s2·h2): softmax-weighted mix over layers + task scaling, learned per task
- Freeze the language model; only the mixture weights train with each downstream task
- Trained on the 1B Word Benchmark (4096-unit LSTMs, 2048 conv filters); domain fine-tuning lowers perplexity
- Results: SOTA on SQuAD/SNLI/SRL/NER/coreference/SST-5; ~98% fewer parameter updates on SRL

---

## Definitions to skim

- **Natural language processing** — computer processing of natural-language information; the CS/AI subfield spanning speech recognition, text classification, language understanding and language generation. (Wikipedia; Date: 2026-10-17 (last edited))

---

## References (link only — verified to resolve)

- Efficient Estimation of Word Representations in Vector Space (Skip-gram, 2013) — https://arxiv.org/pdf/1301.3781
- Distributed Representations of Words and Phrases and their Compositionality (Word2Vec, 2013) — https://arxiv.org/pdf/1310.4546
- GloVe: Global Vectors for Word Representation (website) — https://nlp.stanford.edu/projects/glove/
- GloVe: Global Vectors for Word Representation (2014) — https://nlp.stanford.edu/pubs/glove.pdf
- fastText (website) — https://fasttext.cc/
- Bag of Tricks for Efficient Text Classification (fastText, 2016) — https://arxiv.org/pdf/1607.01759
- Enriching Word Vectors with Subword Information (fastText, 2017) — https://arxiv.org/pdf/1607.04606
- Probabilistic FastText for Multi-Sense Word Embeddings (2018) — https://arxiv.org/pdf/1806.02901
- ELMo (website) — https://github.com/allenai/allennlp-models *(label says "website"; this is the AllenNLP model zoo — see mismatch note at top)*
- Deep contextualized word representations (ELMo, 2018) — https://arxiv.org/pdf/1802.05365
- sklearn.feature_extraction.text.CountVectorizer — https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- sklearn.feature_extraction.text.TfidfVectorizer — https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- gensim.models.word2vec — https://radimrehurek.com/gensim/models/word2vec.html
- gensim.models.fasttext — https://radimrehurek.com/gensim/models/fasttext.html
- Using Gensim Embeddings with Keras and Tensorflow — https://github.com/piskvorky/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow

---

## NotebookLM mirror

- Notebook: **ML-PT 2360 Word Embeddings** — https://notebook.google.com/notebook/cdf1ead4-de8b-432a-b412-5ececc882468 ↔ intranet project **2360**
- Status (2026-10-20): 28 sources — complete coverage of the lists above (was 27 on
  Oct 15; topped up with the GloVe Explained article as a text fallback because
  Medium blocks NotebookLM's website importer — same pattern as the FastText
  source).
- The notebook is shared publicly; transcripts and full texts are notebook-only,
  never committed to this repo.

---

## Quiz Hooks

- word embedding — a map from words to dense real vectors where semantic similarity becomes geometric closeness
- distributional hypothesis — words that share contexts tend to share meaning; the basis of all embeddings
- bag of words — count vectorization over a vocabulary; counts only, no semantics
- binary bag of words — same matrix with 0/1 presence instead of counts
- TF — term frequency: how often a word appears in one document/sentence
- IDF — inverse document frequency: log(N_docs / docs containing the word); kills ubiquitous words
- TF-IDF — TF × IDF weighting that emphasizes rare, locally frequent words
- CBOW — Continuous Bag-of-Words: predict a center word from its surrounding context
- skip-gram — predict surrounding context words from a center word (word2vec's scalable variant)
- window size — how many neighboring words count as context; large → topical, small → syntactic similarity
- negative sampling — train the positive pair plus k sampled "should-be-zero" words instead of the full softmax
- subsampling — randomly dropping frequent words during training (the `sample=0.001` rule)
- unigram^3/4 — the smoothed frequency distribution used to draw negative samples
- co-occurrence matrix — global table of how often word pairs appear together (GloVe's raw material)
- GloVe — Global Vectors: count-based embeddings from factorizing weighted co-occurrence statistics
- character n-grams — subword units (fastText: word + boundary-marked n-grams) enabling OOV embeddings
- hashing trick — bucketing n-grams into a fixed-size table so the vocabulary dictionary disappears
- quantization — compressing fastText models to run small/fast on CPUs with minimal accuracy loss
- contextual embedding — a word's vector depends on its sentence (ELMo), unlike static w2v/GloVe vectors
- ELMo — Embeddings from Language Models: task-weighted mix of a biLM's layer states (γ, s weights)
- highway network — gated layers that ease deep input transforms (used before ELMo's LSTMs)
- polysemy — one word, several senses; the weakness of static embeddings that ELMo/prob-fastText address
