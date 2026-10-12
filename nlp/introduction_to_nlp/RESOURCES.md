# Introduction to NLP Resources

Ingested: 2026-10-12 · Source: [intranet project 3603 — Introduction to NLP](https://intranet-dlh.hbtn.io/projects/3603) (Novice, weight 1)

12 "Read or watch" resources get the full treatment; the 10 "References" are
link-only (all verified to resolve with HTTP 200 on 2026-10-12). Label/destination
mismatches and broken code in the sources are flagged in the bullets.

---

## Text Preprocessing in NLP with Python Codes

URL: https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/ · Date: 2025-04-04 (page last updated) · Status: summary

A worked, end-to-end walk through an NLP cleaning pipeline applied to the SMS
spam/ham dataset, with each step shown as a short pandas-friendly function applied
via `.apply(lambda x: ...)`. It covers punctuation stripping, lowercasing,
tokenization, stopword removal, stemming and lemmatization with NLTK, then closes
with the comparison that motivates the rest of the project (BoW / TF-IDF). The
body is fully readable; the "login to continue" block only appears after the
conclusion and FAQ.

- Stopwords come from `nltk.corpus.stopwords.words('english')`; the shipped list
  starts `['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]`,
  and filtering is a plain list comprehension over tokens. The author explicitly
  warns the stock list is not gospel — a word like "how" can be a stopword for one
  model but the key signal for a customer-query project, so build your own set.
- Punctuation removal uses Python's `string.punctuation` membership test, not a
  regex.
- Caution: the article's tokenizer snippet is `re.split('W+', text)` (missing the
  backslash) — the correct idiom is `\W+`. Treat that snippet as broken rather than
  as the reference pattern.
- Stemming fingerprint (`from nltk.stem.porter import PorterStemmer`):
  `crazy→crazi`, `available→avail`, `entry→entri`, `early→earli` — real output
  that shows the non-word problem.
- Lemmatizing uses `from nltk.stem import WordNetLemmatizer` and
  `wordnet_lemmatizer.lemmatize(word)` — note the call passes **no `pos`**, so the
  article's own demo defaults to noun handling; `goose/geese → goose` both ways.
  Anything needing verb handling must pass `pos` explicitly.
- Useful framing table the author gives: preprocessing helps noise reduction,
  normalization, tokenization, stopword removal, feature extraction and
  dimensionality reduction; the article also notes lowercasing can destroy
  sentiment/emotion signal, so it is not universally desirable.

## NLTK Tokenize: Words and Sentences Tokenizer with Example

URL: https://www.guru99.com/tokenize-words-sentences-nltk.html · Date: unknown · Status: summary

A short tutorial arguing that tokenization is the base step everything else in NLP
hangs off, since stemming, lemmatization and any string-to-numeric conversion all
consume tokens. It demonstrates the two NLTK tokenizers on the identical input
string so the difference between word-level and sentence-level splits is visible
side by side. The motivating use case is deriving a numeric feature (average words
per sentence) by composing both tokenizers.

- `from nltk.tokenize import word_tokenize` on `"God is Great! I won a lottery."`
  yields `['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']` — punctuation
  becomes its own token, which matters if you intend to strip it afterwards.
- `from nltk.tokenize import sent_tokenize` on the same string yields
  `['God is Great!', 'I won a lottery ']` — note the trailing space preserved inside
  the second sentence; `.strip()` before comparing strings.
- Same text = 12 words, 2 sentences; combining both tokenizers gives the
  words-per-sentence feature. Assign the results if you need to reuse them — bare
  calls discard the output.
- Other languages need the matching Punkt model, e.g.
  `sent_tokenize(text, language='french')`; support varies by script.
- Related to the project: the FAQ contrasts these with modern LLM subword
  tokenizers (BPE / WordPiece), which split rare or compound words into pieces to
  keep the vocabulary compact.

## Stemming and Lemmatization in Python

URL: https://www.datacamp.com/tutorial/stemming-lemmatization-python · Date: unknown · Status: summary

A practical tutorial contrasting the two normalisation techniques and then
implementing both with NLTK, using a `program*` word family and a longer sentence
as running examples. Its central claim is that stemming is a cheap suffix-chopping
heuristic while lemmatization consults a lexicon and needs part-of-speech context
to pick the right dictionary form. It closes with a decision table on when each is
worth its cost.

- Stemming: `from nltk.stem import PorterStemmer`, `ps = PorterStemmer()`,
  `ps.stem(word)` — no corpus download needed. Overstemming collapses
  `universal/university/universe` to one stem; understemming leaves
  `alumnus/alumnae/alumni` distinct.
- Lemmatizing needs downloads first: `nltk.download("wordnet")` and
  `nltk.download("omw-1.4")`.
- The POS tag is mandatory in practice: `wnl.lemmatize(word, pos="v")`, with options
  `"n"`, `"v"`, `"a"`, `"r"`. The default is noun, so `lemmatize("running")` returns
  `"running"`; `pos="v"` is what returns `"run"`. For that reason `"programmer"`
  stays `"programmer"` when you ask for verbs.
- If a word is absent from WordNet, or the wrong `pos` is supplied, the word comes
  back unchanged — an unchanged output is a signal to check the POS, not that the
  lemmatizer failed.
- Tokenization in the examples:
  `word_tokenize(sentence.translate(str.maketrans("", "", string.punctuation)))` —
  strip punctuation *before* tokenizing. NLTK 3.9+ wants
  `nltk.download("punkt_tab")` (the old `punkt` id is deprecated).
- Comparison table: stemming is fast/lower-accuracy/context-free and suits search
  engines and spam filters; lemmatization is slow, accurate, context-aware, and
  suits chatbots, sentiment analysis and QA.

## Understanding Word N-grams and N-gram Probability in Natural Language Processing

URL: https://towardsdatascience.com/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing-9d9eef0fa058/ · Date: 2019-11-27 (published) · Status: summary

Note: the URL the intranet link lands on is a shorter/older form that now 404s;
the live article is the different-ID permalink above (Sunny Srinidhi, 5 min read).
It defines an n-gram as a run of n consecutive words, then shows how a bigram
model scores a next-word prediction from corpus counts. It is prose plus arithmetic
with no Python, so the nltk implementation (`nltk.ngrams(tokens, n)`) is left
entirely to the student.

- Naming convention: a two-word run is a 2-gram / bigram ("Medium blog"), three
  words a 3-gram / trigram ("Write on Medium"), four words a 4-gram ("A Medium
  blog post"). `n` is just the window length.
- Conditional-probability formula for a bigram: `P(w1 | w2) = count(w2 w1) / count(w2)`
  — numerator is how often the pair appears, denominator how often the left word
  appears at all in the corpus.
- Worked numbers on a five-sentence corpus: `really` occurs 3 times, and each of
  `really like`, `really appreciate`, `really sorry` occurs once, so all three
  continuations score 1/3 ≈ 0.33. `Thank you for inviting you` is the only instance
  of that bigram, so `you` gets probability 1.0.
- Bigger corpus and larger n both sharpen predictions; a trigram is what lets a bot
  separate "what's the temperature?" from "set the temperature".
- Motivating uses: sentence auto-completion, spell checking, and rough grammar
  checking.
- No smoothing or Laplace correction is introduced here — worth adding yourself if
  a context has zero counts, which otherwise yields division by zero or
  probability 0.

## A Gentle Introduction to the Bag-of-Words Model

URL: https://machinelearningmastery.com/gentle-introduction-bag-words-model/ · Date: 2019-08-07 (published) · Status: summary

A conceptual (no-code) tutorial on turning documents into fixed-length numeric
vectors for machine learning. It builds a worked example from four lines of *A Tale
of Two Cities*, treating each line as a document, and then walks through vocabulary
design, scoring methods and the model's shortcomings. The two design decisions the
student must actually implement — which words form the vocabulary and how each word
is scored — are what the article is organised around.

- BoW needs exactly two things: a vocabulary of known words and a measure of each
  word's presence. Word order and sentence structure are discarded entirely.
- Worked fingerprint: 4 lines / 24 tokens collapse to a 10-word vocabulary
  (`it was the best of times worst age wisdom foolishness`), and "It was the best of
  times" becomes the binary vector `[1,1,1,1,1,1,0,0,0,0]`; unknown words in a new
  document are simply ignored rather than erroring.
- Three scoring schemes in ascending richness: binary presence (0/1), raw counts,
  and normalised frequencies (count divided by document length). sklearn's
  `CountVectorizer` is the canonical count-vectorizer to reproduce.
- Vocabulary-size pressure produces sparse vectors; the "hash trick" / feature
  hashing maps words deterministically into a fixed-size index space to cap
  dimensionality, trading some collision risk for bounded memory.
- N-gram grouping is presented as a vocabulary upgrade: a bigram vocabulary tracks
  pairs like "it was", "was the", "the best" and a bag-of-bigrams is called markedly
  more powerful than a bag of single words for document classification.
- Stated limitations to be aware of: vocabulary design burden, sparsity (costly in
  memory and time), and total loss of semantics — the model cannot distinguish word
  order or synonyms such as "old bike" vs "used bike".

## A Guide to TF-IDF

URL: https://builtin.com/articles/tf-idf · Date: 2024-02-08 (published) · Status: summary

A from-scratch Python implementation of TF-IDF in plain dictionaries and pandas,
followed by the one-line scikit-learn equivalent. It defines the four symbols (term,
document, corpus size N, corpus), derives term frequency then document frequency
then inverse document frequency, and explains why the smoothed log form is the one
to implement. Two short sample sentences about data science serve as the worked
example.

- Exact formulas as printed: `tf(t,d) = count of t in d / number of words in d`;
  `df(t)` = number of documents containing `t` at least once (never the number of
  occurrences); plain `idf(t) = N/df`.
- The smoothed variant is `idf(t) = log(N/(df + 1))`, and the final score is
  `tf-idf(t,d) = tf(t,d) * log(N/(df + 1))`. The `+1` on the denominator exists
  because an out-of-vocabulary term has `df = 0` and you cannot divide by zero; the
  log keeps idf from exploding on huge corpora.
- Their code uses `math.log10(...)` and `count/float(corpusCount)`; `computeIDF` as
  printed initialises every idf to 0 and never accumulates document frequency, so it
  returns the same value for all words — implement the df accumulation yourself.
- Stopword handling: `nltk.download('stopwords')`, `from nltk.corpus import stopwords`,
  `stop_words = set(stopwords.words('english'))`, then a list comprehension over the
  words to drop.
- Case matters before counting: the example keeps both `Data` and `data` as separate
  vocabulary entries, which the author fixes only when moving to sklearn by
  lowercasing every sentence first.
- One-liner equivalent: `from sklearn.feature_extraction.text import TfidfVectorizer`
  then `vectorize.fit_transform([firstV, secondV])` — sklearn applies the smoothed
  idf by default.

## The Illustrated Word2vec

URL: https://jalammar.github.io/illustrated-word2vec/ · Date: 2019-03-27 (published) · Status: summary

Jay Alammar builds word2vec from first principles: an embedding is just a vector
you can compare with cosine similarity, and the vector for a word is learned as a
side effect of training a next-word predictor on running text. The post slides a
fixed-size window across a corpus to generate (context → target) pairs, then flips
the direction twice — CBOW predicts the middle word from its neighbors, skip-gram
predicts the neighbors from the middle word, yielding four samples per window
position. The final piece is negative sampling, which reframes the task as binary
"are these two words neighbors?" and turns the network into a fast logistic
regression instead of a full softmax over the vocabulary.

- Training data is manufactured on the fly: slide a window over text, emit one
  sample per window position; CBOW uses both sides of the window as features with
  the center as label.
- Skip-gram inverts it — the center word is the input and every neighbor inside
  the window becomes a separate (input, output) training sample, so one window
  position produces four pairs.
- Negative sampling replaces "predict the next word" with a 1/0 label (neighbor /
  not neighbor), which is why the third step of the neural LM (softmax over the
  whole vocabulary) is dropped; without negatives a trivial "always 1" model wins
  with 100% accuracy and learns nothing.
- SGNS mechanics: two matrices, `Embedding` and `Context`, both
  `vocab_size × embedding_size`; take dot products of the input embedding with
  each context embedding, pass through sigmoid, and compute
  `error = target - sigmoid_scores` to nudge only the four words involved.
- Hyperparameters: small windows (2–15) make similar words *interchangeable*;
  large windows (15–50) make them *related*; the original paper suggests 5–20
  negatives (2–5 with big data), gensim defaults to window 5 and 5 negatives.
- Analogy arithmetic works as a side effect of the vector space:
  `king - man + woman` lands nearest to `queen`, though not exactly on it.

## What Is FastText? Compared To Word2Vec & GloVe [How To Tutorial In Python]

URL: https://spotintelligence.com/2023/12/05/fasttext/ · Date: 2023-12-05 (published) · Status: summary

Neri Van Otten explains fastText as Facebook AI Research's 2016 answer to the two
weaknesses of word-level embedding: out-of-vocabulary words and heavy morphology.
Instead of learning one vector per word type, fastText decomposes a word into
overlapping character n-grams and learns a vector for each fragment, then sums
them to build the word's representation — so an unseen word can still be embedded
from its pieces. The rest of the post is a side-by-side comparison table against
word2vec and GloVe plus runnable Python snippets for supervised classification and
unsupervised embeddings.

- Subword mechanism: "apple" is split into overlapping character n-grams like
  `ap`, `pp`, `pl`, `le`; a word's embedding is composed from these fragments,
  which is why unseen or rare words still get a usable vector.
- fastText keeps both word2vec training architectures — skip-gram (center word →
  context) and CBOW (context → center) — and adds Hierarchical Softmax (a
  tree-structured output layer) and Negative Sampling for speed on huge
  vocabularies.
- The four head-to-head differences vs word2vec: OOV handling (word2vec simply
  fails on unseen words), representation (subword/morphological vs word-only),
  training speed, and best-fit use cases (language ID, sentiment, morphology vs
  plain semantic similarity).
- Supervised API fingerprint: data lines formatted `__label__<class_name> <text>`,
  then `model = fasttext.train_supervised(input="train.txt", epoch=25, lr=1.0)`
  and `model.predict(text)` returns the label tuple.
- Unsupervised API fingerprint:
  `model = fasttext.train_unsupervised('corpus.txt', model='skipgram')` then
  `model.get_word_vector('example')` returns the embedding.
- Caveat for the study notes: the article names no explicit character-n-gram range
  arguments (`minn`/`maxn`) — the only hyperparameters it shows are `epoch` and
  `lr` — and it lists weak contextual understanding versus BERT as fastText's main
  limitation.

## Bernoulli Naive Bayes, Explained: A Visual Guide with Code Examples for Beginners

URL: https://towardsdatascience.com/bernoulli-naive-bayes-explained-a-visual-guide-with-code-examples-for-beginners-aec39771ddd6 · Date: 2024-08-23 (published) · Status: summary

Samy Baladram introduces the Naive Bayes family as classifiers that multiply
together per-feature probabilities to reach a class decision, with "naive" naming
the assumption that all features are independent even when they aren't. He focuses
on the Bernoulli variant, whose defining assumption is that every feature is
binary, and walks through the whole thing on the classic 14-row golf dataset:
convert Outlook to one-hot, bin Temperature at 80°F and Humidity at 75%, then
compute class priors and per-feature conditional probabilities, optionally smooth,
and multiply. The post also situates Bernoulli against the other two variants —
Multinomial for discrete counts (the text-classification workhorse) and Gaussian
for continuous, normally distributed features.

- Mechanism in four steps: class probability = instances of that class / total;
  per-feature conditional P(feature = 0 or 1 | class); for a new row multiply the
  class prior by every feature conditional; predict the class with the largest
  product.
- Bernoulli assumes binary/boolean features only — in the golf example every
  continuous column is bucketed into a ≤/> threshold before it can be used, which
  is what makes it applicable to presence/absence text features.
- The three variants differ purely by feature-distribution assumption: Bernoulli =
  binary, Multinomial = discrete counts (word counts, used for text), Gaussian =
  continuous with a normal distribution.
- Smoothing (Laplace / `alpha`): add 1 to every numerator and 2 to every
  denominator, which keeps the probabilities summing to 1 while preventing a zero
  frequency from zeroing out the whole product.
- scikit-learn fingerprints: `BernoulliNB(alpha=1)` with `.fit(X_train, y_train)`
  and `.predict(X_test)`; `binarize` is a threshold parameter (values above it
  become 1) and `fit_prior=True` learns class priors instead of assuming a 50/50
  split.
- Stated trade-offs: fast, strong on small/high-dimensional data, good baseline —
  but the independence assumption rarely holds, zero-frequency problems bite
  without smoothing, and results are sensitive to how you choose the binarization
  threshold.

## Performance Metrics: Confusion matrix, Precision, Recall, and F1 Score

URL: https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262 · Date: 2020-09-14 (published) · Status: summary

Vaibhav Jayaswal argues that accuracy is misleading on imbalanced data and that
the 2×2 confusion matrix plus precision/recall/F1 tell you much more. He works
through a 100-patient tumor-detection example to name TP, TN, FP (Type I error) and
FN (Type II error), then derives TPR/FPR/TNR/FNR from them. The payoff is a
decision rule for which metric to prioritise: recall for fraud detection and
medicine, precision for spam filtering, and F1 when you cannot choose.

- Confusion matrix for binary classification: actual labels on one axis,
  predictions on the other; FP = predicted positive but actually negative (Type
  I), FN = predicted negative but actually positive (Type II).
- Precision asks what fraction of predicted positives were real; recall asks what
  fraction of real positives were found and is identical to TPR. Both live in
  [0, 1] and ignore TN entirely.
- Choose recall when a miss is costly — credit-card fraud, missing a tumour —
  accepting lower precision; choose precision when a false alarm is costly, like
  quarantining a legitimate email as spam, accepting missed spam.
- F1 is the harmonic mean of precision and recall, so it penalises both FP and FN
  and behaves sanely under class imbalance; F-beta generalises it, where beta = 2
  means recall counts twice as much as precision.
- Healthy model means TPR and TNR high while FPR and FNR stay near zero.
- This piece is binary-only; multi-class averaging (micro / macro / weighted F1) is
  not covered here — the author links a companion TDS article, "Micro, Macro &
  Weighted Averages of F1 Score, Clearly Explained", as the follow-up reading.

## Text Classification Using NLTK

URL: https://www.geeksforgeeks.org/nlp/text-classification-using-nltk/ · Date: 2025-07-23 (page last updated) · Status: summary

This GeeksforGeeks walkthrough builds an emotion classifier end to end on a small
labelled CSV: load with pandas, clean the text with NLTK's tokenizer and English
stopword list, vectorise, split, fit, and evaluate. Despite the title, the
modelling and vectorisation are done with scikit-learn — NLTK supplies only the
tokenization and stopword removal — which makes it a useful contrast to the
pure-NLTK classifier API (`nltk.NaiveBayesClassifier`,
`feature_extraction.text.BagOfWordsExtractor`, `classify`/`accuracy`) that appears
in the DLH tasks. Note one bug carried in the published snippet: `text = text.lower`
is missing its call parentheses.

- Data prep fingerprint: `nltk.download('punkt')`, `nltk.download('stopwords')`,
  `nltk.download('punkt_tab')`, then `stop_words = set(stopwords.words('english'))`.
- Preprocessing fingerprint: lowercase, `word_tokenize(text)`, keep only
  `word.isalpha() and word not in stop_words`, `" ".join(tokens)` into a
  `clean_text` column.
- Features and model: `TfidfVectorizer().fit_transform(df['clean_text'])` → sparse
  matrix `X`, labels `y`; split with
  `train_test_split(X, y, test_size=0.2, random_state=42)`; `MultinomialNB().fit(X_train, y_train)`.
- Evaluation: `accuracy_score(y_test, y_pred)` for the headline number and
  `classification_report(y_test, y_pred)` for per-class precision, recall and F1;
  prediction reuses `vectorizer.transform([clean])` then `model.predict(vector)[0]`.
- MultinomialNB is the right pick here because TF-IDF values are non-negative
  counts/weights — BernoulliNB expects binary indicators and GaussianNB expects
  continuous normal-ish features.
- Gotcha in the article's own code: `text = text.lower` assigns the bound method
  instead of calling it, so `word_tokenize` receives a method object — write
  `text.lower()` in your own implementation.

## Word Embedding and Word2Vec, Clearly Explained!!!

URL: https://www.youtube.com/watch?v=viZrOnJclY0 · Date: 2023-03-13 (YouTube upload) · Status: summary

Josh Starmer trains a toy next-word-prediction network on four words from two
sentences and then reads the input-to-hidden weights off as the embeddings,
showing that words appearing in the same context drift closer together purely as
a side effect of backpropagation. The punchline is that random numbers per word
waste the learnable structure, whereas embeddings let knowledge about one word
transfer to its synonyms. The second half contrasts word2vec's two
context-widening strategies and quantifies exactly how much compute negative
sampling saves on a Wikipedia-scale vocabulary.

- Architecture: one input per unique word, wired to *N* identity activation
  functions (N = embedding dimension, 100+ in practice); the weights from input to
  those activation functions are the embeddings, and the second layer of weights
  maps activations to vocabulary outputs passed through softmax with cross-entropy
  loss.
- Target construction: set one input to 1 and the rest to 0, then require the
  correct next word's output to be 1; after training on `Troll 2 is great!` and
  `Gymkata is great!`, the words sharing context converge while the next-word
  predictions become exact.
- CBOW uses the surrounding words to predict the middle word (`Troll 2` + `great!`
  → `is`); skip-gram uses the middle word to predict the surroundings
  (`is` → `Troll 2`, `great!`, `Gymkata`).
- Cost arithmetic: ~3M vocabulary words × 100 embedding weights × 2 layers ≈ 600M
  weights to optimize; negative sampling picks a small random subset of words to
  *not* predict (2–20 in practice), so one step only touches ~300 weights instead
  of the full set.
- Why multiple numbers per word: a word like "great" has positive and sarcastic
  uses, and separate embedding dimensions let the network encode both senses
  independently.
- Payoff: similar words with similar embeddings means learning one word's usage
  transfers to its neighbors, cutting the data and model complexity needed to learn
  a language task.

---

## References

Link-only, per the intranet's "References" section. All verified to return
HTTP 200 on 2026-10-12.

| Resource | URL |
|----------|-----|
| NLTK Documentation | https://www.nltk.org/ |
| Emoji | https://carpedm20.github.io/emoji/docs/index.html |
| scikit-learn Text Feature Extraction | https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction |
| Vectorization & Word Embeddings - Gensim (GeeksforGeeks tutorial) | https://www.geeksforgeeks.org/nlp/nlp-gensim-tutorial/ |
| Gensim Word2Vec | https://radimrehurek.com/gensim/models/word2vec.html |
| Gensim FastText | https://radimrehurek.com/gensim/models/fasttext.html |
| WordCloud Library | https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html |
| scikit-learn Naïve Bayes | https://scikit-learn.org/stable/modules/naive_bayes.html |
| scikit-learn Metrics | https://scikit-learn.org/stable/modules/model_evaluation.html |
| Regular Expressions in Python | https://docs.python.org/3/library/re.html |

Label mismatch to note: "Vectorization & Woed Embeddings - Gensim" (intranet
label, typo included) is actually a GeeksforGeeks gensim tutorial, not gensim's
own documentation.

---

## Ingestion Notes

- All 22 rltoken links (12 read-or-watch + 10 references) were resolved inside the
  authenticated browser.
- Two dates are genuinely unknown and were **not** guessed: the Guru99 tokenization
  page publishes no byline date (its only timestamp is a `dateModified` of
  2026-10-10, i.e. cache regeneration, which the ingestion rules forbid trusting),
  and the DataCamp page renders client-side with no machine-readable date.
- Two of the sources ship broken code, flagged above: the Analytics Vidhya regex
  `re.split('W+', text)` (missing backslash) and the GeeksforGeeks
  `text = text.lower` (missing call parentheses). Neither is a reference pattern.
- Two GeeksforGeeks pages print only a "Last Updated" line, so they are labeled
  `(page last updated)` rather than `(published)`.
- The n-grams link resolves to a TDS permalink with a different article ID than the
  intranet's target; the live permalink is used above.

---

## Quiz Hooks

- Tokenization — the base step: `word_tokenize` splits words (punctuation becomes its own token), `sent_tokenize` splits sentences (trailing whitespace survives).
- Stopwords — `nltk.corpus.stopwords.words('english')`; the shipped list is a starting point, not ground truth — "how" can carry the signal in a support-bot corpus.
- Stemming vs lemmatization — stemming chops suffixes heuristically (`crazy→crazi`); lemmatization consults WordNet and needs `pos`, because `lemmatize("running")` returns "running" unless you pass `pos="v"`.
- Over/understemming — overstemming collapses `universal/university/universe`; understemming leaves `alumnus/alumnae/alumni` apart.
- n-gram — a run of n consecutive words; bigram conditional probability `P(w1|w2) = count(w2w1)/count(w2)`.
- BoW — vocabulary + presence measure; order and semantics are discarded, so "old bike" and "used bike" become identical vectors.
- BoW scoring — binary presence, raw counts, or normalised frequency (`CountVectorizer`); feature hashing caps dimensionality at the cost of collisions.
- TF-IDF — `tf(t,d) = count(t,d)/len(d)`, `df(t)` counts *documents* containing t, and the smoothed form is `tf * log(N/(df+1))`.
- Why the `+1` — an unseen term has `df = 0`, so the denominator would divide by zero; the log stops idf exploding on large corpora.
- Word2Vec — embeddings are a side effect of training a next-word predictor; CBOW predicts the centre from neighbours, skip-gram the neighbours from the centre.
- Negative sampling — reframes training as "are these two words neighbours?", turning a vocabulary softmax into logistic regression (5–20 negatives, 2–5 on big data).
- Word window size — small windows (2–15) capture interchangeable synonyms; large windows (15–50) capture topical relatedness.
- FastText — decomposes words into overlapping character n-grams and sums their vectors, so OOV and morphologically rich words still get embeddings.
- Naive Bayes — multiply the class prior by every per-feature conditional; "naive" is the (false) feature-independence assumption.
- Bernoulli vs Multinomial NB — Bernoulli needs binary features (presence/absence); Multinomial takes word counts and is the text-classification default; Laplace smoothing adds 1 to numerators to avoid zero-frequency annihilation.
- Precision vs recall — precision = of what you flagged, how much was real; recall = of what was real, how much you found; pick recall when a miss is costly, precision when a false alarm is costly.
- F1 — the harmonic mean of precision and recall, so it punishes both FP and FN and survives class imbalance; F-beta weights recall at beta = 2.
