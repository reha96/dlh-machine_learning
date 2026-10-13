# Fine-Tuning Large Language Models Resources

Ingested: 2026-10-13 · Source: [intranet project 3689 — Fine-Tuning Large Language Models](https://intranet-dlh.hbtn.io/projects/3689) (Novice, weight 1)

10 "Read or watch" resources get the full treatment (all Status: summary —
no paywalls, no YouTube on this project). This project page has no
"Definitions to skim" section; the 31 unique "References" are link-only
(all resolved inside the authenticated browser on 2026-10-13; direct
HEAD checks return HTTP 200 for 7, HTTP 429 for the 23 huggingface.co
pages — bot rate-limit, not dead links — and 403 for the 1 Medium
article, which rendered fully in the browser).
Label/destination mismatches are flagged in the bullets and notes.

Task map for the bullets below: 0 Load Dataset (`dair-ai/emotion`) ·
1 Load Pretrained DistilBERT (`distilbert-base-uncased`) · 2 Tokenize
(`AutoTokenizer`) · 3 Dynamic Padding (`DataCollatorWithPadding`) ·
4 Evaluation Metrics (accuracy/F1 via sklearn) · 5 `TrainingArguments` ·
6 Training Mode (`Trainer.train()`) · 7 Inference Mode
(`model.eval()` + `torch.no_grad()` / `pipeline`).

---

## What Are Large Language Models (LLMs)? — IBM Think

URL: https://www.ibm.com/think/topics/large-language-models · Date: 2021-10-06 (published) · Status: summary

IBM's topic page defines LLMs as transformer-based deep-learning models
trained on immense text that predict the next token, and walks the full
lifecycle from tokenization and self-supervised pretraining through
supervised fine-tuning, RLHF alignment and instruction tuning. Its
fine-tuning section is the direct theory behind this project's tasks:
pretraining gives broad knowledge, fine-tuning adapts a foundation
model to specific jobs like classification on a small labelled set.
Byline Cole Stryker; dated from the byline (the `Last-Modified` header
shows today's CDN date and is untrustworthy).

- Next-token engine: tokenize the prompt, embed it, generate one token
  at a time from the probability distribution until done — the model
  never "knows" the final answer in advance (tasks 6–7 framing).
- Attention math in three vectors: each embedding is projected into a
  query (what the token seeks), key (what each token contains) and
  value (what flows through, scaled by attention weight); normalized
  query–key similarities become the attention weights.
- Fine-tuning ladder: supervised fine-tuning on a small labelled set
  for new task types (summarization, classification, support) or
  domain customization (medical documents → healthcare QA); RLHF ranks
  outputs for alignment, including stylistic alignment; reasoning
  traces come from RL, not pure imitation.
- No-training alternatives named: prompt engineering, temperature /
  top-k / top-p sampling, context windows, and RAG (retrieved text fed
  into the context window, no retraining).
- Limits to quote: hallucinations (fluent but false), baked-in
  training biases, red-teaming for safety, plus benchmark dimensions
  (accuracy, efficiency, safety, fairness, robustness).

## What is Hugging Face? — IBM Think

URL: https://www.ibm.com/think/topics/hugging-face · Date: 2025-05-13 (published) · Status: summary

IBM's topic page presents Hugging Face as the company plus open-source
community behind the Transformers library and the Hub — a "GitHub for
AI" where users share models, datasets and demos. It covers exactly
the three pillars this project leans on: the Model Hub (find and
download checkpoints), the Transformers library (load any model in a
few lines, fine-tune on custom data), and the surrounding libraries
(Datasets, Evaluate, Gradio). Dated from the `article:published_time`
meta (no byline date printed).

- Hub as marketplace: thousands of pretrained models for text
  classification, generation, summarization, QA and more — download
  and fine-tune in a few lines instead of training from scratch
  (tasks 0–1).
- Simplified-workflow claim: novices can fine-tune powerful models and
  do tokenization, evaluation, distributed training and deployment
  with documented HF tools (tasks 2–6).
- History worth quoting: founded 2016 (Delangue, Chaumond, Wolf),
  Transformers library first, Hub 2020, Datasets library 2021, Gradio
  acquisition 2022, IBM–watsonx.ai collaboration 2023.
- Responsible-AI note: model cards document limitations, biases and
  intended uses — read the card before adopting a checkpoint.
- Deployment angle: serve models to web, mobile or internal systems
  without deep infra background; private repos exist for teams
  (task 7 context).

## Introduction to Hugging Face Transformers — GeeksforGeeks

URL: https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-hugging-face-transformers/ · Date: 2026-05-11 (page last updated) · Status: summary

A beginner survey of the Hugging Face stack organized as seven
components — Tokenizers, Pipeline, Datasets, the Transformers library,
Model Hub, Spaces and pretrained models — plus a click-through signup
walkthrough, application list and challenges. Thin on code but useful
as a vocabulary sheet: every class the project touches
(`AutoTokenizer`, `AutoModelForSequenceClassification`,
`DataCollatorWithPadding`, `Trainer`) lives in the components named
here. GFG "Last Updated" date used.

- Tokenizers: raw text → model-ready IDs, handling padding,
  truncation and attention masks; fast variants are Rust-backed
  (tasks 2–3).
- Pipeline: preprocessing + inference + postprocessing in one call,
  ideal for rapid prototyping — the task-7 inference shortcut.
- Datasets library: fast loading, filtering, streaming and
  memory-efficient processing that plugs into training loops
  (task 0).
- Transformers library core: PyTorch/TensorFlow/JAX support with
  BERT, GPT, T5, ViT; fine-tuning, training and inference behind
  modular reusable APIs (tasks 1, 5–6).
- Challenges quoted honestly: large models need serious compute, and
  community-shared checkpoints vary in quality — assess before use.

## How To Get Started With Hugging Face Models? — Priyanshu Agnihotri (C# Corner)

URL: https://www.c-sharpcorner.com/article/how-to-get-started-with-hugging-face-models · Date: 2023-07-27 (published) · Status: summary

A catalogue-style tour of Hugging Face model families (GPT, BERT,
RoBERTa, DistilBERT, T5, MarianMT, BART, XLNet, ALBERT, ELECTRA, plus
multimodal VisualBERT/CLIP) with the standard six-step load-and-run
recipe: `AutoTokenizer.from_pretrained` + `AutoModel.from_pretrained`,
tokenize with `return_tensors="pt"`, forward pass, read `logits`,
`softmax`/`argmax`, interpret. Its model blurbs tell you which family
fits which job — DistilBERT for small/fast classification being the
project's pick.

- Family picker: BERT for classification/NER, GPT for generation,
  RoBERTa as optimized BERT, DistilBERT as smaller/faster BERT, T5 as
  unified text-to-text (task 1 rationale).
- Canonical snippet shape: tokenizer call → `model(**inputs)` →
  `outputs.logits` → `softmax(dim=-1)` → `argmax(dim=-1)`.
- Encoder-decoder vs sequence-classification split: MarianMT/BART for
  translation/summarization vs encoder-plus-classification-head for
  sentiment/spam/intent (task 1 vs other heads).
- Limitations quoted: bias from training data, hard interpretability,
  privacy sensitivity of memorized data.

## Sequence Classification Using Hugging Face Transformers Library — Madhu Patel (C# Corner)

URL: https://www.c-sharpcorner.com/article/sequence-classification-using-hugging-face-transformers-library · Date: 2024-03-15 (published) · Status: summary

A short beginner piece defining sequence classification (one
label per input sequence: sentiment, spam, intent) and justifying
`AutoModelForSequenceClassification` as the pretrained-plus-classifier-head
shortcut that skips training from scratch. Its main value is the
AutoModel task table mapping each NLP job to its class
(`...ForSequenceClassification`, `...ForTokenClassification`,
`...ForQuestionAnswering`, `...ForCausalLM`, `...ForSeq2SeqLM`,
`...ForCTC`, `...ForImageSegmentation`) — pick the head that matches
the task.

- Why the class exists: it bundles a pretrained encoder with a
  classification head already wired for label prediction, saving the
  time and compute of training from scratch (task 1).
- Input contract: sentences in, category labels out; the model learns
  the mapping from labelled training data and predicts labels for
  unseen sentences (tasks 0, 6).
- Head-selection rule: the `AutoModelFor*` suffix must match the job —
  sequence-level labels here, token-level (NER) or span-level (QA)
  elsewhere.
- Cross-links the two companion articles (models catalogue, loading
  walkthrough) as a three-part reading order.

## How To Load A Pre-trained Model From Hugging Face? — Madhu Patel (C# Corner)

URL: https://www.c-sharpcorner.com/article/how-to-load-a-pre-trained-model-from-hugging-face · Date: 2023-07-24 (published) · Status: summary

A five-step Colab-oriented loading recipe: `pip install transformers`,
pick the checkpoint for your task, `AutoModelForSequenceClassification.
from_pretrained(model_name)`, `AutoTokenizer.from_pretrained(model_name)`,
then `model.eval()` + `torch.no_grad()` inference with
`torch.argmax(logits, dim=1)` mapped through a label dictionary. Uses
`nlptown/bert-base-multilingual-uncased-sentiment` (5-star scale) as
the worked example, including the Hub "Files and versions" download
view. Directly mirrors tasks 1, 2 and 7.

- Install-then-import: `!pip install transformers` in Colab (`!`
  passes the command to the underlying system), then import the
  `Auto*` classes (task 1 setup).
- Checkpoint choice is task-driven: match the model to the job, and
  prefer multilingual variants (e.g. bert-base-multilingual-uncased)
  for non-English text.
- Inference pattern: `tokenizer(text, truncation=True, padding=True,
  return_tensors="pt")` → `model.eval()` → `torch.no_grad()` →
  `argmax` → label map (task 7 in miniature).
- FAQ confirms the transfer-learning premise: yes, fine-tune any
  pretrained checkpoint on your own dataset with HF scripts.

## Text Classification on GLUE — Hugging Face notebooks (Colab)

URL: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb · Date: 2025-09-29 (notebook file last updated) · Status: summary

The official HF fine-tuning notebook: pick a GLUE task and a
checkpoint (`distilbert-base-uncased` is the default), load with
`load_dataset("glue", task)` / `load_metric`, tokenize via
`AutoTokenizer.from_pretrained(checkpoint, use_fast=True)` mapped over
the dataset with `dataset.map(preprocess_function, batched=True)`,
then `AutoModelForSequenceClassification` + `TrainingArguments` +
`Trainer` + `compute_metrics`, `trainer.train()`,
`trainer.evaluate()`, `push_to_hub()`. Content fetched from the raw
`huggingface/notebooks` ipynb (same material the Colab link serves);
dated from the file's last repo commit.

- GLUE map: nine tasks (CoLA, MNLI±mismatched, MRPC, QNLI, QQP, RTE,
  SST-2, STS-B, WNLI) with per-task metric (Matthews for CoLA,
  Pearson/Spearman for STS-B, accuracy elsewhere, F1 added for
  MRPC/QQP) — accuracy-vs-F1 context for task 4.
- `num_labels` rule: 3 for MNLI, 1 for STS-B (regression), 2
  everywhere else (task 1 head sizing).
- Head-swap warning: loading drops the MLM head
  (`vocab_transform`, `vocab_layer_norm`) and randomly initializes
  the classifier — expected, and exactly why fine-tuning must follow
  (tasks 1→6).
- `TrainingArguments` recipe: `eval_strategy`/`save_strategy` per
  epoch, `learning_rate=2e-5`, batch sizes, 5 epochs,
  `weight_decay=0.01`, `load_best_model_at_end=True` with
  `metric_for_best_model` (task 5 template).
- Padding insight stated plainly: the `tokenizer` passed to `Trainer`
  does final padding to uniform length per batch — the dynamic
  padding of task 3, customizable via `data_collator`.
- `compute_metrics` pattern: `np.argmax(predictions, axis=1)` (or
  squeeze for STS-B) then `metric.compute(predictions=…,
  references=…)` (task 4 wiring).
- Bonus sections: Optuna/Ray Tune `hyperparameter_search` (with the
  `model_init` + dataset-shard tips) and `trainer.push_to_hub()`
  sharing.

## Transformers and its Trainer — deepblue research (Medium)

URL: https://dbrpl.medium.com/transformers-and-its-trainer-249508929241 · Date: 2024-06-03 (published) · Status: summary

A parameter-by-parameter reference for `transformers.Trainer`:
`model`/`model_init`, `args`, `data_collator`, `train_dataset`,
`eval_dataset`, `tokenizer`, `compute_metrics`, `callbacks`,
`optimizers`, `preprocess_logits_for_metrics`, each with its default
and contract. Complements the GLUE notebook (which shows the calls)
by stating what each argument must be. Fetched via the authenticated
browser — direct requests get 403 and archive.org holds no snapshot.

- `Trainer` + `TrainingArguments` are the complete training API:
  distributed GPU/TPU training and NVIDIA `torch.amp` mixed precision
  included; `Seq2SeqTrainer`/`Seq2SeqTrainingArguments` inherit for
  summarization/translation (task 6).
- `args` default: omitted args become a basic `TrainingArguments`
  with output dir `tmp_trainer` (task 5 minimal config).
- `data_collator` default chain: `default_data_collator()` with no
  tokenizer, else `DataCollatorWithPadding` (task 3 default).
- Dataset hygiene: non-`forward()` columns auto-removed for HF
  `Dataset`s; dict eval datasets get per-key metric prefixes; sharded
  `IterableDataset`s need identical-seed generators or `set_epoch()`
  for distributed training.
- `compute_metrics` takes `EvalPrediction` (predictions + label IDs)
  and returns a string→value dict; with `batch_eval_metrics=True`
  it must also accept `compute_result` (task 4 contract).
- `optimizers` default: AdamW with a `get_linear_schedule_with_warmup`
  scheduler governed by `args` (task 5 effect).

## F1 Score in Machine Learning: A Balanced Metric for Precision and Recall — Vidhi Chugh (DataCamp)

URL: https://www.datacamp.com/tutorial/f1-score · Date: 2025-11-12 (published) · Status: summary

A tutorial arguing accuracy misleads on imbalanced data (the
always-negative diabetes classifier at 95% accuracy) and building up
the F1 family: harmonic mean of precision and recall, the
binary/multiclass split, and sklearn's `f1_score(...,
average=...)` plus `fbeta_score`. Directly serves task 4's metric
choice; fetched in full, no paywall.

- Core formula: F1 = 2·P·R/(P+R); harmonic (not arithmetic) mean so
  a lopsided precision/recall tanks the score.
- `average=` map: `binary` (positive class only), `micro` (global
  TP/FP/FN pool), `macro` (unweighted per-class mean), `weighted`
  (support-weighted mean), `samples` (multilabel per-instance).
- Fβ generalization: β>1 weights recall (F2 for diagnosis — missing
  disease beats false alarms), β<1 weights precision (F0.5 for spam
  — false positives cost more).
- Worked sklearn code: binary `f1_score` (0.8 on the toy arrays),
  multiclass macro/micro/weighted (0.65/0.67/0.64), and a diabetes
  demo contrasting accuracy 0.88 vs F1 0.8 vs F2 0.71.

## Micro, Macro & Weighted Averages of F1 Score, Clearly Explained — Kenneth Leung (Towards Data Science)

URL: https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/ · Date: 2022-01-04 (published) · Status: summary

A worked-example companion to the DataCamp piece: a 10-image
Airplane/Boat/Car classifier (Boat support = 1, deliberately
imbalanced) traced through confusion matrix → per-class TP/FP/FN →
per-class precision/recall/F1 (One-vs-Rest) → the three averages
(macro 0.58, weighted 0.64, micro 0.60). Its punchline explains a
`classification_report` quirk: micro-F1 = micro-precision =
micro-recall = accuracy for single-label multiclass, so sklearn
prints one accuracy line instead of a micro-avg row. Fetched via
archive.org (TDS blocks direct fetches).

- Macro = unweighted arithmetic mean of per-class F1; treats all
  classes equally regardless of support — the pick when every class
  matters equally on imbalanced data.
- Weighted = per-class F1 mean weighted by support share — the pick
  when big classes should count more.
- Micro = pool all TP/FP/FN first, then one F1 — equals accuracy on
  single-label data; the pick for balanced data where one overall
  number suffices.
- Selection rule of thumb: imbalanced + equal importance → macro;
  imbalanced + size matters → weighted; balanced + overall view →
  micro/accuracy.

---

## References

Link-only, per the intranet's "References" section (31 unique
destinations; 2 rltokens appear twice on the page). All resolved to
live URLs inside the authenticated browser on 2026-10-13. Direct HEAD
checks on the same day: HTTP 200 for sklearn ×3, fs.blog, the HF
homepage, weam.ai, the Colab homepage and GFG; HTTP 429 for all 23
huggingface.co docs/dataset/model pages (bot rate-limit — the pages
rendered fine in the browser); 403 for the Medium Colab-GPU article
(same — rendered fine in the browser).

| Resource | URL |
|----------|-----|
| Hugging Face Hub documentation | https://huggingface.co/docs/hub/en/index |
| Datasets | https://huggingface.co/docs/datasets/en/index |
| Datasets in the Hugging Face Hub | https://huggingface.co/docs/hub/datasets |
| Datasets: dair-ai/emotion | https://huggingface.co/datasets/dair-ai/emotion |
| DistilBERT | https://huggingface.co/docs/transformers/v5.0.0rc0/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification |
| distilbert/distilbert-base-uncased | https://huggingface.co/distilbert/distilbert-base-uncased |
| Tokenizers | https://huggingface.co/docs/transformers/en/fast_tokenizers |
| AutoTokenizer | https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer |
| AutoModelForSequenceClassification | https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification |
| DataCollatorWithPadding | https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorWithPadding |
| EvalPrediction | https://huggingface.co/docs/transformers/en/internal/trainer_utils#transformers.EvalPrediction |
| sklearn.metrics: Classification metrics | https://scikit-learn.org/stable/api/sklearn.metrics.html#classification-metrics |
| sklearn.metrics: accuracy_score | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html |
| sklearn.metrics: precision recall fscore_support | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html |
| TrainingArguments | https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments |
| Trainer | https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer |
| processing_class | https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer.processing_class |
| Sharing pretrained models | https://huggingface.co/docs/course/en/chapter4/3 |
| Pipeline | https://huggingface.co/docs/transformers/en/pipeline_tutorial |
| Text classification | https://huggingface.co/docs/transformers/en/tasks/sequence_classification |
| Fine-tuning a pretrained model: Introduction | https://huggingface.co/learn/llm-course/en/chapter3/1 |
| Fine-tuning a pretrained model: Processing the data | https://huggingface.co/learn/llm-course/en/chapter3/2 |
| Fine-tuning a pretrained model: Fine-tuning a model with the Trainer API | https://huggingface.co/learn/llm-course/en/chapter3/3 |
| Fine-tuning a pretrained model: A full training loop | https://huggingface.co/learn/llm-course/en/chapter3/4 |
| Fine-tuning a pretrained model: Understanding Learning Curves | https://huggingface.co/learn/llm-course/en/chapter3/5 |
| explain to anyone | https://fs.blog/feynman-learning-technique/ |
| Hugging Face | https://huggingface.co/ |
| Hugging Face API Guide | https://opensource.weam.ai/docs/knowledgebase/hugging-face-api |
| Google Colab | https://colab.research.google.com/ |
| How to use gpu in google colab? | https://www.geeksforgeeks.org/python/how-to-use-gpu-in-google-colab/ |
| Understanding Google Colab Free GPU in detail | https://medium.com/data-science-in-your-pocket/understanding-google-colab-free-gpu-in-detail-15074081d494 |

Reference notes: the intranet "Hugging Face API Guide" label lands on
a third-party Weam AI knowledge-base page about creating a HF API key,
not official HF documentation. "explain to anyone" lands on the
Farnam Street Feynman Learning Technique page — generic study advice,
not HF-specific. "Hugging Face Hub" (listed twice, same rltoken) and
"Hugging Face" (listed twice, same rltoken) all land on the
huggingface.co homepage rather than the Hub docs. The raw-URL label
for `dair-ai/emotion` matches its destination exactly.

---

## Ingestion Notes

- All 41 unique rltokens (10 read-or-watch + 31 references) were
  resolved inside the authenticated CDP browser (port 9333, profile
  `/tmp/opencode/chrome-dlh2`); the session stayed alive the whole
  run — project page, Medium article and all resolves returned real
  content, no re-login needed. Chrome window left open.
- Fetch split for the 10: 6 direct (2 IBM, GFG, 3 C# Corner), 1 raw
  GitHub ipynb for the Colab GLUE notebook (same file the Colab link
  serves), 1 via archive.org (TDS micro/macro), 1 via the
  authenticated browser (Medium Trainer — direct 403, no archive
  snapshot), 1 direct that needed no chain at all (DataCamp F1
  rendered fully). Zero fabricated; 0 paywalled; 0 failed-transcript
  (no YouTube on this project).
- Date labels: byline dates marked `(published)`; GFG "Last Updated"
  → `(page last updated)`; the IBM HF page dated from its
  `article:published_time` meta (2025-05-13) since it prints no byline
  date; the GLUE notebook dated from the file's last repo commit via
  the GitHub API → `(notebook file last updated)`. No `Last-Modified`
  headers used anywhere.
- huggingface.co rate-limits headless requests (HTTP 429 on all 23
  docs/dataset/model pages, and several resolve-tab titles also read
  "429 – Hugging Face" transiently) — destinations verified via the
  authenticated browser instead. Same story for the Medium Colab-GPU
  reference (direct 403, browser 200).
- Label/destination mismatches (3): "Hugging Face API Guide" →
  third-party weam.ai page; "explain to anyone" → Feynman Technique
  essay; "Hugging Face (Hub)" labels → site homepage. Flagged in the
  table notes; actual destinations summarized (or link-only for refs).
- Copyright: summaries are own-words (paragraph + bullets, no verbatim
  over one sentence); raw HTML/texts live only in `/tmp/opencode/`
  (`ft_*`), never committed.
- open-notebook mirror: NOT done — local API `http://127.0.0.1:5055`
  connection-refused at session start (same all-day outage); per
  instructions did not block and did not attempt any remote mirror.

---

## Quiz Hooks

- LLM — transformer model trained on immense text that predicts the
  next token; pretraining gives breadth, fine-tuning adapts it.
- Self-attention — query/key/value projections; normalized query–key
  similarity weights how much each value flows into each token.
- Supervised fine-tuning — small labelled set adapts a foundation
  model to new tasks (classification) or domains (medical QA).
- RLHF — humans rank outputs, model learns to prefer higher-ranked
  ones; used for alignment including style.
- Hugging Face Hub — "GitHub for AI": share/discover models, datasets,
  demos; model cards document limits and biases.
- `AutoTokenizer.from_pretrained(checkpoint)` — loads the matching
  tokenizer + pretrained vocabulary (cached after first download).
- `AutoModelForSequenceClassification` — pretrained encoder plus a
  classification head; size it with `num_labels` (2, 3 for MNLI, 1
  for STS-B regression).
- Head-swap warning — loading for classification drops MLM weights
  and randomizes the classifier: always fine-tune before inference.
- Dynamic padding — pad per batch at train time (via the
  `tokenizer`/`DataCollatorWithPadding` in `Trainer`), not the whole
  dataset upfront.
- `TrainingArguments` — output dir + eval/save strategy, lr 2e-5,
  batch size, epochs, weight decay, `load_best_model_at_end`.
- `Trainer` defaults — no `args` → output `tmp_trainer`; no
  `data_collator` + tokenizer → `DataCollatorWithPadding`; default
  optimizer AdamW with linear warmup scheduler.
- `compute_metrics(EvalPrediction)` — argmax logits → dict of
  string→value; `EvalPrediction` carries predictions + label IDs.
- F1 = 2·P·R/(P+R) — harmonic mean; lopsided precision/recall tanks
  it; accuracy misleads on imbalanced data.
- Macro — unweighted per-class F1 mean (equal classes, imbalanced
  data); weighted — support-weighted; micro — pooled TP/FP/FN, equals
  accuracy for single-label multiclass.
- Fβ — β>1 favors recall (F2 diagnosis), β<1 favors precision (F0.5
  spam filtering).
- GLUE — nine sentence/pair tasks (CoLA, MNLI, MRPC, QNLI, QQP, RTE,
  SST-2, STS-B, WNLI), each with its own metric.
- `dataset.map(fn, batched=True)` — preprocess all splits at once;
  results cached; fast tokenizers batch in parallel.
- `push_to_hub()` — upload the fine-tuned model for sharing after
  `trainer.train()` + `trainer.evaluate()`.
