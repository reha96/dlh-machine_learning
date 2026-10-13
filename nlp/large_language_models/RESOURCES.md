# Large Language Models (LLMs) Resources

Ingested: 2026-10-13 · Source: [intranet project 3601 — Large Language Models (LLMs)](https://intranet-dlh.hbtn.io/projects/3601) (Novice, weight 1)

21 "Read or watch" resources get the full treatment (all Status: summary —
no paywalls, no YouTube on this project). This project page has no
"Definitions to skim" section; instead it has a "More" section with 6 items,
treated link-only like the 18 "References" (25-resource cap: 21 full +
24 link-only). All 45 rltoken links were resolved inside the authenticated
browser; label/destination mismatches are flagged in the bullets and notes.

Task map for the bullets below: 0 Load RoBERTa (`RobertaForMaskedLM`) ·
1 Tokenization · 2 Mask indices · 3 Mask logits · 4 Decode mask predictions ·
5 Fill mask (encoder-only) · 6 Translation (encoder-decoder, Helsinki-NLP) ·
7 Text generation (decoder-only, GPT-2/DistilGPT2) · 8 Vision tasks (ViT) ·
9 VLM image captioning (BLIP).

---

## What Are Large Language Models (LLMs)? — IBM Think

URL: https://www.ibm.com/think/topics/large-language-models · Date: 2021-10-06 (published) · Status: summary

IBM's topic page defines LLMs as deep-learning models trained on immense
text that predict the next word, built on the transformer with its
self-attention mechanism, and walks the full lifecycle from tokenization and
self-supervised pretraining through supervised fine-tuning and RLHF
alignment. It closes on choosing foundation models plus the standard
limitations (hallucinations, bias, red-teaming for safety). The visible
byline (Cole Stryker, Oct 2021) predates the page metadata update (2024),
so the byline date is used above.

- Tokenization splits text into words, subwords or characters so rare and
  novel words are handled consistently; each token maps to a numeric vector
  (embedding) enriched layer by layer, plus positional encodings for order.
- Attention math in three vectors: every embedding is projected into a
  query (what the token seeks), key (what each token contains) and value
  (what flows through, scaled by attention weight); query–key similarity
  gives alignment scores, normalized into the attention weights.
- Fine-tuning ladder: supervised fine-tuning on a small labelled set adapts
  a foundation model to summarization/classification/support; RLHF (humans
  rank outputs, model learns to prefer higher-ranked ones) is used for
  alignment, including stylistic alignment (casual, humorous, on-brand).
- Reasoning traces: fine-tuned models that break problems into smaller
  steps before the final output, trained with increasingly sophisticated
  RL rather than pure imitation.
- Limits to quote: hallucinations (fluent but false output), baked-in
  training biases, and safety evaluation via red-teaming — evaluators
  deliberately provoking unsafe/biased responses to expose weaknesses.

## Roadmap to Learning Large Language Models (LLMs) for Beginners — Lekha Priya (Medium)

URL: https://lekha-bhan88.medium.com/roadmap-to-learning-large-language-models-llms-for-beginners-945ec8dd16e7 · Date: 2024-08-13 (published) · Status: summary

A beginner study plan: prerequisites first (maths, Python, ML basics with
linked courses), then a four-step ladder from AI/NLP basics through ML/DL
into LLM architecture and finally fine-tuning plus deployment. Its value
here is the ordered reading/project list, not any single technical claim —
treat it as a syllabus pointer toward the books and courses it names
(Speech and Language Processing, AI: A Guide for Thinking Humans).

- Prereqs named explicitly: linear algebra, calculus, probability; Python
  with NumPy/pandas/Matplotlib; supervised vs unsupervised learning and
  neural networks before touching LLMs.
- Framework pair: PyTorch (ease of use, flexibility) vs TensorFlow
  (performance, scalability), plus Hugging Face Transformers as the
  essential library for pre-trained GPT/BERT-style models.
- Step order: 1) AI/NLP basics → 2) ML/DL algorithms and training →
  3) LLM architectures (GPT, BERT, T5) and how they train →
  4) fine-tuning on a custom set (product reviews, news) and deploying
  behind a web interface or API (Streamlit named).
- First projects kept small: text generation or simple summarization before
  attempting fine-tuning; community upkeep via papers, r/MachineLearning,
  newsletters.

## Large Language Models Explained — NVIDIA Glossary

URL: https://www.nvidia.com/en-us/glossary/large-language-models/ · Date: 2026-10-11 (page last modified) · Status: summary

NVIDIA's glossary entry explains why the transformer won: positional
encoding lets words feed in non-sequentially and self-attention weights
each input part by importance, so the problem decomposes into parallel
GPU-friendly computations over internet-scale unlabeled data. Short and
hardware-flavoured — the parallelism argument is its distinctive point
versus the IBM/AWS versions.

- Transformer = stacks of blocks (self-attention layers, feed-forward
  layers, normalization layers) that decipher input to predict output
  streams at inference; Google's 2017 "Attention Is All You Need".
- Positional encoding embeds token order so words need not feed
  sequentially; self-attention learns, over mountains of data, which parts
  of the input actually matter for each prediction.
- Scale vocabulary: hundreds of billions of parameters on internet-scale
  datasets; GPUs suit the decomposed parallel sub-computations, which is
  what made training at that scale feasible.
- Framing contrast: earlier AI focused on perception/understanding, while
  LLMs unlocked open-ended generation from the same perception base.

## What is LLM (Large Language Model)? — AWS

URL: https://aws.amazon.com/what-is/large-language-model/ · Date: 2026-10-08 (page last modified) · Status: summary

AWS's explainer centers the encoder–decoder transformer with self-attention
and the self-learning pretraining story (grammar, languages and knowledge
absorbed without labels), then sells flexibility: one model answering
questions, summarizing, translating and completing sentences from a few
prompts. Its concrete anchors are the named model sizes and the Common
Crawl / Wikipedia corpus figures.

- Architecture stated as encoder plus decoder with self-attention: the
  pair extracts meanings from a text sequence and the relations between
  words and phrases; training is self-learning on unlabeled data.
- RNN contrast: unlike recurrent nets that process inputs sequentially,
  transformers process whole sequences in parallel, so GPUs cut training
  time dramatically.
- Corpus scale cited: Common Crawl (50B+ web pages), Wikipedia (~57M
  pages); models routinely reach hundreds of billions of parameters.
- Size anchors: GPT-3 at 175B parameters; Claude 2 taking ~100K-token
  inputs. One model, many tasks (Q&A, summarization, translation,
  completion) from relatively few prompt examples.

## What is Hugging Face? — IBM Think

URL: https://www.ibm.com/think/topics/hugging-face · Date: 2025-05-13 (published) · Status: summary

IBM profiles Hugging Face as company plus open-source community: the
Transformers library and the Model Hub marketplace where users find,
download and fine-tune thousands of pre-trained models in a few lines of
code instead of training from scratch. Covers the full loop the project
needs — hub access, simple fine-tuning/distributed/tokenization/eval
workflows, deployment to web/mobile/internal systems, and the community
flywheel of daily model/dataset/tutorial contributions.

- Model Hub behaves like a marketplace: filter by task, download and
  fine-tune (speech recognition, text classification/generation,
  summarization, QA, image generation).
- Workflow claim: novices can fine-tune powerful models and run
  distributed training, tokenization, evaluation and deployment with
  documented, user-friendly libraries.
- Deployment without deep infra background: serve to web, mobile apps or
  internal systems — the startup/enterprise pitch.
- Responsible-AI note: the platform's community norms and tooling are
  presented as part of the offering, matching the ethics learning
  objective of the project.

## Mastering Masked Language Models — Atharv Yeolekar (Medium)

URL: https://medium.com/@atharv6f_47401/mastering-masked-language-models-techniques-comparisons-and-best-practices-33cf061a1693 · Date: 2024-06-24 (published) · Status: summary

A survey of the MLM family tree from BERT's random masking through
SpanBERT's span masking and RoBERTa's dynamic masking to ELECTRA's token
replacement and XLNet's permutation objective, with a comparison table of
objectives, masking strategies and compute trade-offs. Directly relevant to
tasks 0–5: it explains why RoBERTa drops next-sentence prediction and why
whole-word masking matters.

- Core MLM loop: hide (mask) words in a sentence, predict them from both
  left and right context — bidirectional, unlike next-word prediction;
  self-supervised, so vast unlabeled corpora work without manual labels.
- Technique evolution: BERT static random masking → SpanBERT span masking
  (with span-boundary objective) → RoBERTa dynamic masking (mask pattern
  changes across epochs) → ELECTRA replaced-token detection → XLNet
  permutation language modeling.
- Whole-word masking keeps subwords of one word together when masking,
  which helps compound words and whole-word understanding tasks.
- Pretrain–finetune discrepancy: BERT's `[MASK]` never appears at
  fine-tuning time; ELECTRA and XLNet address this, which the article
  credits for part of their transfer-learning gains.
- Cost/complexity trade-off stated: most variants match or beat BERT's
  efficiency except XLNet (computationally expensive); complexity rises
  from BERT toward ELECTRA/XLNet, affecting ease of deployment.

## Masked Language Modeling — ML Recipe Book (bipinkrishnan.github.io)

URL: https://bipinkrishnan.github.io/ml-recipe-book/masked_language_modeling.html · Date: unknown · Status: summary

A code-first recipe that frames MLM as domain adaptation (the MBBS
analogy: pretrain on medical text so a classifier does better on medical
tasks), then implements it on the XSum news dataset: drop summaries, chunk
documents to 128-token blocks, mask, and train with a PyTorch loop ending
in a `logits.argmax(dim=-1)` decode. No date is printed anywhere on the
page, hence unknown. Closest thing on this list to the task 0–4 pipeline.

- "Fill in the blanks" framing: input has `[MASK]` tokens, label is the
  complete sentence; predicting the masks teaches word relationships.
- Data prep fingerprint: `load_dataset('xsum')` (204K train rows,
  document/summary/id), drop `summary`, concatenate documents and split
  into chunks of 128 (max model length 512; truncation avoided by
  chunking, last short chunk dropped).
- Tokenizer: `AutoTokenizer` with `distilbert-base-uncased`.
- Inference fingerprint: tokenize with `return_tensors='pt'`, find mask
  with `tokenizer.mask_token_id` + `torch.where(input_ids ==
  mask_token_id)[1]`, decode `logits.argmax(dim=-1)` at that position,
  `tokenizer.decode(...)` the winner and string-replace `[MASK]`.
- Demo pair: "The goal of life is [MASK]." → "happiness".

## Getting Started with Transformers and Pipelines — Rohan Kotwani (Medium)

URL: https://medium.com/intro-zero/getting-started-with-transformers-pipelines-and-the-hugging-face-model-hub-4bd743c3f0eb · Date: 2024-11-15 (published) · Status: summary

A companion to the Hugging Face video course: transfer learning and the
transformer architecture up front, then the `pipeline` as the highest-level
API bundling preprocessing (text→numbers), model and postprocessing
(predictions→readable labels) in one call. Walks sentiment analysis,
zero-shot classification (87% education example), text generation with
`distilgpt2`, and fill-mask as BERT's pretraining objective. Visible byline
Nov 15, 2024 is used; page metadata was edited later (2025).

- Pipeline = preprocessing + model + postprocessing: models expect
  numbers, not raw text; the pipeline also re-humanizes outputs.
- Sentiment fingerprint: `pipeline("sentiment-analysis")` on a batch
  returns an ordered list of `{label, score}` dicts, one per input.
- Zero-shot: pass `candidate_labels` (education/politics/business) and get
  the winning label with confidence — no task-specific training.
- Generation: any causal-LM model works; `distilgpt2` demo with
  `max_length` and `num_return_sequences` (outputs vary run to run).
- Fill-mask named as BERT's pretraining objective: guess the masked word,
  the encoder-side mirror of generation (tasks 5 vs 7).

## A Hands-On Guide to Hugging Face Pipelines for Machine Learning and NLP — faizan@123 (Medium)

URL: https://medium.com/@faizanfaizy7/a-hands-on-guide-to-hugging-face-pipelines-for-machine-learning-and-nlp-bbb2e8147fdd · Date: 2024-09-16 (published) · Status: summary

A snippet-heavy tutorial: `pipeline("task-name")` for text tasks,
`pipeline(model="...")` to pin a hub model, `return_all_scores=True` for
full distributions, plus audio (`automatic-speech-recognition`) and vision
(`image-classification`, `object-detection`) examples. Its exact model
strings are the takeaway: `FacebookAI/roberta-large-mnli` for
classification and `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
for multilingual sentiment.

- Minimal call: `pipe = pipeline("text-classification")` then
  `pipe("This restaurant is awesome.")` → `[{'label': 'POSITIVE',
  'score': 0.9998}]`.
- Pinning a model: `pipeline(model="FacebookAI/roberta-large-mnli")` on
  the same sentence → `NEUTRAL` at 0.7313 — same input, different head,
  different answer (good quiz hook on task-vs-model).
- Full distribution: `return_all_scores=True` returns every label with its
  score instead of only the top one.
- Cross-modal breadth in one API: speech recognition and
  image-classification/object-detection pipelines share the same
  `pipeline(...)` entry point, prefiguring tasks 8–9.

## Introducing RoBERTa Base Model — Novita AI (Medium)

URL: https://medium.com/@marketing_novita.ai/introducing-roberta-base-model-a-comprehensive-overview-330338afa082 · Date: 2024-05-23 (published) · Status: summary

RoBERTa as "BERT, retrained properly": same encoder architecture, but
160GB of text (10× BERT), dynamic masking, no next-sentence objective,
larger batches and learning rates, and byte-level BPE like GPT-2. Reads as
a spec sheet for task 0 — every pretraining difference versus BERT in one
place. (Fetched via archive.org; live page 403s headless bots.)

- Name = Robustly Optimized BERT Approach (Facebook AI); architecture
  identical to BERT, gains come from training procedure, not structure.
- Four training deltas: 10× data (160GB vs 16GB), dynamic masking, larger
  mini-batches with higher learning rates, no NSP objective.
- Tokenizer delta: byte-level BPE (GPT-2 style) instead of character-level;
  no `token_type_ids` — segments split with `sep_token` (`</s>`).
- Downstream story: same pre-train-then-fine-tune-all-parameters loop as
  BERT; reported wins on translation, classification and QA.

## Introduction of RoBERTa Model — GeeksforGeeks

URL: https://www.geeksforgeeks.org/machine-learning/overview-of-roberta-model/ · Date: 2026-07-10 (page last updated) · Status: summary

A compact BERT-vs-RoBERTa table plus the encoder pipeline
(tokenize with BPE → dense vectors → add positional embeddings). Less
depth than the Novita piece but the comparison table is quotable and it
states the cost side plainly: RoBERTa's longer training on more data costs
more compute. Dated by its "Last Updated" line (original 2020 byline).

- One-line definition used: encoder-only transformer from Meta AI that
  keeps BERT's architecture and optimizes pretraining.
- Table rows: objective MLM+NSP → MLM-only; BookCorpus+Wikipedia →
  +CommonCrawl/OpenWebText; static → dynamic masking; fewer steps and
  smaller batches → longer training, larger batches; generally higher
  accuracy at higher compute cost.
- Input pipeline order: BPE tokenization → dense input embeddings →
  positional embeddings added so the model knows token order without
  recurrence.

## Tweets Sentiment Analysis with RoBERTa — Monica Rotulo (Medium)

URL: https://medium.com/@monica.rotulo/tweets-sentiment-analysis-with-roberta-1f30cf4e1035 · Date: 2022-04-10 (published) · Status: summary

End-to-end three-way tweet sentiment (positive/neutral/negative) with the
pretrained `cardiffnlp/twitter-roberta-base-sentiment` card: install
`transformers` + `scipy`, normalize mentions/URLs to `@user`/`http`,
tokenize, softmax the logits into probabilities. Motivates the hard parts
— conversational text, short length, implicit sentiment in news-like text.
(Fetched via archive.org.)

- Model string: `cardiffnlp/twitter-roberta-base-sentiment`, itself trained
  on 124M+ tweets (Jan 2018–Dec 2021) for self-supervised NLP.
- Preprocessing rule: replace every @mention with `@user` and every URL
  with `http`, because the model ignores usernames and page contents.
- Pipeline roles: tokenizer prepares tensor inputs (tokenization =
  text→pieces); model returns raw scores; softmax (via scipy) turns them
  into per-label probabilities printed next to each label.
- Domain warning: tweets are conversational and short, so sentiment hides;
  deep models like RoBERTa extend sentiment analysis into such harder
  domains (marketing, product, social monitoring uses named).

## Unlocking Language Barriers with Helsinki-NLP Translation Models — Anas Davood TK (Medium)

URL: https://medium.com/@anasdavoodtk1/unlocking-language-barriers-with-helsinki-nlp-translation-models-1fb7a40f9c2d · Date: 2024-02-16 (published) · Status: summary

A short advocacy piece for the University of Helsinki's NLP group models:
1000+ open-source translation language pairs, GitHub repo plus a methods
paper, all runnable from the Hugging Face Hub. Thin on code (integration
steps only), but it is the project's designated pointer for task 6
(encoder-decoder translation) and names the exact discovery path:
repo → pick pair → Hub inference.

- What: open-source pre-trained translation models covering 1000+
  language pairs, including underrepresented languages.
- Provenance: NLP group at the University of Helsinki; methods, language
  coverage and benchmarks in their paper (read before citing numbers).
- Where to run: Hugging Face Hub hosts the models with live inference
  widgets — translate in the browser before writing integration code.
- Usage ladder: explore repo → select the pair's model → follow docs to
  integrate, instead of training translation from scratch.

## Text Generation v/s Text2Text Generation — Sharath Hebbar (Medium)

URL: https://medium.com/@sharathhebbar24/text-generation-v-s-text2text-generation-3a2b235ac19b · Date: 2023-09-27 (published) · Status: summary

A two-minute decoder-only vs encoder-decoder contrast: causal text generation
(GPT family, BLOOM, PaLM — chatbots, completion) against seq2seq
text-to-text (T5, BART — translation, summarization, QA), each with a
minimal `pipeline(...)` snippet (`gpt2` with `max_length=30`,
`num_return_sequences=2`; `google/flan-t5-base` for English→French). Maps
exactly onto tasks 6 vs 7. (Fetched via archive.org; only ~280 words, but
that is the complete article.)

- Text generation = causal language modeling: decoder-only,
  left-to-right; completes sentences and poetry lines.
- Text2Text = sequence-to-sequence: encoder-decoder reading both
  directions; converts one text into another (translation, summary).
- Code fingerprints: `pipeline("text-generation", model="gpt2")` with
  `max_length`/`num_return_sequences`; `pipeline("text2text-generation",
  model="google/flan-t5-base")`.
- Reference pointer: author's Transformers repo
  (github.com/SharathHebbar/Transformers) for fuller examples.

## Scientific Text Generation with HF-DistilGPT2 in TensorFlow — Iva Vrtaric (Medium)

URL: https://medium.com/@ivavrtaric/scientific-text-generation-with-hf-distilgpt2-in-tensorflow-b677f4bef6b6 · Date: 2023-01-13 (published) · Status: summary

Fine-tunes `distilgpt2` (distilled GPT-2) on ML arXiv abstracts in
TensorFlow: `TFAutoModelForCausalLM` + `AutoTokenizer` with
`pad_token = eos_token`, 300-token truncation, exponential-decay schedule,
`model.fit(...)`, then a generation `pipeline`. Opens with a ChatGPT quote
warning that DistilGPT-2 output may not be scientifically reliable — verify
generated text. The pad→eos idiom is the task-7 fingerprint (GPT-2 has no
native pad token).

- Data: Cornell arXiv set (1.7M+ STEM titles+abstracts); HF Hub subset
  `CShorten/ML-ArXiv-Papers` (~100K ML papers); trains on abstracts only,
  `max_length=300` with truncation, uniform padding.
- Model setup: `tokenizer = AutoTokenizer.from_pretrained("distilgpt2")`;
  `tokenizer.pad_token = tokenizer.eos_token`; `TFAutoModelForCausalLM.
  from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)`.
- Training: TF stack (`transformers`, `datasets`, TF), exponential-decay
  learning-rate schedule, `model.fit(train_set, validation_data=...,
  epochs=...)`.
- Caveat to carry: distilled causal models generate fluent but
  unreliable scientific text — generation ≠ knowledge; always verify.

## Vision Transformers (ViT) in Image Recognition — viso.ai

URL: https://viso.ai/deep-learning/vision-transformer-vit/ · Date: 2023-11-25 (published) · Status: summary

A ViT survey pitched at practitioners: pure transformers on image patches
rival CNNs (ResNet, VGG, YOLO, SAM named as the incumbents) with ~4×
compute/accuracy gains, then a benchmark tour (CSWin's cross-shaped-window
attention beating Swin) across classification, detection and segmentation.
Sets up task 8's "pure transformer, no convolutions" premise. Byline
Gaudenz Boesch. (Fetched via archive.org.)

- Thesis: a pure transformer on sequences of image patches does image
  classification exceptionally well — CNN dependence is optional, not
  mandatory; attention otherwise plugs into CNNs as enhancement or partial
  replacement.
- Lineage: transformer standard in NLP (ChatGPT/GPT named) → ViT ports
  self-attention to patch dependencies in vision.
- Efficiency claim: ViTs beat SOTA CNNs by ~4× on the
  computation/accuracy trade-off in image recognition.
- Variant to know: CSWin Transformer with cross-shaped-window
  self-attention, reported as surpassing Swin on vision benchmarks.

## Vision Transformers (ViT) Tutorial — DataCamp (Vaibhav Mehra)

URL: https://www.datacamp.com/tutorial/vision-transformers · Date: 2025-08-28 (published) · Status: summary

A full vanilla-ViT tutorial: image→fixed-patch sequence → flatten → embed
→ transformer encoder with global attention, plus a CNN-vs-ViT table
(local vs global receptive field, strong vs minimal inductive bias,
small-data effectiveness vs large-data hunger) and DeiT/Swin as practical
follow-ups to the 2020 "An Image is Worth 16×16 Words" paper. The
inductive-bias explanation is its best conceptual bit. (Client-rendered;
needed the authenticated browser — plain fetch got a 28-word shell.)

- Mechanism: split image into fixed-size patches, flatten, embed to
  vectors, run a standard transformer encoder; attention models
  patch-to-patch relations globally from the first layer.
- CNN contrast table: convolution+pooling vs transformer encoder;
  receptive field local-but-expanding vs global-from-start; position
  implicit-in-structure vs explicit encodings; CNNs data-efficient,
  ViTs data-hungry; CNNs strong bias, ViTs minimal-but-flexible;
  ViTs more parallel and more scalable to large models.
- Inductive bias glossed: the assumptions a model bakes in to learn from
  less data (CNN locality) — ViTs trade that away for flexibility at the
  cost of needing large datasets.
- History anchors: Google Research 2020 paper; DeiT and Swin as the
  practical variants that spread ViTs across vision tasks.

## What Are Vision Language Models (VLMs)? — IBM Think

URL: https://www.ibm.com/think/topics/vision-language-models · Date: 2025-02-25 (published) · Status: summary

IBM's VLM primer: language encoder + vision encoder (early CNN feature
extractors, modern ViT patch+self-attention) feeding an LLM core, taking
text+image/video in and emitting text (descriptions, visual QA,
grounding). Restates the transformer encoder/decoder embedding story in
multimodal form. Byline Rina Diane Caballar and Cole Stryker.

- Definition: multimodal systems blending computer vision and NLP that map
  text↔visual relations; also called visual language models; usually
  LLM + vision model under the hood.
- Two components: language encoder (transformer — BERT/GPT named — turning
  words into text embeddings) and vision encoder (colors/shapes/textures
  → vector embeddings; CNNs earlier, ViT now).
- ViT recap inside: patches-as-tokens plus self-attention across patches
  to build the visual representation the language side conditions on.
- I/O contract: text and images/video in, text out — captions, answers
  about an image, locating parts/objects; the direct spec of task 9.

## What Are Vision Language Models? — NVIDIA Glossary

URL: https://www.nvidia.com/en-us/glossary/vision-language-models/ · Date: 2026-10-11 (page last modified) · Status: summary

NVIDIA's VLM entry argues from the pain of classic CNN vision (fixed
classes, retrain-for-every-new-class expense, no language understanding)
to zero-shot prompt-driven VLMs built on CLIP-style foundations: chat with
images, change the task by changing the text prompt. The "traditional CV
vs VLM" contrast is its quotable core.

- Old regime: CNN classifiers/detectors/OCRs bound to fixed classes and
  one task; new class or use-case change = collect, label, retrain
  (expensive, slow); zero natural-language understanding.
- New regime: LLM + vision encoder ("sight"); pretrained on text plus
  image/video-caption pairs; strong zero-shot on VQA, classification, OCR
  out of the box.
- Interaction model: same as chatting with an LLM, but prompts interleave
  images; output is text (answer, summarize, explain, reason over the
  image); back-and-forth supported.
- Flexibility claim: nearly any vision use case reachable by rewording the
  prompt — no retraining for the new task.

## BLIP: Bridging the Gap Between Vision-Language Tasks — Dong-Keon Kim (Medium)

URL: https://medium.com/@kdk199604/blip-bridging-the-gap-between-vision-language-tasks-through-unified-pre-training-9536ea1a1407 · Date: 2025-07-28 (published) · Status: summary

A paper explainer for BLIP (Vision-Language Pretraining): the MED
architecture unifying understanding and generation plus the CapFilt data
loop (Captioner writes synthetic captions for web images, Filter drops
noisy ones), beating CLIP/ALBEF/SimVLM-style noisy-data training and
generalizing zero-shot to video. Best conceptual prep for task 9's BLIP
captioning. (Fetched via archive.org.)

- Problem diagnosed: prior VLP leans on noisy web alt-text (weak
  supervision, poor fine-grained understanding); encoder-only models do
  retrieval but not generation, encoder-decoder the reverse; earlier
  unified attempts underperform or train with difficulty.
- MED = Multimodal Mixture of Encoder-Decoder: one architecture with a
  unimodal encoder (image-text contrastive learning, ITC) plus
  image-grounded encoder/decoder via cross-attention — understanding and
  generation in a single model.
- CapFilt = Captioning and Filtering: Captioner (Cap) synthesizes captions
  for web images, Filter (Filt) removes noisy originals — data quality as
  a first-class lever, not just scale.
- Payoff claimed: state of the art across retrieval, captioning and more,
  with zero-shot transfer to video-language tasks.

## Understanding BLIP: A Huggingface Model — GeeksforGeeks

URL: https://www.geeksforgeeks.org/artificial-intelligence/understanding-blip-a-huggingface-model/ · Date: 2026-06-23 (page last updated) · Status: summary

The implementation companion to the BLIP explainer: MED's three blocks
(unimodal encoder, image-grounded encoder with cross-attention,
image-grounded decoder with causal self-attention), the three pretraining
losses (ITC contrastive, ITM matching classification, LM autoregressive),
and copy-paste code for `Salesforce/blip-image-captioning-base` captioning
with `BlipProcessor` + `BlipForConditionalGeneration`. Dated by its "Last
Updated" line. This is the task-9 recipe.

- Expansion: Bootstrapping Language-Image Pre-training; multimodal
  transformer doing captioning, VQA and image-text retrieval from
  large-scale image-text pairs.
- Three losses: ITC aligns visual/textual spaces (match vs negatives),
  ITM classifies whether a text matches an image (fine-grained
  representation), LM generates plausible text from images
  autoregressively.
- Code fingerprint: `pip install torch transformers numpy pillow`;
  `BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')`;
  `BlipForConditionalGeneration.from_pretrained(...)` same string;
  processor handles image+text prep and output postprocessing.
- Role split: processor = preprocessing/postprocessing, model =
  conditional generation for the caption.

---

## References

Link-only, per the intranet's "References" section. All resolved to live
URLs on 2026-10-13 (23 return HTTP 200 to HEAD/GET; StackOverflow,
LinkedIn and the three unsuffixed Medium "More" URLs 403 headless bots
but render correct titled pages in the browser — bot-blocking, not dead
links). The intranet label "(NVIDIA: What Are Vision Language Models" is
missing its closing parenthesis; the destination title is unaffected.

| Resource | URL |
|----------|-----|
| Hugging Face's LLM Course | https://huggingface.co/learn/llm-course/chapter0/1 |
| Prompt engineering | https://huggingface.co/docs/transformers/en/tasks/prompting |
| Masked language modeling | https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling |
| RoBERTa | https://huggingface.co/docs/transformers/en/model_doc/roberta |
| FacebookAI/roberta-base | https://huggingface.co/FacebookAI/roberta-base |
| no_grad | https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html |
| Pipeline | https://huggingface.co/docs/transformers/en/pipeline_tutorial |
| Pipelines | https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/pipelines#transformers.ImageSegmentationPipeline |
| Hugging Face is the home for all Machine Learning tasks | https://huggingface.co/tasks |
| Helsinki-NLP | https://huggingface.co/Helsinki-NLP |
| GPT-2 | https://huggingface.co/docs/transformers/en/model_doc/gpt2 |
| openai-community/gpt2 | https://huggingface.co/openai-community/gpt2 |
| distilbert/distilgpt2 | https://huggingface.co/distilbert/distilgpt2 |
| Setting pad token id to eos token id | https://stackoverflow.com/questions/76633368/how-does-one-set-the-pad-token-correctly-not-to-eos-during-fine-t |
| Vision Transformer (ViT) | https://huggingface.co/docs/transformers/en/model_doc/vit |
| google/vit-base-patch16-224 | https://huggingface.co/google/vit-base-patch16-224 |
| BLIP | https://huggingface.co/docs/transformers/en/model_doc/blip |
| Salesforce/blip-image-captioning-base | https://huggingface.co/Salesforce/blip-image-captioning-base |

Note: "Pipelines" (intranet label) lands on pinned v4.24.0 pipeline docs,
older than the current `en/` pages the sibling "Pipeline" label points to —
same content family, older snapshot.

---

## More

Link-only. The project page carries a fourth section, "More", with 6
further-reading items; with the 25-resource cap filled by Read-or-watch,
these are recorded as links (all destinations verified in-browser).

| Resource | URL |
|----------|-----|
| A tutorial on open-source large language models for behavioral science | https://pmc.ncbi.nlm.nih.gov/articles/PMC11525391/ |
| Preparing Data for Large Language Models (LLMs) | https://www.linkedin.com/pulse/demystifying-tokenization-preparing-data-large-models-rany-2nebc/ |
| Inference in PyTorch | https://medium.com/@whyamit404/inference-in-pytorch-understanding-the-wrappers-and-choosing-the-best-d4f16fbde960 |
| Optimizing inference | https://huggingface.co/docs/transformers/main/en/llm_optims |
| Understanding Masked Language Modeling as a Pretraining Task | https://medium.com/@eraparihar98/understanding-masked-language-modeling-as-a-pretraining-task-4a887ec61c8d |
| A Practical Guide to Using AutoClass in Hugging Face | https://medium.com/@faizanfaizy7/a-practical-guide-to-using-autoclass-in-hugging-face-b2364171600b |

---

## Ingestion Notes

- All 45 rltoken links (21 read-or-watch + 18 references + 6 more) were
  resolved inside the authenticated CDP browser; the session stayed alive
  the whole run (intranet dashboard reachable, no re-login needed).
- No YouTube on this project, so no transcript pipeline was needed. No
  "Definitions to skim" section exists on the 3601 page — glossary terms
  are folded into Quiz Hooks instead.
- Fetch split for the 21: 9 direct (IBM ×3, NVIDIA ×2, AWS, GFG ×2,
  github.io), 6 via archive.org (3 Medium, viso.ai, DataCamp stub, BLIP
  Medium), 6 via the authenticated browser (4 Medium paywall shells +
  roadmap + DistilGPT2 + full DataCamp render). Zero paywalled, zero
  failed-transcript.
- Medium serves headless requests 403 and logged-out browsers a
  members-only shell; the authenticated CDP render returned full text for
  all 10 Medium articles. Trailing story-ID hashes matter: truncated
  Medium URLs render a site 404, so the exact resolved URLs are recorded
  above.
- Archive.org was rate-limiting mid-run (CDX "Temporarily Offline");
  the `/web/2/` snapshots that did resolve were sufficient, and the rest
  came from the authenticated browser instead.
- Date labels: byline dates marked `(published)`; GFG "Last Updated"
  lines marked `(page last updated)`; AWS/NVIDIA Last-Modified headers
  marked `(page last modified)`; the github.io recipe book prints no date
  anywhere → `unknown`. IBM's LLM page shows an Oct-2021 byline under a
  Dec-2024 metadata update — byline used. The pipelines tutorial shows
  Nov-2024 in the byline under a May-2025 metadata edit — byline used.
- Reference hygiene: Hugging Face rate-limited the browser mid-run (page
  titles read "429"), but all HF URLs are canonical and return HTTP 200
  to direct requests. "Pipelines" pins v4.24.0 docs (flagged above).
- Copyright: summaries are own-words; raw HTML/texts live only in
  `/tmp/opencode/` (`llm_*.html`, `llm_texts/`), never committed.

---

## Quiz Hooks

- LLM — giant next-word prediction machine on transformer architecture,
  trained self-supervised on billions/trillions of tokens.
- Tokenization — text → words/subwords/characters; each token becomes an
  embedding vector, plus positional encodings for order.
- Self-attention — per token, query seeks, key advertises, value flows;
  query–key similarity → alignment scores → attention weights.
- Pretraining vs fine-tuning — pretraining gives broad knowledge;
  supervised fine-tuning on a small labelled set adapts to a task.
- RLHF — humans rank outputs, model learns to prefer higher-ranked ones;
  used for alignment including stylistic (casual/humorous/on-brand).
- Hallucination — fluent but false output; verify, and red-team for
  safety/bias weaknesses.
- MLM — mask words, predict from both directions; the fill-in-the-blanks
  pretraining behind BERT/RoBERTa (tasks 0–5).
- Dynamic masking — the mask pattern changes across epochs (RoBERTa),
  unlike BERT's fixed static masks.
- NSP dropped — RoBERTa removes next-sentence prediction, trains on MLM
  only, with 160GB text, bigger batches, byte-level BPE.
- `RobertaForMaskedLM` — the task-0 class: a RoBERTa model with an MLM
  head ready for inference.
- Mask decode loop — `tokenizer.mask_token_id` + `torch.where` to find
  `[MASK]`, `logits.argmax(dim=-1)` to pick the word, `tokenizer.decode`
  to read it (tasks 2–4).
- `pipeline("sentiment-analysis")` — top call returns
  `[{'label': 'POSITIVE', 'score': 0.9998}]`; `return_all_scores=True`
  returns the full distribution.
- Zero-shot classification — pass `candidate_labels`, get the winner plus
  confidence with no task training.
- Encoder-only fill-mask vs decoder-only generation — `fill-mask`
  (BERT's objective) guesses hidden words; `text-generation` (GPT-2's)
  extends prompts left-to-right (tasks 5 vs 7).
- Text2Text — encoder-decoder (T5, BART) converting text to text:
  translation, summarization, QA (task 6).
- Helsinki-NLP — 1000+ open-source translation pairs from the University
  of Helsinki group, runnable from the HF Hub (task 6).
- GPT-2 has no pad token — set `tokenizer.pad_token = tokenizer.eos_token`
  (and `pad_token_id`) before batched TF training (task 7).
- DistilGPT2 — distilled GPT-2 for generation; fluent output still needs
  fact verification, especially scientific text.
- ViT — image as patch sequence → flatten → embed → transformer encoder;
  global attention from layer one (task 8).
- Inductive bias — baked-in assumptions (CNN locality) that cut data
  needs; ViTs trade it for flexibility and need large datasets.
- `google/vit-base-patch16-224` — the canonical pretrained ViT card for
  task 8 (16×16 patches, 224px input).
- VLM — LLM + vision encoder (ViT now, CNNs earlier); text+image in,
  text out: captions, visual QA, grounding (task 9).
- BLIP/MED — Multimodal Mixture of Encoder-Decoder: unimodal encoder for
  contrastive ITC plus image-grounded encoder/decoder for ITM and LM.
- CapFilt — Captioner writes synthetic captions for noisy web images,
  Filter drops the bad ones; data quality as a lever alongside scale.
- `BlipProcessor` + `BlipForConditionalGeneration` — the task-9 pair on
  `Salesforce/blip-image-captioning-base` (processor preps, model
  generates).
