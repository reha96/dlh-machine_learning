# NLP & Large Language Models

The natural-language end of the ML specialization: classical text processing,
transformer/LLM mechanics, fine-tuning, and LangChain application building.

| Project | Intranet | Window | Tasks | Status |
|---------|----------|--------|-------|--------|
| [Introduction to NLP](introduction_to_nlp/) | [3603](https://intranet-dlh.hbtn.io/projects/3603) | Sep 14 → Sep 27 | 13 | scaffolded |
| [Large Language Models (LLMs)](large_language_models/) | [3601](https://intranet-dlh.hbtn.io/projects/3601) | Sep 21 → Oct 04 | 10 | scaffolded |
| [Fine-Tuning Large Language Models](fine_tuning_llms/) | [3689](https://intranet-dlh.hbtn.io/projects/3689) | Sep 28 → Oct 15 | 8 | scaffolded |
| [LangChain](langchain/) | [3602](https://intranet-dlh.hbtn.io/projects/3602) | Oct 05 → Oct 18 | 16 | scaffolded |

Sequence-modeling material lives one level up with the rest of the supervised
work: [`../supervised_learning/rnn/`](../supervised_learning/rnn/) (RNNs, 2337).

---

## Progression

1. **Introduction to NLP** — string cleaning → tokens → features → embeddings
   (TF-IDF, Word2Vec, FastText).
2. **Large Language Models** — masked language modeling internals (RoBERTa),
   then encoder / encoder-decoder / decoder-only usage, then ViT + BLIP.
3. **Fine-Tuning LLMs** — DistilBERT sentiment fine-tuning with `Trainer`.
4. **LangChain** — prompts, agents, and RAG pipelines on top.

---

## Setup

```bash
source my-venv/bin/activate        # for the classical NLP tasks
source tf-venv/bin/activate        # for transformers / torch / TF tasks
```

Hugging Face models and datasets are downloaded at runtime; never commit the
cache (see the root `.gitignore`).
