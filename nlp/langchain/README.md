# LangChain

Building LLM applications with LangChain: prompt templates, chat-style prompts,
tool-using agents (including a pandas DataFrame agent), then a full
retrieval-augmented-generation (RAG) stack — loaders, chunking, embeddings,
vector store, document chains, and history-aware conversational retrieval.

| | |
|---|---|
| Intranet project | [3602 — LangChain](https://intranet-dlh.hbtn.io/projects/3602) |
| Difficulty | Novice · Weight 1 |
| Author | Malek Mrabti |
| Window | Oct 05 → Oct 18 (current week) |
| Stack | `python3` 3.11, `pycodestyle` 2.14.0, `langchain`, Hugging Face endpoints |
| Status | Scaffolded 2026-10-12 — no task files yet |
| Resources | [`RESOURCES.md`](RESOURCES.md) |

---

## Tasks

| # | Task | Status |
|---|------|--------|
| 0 | Initialize Hugging Face LLM in LangChain | not started |
| 1 | Load Mistral using LangChain | not started |
| 2 | Retrieve LLM Response | not started |
| 3 | Build a Message Sequence | not started |
| 4 | Create and Use a Prompt Template | not started |
| 5 | Setup Chat-Style Prompt | not started |
| 6 | Multi-Tool Agent | not started |
| 7 | Pandas DataFrame Agent | not started |
| 8 | (RAG) Document Loader | not started |
| 9 | (RAG) Chunking | not started |
| 10 | (RAG) Load Embedding Model | not started |
| 11 | (RAG) Create Vector Store | not started |
| 12 | (RAG) Get Answer Using Document Chain | not started |
| 13 | (RAG) History-Aware Retriever | not started |
| 14 | (RAG) Conversational Retrieval Chain | not started |
| 15 | (RAG) Session-Aware Q&A Chatbot | not started |

---

## Requirements (from the spec)

- Ubuntu 20.04 LTS, `python3` 3.11.
- First line of every file is exactly `#!/usr/bin/env python3`; files end with a newline.
- `README.md` at the project root is mandatory; `pycodestyle` 2.14.0 clean.
- Module, class and function docstrings everywhere.

---

## Notes for this project

- Needs a Hugging Face inference token; keep it out of the repo (env var only).
- LangChain APIs churn fast — the task spec is the source of truth for import
  paths and class names, not the current LangChain docs.
- Verification environment: `tf-venv/bin/python`.
