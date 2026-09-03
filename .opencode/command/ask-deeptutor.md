---
description: Answer a question grounded in the Hands-On ML book via DeepTutor ai-book-kb
---

Answer the user's question using the **ai-book-kb** knowledge base on the `deeptutor` MCP server. The question:

$ARGUMENTS

Knowledge base (exact — 1181 chunks, llamaindex, BGE-small 384d, Gemma 73728 ctx):
`ai-book-kb` (`~/Documents/deeptutor-workspace/data/knowledge_bases/ai-book-kb`)

If the question above is empty, ask the user what they want to know and stop.

## Workflow (read-only)

1. **Orient** — `list_knowledge_bases` to confirm `ai-book-kb` is `ready`; `get_knowledge_base_info` with `name: ai-book-kb` for document count and index version if needed.
2. **Locate** — `search_knowledge_base` with `name: ai-book-kb`, `query: <question>` (mode `hybrid`) to retrieve relevant passages with page refs and scores. Cite these.
3. **Ask** — `ask_deeptutor` with `message: <question> grounded in Hands-On ML book (cite pages)`, `knowledge_bases: ["ai-book-kb"]`, `capability: "chat"`. Note the returned `session_id` and reuse it for follow-ups (`ask_deeptutor` with `session_id`). For multi-step tasks use `capability: deep_solve` or `deep_question` if needed.
4. **Report** — give the answer grounded in the KB and name the source pages/sections that supported it. If the KB does not cover the question, say so plainly instead of guessing.

## Hard limits

- Never create, update, or delete knowledge bases, documents, or settings from this command.
- Never upload or re-index a document without the user's explicit approval for that specific action.
- Env is `DEEPTUTOR_HOME=~/Documents/deeptutor-workspace`; gateway is local `172.18.0.1:18080/18081` (no external key).
