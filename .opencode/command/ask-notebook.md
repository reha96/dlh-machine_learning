---
description: Answer a question grounded in the AI Book notebook (open-notebook MCP)
---

Answer the user's question using the **AI Book** notebook on the
`open-notebook` MCP server. The question:

$ARGUMENTS

Notebook ID (exact — IDs contain a `:`, quote them in shell contexts):
`notebook:rxx4byfysdltkrffq02y`

If the question above is empty, ask the user what they want to know and stop.

## Workflow (read-only)

1. **Orient** — `list_sources` with the notebook ID above to see what the
   notebook contains; `get_source` for details on one source.
2. **Locate** — for a specific question, `search` (vector or text, scoped
   with the notebook ID) to find relevant passages.
3. **Ask** — prefer the session route: `list_chat_sessions` for the notebook
   and reuse a fitting session ID; if none, `create_chat_session` (notebook
   ID + short title). Then `execute_chat` with the session ID and the
   question. Note the session ID and reuse it for follow-ups so context
   carries over. Default models apply — no model IDs needed on this path.
4. **Report** — give the answer grounded in the notebook and name the
   source(s) that supported it. If the notebook does not cover the question,
   say so plainly instead of guessing.

## Fallback: one-shot ask tools

`ask_simple` / `ask_question` require explicit model IDs. Pass
`model:wjn7c6g5loecxcktll4r` (the project's chat model) as `strategy_model`,
`answer_model`, and `final_answer_model`, plus the notebook ID.

## Hard limits

- Never create, update, or delete notebooks, sources, notes, models, or
  settings from this command.
- Never upload or ingest a source without the user's explicit approval for
  that specific action.
