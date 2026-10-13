# LangChain Resources

Ingested: 2026-10-13 · Source: [intranet project 3602 — LangChain](https://intranet-dlh.hbtn.io/projects/3602) (Novice, weight 1)

15 "Read or watch" resources get the full treatment (14 Status: summary,
1 paywalled — the Kaggle notebook is login-walled). This project page has no
"Definitions to skim" and no "More" section; the 26 "References" are
link-only (all verified to return HTTP 200 on 2026-10-13).
Label/destination mismatches are flagged in the bullets and notes.

Task map for the bullets below: 0 HF LLM init (`HuggingFaceEndpoint`) ·
1 Load Mistral · 2 Invoke / response · 3 Message sequence
(`SystemMessage`/`HumanMessage`) · 4 Prompt template · 5 Chat-style prompt ·
6 Multi-tool agent · 7 Pandas DataFrame agent · 8 Document loader ·
9 Chunking · 10 Embedding model · 11 Vector store · 12 Document chain ·
13 History-aware retriever · 14 Conversational retrieval chain ·
15 Session-aware chatbot (`RunnableWithMessageHistory`).

---

## What Is LangChain? — IBM Think

URL: https://www.ibm.com/think/topics/langchain · Date: 2024-12-02 (published) · Status: summary

IBM's topic page presents LangChain as an open-source orchestration
framework (Python and JavaScript) that chains modular abstractions —
models, prompt templates, chains, indexes, memory, tools and agents —
so developers can build LLM apps without hand-writing every integration.
It walks the whole stack the project needs: importing any LLM behind a
standard interface, formalising prompts, sequencing calls in chains,
grounding models with document loaders/vector stores/retrievers, adding
memory, and giving agents tools; it closes on LangGraph and LangSmith.
Byline Dave Bergmann / Cole Stryker; dated from the
`article:published_time` meta (the `Last-Modified` header shows today's
CDN date and is untrustworthy).

- Abstraction thesis: each LangChain component encapsulates a complex
  process (like `π` for a ratio), and components "chain" so output of
  one step feeds the next — power at the cost of some fine control.
- Model uniformity (tasks 0–1): the `LLM` class gives one interface for
  any provider behind an API key; `CustomLLM` wraps your own, and
  `WatsonxLLM` is the IBM-side example of a provider wrapper.
- Prompt/chains core (tasks 4–5): `PromptTemplate` holds context,
  instructions, few-shot examples and output format in named
  `input_variables`; the minimal chain calls one model plus one prompt
  template, and sequential chains pipe output forward.
- Indexes = the RAG half (tasks 8–11): document loaders for third-party
  sources, vector databases holding embeddings plus metadata for
  low-latency search, text splitters for chunking, and retrievers that
  take a string query and return `Document` objects.
- Memory/tools/agents (tasks 6–7, 13–15): memory utilities from full
  history to last-n summaries; tools (Wolfram, Search, Weather,
  Wikipedia) cover the LLM gaps of stale knowledge, domain expertise
  and maths; agents decide, plan and act instead of emitting one reply.

## LangChain Course — Course Introduction (KodeKloud notes)

URL: https://notes.kodekloud.com/docs/LangChain/Introduction/Course-Introduction/page · Date: unknown · Status: summary

Study notes for a KodeKloud LangChain course (instructor Janakiram MSV)
that teach the framework as composable building blocks shared by apps
like ChatGPT, Gemini and Copilot: models, inputs, outputs, LCEL, chains,
memory, tools and agents, each topic following a theory → demo → lab
pattern. The notes page prints no byline date, hence unknown. Its
compact chat-prompt example is the closest thing here to the task 3–5
message/prompt pattern.

- Building-blocks framing: every LLM app combines common elements, and
  LangChain's job is composing them — models, inputs, outputs plus the
  LangChain Expression Language (LCEL) for wiring calls together.
- Module table worth quoting: Model (chat models, temperature,
  streaming) · Input (prompts, system/human messages, LCEL) · Output
  (parsers, structured responses) · Chains (sequential, conditional,
  map-reduce) · Memory (conversation memory, vector stores) ·
  Tools & Agents (APIs, search, tool selection).
- Code fingerprint: `ChatPromptTemplate.from_messages` with
  `SystemMessagePromptTemplate.from_template("You are a {subject}
  teacher")` and `HumanMessagePromptTemplate.from_template("Tell me
  about {concept}")`, then `format_messages(subject="Chemistry",
  concept="Periodic Table")`.
- Lab hygiene: API keys via environment variables
  (`os.environ["OPENAI_API_KEY"]`), never committed — same rule as the
  HF token in this project.

## How to Invoke LLM from LangChain — Terry Cho (Medium)

URL: https://medium.com/@terrycho/3-how-to-invoke-llm-from-langchain-7325988e3097 · Date: 2023-12-20 (published) · Status: summary

Third post of a LangChain series explaining the two model abstractions
(completion-style LLM vs chat-style) and then the four call patterns for
actually invoking them: single synchronous `invoke`, multi-prompt
`batch`, token-by-token `stream`, and the async trio
(`ainvoke`/`abatch`/`astream`). Directly relevant to task 2. (Fetched
via the authenticated browser; archive.org holds no snapshot.)

- Model split: completion models answer/summarise from a prompt, chat
  models converse; LangChain abstracts both, with per-provider tunable
  parameters (temperature for OpenAI-style, plus top-k/top-p elsewhere).
- `llm.invoke(prompt)` — one prompt, synchronous; `llm.batch(prompts)`
  — a Python list of prompts in one call instead of a loop.
- `llm.stream(prompt)` yields chunks as generated for slow/large
  responses (the chatbot pattern); each of the three has an async twin
  for concurrent calls.
- Timing demo: 10 identical prompts take ~1.9s with `asyncio.gather`
  over `ainvoke` vs ~5.8s looping `invoke` — the quantitative argument
  for async in batch workloads.
- Caveat carried: supported call patterns differ per model integration —
  check the provider's integration page before assuming `batch`/`stream`
  exist.

## Prompt Engineering Overview — Elvis Saravia (YouTube)

URL: https://www.youtube.com/watch?v=dOxUroR57xs · Date: 2023-02-19 (YouTube upload) · Status: summary

A 64-minute lecture (DAIR.AI prompt-engineering course, presented by
Elvis Saravia) that formalises prompts as instruction + context + input
data + output indicator, then tours techniques from zero/few-shot and
role-playing through chain-of-thought and ReAct, plus the operational
side: temperature/top-p, tool use, and failure modes (injection,
leaking, jailbreaking). Read it as the theory behind tasks 4–5: every
prompt template in the project is one of these patterns written down.

- Prompt anatomy: instruction (e.g. "classify into neutral/negative/
  positive"), input data (the text), output indicator (where the answer
  goes), and context — the lever whose presence or absence moves
  performance most.
- Sampling controls: temperature and top-p set determinism; keep them
  low for factual Q&A, high for poetry/diverse generation; change one,
  not both, per the provider guidance quoted.
- Technique ladder: zero-shot → few-shot with examples → role-playing →
  code generation → reasoning (chain-of-thought as an emergent,
  scale-dependent ability) → generated knowledge → ReAct (interleaved
  reasoning traces and tool actions against knowledge bases).
- Limits to quote: prompt injection (a pasted "ignore the above…"
  hijacking a bot into threats), prompt leaking, jailbreaking; no
  delimiter trick fully prevents them.
- Forward look: multimodal prompting and graph prompting named as the
  bleeding edge, alongside LLM-plus-agent planning.

## LangChain: Pandas DataFrame Agent Guide — KS Mooi (Kaggle)

URL: https://www.kaggle.com/code/ksmooi/langchain-pandas-dataframe-agent-guide · Date: unknown · Status: paywalled

A Kaggle notebook (titanic dataset, ~9.4K views, Python) whose table of
contents runs Introduction → Preparation → Examples → Best Practices →
Conclusion. The code cells render only for signed-in users: plain fetch,
the authenticated browser, and a scroll-and-wait retry all return the
same ~140-word chrome (title, author, runtime "4m 41s", dataset tag).
Per policy only the title/URL/ToC are recorded — no summary is
fabricated from the shell. Open the notebook signed in before task 7;
its Examples section is the intended recipe.

- What is visible: notebook on the titanic dataset, Apache 2.0 license,
  tags Programming / Data Analytics / Text Generation / Language.
- What is NOT visible without login: every code cell, so the agent
  constructor, prompt used, and best-practice list cannot be quoted here.
- Task-7 fallback: the "create pandas dataframe agent" reference link
  below (readthedocs, LangChain 0.0.107) is openly readable and covers
  the same API.

## Build a Semantic Search Engine with LangChain — LangChain docs

URL: https://docs.langchain.com/oss/python/langchain/knowledge-base · Date: unknown · Status: summary

The official tutorial for the retrieval half of RAG: make `Document`
objects, embed them, load and split a PDF (a Nike 10-K), index chunks in
a vector store, and query by similarity — then wrap the store as a
runnable retriever. The worked numbers (107 pages → 516 chunks) give a
feel for chunking ratios. No `Last-Modified` header is sent and the page
prints no date, hence unknown. Core prep for tasks 8–11.

- `Document(page_content, metadata, id?)`: the unit of text; metadata
  keeps source/page/`start_index` so answers stay citable.
- Splitter fingerprint: `RecursiveCharacterTextSplitter(chunk_size=1000,
  chunk_overlap=200, add_start_index=True)` — the recommended generic
  splitter, recursing on newlines and friends.
- Embeddings are provider-pluggable (`OpenAIEmbeddings(model=
  "text-embedding-3-large")`, HF `sentence-transformers/
  all-mpnet-base-v2` with normalised embeddings, and a dozen more tabs);
  `embed_query` text and chunks must share dimensions.
- Store → retriever ladder: `vector_store.similarity_search` (plus
  `_with_score`, `_by_vector` variants) → the `@chain` wrapper or
  `vector_store.as_retriever(search_type="similarity",
  search_kwargs={"k": 1})`, which is a `Runnable` and therefore batch-
  and async-capable.
- Search-type note: `"similarity"` (default), `"mmr"` for diverse hits,
  `"similarity_score_threshold"` to cut weak matches; score semantics
  vary by provider (here a distance, lower = closer).

## Build a RAG Application with LangChain — LangChain docs (Deep Agents)

URL: https://docs.langchain.com/oss/python/deepagents/rag · Date: unknown · Status: summary

Note: the intranet label ("Build a RAG application with LangChain")
lands on the **Deep Agents** RAG guide, not the classic LangChain RAG
tutorial — an agentic, multi-step pattern (retrieve → offload chunks to
a filesystem backend → delegate analysis to subagents → synthesise),
not the `create_retrieval_chain` of tasks 12–14. Still useful as the
"what comes after the project" view, but do not cite it as the task
recipe. No date header or byline, hence unknown.

- Four RAG patterns named: skills-guided retrieval, rubric-checked
  grounding (`RubricMiddleware`), todo-driven investigation, and the
  worked retrieve-offload-delegate flow.
- Indexing in four verbs: load (`Document` per docs page) → split
  (`RecursiveCharacterTextSplitter(1000, 200)`) → embed → store; the
  demo indexes 14 docs pages into 782 chunks.
- Runtime shape: a `search_documentation` tool writes top-4 chunks to
  `/retrieved/<batch>/chunk_N.md` and returns paths; `chunk-analyst`
  subagents summarise one file each; the orchestrator merges summaries
  with source links, keeping full text out of its context.
- Security warning to carry into task 15: indirect prompt injection —
  retrieved chunks share the context window, so treat them as data, head
  each with a `# Source:` line, and validate cited paths before showing
  answers.

## Beginner's Guide To Conversational Retrieval Chain Using LangChain — Vijaykumar Kartha (Medium)

URL: https://vijaykumarkartha.medium.com/beginners-guide-to-conversational-retrieval-chain-using-langchain-3ddf1357f371 · Date: 2024-04-29 (published) · Status: summary

A beginner walkthrough of the exact three-function assembly behind tasks
12–14: `create_history_aware_retriever` (chat history + input → search
query) feeding `create_stuff_documents_chain` (context + history +
input → answer), combined by `create_retrieval_chain`. Builds on the
author's earlier retrieval-chain post, using a `dasa.org` vector store
and OpenAI. (Fetched via archive.org.)

- Ingredient list: the three `create_*` functions plus an LLM, a
  retriever, and two prompts — one to generate the search query, one to
  get the answer.
- Search-query prompt fingerprint: `ChatPromptTemplate.from_messages`
  with `MessagesPlaceholder("chat_history")`, `("user", "{input}")`,
  and the instruction "Given the above conversation, generate a search
  query to look up to get information relevant to the conversation".
- Answer prompt fingerprint: `("system", "Answer the user's questions
  based on the below context:\n\n{context}")` + history placeholder +
  `("user", "{input}")`.
- Demo proof: with `chat_history = [HumanMessage("Does DASA allow
  anybody to be certified?"), AIMessage("Yes")]`, the bare follow-up
  `"How?"` returns DASA certification programmes — history resolved
  into a standalone query.
- `MessagesPlaceholder` defined plainly: a prompt template whose
  variable is a *list of messages*, so the history slots into the prompt
  at a chosen position.

## RAG Updated — Q&A and Converstaional — Ashish Malhotra (Medium)

URL: https://medium.com/@mrcoffeeai/rag-updated-q-a-and-converstaional-99b6af6ddd8b · Date: 2024-06-13 (published) · Status: summary

The migration post the project implicitly follows: the old
`ConversationalRetrievalChain` / `RetrievalQA` are deprecated since
0.1.17 (removal slated for 0.3.0) in favour of the LCEL-style
`create_history_aware_retriever` + `create_retrieval_chain` pair, built
step by step over a story-chunk vector store (FAISS/Chroma). The
"Converstaional" typo is the author's own title spelling, kept as-is.
(Fetched via the authenticated browser; archive.org holds no snapshot.)

- Deprecation facts: both legacy chains deprecated as of 0.1.17, removal
  in 0.3.0; new development targets the `create_*` LCEL functions for
  cleaner interfaces and consistent naming.
- `create_stuff_documents_chain(llm, prompt)` first: combine retrieved
  `Document` objects with a `{context}`/`{input}` template — demoed on a
  toy sun/moon context before the real chunks.
- Q&A assembly: `retriever = vectorstore.as_retriever(search_type=
  "similarity", search_kwargs={"k": 3})`, then
  `create_retrieval_chain(retriever, document_chain)`; the response dict
  carries `input`, `context` and `answer`.
- Conversational upgrade: the history-aware retriever reformulates with
  the chat-history prompt, the answer prompt gains a history
  placeholder, and `invoke({"chat_history": [HumanMessage(...),
  AIMessage(...)], "input": "Tell me more about it!"})` answers from
  context the bare input never states.
- Full code linked to the author's `RAG_updated` GitHub repo
  (github.com/y-pred).

## RAG Chatbot — Mr Coffee AI (YouTube)

URL: https://www.youtube.com/watch?v=hskICWOKkn8 · Date: 2024-10-02 (YouTube upload) · Status: summary

The video companion to the article above: ~17 minutes of live coding the
same conversational chain over a Little Red Riding Hood PDF — history
retriever prompt, answer prompt, stuff-documents chain, retrieval chain,
invocation with a hardcoded `chat_history` — including the debugging
moments (a misspelled variable, a dict-syntax slip) left in. Best
watched after reading the article; the transcript alone reproduces the
code only approximately.

- Constructor reminder read off the docs: the history-aware retriever
  takes `llm`, `retriever`, `prompt`, and is invoked with `{"input",
  "chat_history"}`.
- Two-prompt structure: a retrieval prompt (history placeholder +
  "generate a search query…" instruction) and an answer prompt (system
  line "takes into account the context", history placeholder, user
  input).
- History mechanics demo: empty history + "what is this about?" yields
  the tale summary; then that Q&A becomes the `HumanMessage`/`AIMessage`
  history and the vague "what is the happy ending to this story?" pulls
  the woodcutter rescue — context carried across turns.
- Honest pedagogy: the video keeps its typos and `==`-vs-`:` errors in,
  a reminder to test each chain piece before composing.
- Trailer for the sequel: the next video wraps this in Streamlit with
  `session_state` as the chat history — which is exactly the following
  article below.

## How Does a History-Aware Retriever Work? — Guilherme Pereira Alves (dev.to)

URL: https://dev.to/guilhermecxe/how-a-history-aware-retriever-works-5e07 · Date: 2024-09-01 (published) · Status: summary

A microscope on task 13's key object: what `create_history_aware_
retriever` does when invoked, traced through three worked cases against
a Lilian Weng agent-survey page (WebBaseLoader → 500-char chunks →
Chroma). Its four-step conclusion is the quotable definition of the
component. Dated from the "Posted on Sep 1, 2024" byline.

- Setup fingerprint: `ChatOpenAI(model="gpt-4o-mini")`, chunking with
  `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0,
  add_start_index=True)`, and the "rephrase, don't answer" system
  instruction ("formulate a standalone question… Do NOT answer").
- The four-step workflow: substitute `input`/`chat_history` into the
  prompt → send to the LLM for a rephrased standalone question → send
  that to the vector-store retriever → return the documents.
- Three demos proving each regime: no history (pure planning chunks),
  history steering ("planning" + task-decomposition chunks after a
  steering message), and pure anaphora ("It is a way of doing what?"
  resolving to ReAct chunks via the prior exchange).
- Embedding note: the vector embedding is whatever `Chroma.
  from_documents` was given — default Chroma embeddings here, not
  OpenAI's, since none was passed.
- Vocabulary: "standalone question" = one understandable without the
  history; that rephrasing is the retriever's entire LLM job.

## Build a Context Aware Chatbot That Chats with Your Database Using Streamlit — Ashish Malhotra (Medium)

URL: https://medium.com/@mrcoffeeai/build-a-context-aware-chatbot-that-chats-with-your-database-using-streamlit-9a56ef269815 · Date: 2025-06-07 (published) · Status: summary

The deployment sequel: the conversational RAG chain served behind a
Streamlit UI whose sidebar gate (password `"SQL"`), employee-directory
table, and chat widgets are all wired to `st.session_state`, with a
Neon PostgreSQL `employees` table as the live data source. Newest
Read-or-watch item on the page and the direct model for task 15's
session handling. (Fetched via the authenticated browser.)

- Data path: `SQLDatabase.from_uri(neon_connection_uri)` →
  `SELECT id, first_name, last_name, email, hire_date FROM employees` →
  one `Document` per row (`"id: …, name: …"`) → `OpenAIEmbeddings` →
  FAISS; the same rows also render as a sidebar dataframe.
- Session pattern: `st.session_state.chat_history` seeded with
  `AIMessage("I am a bot, how can I help you?")` after the password
  gate; `st.session_state.vector_store` built once and reused.
- Chain split into two builders: `get_retreiver_chain(vector_store)`
  (history-aware retriever) and `get_conversational_rag(...)` (answer
  prompt + stuff chain + `create_retrieval_chain`), invoked with
  `{"chat_history": st.session_state.chat_history, "input": user_input}`
  and answered from `response["answer"]`.
- Rendering loop: replay `chat_history` through `st.chat_message`
  ("AI"/"Human" by `isinstance` check on `AIMessage`), read new turns
  with `st.chat_input("Type your message here...")`.
- Hygiene flag: the printed Neon connection string embeds a live
  password — never copy credentials into code; use env vars.

## Creating Retrieval Chain: RAG Chatbot with LangChain — Bhargob Deka (Level Up Coding)

URL: https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85 · Date: 2024-05-24 (published) · Status: summary

A Blog-#2 beginner tutorial building the full classic pipeline end to
end on the 2024 Canada budget PDF: PyPDFLoader → recursive splitting →
OpenAI embeddings → Chroma → MMR retriever → stuff-documents chain →
`create_retrieval_chain`, motivated by the LLM's failure on post-cutoff
budget questions. The chef/sous-chef analogy (LangChain cooks,
retriever fetches the missing ingredient) is its memorable framing.
(Fetched via the authenticated browser; archive.org holds only a JS
redirect shell.) The page shows "May 25, 2024" in the reader; the
byline metadata reads May 24 — byline used.

- Setup: `%pip install langchain langchain-openai langchain-community
  langchain-chroma chromadb langsmith`, keys from `.env`
  (`OPENAI_API_KEY`, LangSmith tracing flags).
- Splitter with explicit defaults: `RecursiveCharacterTextSplitter(
  chunk_size=1000, chunk_overlap=200, length_function=len,
  is_separator_regex=False)` — "keeps relevant information close
  together"; whole-document-as-one-vector noted as a token-limit trap.
- Retriever choice argued, not defaulted: `as_retriever(search_type=
  "mmr")` over plain similarity, because MMR's
  relevance-plus-novelty criterion diversifies hits on single-document
  search.
- Prompt hardening worth copying: the system template answers from
  `<context>{context}</context>` and, when nothing is relevant, says "I
  don't know" instead of inventing — proved by invoking with an empty
  context list.
- Finale via the Hub: `hub.pull("langchain-ai/retrieval-qa-chat")` +
  `create_stuff_documents_chain` + `create_retrieval_chain`, invoked
  with `{"input": ...}`, answer at `response["answer"]`.

## Migrating from ConversationalRetrievalChain — intranet label → LangChain overview docs

URL: https://docs.langchain.com/oss/python/langchain/overview · Date: unknown · Status: summary

Note: the intranet label promises a migration guide, but the link lands
on the generic **LangChain overview** page (same URL as the
"LangChain overview" reference below) — there is no migration content
at this destination. Summarised as the actual page: the current
positioning of the framework as `create_agent`, a minimal configurable
agent harness ("Agent = Model + Harness") composed of model, tools,
prompt and middleware. No date header or byline, hence unknown.

- Positioning delta vs the IBM article: LangChain the library is now
  framed as the customisable harness, Deep Agents as batteries-included
  (filesystem, subagents, compression), LangGraph as low-level
  orchestration — pick by how much control you need.
- Minimal agent fingerprint: a plain `get_weather(city)` function plus
  `create_agent(model="openai:gpt-5.5", tools=[get_weather],
  system_prompt="You are a helpful assistant")`, invoked with
  `{"messages": [{"role": "user", "content": ...}]}`.
- Ecosystem constants: one model interface across providers
  (`create_agent` model strings like `google_genai:…`,
  `huggingface:…`), LangSmith for trace/debug/evaluate, LangGraph
  underneath for durable execution and human-in-the-loop.
- For the actual migration content the label implies, use the "RAG
  Updated" article above instead.

## Understanding MessagePlaceholder Vs RunnableWithMessageHistory — Ashish Malhotra (Medium)

URL: https://medium.com/@mrcoffeeai/understanding-messageplaceholder-vs-runnablewithmessagehistory-f2eaf41ed0d0 · Date: 2024-06-22 (published) · Status: summary

A short conceptual contrast for task 15's vocabulary: history management
(`RunnableWithMessageHistory`, the stateful wrapper that keeps past
exchanges and feeds them back) versus templating (`MessagesPlaceholder`,
the slot in a prompt where a message list is injected). The framing is
right, but both code samples are pseudo-code, not the real API —
`RunnableWithMessageHistory()` takes no such constructor and has no
`add_message`/`run_task`, and the `MessagePlaceholder` f-string demo is
invented; the real patterns are `RunnableWithMessageHistory(runnable,
get_session_history)` and `MessagesPlaceholder("chat_history")` inside
`ChatPromptTemplate.from_messages`. (Fetched via the authenticated
browser.)

- One-line split: history class = remembers and reasons over the past;
  placeholder = a named hole in a prompt filled at invoke time.
- Correct pairing the article gestures at: placeholder for dynamic
  insertion *within* a prompt, history wrapper for persistence *across*
  turns — combine them (placeholder naming the history key the wrapper
  populates).
- Streamlit bridge: the article notes placeholders pair with an external
  store like `st.session_state` — exactly the previous article's
  architecture.
- Read the prose, not the snippets: treat both `Example Usage` blocks
  as illustrations and copy constructor calls from the reference docs
  below.

---

## References

Link-only, per the intranet's "References" section. All resolved to live
URLs on 2026-10-13 (all 26 return HTTP 200).

| Resource | URL |
|----------|-----|
| Text Generation | https://huggingface.co/tasks/text-generation |
| LangChain overview | https://docs.langchain.com/oss/python/langchain/overview |
| Hugging Face integrations | https://docs.langchain.com/oss/python/integrations/providers/huggingface |
| Hugging Face local pipelines integration | https://docs.langchain.com/oss/python/integrations/llms/huggingface_pipelines |
| ChatMistralAI integration | https://docs.langchain.com/oss/python/integrations/chat/mistralai |
| SystemMessage | https://reference.langchain.com/python/langchain-core/messages/system/SystemMessage |
| HumanMessage | https://reference.langchain.com/python/langchain-core/messages/human/HumanMessage |
| PromptTemplate | https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate |
| ChatPromptTemplate | https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate |
| load tools | https://reference.langchain.com/python/langchain-community/agent_toolkits/load_tools/load_tools |
| create tool calling agent | https://reference.langchain.com/python/langchain-classic/agents/tool_calling_agent/base/create_tool_calling_agent |
| AgentExecutor | https://reference.langchain.com/python/langchain-classic/agents/agent/AgentExecutor |
| Pandas Dataframe integration | https://docs.langchain.com/oss/python/integrations/tools |
| create pandas dataframe agent | https://langchain-doc.readthedocs.io/en/latest/modules/agents/agent_toolkits/pandas.html |
| handle parsing errors | https://github.com/langchain-ai/langchain/issues/14947 |
| DirectoryLoader | https://reference.langchain.com/python/langchain-community/document_loaders/directory/DirectoryLoader |
| RecursiveCharacterTextSplitter | https://reference.langchain.com/?lang=python |
| HuggingFaceEmbeddings | https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub |
| FAISS | https://reference.langchain.com/python/langchain-classic/vectorstores/faiss |
| from documents | https://reference.langchain.com/python/langchain-community/vector-stores |
| as retriever | https://reference.langchain.com/python/langchain-community/vector-stores |
| create stuff documents chain | https://reference.langchain.com/python/langchain-classic/chains/combine_documents/stuff/create_stuff_documents_chain |
| MessagesPlaceholder | https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder |
| create history aware retriever | https://reference.langchain.com/python/langchain-classic/chains/history_aware_retriever/create_history_aware_retriever |
| create retrieval chain | https://reference.langchain.com/python/langchain-classic/chains/retrieval/create_retrieval_chain |
| RunnableWithMessageHistory | https://reference.langchain.com/python/langchain-core/runnables/history/RunnableWithMessageHistory |

Reference notes: "Migrating from ConversationalRetrievalChain"
(Read-or-watch) and "LangChain overview" here land on the same overview
URL. "Pandas Dataframe integration" lands on the generic Tool
integrations index, not a pandas page. "create pandas dataframe agent"
lands on readthedocs for LangChain 0.0.107 (old but openly readable —
the task-7 fallback while Kaggle is login-walled). "handle parsing
errors" is GitHub issue #14947 (`An output parsing error occurred` even
with `handle_parsing_errors=True`). "RecursiveCharacterTextSplitter"
resolves to the reference-docs homepage, not the splitter page — a
stale deep link. "from documents" and "as retriever" both land on the
vector-stores index (method names, not pages).

---

## Ingestion Notes

- All 41 rltoken links (15 read-or-watch + 26 references) were resolved
  inside the authenticated CDP browser; the session stayed alive the
  whole run (dashboard reachable, no re-login needed).
- Fetch split for the 15: 4 direct (IBM, KodeKloud, dev.to,
  3 docs.langchain.com pages), 1 via archive.org (Vijaykumar), 6 via the
  authenticated browser (Terry Cho, 3 mrcoffeeai Medium posts, LevelUp,
  Kaggle shell), 2 YouTube transcripts via `youtube-transcript-api`.
  Zero fabricated; 1 paywalled (Kaggle).
- Medium serves headless requests 403 and archive.org holds no snapshot
  for 4 of the 5 Medium articles here — the authenticated CDP render
  returned full text for all of them. Exact resolved URLs (with full
  story-ID hashes) are recorded above; truncated Medium URLs render a
  site 404.
- Date labels: byline dates marked `(published)`; the IBM page dated
  from its `article:published_time` meta (its `Last-Modified` header
  shows today's CDN date and was rejected); docs.langchain.com sends no
  `Last-Modified` and its pages print no date → `unknown`; KodeKloud
  notes print no date → `unknown`; LevelUp shows May 25 in the reader
  but May 24 in the byline meta — byline used. Kaggle shows only "2y
  ago" → `unknown`.
- The intranet "Migrating from ConversationalRetrievalChain" label does
  not match its destination (generic overview); the actual migration
  content lives in the "RAG Updated" article. The intranet "Build a RAG
  application" label lands on the Deep Agents RAG guide, not classic
  RAG — flagged in both summaries.
- The MessagePlaceholder-vs-history article's code blocks are
  illustrative pseudo-code, not runnable API — flagged above with the
  real constructor forms.
- The Streamlit article prints a live Neon database password inline —
  noted, not reproduced.
- Copyright: summaries are own-words; raw HTML/texts/transcripts live
  only in `/tmp/opencode/` (`lc_*`), never committed.

---

## Quiz Hooks

- LangChain — open-source orchestration framework: modular abstractions
  (models, prompts, chains, indexes, memory, tools, agents) chained so
  one step's output feeds the next.
- LLM vs chat model — completion models answer from a prompt, chat
  models converse; LangChain abstracts both behind one interface.
- `invoke` / `batch` / `stream` — one prompt sync, many prompts at
  once, tokens as generated; each has an async twin (`ainvoke`…).
- Async payoff — 10 prompts in ~1.9s via `asyncio.gather(ainvoke)`
  vs ~5.8s looping `invoke`.
- Prompt anatomy — instruction + context + input data + output
  indicator; context moves performance most.
- Temperature / top-p — low for factual answers, high for diverse
  generation; change one, not both.
- `PromptTemplate` — named `input_variables` holding context,
  instructions, few-shot examples and output format for reuse.
- `Document` — `page_content` + `metadata` (+ optional `id`); metadata
  keeps source/page/`start_index` for citations.
- Chunking — `RecursiveCharacterTextSplitter(chunk_size=1000,
  chunk_overlap=200, add_start_index=True)`; 107 PDF pages → 516 chunks
  in the worked example.
- Embeddings — text → same-dimension numeric vectors; query and chunks
  must share the model/dimensions.
- `as_retriever` — turns a vector store into a `Runnable`;
  `search_type` "similarity" (default), "mmr" (diverse), or
  "similarity_score_threshold" (cut weak hits).
- `create_stuff_documents_chain` — stuffs retrieved `Document`s into a
  `{context}`/`{input}` prompt and calls the LLM.
- History-aware retriever — substitutes `input`/`chat_history` into a
  "rephrase, don't answer" prompt, sends the standalone question to the
  vector store, returns documents.
- Standalone question — one understandable without the chat history;
  the history-aware retriever's entire LLM job.
- `create_retrieval_chain(retriever, document_chain)` — one-step
  retrieve-then-answer; response dict holds `input`, `context`,
  `answer`.
- DePRECATED pair — `ConversationalRetrievalChain`/`RetrievalQA`
  deprecated since 0.1.17, removal in 0.3.0; use the `create_*` LCEL
  functions.
- `MessagesPlaceholder("chat_history")` — the prompt slot where the
  message list is injected at invoke time.
- `RunnableWithMessageHistory` — stateful wrapper persisting turns
  across invocations; the placeholder names the key it populates.
- "I don't know" prompt — system template refusing to invent when
  `<context>` lacks the answer; proved with an empty-context invoke.
- ReAct — LLMs interleaving reasoning traces with tool actions against
  knowledge bases.
- Prompt injection — pasted "ignore the above…" hijacking output; no
  delimiter trick fully prevents it (nor its RAG cousin, indirect
  injection via retrieved chunks).
- MMR — relevance-plus-novelty retrieval for diverse hits on
  single-document search.
