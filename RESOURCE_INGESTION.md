# Resource Ingestion Workflow

For every Holberton ML project, scrape the intranet Resources, summarize each
item, and capture the crucial bits. Output goes to `<project>/RESOURCES.md`.

## When It Runs

- Once per project, at the first task's Prepare step (AGENTS.md Step 1b).
- Later tasks in the same project skip ingestion and use the existing
  `RESOURCES.md`.
- Never ingest mid-task. For an in-flight project (e.g. classification at
  task 15), wait until the task is done, then ingest as a standalone session
  before the next task's Prepare.
- Weekly cadence: 2-3 backlog projects (newest first) plus the current
  project, until the backlog since the first commit is exhausted.

## The Loop

1. Run the session health check (below). Stop and ask the user for a FIDO
   sign-in if the session is dead.
2. Open the project page, snapshot, extract the Resources lists.
3. Classify every resource by section and type (below).
4. Walk the rltoken links in the same browser context.
5. Summarize articles, pull transcripts for videos.
6. Write `RESOURCES.md` for the project.

## Access

- Base URL: `https://intranet-dlh.hbtn.io` (DLH campus; not
  `intranet.hbtn.io`).
- Snapshot rltoken URLs are relative (`/rltoken/...`). Prefix the base before
  following. The links resolve only inside the authenticated browser context.
- Use the playwright MCP context only. It owns the MSAL session; no other
  tool has intranet auth.

## Session Health Check

Run first, before snapshotting:

- Navigate to `https://intranet-dlh.hbtn.io/projects/current`.
- If the page redirects to `/auth/login` or shows "Sign in with Microsoft",
  the session is dead. Stop and ask the user for one interactive FIDO
  sign-in. Do not walk rltokens on a dead session; every link returns the
  login page and the summaries come out garbage.
- The active profile is the one with the freshest `Default/Cookies` mtime,
  not the newest profile directory. Sessions do not survive across days;
  re-check at every run.

## Classification

The project-page snapshot has three resource sections:

- **Read or watch** — full treatment: fetch and summarize, or pull the video
  transcript.
- **Definitions to skim** — one-line glossary entry each.
- **References** — link only. Verify the URL resolves; no summary.

Within Read or watch, split by destination: article / YouTube / other video.

## Articles

- Fetch with webfetch, or system `python3` (bs4 + requests).
- Paywall policy: detect signals (body under 300 words ending mid-sentence,
  "members-only", "create a free account"). On a paywall, in order:
  1. `https://web.archive.org/web/2/<url>`
  2. retry in the playwright context (may carry a Medium cookie)
  3. mark `paywalled` in RESOURCES.md with title, URL, and the first
     paragraph only.
- Never fabricate a summary from a paywall stub.
- Crucial bits: key concepts, formulas, vocabulary, anything the spec or
  checker fingerprints (quirks, exact error strings, dtype/shape demands).

## Recency (Date line)

Every Read-or-watch resource gets a `Date:` line in RESOURCES.md. Per type:

- **YouTube**: upload date via `yt-dlp` (in `my-venv`) —
  `extract_info(url, download=False)['upload_date']`. Label
  `(YouTube upload)`.
- **Articles with a byline**: the publication date printed on the page.
  NEVER use `Last-Modified` headers on WordPress/CDN sites — they reflect
  cache regeneration (PyImageSearch showed 2026 for a 2019 article).
- **Docs sites** (tensorflow.org, keras.io): `curl -sI -L <url>` and read
  `Last-Modified`. Label `(page last modified)`.
- **Wikipedia**: the "last edited" date in the page footer. Label
  `(last edited)`.
- No date obtainable: write `Date: unknown`. Never guess.

Note the distinction in the label: YouTube and byline dates are publication
dates; docs and Wikipedia dates are recency-of-page, not content age.

## YouTube

- `youtube-transcript-api` first (fast, plain text).
- On `NoTranscriptFound`, `CouldNotRetrieveTranscript`, or age-restriction:
  fall back to `yt-dlp --write-auto-subs --skip-download --sub-format vtt`,
  then strip timestamps.
- If both fail (captions disabled, region-locked): mark `failed-transcript`
  with the title and duration so the user can watch manually.
- Do not run full speech-to-text for YouTube videos.

## Other Videos

- Download with yt-dlp, transcribe locally (noScribe bundle or
  faster-whisper).
- The speech-to-text stack is not installed by default. Install it only when
  the first non-YouTube video appears; verify Python 3.14 wheel availability
  first (ctranslate2 lags on new Pythons).
- noScribe models are absent (~1.6 GB download):
  `faster-whisper-large-v3-turbo` from `mobiuslabsgmbh`.

## Copyright (public repo)

This repo is public. Only derived content may be committed:

- Summaries in your own words: one paragraph plus 3-6 bullets per resource.
- No verbatim text longer than one sentence.
- Raw transcripts never enter the repo. Keep them in `.playwright-mcp/`
  (gitignored) or `/tmp/opencode/`.
- Paywalled articles: mark `paywalled`; commit title, URL, first paragraph,
  no summary that could only come from paid access.

## Output Format

`<project>/RESOURCES.md`, one block per resource:

```markdown
## <Title>

URL: <final url>  ·  Date: YYYY-MM-DD (<label>)  ·  Status: summary | paywalled | failed-transcript | deferred

One-paragraph summary in own words.

- crucial bit 1
- crucial bit 2
- (3-6 bullets)
```

End the file with a quiz-hook bank for the separate Kahoot workflow:

```markdown
## Quiz Hooks

- <vocabulary term> — one-line definition
- <formula> — what it computes, why
- <concept> — one-line takeaway
```

The end-of-project README (per STUDY_GUIDE_TEMPLATE.md) links resources by
URL only and points to RESOURCES.md for detail. Do not duplicate summaries
into the README.

## Budget

- Target 90 minutes per project; hard cap 25 resources per session.
- Process Read or watch first, then Definitions to skim. References are
  link-only; defer them if time runs out, marked `deferred`.
- If a session passes two hours, stop and resume next session. Never rush a
  summary.
- Weekly target: 3-4 projects (2-3 backlog + current), about 5-6 hours.

## Privacy

`.playwright-mcp/` snapshots carry the user's name, scores, and correction
URLs. Never paste raw snapshot content into shared channels without
redacting.

## Verification

- Every Read-or-watch resource has a summary or an explicit status marker.
  No silent drops.
