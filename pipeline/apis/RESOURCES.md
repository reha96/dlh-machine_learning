# Data Collection - APIs — Resources

Project: Data Collection - APIs (intranet #2372, `pipeline/apis`).
Only one Read-or-watch item on the project page; task bodies link the two
APIs used. No Definitions-to-skim or References sections on the page.

## Requests: HTTP for Humans — Quickstart

URL: https://docs.python-requests.org/en/latest/user/quickstart/  ·  Date: 2026-08-17 (page last modified)  ·  Status: summary

The official Requests quickstart shows the whole request/response cycle in a
few calls: `requests.get/post/put/delete/head/options`, query params via the
`params` dict, and reading the body as text, bytes, or parsed JSON with
`r.json()`. It stresses checking success explicitly with `r.status_code` or
`r.raise_for_status()`, since decodable JSON does not mean success, and
reading response metadata from the case-insensitive `r.headers` dict.

- `requests.get(url)` returns a Response; `r.json()` parses a JSON body
- success is NOT implied by `r.json()` — check `r.status_code` or call `r.raise_for_status()`
- `r.headers` is case-insensitive; rate-limit info arrives as headers
- network problems raise `requests.exceptions.RequestException` subclasses (`ConnectionError`, `Timeout`, `HTTPError` via `raise_for_status()`)
- pass `timeout=` on nearly every request so code cannot hang forever
- always send query strings via `params={...}`, never by hand-building URLs

## SWAPI — The Star Wars API (Holberton mirror)

URL: https://swapi-api.hbtn.io/  ·  Date: unknown  ·  Status: summary

Task reference for tasks 0 and 1 (intranet Swapi API rltoken resolves here).
Live probe: `/api/starships/` holds 36 ships, `/api/species/` holds 37
species, both in pages of 10 with `count/next/previous/results` envelope and
`next` holding the following page URL (`null` on the last page).

- starship record keys include `name` and `passengers` (a string, not a number)
- `passengers` raw values mix plain numbers, one comma-formatted value (`843,342`), and the sentinels `n/a` / `unknown`
- species record keys include `name`, `classification`, `designation`, and `homeworld` (a planet URL, possibly `null`)
- sentience is flagged by the word `sentient` in `classification` OR `designation`
- use this mirror host, not the retired `swapi.dev` seen in some forks

## REST API endpoints for users — GitHub Docs

URL: https://docs.github.com/en/rest/users  ·  Date: unknown  ·  Status: deferred (reference, link-only; verified resolves HTTP 200)

Task reference for task 2 (intranet GitHub API rltoken resolves here). Live
probe of `api.github.com`: unauthenticated core quota is 60 requests/hour;
`GET /rate_limit` reports it without consuming quota; a missing user returns
404 with body `{"message": "Not Found", ...}`; a found user carries
`X-Ratelimit-Reset` (epoch seconds) in the response headers and a `location`
string in the JSON body.

- unauthenticated quota: 60 requests/hour — analyze first, then code
- missing user → 404, body message `Not Found`
- rate-limited → 403 with `X-Ratelimit-Reset` epoch header; minutes left = (reset − now) / 60
- user JSON has a `location` field (may be `null`)

## The Feynman Learning Technique

URL: https://fs.blog/feynman-learning-technique/  ·  Date: 2021-02-22 (publication)  ·  Status: summary

The "explain to anyone" link behind the Learning Objectives. Essay on
Feynman's four-step study loop: teach the concept simply, spot the gaps,
organize and simplify, then transmit to a real person. Knowing a name is not
knowing the thing — an idea is learned when it can be rebuilt from scratch in
plain words.

- step 1: teach it to a sixth-grader (or a rubber duck), simple words only
- step 2: gaps found while teaching mark exactly what to re-study
- step 3: organize the notes into a story told end to end, out loud
- step 4 (optional): transmit to someone unfamiliar; their questions finish the lesson
- jargon that cannot be unpacked is a mask for not understanding

## Quiz Hooks

- `requests.get` + `r.json()` — fetch a URL and decode its JSON body
- `r.status_code` / `raise_for_status()` — JSON that decodes is not proof of success
- `r.headers` case-insensitive — where `X-Ratelimit-Reset` arrives
- SWAPI pagination — follow `next` until it is `null`
- `passengers` cleaning — commas out, skip `n/a` / `unknown`, then compare as numbers
- `sentient` in `classification` OR `designation` — the species filter
- `X-Ratelimit-Reset` — epoch seconds; `(reset − now) // 60` = minutes to print
- Feynman step 1 — explain it simply enough for a sixth-grader
