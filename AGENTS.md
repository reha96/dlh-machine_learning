# DLH Workflow Contract

This file governs every DLH (Holberton ML) task in this repo. Follow the 5-step
rhythm in order. The goal is one consistent loop: explain, template, student
codes, verify, checker.

## The Task Rhythm

1. **Prepare**
   - Fetch the reference implementations from both known-good forks:
     `Teheremiti/holbertonschool-machine_learning` and
     `dassantoss/holbertonschool-machine_learning` (raw.githubusercontent.com).
   - Inspect the project's data files (shapes, dtypes, keys) if present.
   - Create the template file: full class skeleton from the previous task,
     intranet-spec docstrings, `pass`/stub bodies for the new method, header
     comment `(Based on N-*.py)`. **No solution code in the template.**
2. **Explain** — the 3-tier ladder (below). Concept first, always.
3. **Student codes** — the student fills the stubs step by step. Answer
   questions; never put solution code into a concept explanation unless asked.
4. **Verify** — run the verification battery (below) before the student submits.
5. **Check** — student runs the official checker; diagnose any failure against
   the reference forks.

## The 3-Tier Explanation Ladder

Explain every concept in this order, no skipping tiers:

1. **ELI5** — one everyday analogy, no jargon, one image (e.g. "cost is the
   measure of how wrong the guess is").
2. **Intuition** — plain-language mechanics: what it does, why each piece
   exists, shapes and their meaning, the project's vocabulary.
3. **Math** — exact formulas, why each term is there, shape alignment
   (broadcasting, the role of m, why W is a row / b a column).

## Question-Handling Contract

- Answer "why" before "what".
- Concept explanations stay code-free unless the student asks for code.
- Every code review includes a check of the student's comments for accuracy.

## Conventions and Lessons

- Error messages must match the intranet spec character-for-character.
- Type checks before value checks; validation order per the spec.
- Student comment style: short plain-language `#` comment per block.
- Reference-fork quirks are checker fingerprints — match them, don't "fix"
  them: `1.0000001 - A` in cost, `range(iterations + 1)` loop, `argmax + 2`
  (PCA), `np.random.standard_normal` == `np.random.randn` (both N(0,1)).
- **Never leave module-level executable code** in task files. Import must be
  silent and fast (<1s). Test harnesses go in a separate file or not at all.
- Data files go in `.gitignore` with explicit paths (mnist txt files, npz
  datasets). Never commit them.
- When a project's tasks are done, rewrite its README per
  `STUDY_GUIDE_TEMPLATE.md`.
- Verification environment: **always** run tests and code with `my-venv`
  (`my-venv/bin/python`, numpy 2.4.4, matplotlib 3.11.0). Never system
  `python3` for task code, tests, or verification runs.

## Verification Battery

Run before the student submits, for every task file:

- [ ] `pycodestyle` clean
- [ ] compile + import smoke test: no module-level side effects, no output,
      no `RuntimeWarning` under `python3 -W error`
- [ ] attribute shapes and initial values match the spec
- [ ] every exception type and message exact
- [ ] behavioral anchors (cost decreases after training, sigmoid output in
      (0,1), threshold at 0.5, gradient matches finite differences)
- [ ] for graph tasks: mocked-`plt` call order (plot, xlabel, ylabel, title,
      show)

## Lessons Log

Append new lessons here as they are learned; the file is meant to grow.

- 2026-08-07: verification and test runs always use `my-venv`, never system
  `python3` (keeps numpy 2.4.4 / matplotlib consistent with the checker's
  tooling and available everywhere).
- 2026-08-07: a test harness pasted at the bottom of a task file runs on every
  `import` (7-neuron.py: 20s import + stdout pollution). Harnesses live
  outside task files, or in a `__main__` guard in a scratch file.
