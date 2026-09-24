# AGENTS.md — working on handoff

handoff trains a personal cheap/strong router on the user's own AI chat history, on top of Laya's frozen encoder.

## Map
```
src/handoff/
  extract.py   history → only the human's prompts; redact() blanks secrets BEFORE anything is written
  label.py     policy-driven labelling (claude -p / manual / csv); read_jsonl skips corrupt lines
  metrics.py   plain-Python scoring: confusion, gate_table, pick_gate, split3 (easy to test)
  train.py     embed with Laya → logistic head; l2 by 5-fold CV; gate by out-of-fold preds on dev; report on test
  serve.py     127.0.0.1 service: token + Host + Content-Type checks, lazy load, idle unload, weights_only load
  cli.py       every command; route fails open to "strong"; gate --cheap-share; pidfile-based stop
  paths.py     ~/.handoff layout (0700/0600), HANDOFF_HOME sanity check, token()
  policy.md    default policy (package data)
examples/claude-code-router/handoff-router.js   CCR custom router (fails open)
tools/make_visuals.py                           regenerates docs/ images from ~/.handoff/model/report.json
tests/                                          pytest; never loads the real model or touches real ~/.handoff
```

## Rules
1. **Never commit personal data.** `*.jsonl` is git-ignored except `tests/fixtures/`. No real prompts, labels,
   tokens or reports in the repo. Fixtures build fake secrets at runtime.
2. **The test set is sacred.** Nothing may be chosen (threshold, gate, hyper-parameter) using test-set numbers.
3. **Fail open.** Any path that can't decide must answer `strong`.
4. **The service stays local and authenticated.** Keep the 127.0.0.1 bind, token, Host and Content-Type checks.
5. **Numbers in README/docs come from a real `handoff train` run** — regenerate visuals with
   `python tools/make_visuals.py` after retraining, never hand-edit numbers.
6. Keep dependencies to `laya` (+ pytest for tests).

## Verify
```bash
pytest -q                                   # must stay green
handoff train && handoff report             # real run (needs labelled data in ~/.handoff)
handoff route "hello" && handoff stop       # service round-trip
```
