# Routed-queries loop: the first confirmed end-to-end filing win

**Result: rank → route → judge → place beats rank-alone, confirmatory.** On the standing 1200-query
manifest (sha `fcf5e1d6…`, e5 @ K=100, title-equivalence grading), routing low-margin queries
(t = 0.02 → 695/1200 routed) to a cheap judge (Haiku, title-only 10-item menus) yields

- policy R@1 **0.258** vs no-routing baseline **0.203**
- **delta +0.055, paired bootstrap 95% CI [+0.037, +0.072] — EXCLUDES ZERO.**

This is the filing program's first intervention with a confirmatory CI excluding zero (every
ranker-side blend was null). It lands where the ceilings said the value lives: the perfect-judge
ceiling at (t=0.02, N=10) is 0.389 (+0.186); Haiku captured ~30% of that headroom, rescuing
193 of the 351 menus that contained the true folder (55% rescue-pool conversion; 87/695 nulls).

## Protocol (routed_queries.py)

- `ceilings`: perfect-judge upper bounds over (t × N). Baseline 0.203; 0.389 @ t=0.02/N=10;
  0.512 @ t=0.03/N=20 — routing, not ranking, is where filing performance lives.
- `emit`: outcome-blind task JSONL (menus in e5 order, truth never marked) → private `~/mu_data`.
- Judge spend: 4 parallel Haiku subagents (the roster's cheap judge), strict `{qid, pick|null}`
  contract, 695/695 scored.
- `score`: fail-closed ingest (manifest match, each routed qid exactly once, range-checked),
  policy R@1 with paired per-query bootstrap (2000 draws) on Δ(correct) vs the no-routing
  baseline.

## Honest caveats

- Ground truth = recorded placement; a judge pick that is topically defensible but differs from
  the recorded folder scores as wrong — 0.258 is a LOWER bound on subjective quality.
- Single judge pass; per-chunk null rates varied (9/15/49/14 across the 4 subagent chunks) —
  judge-draw variance is not in the CI (which is over queries). A second judge/seed pass and a
  stronger judge (sonnet-tier) are the obvious next spends.
- Menus are title-only. Judge cards / folder lineage context (the §7 candidate-lineage machinery)
  are unexploited headroom, as is menu size (ceiling at N=20 is 0.453 for this t).
- Thresholds t, N chosen from the descriptive ceiling grid before judging, but not preregistered.

## Where this leaves the program

The deployed loop is filing_assistant (e5 @ K=100 + margin gate) + judge escalation, worth a
confirmed +0.055 R@1 today. Remaining levers, in ceiling order: stronger/contextual judge
(up to +0.13 more at this t/N), larger menus (N=20 ceiling 0.512 at t=0.03), DAG completion
(the `missing` stratum), and gap-directed training (REPORT_hybrid_candidates §B′/B″).
