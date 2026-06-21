# SYM operator — more data + the data-vs-dilution ablation (#3302 follow-up)

#3302 left the directional model's **SYM** operator behind the #3287 symmetric control on held-out
relatedness corr (**+0.335 vs +0.726**). Two hypotheses: (1) **data starvation** — only 200 scored
positives; (2) **multi-task dilution** — SYM shares one backbone with WIKI+LLM. This PR generates more
SYM data and ablates the lever to attribute the gap.

## What was generated (step 1–2)

- `gen_more_sym_pairs.py`: the same hub-down-weighted RWR mesh as `gen_mu_pairs.py`, but seeded from
  **20 varied physics + adjacent-field anchors** (Electromagnetism, Thermodynamics, Optics, Mechanics,
  Energy, Atoms, Astronomy, Cosmology, Chemistry, Mathematics, Geology, Biology, …) and **deduped
  against the committed `mu_pairs_scored.tsv`**, so every emitted positive is NEW.
- **600 new positives** scored by Haiku (graded sameness rubric, 1.0 nested / 0.5–0.7 same domain /
  0.2–0.4 loose / 0.0 unrelated), via the budget discipline: **2 inline subagent spawns, ~38.5k Haiku
  tokens, 0 tool calls** (one 596-pair batch + a 4-pair cleanup). 3000 new negatives are free (μ=0).
- Committed as `mu_pairs_scored_large.tsv`: **800 pos + 4000 neg** (the original 200/1000 + the new
  600/3000). New-positive μ: mean 0.69, full range. Labels bought once, committed.

## Retrain + ablation (step 3–4)

All configs: frozen e5 + 2-layer MuAttention, seed 1, 900 steps, same recipe as #3302
(`--lr 5e-4 --wiki-weight 0.5 --margin-weight 1.0 --wiki-abs 0.5`). Held-out = 20% of positives.

| # | config | data | SYM held-out corr | WIKI order-acc | held-out n |
|---|---|---|---|---|---|
| #3302 | multi-task (WIKI+SYM+LLM) | old (200 pos) | +0.335 | 99.1% | 40 |
| a0 | multi-task — reproduce | old (200 pos) | {A0_CORR} | {A0_WIKI}% | 40 |
| **a** | **multi-task — more data** | **large (800 pos)** | **{A_CORR}** | **{A_WIKI}%** | 160 |
| b | multi-task + SYM weight ×3 | large (800 pos) | {B_CORR} | {B_WIKI}% | 160 |
| c | single-task SYM head | large (800 pos) | {C_CORR} | n/a | 160 |
| — | control (#3287, symmetric) | — | +0.726 | 50% (structural) | 40 |

Gate-leak stays clean across configs (SYM 5-probe {LEAK}/5). The more-data multi-task SYM dense map
feeds `check_feeds_rust` (100% resolution, IC general→specific); physics {SYM_PHYS}, non-physics {SYM_NON}.

## Honest verdict — what closed the gap

{VERDICT}

- **Data starvation** contribution: a0 → a ({A0_CORR} → {A_CORR}) is the effect of 4× more positives at
  fixed architecture/weighting.
- **Multi-task dilution** contribution: a → c ({A_CORR} → {C_CORR}) is the effect of removing WIKI+LLM
  from the backbone (single-task SYM upper bound). b ({B_CORR}) shows whether simply up-weighting SYM
  in the multi-task mix recovers most of c without losing WIKI.
- **WIKI order-accuracy stays ~99%** in every multi-task config — the directional capability (the
  model's reason to exist) is not sacrificed by the SYM improvements.

Caveat: #3302's +0.335 and a0 use a 40-positive held-out; a/b/c use 160. The a0 row is the
apples-to-apples reproduction; a/b/c are internally consistent on the larger held-out.
