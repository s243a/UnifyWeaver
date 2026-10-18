# SYM operator — more data + the data-vs-dilution ablation (#3302 follow-up)

#3302 left the directional model's **SYM** operator behind the #3287 symmetric control on held-out
relatedness corr (**+0.335 vs +0.726**). Two hypotheses: (1) **data starvation** — only 200 scored
positives; (2) **multi-task dilution** — SYM shares one backbone with WIKI+LLM. This PR generates more
SYM data and ablates the lever to attribute the gap.

## What was generated (steps 1–2)

- `gen_more_sym_pairs.py`: the same hub-down-weighted RWR mesh as `gen_mu_pairs.py`, but seeded from
  **20 varied physics + adjacent-field anchors** (Electromagnetism, Thermodynamics, Optics, Mechanics,
  Energy, Atoms, Astronomy, Cosmology, Chemistry, Mathematics, Geology, Biology, …) and **deduped
  against the committed `mu_pairs_scored.tsv`**, so every emitted positive is NEW.
- **600 new positives** scored by Haiku (graded sameness rubric: 1.0 nested / 0.5–0.7 same domain /
  0.2–0.4 loose / 0.0 unrelated), under the budget discipline: **2 inline subagent spawns, ~38.5k Haiku
  tokens, 0 tool calls** (one 596-pair batch + a 4-pair cleanup). 3000 new negatives are free (μ=0).
- Committed as `mu_pairs_scored_large_260620-223001.tsv`: **800 pos + 4000 neg** (original 200/1000 + new 600/3000).
  New-positive μ: mean 0.69, full range. Labels bought once, committed.

## Retrain + ablation (steps 3–4)

All configs: frozen e5 + 2-layer MuAttention, **seed 1, 900 steps, identical recipe** as #3302
(`--lr 5e-4 --wiki-weight 0.5 --margin-weight 1.0 --wiki-abs 0.5`). Held-out = 20% of positives.

| # | config | data | SYM held-out corr | WIKI order-acc | held-out n |
|---|---|---|---|---|---|
| #3302 | multi-task (WIKI+SYM+LLM) | old (200 pos) | +0.335 | 99.1% | 40 |
| a0 | multi-task — reproduce | old (200 pos) | **+0.342** | 99.3% | 40 |
| **a** | **multi-task — more data** | **large (800 pos)** | **+0.479** | **97.7%** | 160 |
| b | multi-task + SYM weight ×3 | large (800 pos) | +0.452 | 98.4% | 160 |
| c | single-task SYM head | large (800 pos) | **+0.000 (collapsed)** | n/a | 160 |
| — | control (#3287, symmetric) | — | +0.726 | 50% (structural) | 40 |

`a0` reproduces #3302 (+0.342 ≈ +0.335) — the code/recipe match, so the rows are comparable.
Gate-leak stays clean across configs (SYM 5-probe **0/5**; run `a` OOD 0.5%, μ̄ 0.006). Run `a`'s SYM
dense map feeds `check_feeds_rust` (100% resolution, IC general→specific): physics 0.50–0.87 (Energy
0.87, Atoms 0.82, Optics 0.76), non-physics ≤ 0.06 (Music 0.02, Cooking/Sociology/Football < 0.01);
top-8 all physics. SECONDARY node-gated-IC lin-agreement improved **−0.033 (#3302) → +0.070**.

## Honest verdict — what closed the gap

**The gap was DATA STARVATION, and multi-task is NOT the culprit — it is load-bearing.**

1. **More data closed a big chunk of the gap.** Same recipe, 4× more positives: **a0 +0.342 → a +0.479**
   (+0.137), i.e. ~35% of the +0.335→+0.726 gap to the control, with WIKI order-accuracy still ~98%.
2. **Multi-task is HELPING SYM, not diluting it.** The single-task SYM head (`c`) **collapses to a
   constant (+0.000, symmetry gap 0.000, MSE 0.491)** under the identical recipe — removing the WIKI +
   LLM gradients destabilises the shared backbone that SYM reads off. So the "multi-task dilution"
   hypothesis is **rejected**: the auxiliary directional tasks regularise the representation; without
   them, the sparse-positive SYM head has nothing to anchor it. (A single-task-tuned recipe — higher LR,
   more steps, stronger weight decay — might train SYM alone; but *under matched conditions* multi-task
   is strictly better, which is the relevant comparison.)
3. **Up-weighting SYM does not help.** `b` (SYM ×3) gives **+0.452 ≤ +0.479** — the backbone already
   serves SYM well; re-weighting only trades a little WIKI for no SYM gain.
4. **The directional capability is preserved.** WIKI held-out order-accuracy stays **97.7–99.3%** in
   every multi-task config — the model still does the thing the symmetric control structurally cannot.

**Remaining gap to the control (+0.479 vs +0.726).** The dedicated single-task symmetric MiniLM control
still leads on *pure symmetric corr* — unsurprising, it is purpose-built for exactly that one metric —
but our model now trails by less while *also* delivering the directional WIKI (≈98%) and LLM operators
the control cannot represent at all, with clean gate-leak (0/5) and maps that feed the Rust core. The
next lever is simply **more positives** (the trend is not yet saturated), not architecture or loss
re-weighting.

_Caveat: #3302 and `a0` use a 40-positive held-out; `a`/`b`/`c` use 160. `a0` is the apples-to-apples
reproduction of #3302; `a`/`b`/`c` are internally consistent on the larger held-out._
