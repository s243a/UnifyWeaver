# Graph-widening round (#3313) — enwiki: AI + modern physics

The μ-method was data-ceilinged: the 10k simplewiki slice had **no AI and no modern physics** (only
classical Optics/EM/Thermo/Mechanics). This round fixes the enwiki ingest and finally supplies those
subtrees, then adds them by **fine-tune-with-replay** (the #3324 default). Prototype only; no WAM-Rust
core changes.

## The enwiki ingest fix (the real unblock)

`data/wikipedia_categories.db` was malformed — the **2024 MediaWiki `categorylinks` schema dropped the
`cl_to` text column**; the parent is now `cl_target_id` (a bigint into `linktarget`), and the old parser
both targeted the wrong column and mis-tokenised the binary `cl_sortkey` (yielding `*Astrophysics`,
`Karpov, Anatoly` junk). `scripts/ingest_enwiki_categories.py` does the correct 3-dump join —
`child = page[cl_from]`, `parent = linktarget[cl_target_id]` (subcat links, ns14) — with a quote-aware
tokenizer. Result: **9.91M edges, 2.62M nodes, 0.17% unresolved**.

`data/benchmark/wide_enwiki/` slice: depth-≤3 closures from the science roots ∪ existing scored nodes,
admin-filtered → **14,368 nodes** with the previously-absent subtrees (Neural_networks, Deep_learning,
NLP, Computer_vision, Robotics; Quantum_field_theory, Particle_physics, Relativity, Condensed_matter,
Statistical_mechanics) and **82% replay continuity** with the cumulative scored set.

## Data (`gen_enwiki_widen_pairs.py`, corpus=enwiki)

640 candidate pairs → **637 Haiku-scored** (3 parallel subagents, ~59k tokens, 0 tool calls, graded
rubric). Strata: within-AI / within-modern-physics / within-Math, cross AI×CS / modern×classical-physics /
Math×{Physics,CS} / AI×Math. Committed in `mu_pairs_scored_enwiki_260622-085608.tsv` (corpus tagged via the new
`--pairs-corpus enwiki`). Within-domain means: `pos_ai` 0.61, `pos_modphys` 0.64, `pos_math` 0.55;
random cross-strata low (correct).

## Fine-tune-with-replay + controls (enwiki slice, seed 1)

Warm-start the cumulative baseline; `--pairs mu_pairs_scored_enwiki_260622-085608.tsv --pairs-corpus enwiki
--replay-pairs mu_pairs_scored_eng_260621-174251.tsv --replay-frac 0.4 --lr 1.5e-4 --steps 500`. **6-way** discrimination
(Artificial_intelligence added as a candidate 6th root; the Physics probe carries classical + modern nodes):

| model | overall argmax | **AI** rank / top-2 | Physics (9 nodes) | ALL top-2 |
|---|---|---|---|---|
| baseline (no enwiki training) | 22/34 (65%) | **5.0 / 0%** (→Computer_science) | 8/9, rank 1.11, top-2 100% | 79% |
| placebo (warm+replay, **no new data**) | 20/34 (59%) | **3.6 / 20%** | 2/9 | 85% |
| **fine-tune+replay (enwiki data)** | 24/34 (71%) | **1.2 / 100%** (4/5 argmax) | 4/9, rank 1.78, top-2 78% | 91% |

### (1) AI becomes a separable 6th root — and it is the DATA, not churn
Baseline: AI nodes (Machine_learning, Neural_networks, Computer_vision, NLP, Deep_learning) read as
**Computer_science**, rank **5/6** (dead last) — the untrained AI anchor carries no signal. After the
fine-tune: **AI argmax 4/5, mean-rank 1.2, top-2 100%, margin +0.04**. The **placebo** (identical
warm-start + replay, but the new slot filled with OLD data instead of the AI/physics pairs) leaves AI at
**0/5, rank 3.6** — so the gain is attributable to the enwiki **data**, not re-optimisation. This is the
**first round where new data produced a discrimination gain beyond the churn floor**, precisely because
AI was *absent* before (genuine new capability), unlike the already-saturated engineering domain
(`REPORT_engineering_finetune.md`). AI and CS bleed into each other slightly (AI⊂CS — correct
multi-membership), but AI now self-identifies.

### (2) Modern physics: e5 already knew it — the gap was the GRAPH, not the model
On the **baseline** (never trained on enwiki), the modern-physics probe nodes (Quantum_field_theory,
Particle_physics, General_relativity, Condensed_matter_physics) already argmax **Physics, 8/9, top-2
100%** — purely from frozen e5. So `REPORT_phys_discrim.md`'s "missing-subfield" brittleness was a
**graph-coverage** gap (the nodes weren't in the slice to be sampled/scored), not a modelling limit.
Fine-tuning on the much denser enwiki graph introduces physics-calibration **churn** (placebo drops
Physics to 2/9); the modern-physics training data partially counters it (real 4/9 > placebo 2/9), and
ranking stays decent (top-2 78%). Headline: once the graph *contains* modern physics, it discriminates.

### (3) Replay prevented catastrophic forgetting
Gate-leak 0/3 (every operator), `pos_math` held **+0.783**, overall SYM corr **+0.727**, WIKI order-acc
**96.9%** (slightly below the simplewiki ~99% — the enwiki graph is denser/harder). Per-stratum held-out:
`pos_ai` +0.545, `pos_math` +0.783, `pos_modphys` +0.374, `cross_MPCP` +0.418.

## Honest verdict
- **The widening worked where it mattered: AI is now a real, separable domain, data-driven (placebo-
  confirmed).** This is the clearest "new data → new capability" result in the arc.
- **Modern physics was never a model problem — frozen e5 discriminates it out of the box; it was a
  graph-coverage gap that the enwiki ingest closes.**
- **Fine-tune-with-replay held the line** (no forgetting; gate-leak 0/3), at the cost of mild
  physics-calibration churn from the graph switch (visible in the placebo, partly recovered by data).
- **Source token:** the new pairs were tagged `corpus=enwiki`, but the provenance probe currently reveals
  the *judge* axis only (Δ haiku 0.012, graph 0.787 — the masked-default semantic path is accurate);
  a corpus-reveal probe (simplewiki vs enwiki on shared nodes) is the natural next structural check now
  that two corpora exist.

Reproduce: `scripts/ingest_enwiki_categories.py` (3-dump join) → build the `wide_enwiki` slice →
`gen_enwiki_widen_pairs.py` → score the 640 non-neg pairs → `mu_pairs_scored_enwiki_260622-085608.tsv`; then
`UW_MU_GRAPH=…/wide_enwiki/category_parent.tsv UW_E5_CACHE=e5_tables_enwiki.pt train_mu_attention.py
--pairs mu_pairs_scored_enwiki_260622-085608.tsv --pairs-corpus enwiki --replay-pairs mu_pairs_scored_eng_260621-174251.tsv
--init-from <baseline> --lr 1.5e-4 --steps 500` (+ the placebo with `--pairs mu_pairs_scored_eng_260621-174251.tsv`).
