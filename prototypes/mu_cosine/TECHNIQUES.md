# Techniques guide — μ-attention: operators, data acquisition, and the weak-node loop

A practical map of the machinery added on top of the base μ-cosine prototype: the **multi-operator /
multi-corpus model**, the **data-acquisition toolchain** (category slices → page frontier → Pearltrees
gap-fill), and the **find-weak → supplement → retrain loop**. Theory here is in brief with pointers; the
deep treatments are `DESIGN_calibrated_judges.md` (judges, calibration, range targets, node-types) and
`DESIGN_sampling_and_grading.md` (sampling + instruct grading + test matrix). Findings live in the
`REPORT_*` files (cited inline).

---

## 1. The conceptual model

Everything trains one quantity: **μ(node | root) ∈ [0,1]**, directional fuzzy membership. Four codebooks
condition it (`mu_attention.py`), each answering a different question — all factored, additive, and
(provenance) **maskable**, so the model learns a provenance-agnostic default plus conditioned variants:

| axis | question | values (today) | mechanism |
|---|---|---|---|
| **operator** (`OPS`) | what *relation*? | `SYM` (graded symmetric), `WIKI` (subcat, directional, free), `ELEM` (page/collection membership, directional+graded), `LLM` (frozen fixture) | operator token + per-op **readout row** on a shared trunk |
| **corpus** (`CORPORA`) | which *source graph*? | `simplewiki`, `enwiki`, (`pearltrees`) | `corpus_emb`, in the maskable provenance token |
| **judge** (`JUDGES`) | who *graded* it? | `haiku` (bought LLM), `graph` (free structural), (`human`) | `judge_emb`, same token |
| **node-type** (planned, §7 of the design doc) | what is each *endpoint*? | category, page, mindmap, pearltrees | factored token (not yet implemented) |

The **operator is the key idea**: a relation gets its own token + readout on a *shared* transformer trunk,
so the trunk learns the common "membership geometry" once and each operator embedding carries only the
delta. That's why page-membership needed `ELEM` (see §3). Full theory — including calibrating the graph
judge's target and supervising **ranges** instead of points — in `DESIGN_calibrated_judges.md`.

---

## 2. Data-acquisition toolchain (how-to)

All tools are reusable and live in this directory. The pair-file format is
`a · b · stratum · wl · μ · [relation · a_type · b_type · corpus · url]` — the first 5 columns are read by
`load_pairs`; extras are ignored (so tagged page/pearltrees rows stay backward-compatible).

| step | tool | what it does |
|---|---|---|
| 1. slice | `build_slice.py` | streaming-BFS neighbourhood of seed roots over the 615 MB full enwiki graph (downward closure + one level up + apex-capped siblings). The full graph OOMs if loaded. |
| 2a. category pairs | `gen_boundary_pairs.py`, `gen_*_pairs.py` | down/bidir candidate pairs over a slice, e5-coherence-filtered, deduped vs the cumulative |
| 2b. **page** pairs | `fetch_category_pages.py` → `gen_page_pairs.py` | pull category **member pages** via the MediaWiki API (`cmtype=page`), emit `element_of` candidates (page→category) |
| 2c. **pearltrees** pairs | `fetch_pearltrees_tree.py` → `gen_pearltrees_pairs.py` | harvest a Pearltrees tree (cookie session; collections/pages/shortcuts) for topics enwiki has no category for; PagePearl `url`s carry the enwiki anchor |
| 3. grade | (inline Haiku subagents) | 6-band centrality rubric — see `DESIGN_sampling_and_grading.md` §3 (the "instruct grading" technique) |
| 4. select (large scale only) | `select_diverse.py` (farthest-point ≈ DPP-MAP) **or** `select_svd_coverage.py` (top-K SVD axes × μ-coverage, with `--min-pages` sufficiency gate) | optional pruning for big page pools; at current scale **keep all** (policy in §5). See `REPORT_page_selection.md` |
| 5. assemble | `make_cumulative.sh` | union all `mu_pairs_scored_<round>_<yymmdd-HHMMSS>.tsv` into the (gitignored) cumulative + prior replay sets |

Filenames: scored rounds are timestamped `mu_pairs_scored_<round>_<yymmdd-HHMMSS>.tsv` (first-commit time,
**Mountain Time**); candidate + cumulative files are gitignored (regenerable). After adding a round, list
it in `make_cumulative.sh`'s `ROUNDS`.

### Example — a category round

```bash
python3 build_slice.py --root Network_theory --root Dynamical_systems --depth 2 \
    --out ../../data/benchmark/wide_enwiki_systheory/category_parent.tsv
UW_MU_GRAPH=../../data/benchmark/wide_enwiki_systheory/category_parent.tsv \
  python3 gen_boundary_pairs.py --down Network_theory --bidir Networks \
    --coh-keep Mathematics,Computer_science,Engineering --out mu_pairs_systheory.tsv
# → Haiku-grade the non-neg pairs → mu_pairs_scored_systheory_<ts>.tsv → add to make_cumulative.sh
```

### Example — a page round (the page frontier)

```bash
python3 fetch_category_pages.py --cat Bifurcation_theory --cat Ergodic_theory --out page_members.tsv
python3 gen_page_pairs.py --members page_members.tsv --out mu_pairs_pages.tsv
# → Haiku-grade CENTRALITY (element-of template) → mu_pairs_scored_pages_<ts>.tsv
```

`element_of` rows are tagged `relation=element_of, a_type=category, b_type=page`; the membership *fact* is
free (a listed page IS a member) + free μ=0 negatives, **Haiku grades the centrality** (core 1.0 →
peripheral 0.4 → mis-categorized 0.0). See `REPORT_train_consolidation.md` for why this matters.

---

## 3. The element-of operator (`ELEM`)

**Why.** Page-membership ("article is *about* topic") is a different relation than subcategory membership
("subfield is *part of* topic"). Fed as undifferentiated `SYM` positives, page labels were **inert** —
the model couldn't rank centrality (corr +0.0–0.4 vs +0.85+ for categories), and a fine-tune barely beat
a placebo that never saw them (`REPORT_train_consolidation.md`).

**What.** `OPS["ELEM"]=3` — a 4th operator token + readout row on the shared trunk. **Directional** like
`WIKI` (μ(page|category) high, reverse low) but **graded** like `SYM` (Haiku centrality target). Loss:
`MSE(μ(page|cat), target) + margin·relu(m − (μ(page|cat) − μ(cat|page)))` on positives. `--elem-weight`
tunes it. Rows route by their `relation` column. Warm-start across the op-count change is **partial**:
overlapping weights copy, op-indexed tensors grow, and the new `ELEM` row is **seeded from `SYM`**.

**Result + the capacity finding.** ELEM lifts page-centrality sharply (`pos_pageof_nonlinear`
+0.12 → +0.795). At 2 layers, full ELEM weight cost cross-domain discrimination (90%→79%, though top-2
stayed 100% — all misses razor-thin multi-membership flips). **Adding a 3rd layer recovered discrimination
fully (90%) at full ELEM weight while keeping the page gains** → it was a *capacity* limit, not data, and
frozen e5-small (384-d) is rich enough. **`--layers` default is now 3.** Full grid:
`REPORT_element_operator.md`.

---

## 4. Training & evaluation (how-to)

Fine-tune-with-replay (continual learning — warm-start, mix a replay fraction of old data so the head
doesn't forget):

```bash
UW_MU_GRAPH=…/wide_enwiki_math/category_parent.tsv UW_E5_CACHE=e5_tables_train_all.pt \
python3 train_mu_attention.py \
  --pairs mu_pairs_scored_cumulative.tsv --replay-pairs mu_pairs_scored_prior.tsv \
  --init-from <baseline>.pt --pairs-corpus enwiki --lr 1.5e-4 --steps 600 --layers 3 --save model_new.pt
```

- **e5 union (important):** `train`/`eval` union every `--pairs`/`--replay` node into the e5 build, so
  cold-start endpoints (cross-slice, page, pearltrees nodes) get a frozen embedding instead of being
  silently dropped by the `in idx` filter. This is what makes new data actually train.
- **Built-in eval** (per-operator): `[SYM]` held-out μ corr, `[ELEM]` held-out centrality corr + direction,
  `[WIKI]` edge order-accuracy, gate-leak (5-probe + OOD), 6-domain discrimination + ranking/margin,
  provenance Δ.
- **Per-stratum corr:** `eval_per_stratum.py --model M.pt --pairs <scored>` — routes `element_of` strata
  to `ELEM` directionally, the rest to `SYM`. The honest read on *which* rounds' ranking generalised.
- **Drift control (always):** a **placebo** run (`--pairs <prior>`, no new data) on the same warm start —
  any movement above placebo is the new data's real contribution, not churn. This pattern recurs across
  every `REPORT_*`.

---

## 5. The weak-node loop (the acquisition policy)

The data strategy, end to end: **find where the model is weak in an area of interest, supplement those
nodes with page data, retrain, re-measure.**

```bash
# 1. find targets: THIN (few subcats) ∧ in REGION (closure of seeds) ∧ WEAK (low e5 domain margin)
python3 find_data_gaps.py --region-root Systems_theory --region-root Dynamical_systems \
    --max-subcats 3 --margin-max 0.18 --top 30 --page-counts
# 2. harvest their pages → 3. Haiku centrality → 4. fold into cumulative → 5. train --layers 3 → re-eval
```

- **Why e5-margin = "weak":** the model's discrimination is e5-driven, so a node with low top1−top2 cosine
  to the domain roots is one e5 (and thus the model) can't place confidently — the ex-ante predictor of
  model weakness. Validated: the finder independently flagged `Ergodic_theory`, the exact stratum the
  trained model ranked worst, and supplementing it nudged its corr up (`REPORT_page_selection.md`).
### Page-sampling policy (how much page data to keep, and why)

There is **far more page data than category data** (a thin category can have 60+ member pages), which would
normally argue for subsampling the majority class. The key insight: **because page-membership trains on its
own `ELEM` operator, the imbalance is *contained* in `ELEM` rather than drowning the category operators
(`SYM`/`WIKI`).** That structural isolation — not a sampling trick — is what neutralises the imbalance. So
the policy is:

1. **Don't filter page data for `ELEM` — use all of it.** The A/B (`REPORT_page_selection.md`) confirmed
   keep-all beat a diversity-pruned subset at current scale (the operator is data-hungry; "redundant"
   filter/wavelet pages carry distinct centrality; frozen e5 means it can't overfit duplicates). Filtering
   is the wrong lever.
2. **Monitor cross-operator degradation every run.** `ELEM` shares the trunk, so heavy `ELEM` learning can
   pull the shared representation and degrade `SYM`/`WIKI` — exactly the capacity interference measured at
   2 layers (discrimination 90%→79%, `REPORT_element_operator.md`). Watch the `[SYM]` corr and 6-domain
   discrimination as `ELEM`'s share of the data grows.
3. **If it degrades the others, throttle `ELEM`'s gradient — don't cut the data.** Reduce `ELEM`'s pull on
   the shared trunk via **`--elem-weight`** (existing lever; 0.4 recovered discrimination) or, as a
   stronger next lever, a **per-operator learning rate / scaled backprop on `ELEM`** (not yet implemented).
   Adding **capacity (a layer)** is the cleanest fix when available — 3 layers removed the interference at
   full `ELEM` weight without throttling — which is why `--layers` defaults to 3.

**The selection method, precisely (for the larger-scale "must filter" regime).** `select_diverse.py` is
**greedy quality-weighted farthest-point** on e5 cosine — drop μ<0.4 junk, seed the highest-Haiku page,
then add the page maximising `centrality · (1 − max cosine to already-picked)`. This is a cheap **DPP-MAP
proxy; it is NOT a PCA/SVD decomposition.** The **second method, `select_svd_coverage.py`, IS the
SVD/μ-coverage one:** take the **top-K SVD axes** of the category's page-embedding matrix (K from the
variance elbow = intrinsic dimension, not a fixed `--frac`) and keep a subset that **covers the
*(e5-axis × μ)* joint** — both ends of each principal axis × low/mid/high centrality — so the operator
learns how membership varies along each subtopic direction; junk (μ<min) dropped, negatives passed
through. It adds the two design preconditions: a **sufficiency threshold** (`--min-pages` — below it a
category is kept whole, sampling isn't worthwhile) and **subcategory augmentation** (push a thin category
over the threshold by pooling its subtree's pages via `fetch_category_pages.py --recurse-subcats`). Both
selectors are the **larger-scale "must filter" knob**; at current scale the policy is keep-all (above).

---

## 6. Where to read more

| topic | doc |
|---|---|
| judges, calibration, range targets, node-types, relation operators, page frontier (theory) | `DESIGN_calibrated_judges.md` |
| sampling modes, instruct grading rubric, data-rejection, test matrix, storage | `DESIGN_sampling_and_grading.md` |
| directional attention architecture | `DESIGN_directional_attention.md` |
| element-of operator + capacity finding | `REPORT_element_operator.md` |
| consolidation train+eval (page-as-SYM is inert) | `REPORT_train_consolidation.md` |
| targeted supplementation + diversity A/B | `REPORT_page_selection.md` |
| per-region round findings | `REPORT_math_fields.md` (§§7–12), `REPORT_widen_enwiki.md`, … |

Everything here is the **prototype** (`prototypes/mu_cosine/`); none of it touches the WAM-Rust core.
