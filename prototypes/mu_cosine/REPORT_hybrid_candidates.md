# Hybrid candidate generation: μ-certainty precondition test + the recall-ceiling control

**Verdict up front: two honest nulls that redirect the program.** (1) The precondition for
certainty-weighted μ ranking — μ must beat e5 at least in densely-trained regions — **fails in every
stratum**, including the densest (h1, sib) and independent of training mass. (2) The
candidate-recall "ceiling" (0.680 @ K=50) is **not a candidate-generation problem**: it is pure
truncation of e5's own list (e5@100 = 0.801), and at any matched pool size pure e5 dominates every
μ- or graph-augmented pool. Certainty-weighted hybrid generation is dead on this evidence; the
lever is K (plus ranking precision within a longer list), not a second source.

Harness: `hybrid_candidates.py` (measurement only; no fitted combiner). Query sampling, catalog,
folds-free descriptive design mirror `filing_ranker.py` (seed 7, 1200 queries, manifest sha
`fcf5e1d6…`, 335-folder catalog); the harness reproduces the standing e5@50 = 0.680 exactly.
μ = `model_pt_filing_lin.pt` (the deployed single-path LINEAGE fine-tune), scored as
μ_LINEAGE(judge=graph) over the full catalog via `eval_pearltrees_filing.score_cond`. Graph source
= candidates by d_sym(anchor→f) on the record-majority principal-parent graph (anchor = bookmark's
e5-nearest graph folder, outcome-blind). Region bins via `fit_bias_states.pair_distance_features`
+ `soft_bin_weights` (hard argmax, τ=0.25). Training mass = campaign TRAIN-side rows (node-disjoint
seed-0 split — the fine-tune's own) touching the true folder's title ±1 DAG hop.

## A. Budget-matched recall@50 (candidate generation)

| pool (budget 50 unless noted) | recall | vs e5@50 |
|---|---|---|
| e5@50 (baseline) | 0.680 | — |
| e5@40 + graph@10 | 0.681 | +7 rescued, −6 displaced |
| e5@40 + mu@10 | 0.682 | +31 rescued, −29 displaced |
| e5@40 + mu@5 + graph@5 | 0.677 | +20 rescued, −23 displaced |

Matched-budget nulls: every fill source displaces about as many e5 hits as it rescues.

**Pool-size control (the decisive table)** — same #candidates per row:

| pool | recall |
|---|---|
| e5@60 | 0.709 |
| e5@75 | **0.745** |
| e5@100 | **0.801** |
| e5@150 | 0.870 |
| e5@50 ∪ mu@50 (pool 100) | 0.745 |
| e5@75 ∪ mu@25 (pool 100) | 0.773 |

e5@50∪mu@50 (0.745) merely matches e5@75 and loses to e5@100 by −0.056. The 78 μ-"rescued"
queries' true folders sit at e5 rank median 122 (IQR 84–171) — μ surfaces folders e5 itself
reaches by K≈150, just later in its own ordering. **There is no complementary μ coverage beyond
list extension.** The unmatched upper bounds (e5@50∪graph = 0.686; e5@50∪mu@50 = 0.745) are
pool-size artifacts, reported only to close the loop.

## B. The certainty precondition (DESCRIPTIVE; stratification uses the true folder)

Per-stratum MRR over the identical full catalog, e5-cos vs μ_LINEAGE:

| region bin (anchor→true) | n | MRR_e5 | MRR_μ | med rank e5 | med rank μ | μ wins |
|---|---|---|---|---|---|---|
| h1 | 23 | 0.667 | 0.389 | 1 | 6 | 0.17 |
| h2 | 4 | 0.084 | 0.092 | 12 | 76 | 0.25 |
| sib | 122 | **0.912** | 0.309 | 1 | 8 | 0.02 |
| cous | 4 | 0.507 | 0.084 | 25 | 22 | 0.50 |
| rand | 154 | 0.227 | 0.080 | 15 | 84 | 0.25 |
| missing | 893 | 0.211 | 0.085 | 20 | 85 | 0.25 |

| train mass of true folder | n | MRR_e5 | MRR_μ |
|---|---|---|---|
| zero | 449 | 0.291 | 0.118 |
| ≤ median | 397 | 0.299 | 0.102 |
| > median | 354 | 0.291 | 0.118 |

| true folder in μ's train split? | n | MRR_e5 | MRR_μ |
|---|---|---|---|
| train-node | 362 | 0.281 | **0.092** |
| held-node | 838 | 0.299 | 0.122 |

Three findings, each fatal to certainty-weighted μ ranking:

1. **μ loses everywhere**, including exactly where it should be strongest: h1 (direct
   principal-parent, the LINEAGE training relation itself) 0.389 vs 0.667, and sib 0.309 vs 0.912.
2. **Training density does not modulate μ quality**: MRR_μ is flat (0.118 / 0.102 / 0.118) across
   zero/low/high training mass. The epistemic-certainty weight `w_μ(region)` would multiply a
   signal that is uniformly inferior — there is no region where the gate would open.
3. **No train-side advantage** (train-node 0.092 vs held-node 0.122 — if anything reversed), so
   the failure is not a generalization gap; the head's catalog-wide ordering is simply weak,
   consistent with the standing "μ heads ≤0.11 as rankers" result.

## Program consequences

- **Retire** the certainty-weighted hybrid-generator idea (this was the cheap test that killed it
  before build-out — the precondition was checked first, as intended).
- **Recall lever = K.** Raising K 50→100 buys +0.121 recall for free; the binding problem moves to
  ranking precision within the longer list and the escalation/routing policy
  (`filing_ranker.py`'s margin gate), not to candidate sourcing.
- The failure mass lives in the `missing` stratum (893/1200 queries: anchor and true folder share
  no graph relation within the horizon; e5 MRR 0.211 there vs 0.912 for sib). That stratum is
  partly an artifact of the **partial DAG recovery** (396/880 multi-parent folders; RDF
  truncation) — completing the harvest may convert `missing` rows into graph-covered rows, which
  is where graph features actually work. Data completion, not modeling, is the path into that mass.
- Where the anchor IS graph-related to the true folder (sib/h1), e5 already solves filing
  (MRR 0.9 / 0.67) — consistent with the blend's null: there was little headroom to begin with.

## B′. Corrected density axis: wiki-semantic proximity (owner's objection, upheld in part)

The owner's objection to B's density axis was correct: μ's semantic competence comes from the
Wikipedia-trained base (`model_prod_namecond_full.pt`, enwiki/simplewiki pairs; the Pearltrees
fine-tune is a thin 800-step corpus onboarding over 799 rows), so "training near the region" must
be measured as SEMANTIC proximity to the full wiki training cloud, not Pearltrees graph-touch.
Redone with density = mean top-5 e5 cosine to the 8,964 campaign training titles
(`~/mu_data/campaign_100k_e5.pt`), terciles:

| bookmark wiki-density | n | MRR_e5 | MRR_μ | med rank μ | μ/e5 ratio | μ wins |
|---|---|---|---|---|---|---|
| near | 400 | 0.355 | 0.175 | 30 | 0.49 | 0.22 |
| mid | 400 | 0.238 | 0.090 | 90 | 0.38 | 0.24 |
| far | 400 | 0.287 | 0.074 | 100 | 0.26 | 0.21 |
| both near (bm ∧ folder) | 187 | 0.411 | 0.219 | — | 0.53 | 0.22 |

(True-folder density shows the same direction, weaker: MRR_μ 0.131 near → 0.089 far.)

Two-sided finding:
- **The generalization hypothesis is CONFIRMED directionally**: μ's MRR more than doubles far→near
  and its relative gap to e5 halves (ratio 0.26 → 0.53). Cross-corpus semantic transfer from the
  wiki training is real and measurable; B's "density doesn't modulate μ" was an artifact of the
  Pearltrees-only row count. Certainty-∝-training-proximity is the right model for μ.
- **The ranking precondition still fails**: e5 rises on the same axis and μ never crosses — best
  stratum 0.219 vs 0.411, μ wins 22%. A wiki-density certainty gate would be correctly ordered
  but never opens (μ ≥ e5 nowhere).

Consequence: to make μ cross anywhere, the lever is TRAINING-CLOUD COVERAGE, not weighting —
i.e. gap-directed data (the tail-augmentation result: Haiku tail-augmentation helped at ~30%
tail-weight; and the #3936 privacy-aware public STEM gap-harvest plan, for which the far stratum
here is the empirical target list).

## B″. The gap-target list (emit_gap_targets.py)

`emit_gap_targets.py` materializes §B′ into the #3936 work order: all 1,515 harness titles
(1,192 bookmarks + 323 folders) ranked by ascending wiki-density, with each row's 3 nearest
training-cloud titles → `~/mu_data/gap_targets.tsv` (PRIVATE — personal titles; the harvest
itself targets public sources near these regions). Two findings from the list:

- **The gap mass is NOT STEM.** The farthest rows are news/politics/current-events commentary
  (whistleblower, election, healthcare-policy headlines). A gap harvest scoped to STEM would miss
  the actual gap — the target branches are politics/journalism/current-events categories.
- **The training cloud contains admin junk.** Nearest-cloud neighbors include Wikipedia
  maintenance categories (`Monthly_clean-up_category…`, `AOC_profile_template_missing_ID…`) —
  the campaign predates the enwiki correct-ingest rule (content subtree excludes ~14% admin
  cats), so some training mass is spent on categories with no semantic content, and the density
  measure is slightly polluted by them. A cloud rebuild on the correct-ingest subtree is the
  clean upgrade. (Minor: harvested bookmark titles carry unescaped HTML entities, e.g. `&#39;`.)

## Caveats

Single seed (sampling seed 7, deterministic scores — no fit, so fold variance does not apply);
stratification is outcome-aware (diagnostic only); μ tested is the single-path LINEAGE fine-tune
under judge=graph conditioning — other heads/judges were already ≤0.11 as rankers in
REPORT_pearltrees_candidate_lineage.md and were not retested here. B′ density uses the campaign
training titles as the cloud (the base model also saw earlier graded data not counted here);
underscored enwiki titles vs natural-text Pearltrees titles shift e5 cosines slightly but
identically across strata.
