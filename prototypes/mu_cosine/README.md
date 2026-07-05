# μ-cosine embedding prototype

A prototype for **generating dense fuzzy membership `μ` from learned category embeddings**, so the
μ-gated cone / similarity machinery (`descendant_mu_mass_gated`, `gated_ic`, `lin_from_ic` in the
WAM-Rust target) can run on domains where only a sparse set of categories has been LLM-scored.

This is a **separate project, prototyped on a branch** — it is Python/ML, not the Rust/Prolog core.

> **New here? Start with [`TECHNIQUES.md`](TECHNIQUES.md)** — a practical guide to the multi-operator model
> (`SYM`/`WIKI`/`ELEM`/`LLM`), the data-acquisition toolchain (category slices → page frontier → Pearltrees
> gap-fill), and the find-weak → supplement → retrain loop, with runnable commands. Deep theory:
> [`DESIGN_calibrated_judges.md`](DESIGN_calibrated_judges.md).
>
> **Doing uncertainty / confidence / multi-source estimation work?** Read
> [`DESIGN_uncertainty_estimation_playbook.md`](DESIGN_uncertainty_estimation_playbook.md) **first** — the
> calibrated-joint-posterior + margin-gate approach and the five pitfalls (correlated sources, margin-not-level,
> held-out node-disjoint, …) the project learned the slow way. Don't hand-set independent confidence weights.

## Status & handoff (read this first)

**Status:** merged to `main` (via #3280); the torch port + direct-embedding comparison are folded in
here. New work: branch from `main`, open your own PR. Last verified Python 3.11 (stdlib pieces) /
torch 2.x + sentence-transformers (ML pieces).

| piece | state |
|---|---|
| training objective (cosine ≈ μ) | ✅ **proven** on real data — `train_cosine_mu.py`, corr 1.00 |
| distance-biased sampler | ✅ implemented (in the trainer) |
| pairwise label *generator* | ✅ **done** — `gen_mu_pairs.py` (emits candidate pairs; no LLM cost) |
| encoder architecture (MLP/MoE, not a transformer) | ✅ **forward pass** only — `mu_encoder.py`, runs at 384-dim |
| MiniLM init | ✅ **done** — `mu_encoder_torch.py` `build_minilm_init` + `embed()` (ML env w/ HF egress) |
| training the full encoder | ✅ **done** — `train_cosine_mu_torch.py` (torch port; objective + held-out generalisation) |
| **scoring the pairs (μ labels)** | ✅ **sampler labels done** — `mu_pairs_scored.tsv` (200 Haiku-scored positives + 1000 free μ=0 negatives) |
| **cutoff-band rescore (Prompt C / stream V)** | ✅ **done** — `tests/fixtures/wikipedia_physics_boundary_haiku.tsv` (654 Haiku membership labels over the e5 decision band); **42.7% decision-flip**, node-vs-root Lin confirmed degenerate, calibration stable (~34k Haiku tokens). `boundary_band.py` + `fuse_and_validate.py` |
| training on the scored pairs | ✅ **done** — `train_cosine_mu_torch.py --mode pairs --minilm` on `mu_pairs_scored.tsv`; held-out pos corr **+0.726**, lin-agreement moved **−0.13 → +0.10** |
| wiring dense μ back to the Rust core | 🟡 **emitter done** — `emit_dense_mu.py` / `dense_mu_direct.py` (clamped, verbatim names, 100% coverage); Rust consumption verified by `check_feeds_rust.py`, not run end-to-end in CI |
| control baseline + theory questions | ✅ **done** — `validate_control_baseline.py` → `REPORT_control_baseline.md` (the #3287 control arm the directional model must beat; lin-saturation, decision-flip, cold-start all resolved) |
| gated similarity correctness | ✅ **node-gated IC** (#3296) — `gated_ic_node_filtered` de-saturates Lin/Resnik/FaITH (path-gated 96.7% → **0.1%**); theory in `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` §5f, proof `node_gated_ic.py`. Directional model's Lin-agreement is validated on this (#3297). |
| **directional multi-relational μ** | ✅ **done** — `mu_attention.py` + `train_mu_attention.py` → `REPORT_directional.md`. Frozen e5 + learned tags + 2-layer attention, multi-task over operators. **WIKI directional edge order-accuracy 99.1%** (control structurally 50%), gate-leak 0/5 all operators, every map feeds `check_feeds_rust`. SYM held-out corr +0.335 (< control +0.726 — the multi-task trade-off). No new LLM budget. |
| **SYM more-data + ablation** | ✅ **done** — `gen_more_sym_pairs.py` + `mu_pairs_scored_large_260620-223001.tsv` (+600 Haiku pairs, ~38.5k tokens) → `REPORT_sym_more_data.md`. More data raised SYM held-out corr **+0.342 → +0.479** (WIKI stays ~98%); single-task SYM **collapses** ⇒ the gap was **data starvation, not multi-task dilution** (multi-task is load-bearing). |
| **multi-domain (Physics + Chemistry)** | ✅ **done** — child-only sampler + μ-coherent pools (`gen_multidomain_pairs.py`, +650 Haiku pairs) → `REPORT_multidomain.md`. SYM `pos`-stratum corr **+0.479 → +0.695** (control +0.726; gap ~0.03), overall +0.822; **Physics-vs-Chemistry discrimination 10/10**; WIKI 99.7%; lin-agreement +0.183 (> control). |
| **four roots (Phys/Chem/Math/CS)** | ✅ **done** — depth-bounded closures ∩ μ-coherence + #3309 bidir batch (`gen_multidomain_pairs.py`, +885 Haiku pairs) → `REPORT_4roots.md`. **4-domain discrimination 18/20 (90%)** — Physics/Chemistry/CS clean 5/5, **Math 3/5** (Calculus & Diff-eq leak to Physics — thin 9-node pool). WIKI 98.6%; lin-agreement +0.237. `pos` stratum regressed +0.695→+0.570 (breadth traded vs physics fidelity). |
| **more Math DATA + Engineering** | ✅ **done** — deepened *inclusive* Math pool (9→13, keeps math-of-physics) + new Engineering domain (`gen_math_eng_pairs.py`, +404 Haiku pairs ~24.8k tok) → `REPORT_matheng.md`. **Physics SYM corr recovered +0.570→+0.838** (more DATA, not breadth — Part-A hypothesis confirmed); `pos_math` +0.818, `cross_MP` +0.780 (math-of-physics learned **high-to-both**); `Calculus` now argmax-Math; Engineering clean 4–5/5; WIKI 99.2%. 5-way argmax **seed-sensitive 56–84%** — Physics is the brittle one (1–2/5) because it's the spine's connective domain (high-μ to several roots → argmax ill-posed even as ranking stays strongest). |
| **core-physics discrimination** | ✅ **done** — re-measured with a **ranking/margin** metric + targeted sampling (`gen_core_physics_pairs.py`, +152 Haiku pairs ~15.6k tok) → `REPORT_phys_discrim.md`. Re-eval: argmax swings 64–92% but **ranking is robustly strong (top-2 92–100%; physics margins ±0.08)** — the #3314 "brittleness" is mostly a **metric artifact (correct multi-membership)**. Targeted classical-physics data **did NOT help** (argmax mean 81%→71%; physics SYM held +0.823); the clean classical core is only ~23 nodes and the **modern subfields (Quantum/Stat-mech/Relativity) are absent**. Verdict: brittleness is **structural (connective spine) + a missing-subfield problem** — the real fix is graph widening (#3313). |
| **Engineering build-out (fine-tune+replay)** | ✅ **done** — real 36-node Engineering pool + Mech/Thermo×Eng high-to-both boundary (`gen_engineering_pairs.py`, +300 Haiku pairs ~33k tok) → `REPORT_engineering_finetune.md`. **Continual learning**: warm-start the cumulative checkpoint, fine-tune on new data with 40% **replay** (`--init-from`/`--replay-pairs`, lr 1/3, 400 steps). New data taught the held-out **ranking** (`pos_eng` +0.888, `cross_EP` +0.699); Engineering discriminates 5/5 every seed. **Replay prevented forgetting** (physics SYM +0.824→**+0.872**, WIKI 98.8%, gate-leak 0/5). **Multi-seed (1/7/23):** fine-tune+replay argmax 80–92 / top-2 96–100 — far more **stable** than full retrain (60–96, collapsed at seed 7). **Placebo/churn control** (`drift_control.py`): a no-new-data fine-tune moves the discrimination probe **0.90×** as much as the real data (and argmax 64%→84% on churn alone) → the argmax gains are re-optimization, **not** the new data; new data helps the **ranking**, not the already-saturated argmax. Data-gap: Chem/CS/Eng saturated (100% top-2 every seed); **Physics & Math** are the data-hungry domains (+ Electrical/Process/Chemical/Software/Aerospace eng absent). |
| **graph widening — enwiki (AI + modern physics)** | ✅ **done** → `REPORT_widen_enwiki.md`. Fixed the enwiki ingest (`scripts/ingest_enwiki_categories.py` — 2024 `categorylinks` schema dropped `cl_to`; parent is now `cl_target_id`→`linktarget`): clean **9.9M-edge** graph → `wide_enwiki` slice (14,368 nodes) with the previously-absent AI + modern-physics subtrees. +637 Haiku pairs (~59k tok, `corpus=enwiki`), fine-tune+replay. **AI became a separable 6th root: 0/5 (rank 5, →CS) → 4/5 (rank 1.2, top-2 100%)** — and the **placebo (no new data) stays 0/5**, so it's the DATA, not churn (first clear new-data→new-capability win). Modern physics: frozen e5 **already** discriminates it (baseline 8/9) — the gap was graph-coverage, not the model. No forgetting (gate-leak 0/3, `pos_math` +0.783, WIKI 96.9%). |
| **math-deepening (core subfields)** | ✅ **done** → `REPORT_math_fields.md`. `wide_enwiki_math` slice (+813 math nodes: analysis/algebra/geometry-topology/foundations/discrete, e5-coherence-filtered) + 533 Haiku pairs (~56k tok). Within-area > cross-area > cross-domain μ-gradient (areas have real structure). **But Math discrimination was ALREADY saturated** — all 10 probe nodes incl. the deepened subfields (Number_theory/Group_theory/Topology/Real_&_Complex_analysis) argmax Math **10/10 on the baseline** (margin +0.29); deepening left it flat (placebo identical). Data's value is the **within-subfield ranking** (`pos_discrete` +0.922, `pos_foundations` +0.804); no forgetting (SYM +0.827). **But the SUBFIELD axis is not saturated**: μ(node\|Real_analysis) vs μ(node\|Algebra/Topology/…) is only 54% on e5 alone → fine-tune 62% / top-2 79% (placebo 59% — churn drives most, data adds a modest real increment). **Meta-finding:** frozen e5 saturates *coarse* (cross-domain) discrimination but *blurs sibling subfields* — new data adds capability for e5's blind spots (AI) and a modest gain for fine intra-field distinctions, not for coarse separations it already nails. |
| **retrieval APPLICATION — hybrid μ+e5** | ✅ **done** (#3388) → `DESIGN_model_applications.md`, `eval_hybrid.py`. "Find my container": e5 coarse shortlist → μ re-rank. The score is a **non-linear OR over separately-computed operator queries** — `max(μ-elem, μ-wiki, μ-sym)` — NOT pure ELEM (over-generalises) and NOT the internal μ-super (one blended forward). Tuned grid (μ-part × α=e5-weight, 1000 q): best `max(elem,wiki,sym)` @ α≈0.9 → **MRR 0.358 vs e5-cos 0.286 (+25%)**, container-vs-sibling μ 72% vs e5 67%. Prefixed e5 shared with e5-cos (one encode). Default keeps 10% e5 as **coverage insurance** for untrained/OOD regions. |
| **LLM re-rank — two-stage pipeline** | ✅ **done** (#3389) → `eval_rerank.py`. Blend top-K → Haiku picks precision@1. 60 q, top-5: blend #1 (no LLM) **0.267 → blend→LLM 0.350 (+31%)**, ceiling recall@5 0.50. **The re-rank doubles as a μ-quality eval** (user's framing): 50% LLM↔μ agreement (kept μ's #1); when it overrides it moves only ~1 rank (to 2.2) — μ's ordering is close even when #1 is wrong. Head-to-head blend→LLM vs e5→LLM coded but unrun (deferred on Haiku spend). |
| **confidence-adaptive blend** | ✅ **done** (this branch) → `DESIGN_model_applications.md`, `eval_hybrid.py`. Per-query α from μ's **own top-score** (calibrated [0,1] degree = free confidence signal). **Mechanism validated**: μ earns its weight where confident (α0.9 vs 0.3 worth **+0.037** MRR) vs not (**+0.003**) — μ knows when it's useful. But adaptive α∈[0.3,0.9] beats fixed α=0.9 by only **+0.006** in-dist, because low-confidence μ is **neutral, not harmful** (e5-heavy even slightly worse) → fixed high-α already near-optimal. **Real payoff is OOD** (where low confidence coincides with μ being *wrong*); confirms the coverage-insurance testable. Correct and never hurts. |

> **Methodology** (how every round above works — sampling, the graded-rubric "instruct grading", the
> data-rejection rules, fine-tune-with-replay, the placebo control, and the standing **test matrix** of
> roots/subroots + subsampling policy): [`DESIGN_sampling_and_grading.md`](DESIGN_sampling_and_grading.md).

### Progress — directional, multi-relational μ (`MuAttention`, DESIGN_directional_attention.md realized)

The symmetric encoder computes `μ(a,b)=cos(f(a),f(b))`, which **cannot** represent order
(`μ(Optics|Physics) ≫ μ(Physics|Optics)`). `MuAttention` (`mu_attention.py`) is a tiny
permutation-invariant transformer over a **set** of tagged tokens — `operator`, `anchor(root)`,
`node@gen0`, `{ancestors}@gen_d` (min-hop on the DAG), off-manifold-noise absent slots — on **frozen e5**
inputs, learning only the tags (`op_emb`/`anchor_tag`/`gen_emb`), a 2-layer attention block, and a
per-operator sigmoid readout. `train_mu_attention.py` trains it multi-task over operators (WIKI margin +
SYM order-invariant MSE + an optional LLM operator from the already-bought boundary fixture, no new
spend). All 8,247 nodes covered (frozen e5 + shared tags). Results (`REPORT_directional.md`):

| operator | metric | value | control |
|---|---|---|---|
| **WIKI** | held-out edge order-accuracy `μ(child\|parent)>μ(parent\|child)` | **99.1%** | 50% (symmetric ceiling) |
| **SYM** | held-out μ corr / symmetry gap | +0.335 / 0.159 | +0.726 |
| **LLM** | in-sample directional-μ corr (654 band) | +0.963 | — |
| all | gate-leak 5-probe / OOD | **0/5** / 0.2–1.5% | 0/5 / 1.1% |

**Wins decisively on the directional task** (the design's reason to exist; the symmetric control is
pinned at 50%); matches on the gate-leak probe; **behind on pure symmetric corr** (multi-task trade-off —
SYM shares the backbone with WIKI+LLM). Every operator's dense map feeds `gated_ic`/`lin_from_ic`. The
SECONDARY node-gated-IC (#3296) lin check confirms the saturation fix (0.1% vs 96.7% path-gated).

**Provenance token** (`REPORT_provenance.md`) — the deferred judge axis, realized. One **maskable**
token with a **factored** embedding `corpus_emb[corpus] + judge_emb[judge] + prov_tag` records each
label's source (corpus `simplewiki`/`enwiki`; judge `haiku` for bought SYM/LLM labels, `graph` for WIKI
edges + free μ=0 negatives). **Masked by default** (off-manifold noise → provenance-agnostic μ,
marginalizing over sources — the default inference path). Single-corpus so validated **structurally**:
the slot is read (revealing `judge=graph` collapses μ by 0.66 — it learned graph-judged SYM = the μ=0
non-edges; `judge=haiku` near-constant 0.027), and a `--prov-mask 1.0` ablation confirms it does **not
regress** Part A (`pos_phys` +0.831 vs +0.838). Ready to carry real signal once `enwiki`/a 2nd judge
arrives — no architecture change.

### Progress — ML-environment port (folded in from #3283)

The handoff's "take it into the real ML environment" steps are implemented in torch (the steps that
spend **no** LLM budget). Gating checks passed first: `pip install -r requirements.txt` succeeds and
HuggingFace is reachable (MiniLM downloads + encodes, 384-d).

| new file | what it does |
|---|---|
| `mu_encoder_torch.py` | torch port of the MLP/MoE encoder; **MiniLM init wired into `embed()`** (`build_minilm_init`); MoE load-balancing aux loss; `to_membership` clamp |
| `train_cosine_mu_torch.py` | the cosine-μ objective in torch (Adam / SparseAdam). `--free-vectors` reproduces `train_cosine_mu.py` (**corr → 1.00**); `--minilm --holdout` is the generalisation test; `--mode pairs` for scored `gen_mu_pairs.py` (refuses unscored rows) |
| `emit_dense_mu.py` | step 5 — dense `name<TAB>μ` for the whole graph from the trained encoder; **clamps [0,1]**, **emits names verbatim**, asserts coverage |
| `validate_lin_agreement.py` | step 4b — faithful python port of `gated_ic`/`lin_from_ic` (threshold 0.3, Σμ denom), compares graph-side Lin vs semantic cosine |

**Validation results (no LLM budget spent):**
- **Objective**: `--free-vectors` → train MSE 0.0000, corr **+1.00** (torch matches the stdlib proof).
- **Generalisation** (frozen MiniLM init + shared 1-layer encoder, 0.2 held-out): **held-out corr +0.946**,
  MSE 0.023 — μ predicted for categories whose label the encoder never saw. Even the *untrained*
  MiniLM-cosine already scores held-out corr +0.88.
- **Dense μ**: emitted for all 8 247 categories, 100% name resolution; highest-μ are exactly physics
  topics (Light, Electromagnetism, Thermodynamics, Acoustics…); 262 in the [0.3,0.7] boundary band.
- **Lin agreement — the Step-4 claim, corrected.** The handoff's step 4 said the dense μ should
  *converge* to the graph-side `lin_from_ic` (`μ(X|root) == Lin(X, root)`). **That validation is
  degenerate and has been dropped:** node-vs-root Lin saturates at 1.0 for all 51 scored nodes (the
  root is the most-general node, so every node's MICA-with-root *is* the root). The *pairwise* graph-Lin
  vs semantic-cosine correlation is **weak** for the single-anchor-trained encoder (Pearson ≈ −0.13):
  single-anchor μ(X|Physics) training collapses physics nodes onto the Physics direction, so pairwise
  cosine saturates too. **This is the concrete motivation for the pairwise scored labels**
  (`gen_mu_pairs.py`, the budget step) — the held-out *single-anchor* μ regression is strong, but
  capturing pairwise/Lin structure needs the varied-anchor pairwise μ. Do **not** gate the encoder on
  agreement with node-vs-root Lin.

**Next (needs the budget step):** score `mu_pairs.tsv` (`gen_mu_pairs.py score_stub`, ~1200 pairs) →
`train_cosine_mu_torch.py --mode pairs --minilm` → re-run `validate_lin_agreement.py` (expect the
pairwise agreement to improve). **Confirm with the user before scoring.**

### Progress — dense μ *without* training, model comparison (Prompt A, realized)

The fastest path to unblock the graph work (Prompt A below) is a dense μ map by **direct asymmetric
embedding** (encode the root as a *query*, each category as a *document*, cosine, clamp) — picking the
embedder that minimises future Haiku re-scoring.

| new file | what it does |
|---|---|
| `dense_mu_direct.py` | dense `name<TAB>μ` with no training, asymmetric query/doc prefixes; presets `minilm` / `e5` / `nomic`; `--compare` reports each model's **decision band** + fixture discrimination |
| `check_feeds_rust.py` | confirms a chosen map feeds `gated_ic` / `lin_from_ic` (faithful port): names resolve verbatim, μ∈[0,1], gated IC finite + general→specific, membership separates physics/non-physics |
| `dense_mu_e5.sample.tsv` | committed illustrative 20-row sample of the chosen map (full maps are git-ignored, regenerable) |

**Decision-band metric & the trap (this corrects Prompt A's "smallest band wins").** The budget metric
is the **decision band** — categories with μ ∈ [0.2, 0.45] straddling the 0.3 gate, the ones a later
Haiku pass must re-score. But the **raw** band is misleading: asymmetric retrieval models have a high
cosine *floor*, so they pile everything above the gate (e5 raw band = **0** — but Music=0.84 ≈
Optics=0.87, no discrimination at all; "smallest raw band" would pick the *worst*, most-compressed
model). The fix (no budget): **calibrate** each model's cosine→μ against the existing 90-node Haiku
fixture (linear fit) **before** counting the band — only then is the band comparable and reflective of
genuine ambiguity:

| model | dim | raw band | **calibrated band** | fixture r | gate leak (non-phys passing 0.3) |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 876 | 2028 | +0.573 | 4/5 |
| **`e5-small-v2`** | **384** | 0 | **653** | **+0.665** | **3/5** (Cooking, Religious correctly out) |
| `nomic-embed-text-v1.5` | 768 | 4109 | 1049 | +0.647 | — |

**Pick: `intfloat/e5-small-v2`** — smallest *calibrated* band, best fixture correlation, cleanest gate
separation, and 384-d (lightweight). `check_feeds_rust.py` confirms its map feeds the Rust core
(IC general→specific: Physics 2.85 < Optics 5.12 < Thermodynamics 7.18). Caveat: pairwise `lin_from_ic`
**saturates** toward 1.0 for many pairs on this graph (all maps) — the membership μ, not pairwise Lin,
is the clean separator (same finding as the Step-4 correction above). Don't use BERT/ModernBERT
(logit/entropy encoders, not sentence embedders). e5 still leaks a couple of loosely-related apexes
near the gate (e.g. Music ≈ 0.52, Politics ≈ 0.50) — exactly what **Prompt C** (cutoff-band Haiku
rescore) is for.

```bash
python3 dense_mu_direct.py --compare                              # the table above
python3 dense_mu_direct.py --model e5 --out dense_mu_e5.tsv       # emit the chosen (calibrated) map
python3 check_feeds_rust.py --mu-file dense_mu_e5.tsv             # sanity-check it feeds the Rust core
```

### Progress — sampler training labels scored (`mu_pairs_scored.tsv`)

The first batch of **pairwise** training labels is generated and scored (step 2 of the ordered plan).
`gen_mu_pairs.py --seeds Physics` produced 1200 candidate pairs; the **200 positives** (hub-down-weighted
RWR mesh) were scored with parallel Haiku subagents (graded sameness/relatedness 0..1 — 1.0 same/nested
topic, 0.5–0.7 same broad domain, 0.2–0.4 loosely related, 0.0 unrelated), and the **1000 negatives**
(domain × uniform-random noise) take μ=0 for free (SGNS — negatives are sampled, not labelled). Committed
as `mu_pairs_scored.tsv` (small, expensive, reusable — the commit-vs-regenerate rule; the unscored
`mu_pairs.tsv` is git-ignored). Positive-μ mean 0.61, ~72% ≥0.5 — a graded spectrum (long mesh walks
drift out of physics and correctly score low, e.g. `Physics/Creation_myths` 0.0, `Physics/Time_travel`
0.1, vs `Physics/Heat` 0.95). **These varied-anchor pairwise labels are exactly the signal the Step-4
note says is missing** (single-anchor μ(X|Physics) training collapses pairwise structure).

**Next:** in an **HF-egress env**, `train_cosine_mu_torch.py --mode pairs --minilm` on `mu_pairs_scored.tsv`,
then re-run `validate_lin_agreement.py` (expect pairwise agreement to improve over the single-anchor
encoder). The **cutoff-band rescore (Prompt C)** is the complementary label set — it needs the e5 dense
map (also HF-egress), so it runs in the same env.

### Progress — trained on the scored pairs (this PR)

Ran the "Next" above. `train_cosine_mu_torch.py --mode pairs --pairs mu_pairs_scored.tsv --minilm`
(`n_layers=1`, `n_experts=1`, AdamW, **no** `--dist-bias` — the explicit negatives already encode
distance). Two trainer changes made the run honest:
- **Positive-stratified holdout** (`--holdout` now holds out a fraction of *positives* only; the 1000
  μ=0 negatives all stay in training and are excluded from the held-out metric — a held-out corr over
  μ=0 rows is meaningless and would dilute the signal).
- **`--weight-decay`** (AdamW) regularises the 1.2M-param shared encoder toward the identity residual
  (≈ raw MiniLM). Without it the encoder memorises the 160 training positives (train corr → 1.00) and
  held-out corr *drops below* the untrained MiniLM baseline (+0.65 → +0.46). `wd=1.0` fixes this.

**Results (no Haiku budget — training is local compute):**

| metric | untrained MiniLM | single-anchor encoder | **pairs-trained (wd=1.0)** |
|---|---|---|---|
| held-out *pairwise* μ corr (40 held-out positives) | +0.650 | — | **+0.726** (MSE 0.065) |
| lin-agreement: graph-Lin vs semantic-cosine, Pearson (1275 pairs) | — | **−0.13** | **+0.098** |
| lin-agreement Spearman | — | −0.10 | **+0.113** |
| dense-μ gate leak (non-physics passing 0.3, of 5 probes) | — | — | **0/5** (Music 0.19, Cooking 0.22, Religious 0.00) |

The **lin-agreement delta is +0.23 Pearson** (−0.13 → +0.10): the varied-anchor pairwise labels broke
the single-anchor encoder's collapse-onto-the-`Physics`-direction (which had made pairwise cosine
saturate and anti-correlate with graph-Lin). It is now *weakly positive* — the explicit negatives also
give the dense map clean in/out separation (0/5 leak vs 3–4/5 for the single-anchor / e5-direct maps).
`emit_dense_mu.py` → `check_feeds_rust.py`: 100% coverage, IC general→specific
(Physics 3.49 < Electromagnetism 5.69 < Thermodynamics 6.99 < Optics 9.21), `lin ∈ [0,1]`, guards OK.
Pairwise `lin_from_ic` still **saturates** toward 1.0 for many pairs (graph property, all maps) — the
ceiling on how far this correlation can climb; membership μ remains the clean separator.

```bash
python3 train_cosine_mu_torch.py --mode pairs --pairs mu_pairs_scored.tsv --minilm \
    --holdout 0.2 --weight-decay 1.0 --epochs 2000 --lr 0.01 --save-encoder enc_pairs.pt
python3 validate_lin_agreement.py --encoder enc_pairs.pt        # lin-agreement delta
python3 emit_dense_mu.py --encoder enc_pairs.pt --out dense_mu_pairs.tsv
python3 check_feeds_rust.py --mu-file dense_mu_pairs.tsv
```

**Next architecture (design, not yet built):** the symmetric `cos(f(a),f(b))` encoder above cannot model
the *order-dependence* of membership (`μ(X|root) ≠ μ(root|X)`). The directional successor — frozen e5
inputs + learned **operator** (symmetric / Wikipedia-order / LLM-judged), **anchor**, and
**per-generation ancestry** tags (the node + its parent/grandparent lineage as a soft cone) read by a
small attention block, with cold-start coverage preserved — is specified in
[`DESIGN_directional_attention.md`](DESIGN_directional_attention.md). #3287 (MiniLM symmetric) is its
control arm.

**Reproduce what exists (no deps):**
```bash
cd prototypes/mu_cosine
python3 train_cosine_mu.py        # learns cos≈μ on the 90-node fixture; prints corr → 1.00
python3 mu_encoder.py             # forward pass of the MLP/MoE encoder at d_model=384
python3 gen_mu_pairs.py           # emits 1200 candidate pairs (200 pos / 1000 neg, 5:1) to mu_pairs.tsv
```
(`mu_pairs.sample.tsv` is a committed 30-line example of the generator's output.)

**To take it into the real ML environment (ordered):**
1. `pip install torch` (or numpy) — and obtain MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, 384-d)
   for the per-category init embeddings. Wire it into `MuEncoder.embed` (currently random fallback).
2. **Score the candidate pairs — but only the *positives*.** `gen_mu_pairs.py` emits a
   graded-word2vec-SGNS design (~5:1 negatives:positives; positives a hub-down-weighted random-walk
   *mesh*; negatives uniform noise). **Do not LLM-score the negatives** — in SGNS negatives are
   *sampled*, not labelled; a random `(Geology, 2022_movies)` pair is `μ≈0` by construction, and paying
   Haiku to confirm 1000 obviously-unrelated pairs is wasted budget (it also contributes ~no gradient
   signal). So score the `stratum=pos` rows only (~200), assign the `neg` rows `μ=0`. Fill the `μ`
   column with a Haiku subagent (`score_stub` shows the prompt/format; same discipline as
   `tests/fixtures/wikipedia_physics_*`). **Spends LLM budget — confirm with the user first.** *Budget
   priority:* the cutoff-decision band (Prompt C) is higher-value-per-label than these mesh positives —
   if budget is tight, do C first.
3. Port `mu_encoder.py`'s forward to torch (it is a faithful spec) and add the training loop — the
   objective and gradient shape are validated by `train_cosine_mu.py`. Start with `n_layers=1`,
   `n_experts=1` (plain MLP / projection head); add MoE experts or neighbour context only if it
   underfits. Notes: use **Adam, not SGD** (an embedding table wants per-row adaptive steps, with
   sparse gradients — see batching); **do *not* re-apply `train_cosine_mu.py`'s `--dist-bias`** when
   training on `gen_mu_pairs.py` output — those pairs already encode the distance preference via the
   explicit negatives, so a distance bias on top double-penalises far pairs.
4. Validate: held-out `μ` regression (this is the real generalisation signal). **Do _not_ validate
   against node-vs-root `lin_from_ic`** — that cross-check is degenerate (root is the most-general
   node, so node-vs-root Lin saturates at 1.0); see the "ML-environment port" progress note above. If
   you want a graph-side cross-check, use *pairwise* μ(a,b) for varied anchors, and expect it to need
   the pairwise scored labels (step 2) before it agrees.
5. Emit a dense `μ` map (`category → μ`) for the whole graph — **clamp cosine to `[0,1]`
   (`to_membership`) and emit names verbatim** (both load-bearing — see Integration) — and feed it to
   the Rust core.

**Compute & batching (so the GPU sizing is not guessed):**
- The sampler's **mesh size** (e.g. 324 nodes) is *coverage*, **not** a batch size — it sets label
  diversity, nothing about training memory. The training **batch** is a count of *pairs* per step, an
  independent knob.
- This prototype model is **not memory-bound**: 1 block (~1.77M params ≈ 7 MB) + the 8.2k-cat
  embedding table (~13 MB) + Adam (~25 MB); activations scale at **~24 KB/pair**. The whole 1.2k-pair
  set is ~30 MB — you can full-batch it. Choose batch (256–1024) for SGD noise vs. stability, not
  memory. Bigger batches also give more **in-batch negatives** (SGNS), so prefer larger on a big GPU.
- Activations/pair scale with `layers × seq_len × 2`, so a **500 MB activation budget** holds
  ~20k pairs for this 1-layer single-token model, ~3k for a 6-layer single-token one, ~200 for a
  6-layer name-token-sequence one — all comfortable batches. Don't assume the prototype's tiny
  footprint if you deepen the model or feed token sequences.
- The real memory cost is the **embedding table at full Wikipedia scale**: ~1M cats × 384 × 4B ≈
  1.5 GB (+~3 GB dense Adam). Use **sparse embedding gradients** (only the rows a batch touches) — the
  lever is the table, not the pair-batch (still 1k–4k).

**Resolved design decisions (and the reasoning, so they aren't re-litigated):**
- **MLP/MoE encoder, not multi-head attention.** A single category vector ⇒ attention softmax ≡ 1 ⇒
  Q/K dead, V/O collapse to one Linear; the region-dependent computation is the MLP nonlinearity
  (and an optional gated MoE for explicit region routing). So no attention in the base model.
- **Attention is the future upgrade, gated on a *sequence* input** — feed the category + a few
  neighbour tokens and *learn* the pooling weights (vs the cheap fixed mean-pool already in
  `encode(..., neighbors=...)`). Only worth it if the fixed-weight context underfits.
- **Start at `n_layers=1`, `n_experts=1`** (plain MLP / projection head): few params generalise from a
  small label set; a *shared* encoder on (mostly-frozen) MiniLM embeddings is what produces dense μ for
  *unlabelled* categories — big/per-category capacity overfits the labels and leaves the rest at init.
- **If you turn on MoE (`n_experts > 1`):** the prototype's *soft* gate (all experts, softmax blend) is
  interpretable but can **collapse** all gate weight onto one expert without a load-balancing auxiliary
  loss (`aux = n_experts · Σ_i f_i·P_i`, fraction-routed × mean-gate) — add it. *Top-k* (sparse) routing
  is the throughput upgrade for production; soft routing is fine while prototyping.

## The idea

`μ(X | root)` — how much category `X` belongs to a domain anchored by `root` (e.g. `Physics`) — is a
**relational** quantity: the same category has different `μ` under different domains. So it is not a
single per-node scalar but the **cosine between two learned vectors**, one for `X` and one for the
`root`:

```
μ(X | root) ≈ cos( encode(X), encode(root) )
```

Learn the encoder so that cosine matches the LLM-provided `μ`. Because the root is just another encoded
category, the *same* node embeddings serve any domain — swap the root vector, swap the domain. This is
*analogous* to the graph-similarity work — `μ(X | root)` plays the role `Lin(X, root)` plays on the
category graph, but measured in semantic (vector) space. **It is not a literal identity:** the
node-vs-root `Lin` is degenerate (saturates at 1.0 — the root is the most-general node), so don't try
to validate the embedding μ against graph-side node-vs-root Lin (see the Step-4 note in the status
section).

## Architecture (`mu_encoder.py`) — an MLP, not a transformer

We worked out (see the design discussion) that **multi-head attention is the wrong tool here**: each
category is a single MiniLM-pooled vector, and self-attention over one token has softmax ≡ 1 — its
query/key projections become dead weight and value/output collapse to one linear layer (a redundant
`Linear` in an attention costume). The region-dependent behaviour we want — different units firing in
different parts of semantic space — is the **MLP nonlinearity**, optionally made explicit and
interpretable by a **gated MoE** (route to a region expert). So the encoder is an MLP/MoE, not a
transformer.

| knob | value | rationale |
|---|---|---|
| `d_model` | 384 | MiniLM scale |
| `n_layers` | 1 (configurable) | start small — few params generalise from few labels |
| `n_experts` | 1 (→ plain MLP; >1 → gated MoE) | explicit, inspectable region routing |
| init | MiniLM per-category embedding by Wikipedia id | warm start; random fallback here |

`encode(id) = blocks(init_embedding[id])`; `μ(a|root) = cosine(encode(a), encode(root))`. Each block
is `x + FFN(LN(x))` (FFN = MLP, or a soft gated mixture of experts). **Neighbour context** is folded in
cheaply by mean-pooling the category with a few neighbour vectors (a fixed-weight "sum of vectors");
*learning* those pooling weights is exactly attention-over-neighbours — the documented future upgrade,
the only place attention earns its keep. `python3 mu_encoder.py` runs the forward (MLP and MoE).

## Training objective — proven runnable (`train_cosine_mu.py`)

`python3 train_cosine_mu.py` learns per-category vectors directly (the simplest "encoder") to isolate
and **prove the objective on the real Haiku-scored fixture**: minimise `(cos(v_X, v_root) − μ_X)²` with
the **distance-biased sampling** from the design (categories closer to the anchor in the Wikipedia
graph are sampled more often). Pure stdlib, closed-form cosine gradient (finite-difference checked).

Result on `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (90 nodes, anchor `Physics`):

```
initial  MSE 0.41   corr(cos, μ) +0.05
final    MSE 0.0002 corr(cos, μ) +1.00
  μ 0.40  cos +0.40  Arsenic_compounds
  μ 0.90  cos +0.90  Atoms
  μ 1.00  cos +0.96  Sound
```

So `μ` is faithfully expressible as the cosine of learned vectors. **Read this as *capacity*, not
*inductive bias*:** 91 vectors × 16 dims = 1,456 free parameters fitting 90 targets (~16× over-
parameterised), so `corr → 1.00` proves the representation *can* encode μ, nothing about
generalisation. The real test — predicting μ for *unscored* categories — needs the shared MLP encoder
on MiniLM init, and is the separate project.

## What this prototype does NOT do (the separate project)

This environment has **no numpy / torch / network**, so:

- **MiniLM initialisation** needs the model weights (HuggingFace egress blocked here) — the encoder
  falls back to random init.
- **Training the full transformer** wants numpy/torch for speed (the pure-Python forward is for
  inspecting the architecture, not gradient descent at 384 dims).
- **Generalisation** needs more `μ` labels than the 90 single-anchor scores — cheap to generate with
  the existing Haiku pipeline (`scripts/physics_random_walk_candidates.py` + scoring), ideally
  **pairwise** `μ(a, b)` for varied anchors, sampled with the distance bias.

## Integration with the WAM-Rust core

Once the encoder produces dense `μ` for *all* categories (not just the scored 90), it removes the
density caveat on `descendant_mu_mass_gated` (gating no longer prunes through unscored connectors) and
feeds `gated_ic` / `lin_from_ic` directly — the same `μ` map, just dense and domain-swappable.

### Realized (no training): `gen_dense_mu.py` + `sanity_check_rust.py`

Kickoff prompt **A** below is now implemented as the *fastest path to unblock the graph work*:

- `gen_dense_mu.py` MiniLM-encodes every category name (`all-MiniLM-L6-v2`), cosines it to the
  `Physics` anchor, clamps via `to_membership`, and emits `dense_mu_physics.tsv` (`name<TAB>μ`, the
  fixture format, names verbatim). It is a **raw pretrained-embedding prior** — coarse semantic
  proximity, NOT the trained cosine-μ encoder and NOT authoritative (e.g. untrained MiniLM scores
  `Music ≈ 0.34`, where the curated fixture has it at `0`). Its job is *density*.
- `sanity_check_rust.py` renders this very template into a throwaway crate and runs the **real**
  `condense_scc → lift_mu_to_components → gated_ic → lin_from_ic` on the dense map (the loader is
  copied verbatim from `wikipedia_gated_similarity_tracks_physics_relatedness`). It checks the
  integration *contract* (coverage, finite/`+∞` IC, `lin ∈ [0,1]`), not the training-dependent physics
  ordering. Last run: **8247/8247 names resolve (100% coverage)**, `gated_ic` (threshold `0.3`) →
  7429 finite IC + 775 out-of-domain, `lin_from_ic(Electromagnetism, Optics) = 0.85`.

```
pip install -r requirements.txt          # torch, sentence-transformers (+ HF egress)
python3 gen_dense_mu.py --anchor Physics --out dense_mu_physics.tsv
python3 sanity_check_rust.py --fuzzy dense_mu_physics.tsv   # needs cargo
```

**Two things that will silently corrupt the integration if missed (review-flagged):**
1. **Clamp the cosine to `[0,1]` before emitting** (`to_membership` in `mu_encoder.py`). Cosine is in
   `[-1,1]`, but the Rust mass functions assume `μ ≥ 0`: `descendant_mu_mass` and `sketch_mu_mass` *sum*
   μ as mass, so a negative weight corrupts the mass / KMV estimate; only `descendant_mu_mass_gated`
   tolerates it (a negative just fails the gate). Training targets `μ ∈ [0,1]` so the model is only
   *asked* for non-negative, but an unlabelled very-dissimilar pair can produce a negative cosine at
   inference — hence the clamp at emission.
2. **Category-name strings must match `category_parent.tsv` exactly** (case- and underscore-sensitive).
   The Rust loaders intern names → integer ids; a name absent from the graph hits `unwrap_or(0.0)` and
   silently becomes `μ=0`, **re-activating the density caveat this whole project exists to close.**
   Emit names verbatim from the TSV, and assert your coverage (how many emitted names resolved).

**Exact pointers** (so a follow-up doesn't have to hunt):
- Consumers of `μ` live in `templates/targets/rust_wam/boundary_cache.rs.mustache`:
  `descendant_mu_mass` / `descendant_mu_mass_gated` (cone mass, gated), `gated_ic` (per-node IC),
  `resnik_from_ic` / `lin_from_ic` / `faith_from_ic` (similarity). They take `μ` as
  `HashMap<u32, f64>` keyed by the internal node id.
- The category graph (this prototype loads it): `data/benchmark/10k/category_parent.tsv`
  (`child<TAB>parent`). The sparse μ fixture: `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv`
  (`name<TAB>μ`). Candidate surfacing: `scripts/physics_random_walk_candidates.py`.
- Format the dense μ the same way (`name<TAB>μ`, '#'-comment header) so the existing Rust loaders
  (e.g. `wikipedia_gated_similarity_tracks_physics_relatedness`) consume it unchanged — just larger.

## Starting a new session (ML environment)

The pure-stdlib pieces run anywhere. To train the real encoder you need an environment with
`pip install -r requirements.txt` (torch, numpy, sentence-transformers) **and** HuggingFace egress
(for `all-MiniLM-L6-v2`). On Claude-Code-on-the-web that means an environment whose **network policy**
allows `huggingface.co` and whose **setup script** installs the requirements (an HF MCP server is
optional — it's an interface, not a bypass of the network policy). Large artifacts (full-graph
embeddings ~1.5 GB, weights) should be hosted externally and downloaded at runtime, **not** committed
to git — the cloud container is ephemeral, so they re-fetch per session. **Coordination:** the
prototype was merged to `main` (via #3280), so each new session **branches from `main` and opens its
own PR** — parallel sessions (A and B below) coordinate by cross-referencing their PRs.

Two ready-to-paste kickoff prompts:

**A — dense μ *without* training (fastest path to unblock the graph work).** ✅ **Realized** — see the
"dense μ *without* training, model comparison" progress section above (`dense_mu_direct.py`; **e5-small
picked** on smallest *calibrated* band). Kept here for the reasoning; re-run only to re-pick the model.
Use an **asymmetric** embedder, not MiniLM. μ(X | root) is *directional* ("is X a member of *root*'s
domain"); MiniLM is symmetric (generic relatedness — it conflates loose-relatedness with membership,
e.g. `Music ≈ 0.34`). e5-small and nomic have query/passage prefixes for exactly this. The choice
should be made on **decision-band size** (the budget driver: fewer categories near the cutoff ⇒ fewer
Haiku rescores in Prompt C), which is *not* the same as cosine-to-target — on prior project reports
e5-small wins cosine-to-target (0.953) but has poor *discrimination* (rank 84) while nomic has the best
discrimination (rank 1.2). So **measure it on our data** (zero LLM cost):
> In the UnifyWeaver repo, produce a dense μ map by running an embedding model directly — no training.
> For every category in `data/benchmark/10k/category_parent.tsv`, encode its name with an **asymmetric**
> model — try both `intfloat/e5-small-v2` (384-d, lightweight, `query:`/`passage:` prefixes) and
> `nomic-ai/nomic-embed-text-v1.5` (768-d, `search_query:`/`search_document:`). Encode the root as the
> **query** (`query: Physics`) and each category as the **document** (`passage: <name>`); take the
> cosine, clamp to `[0,1]` (`prototypes/mu_cosine/mu_encoder.py:to_membership`), emit `name<TAB>μ` in
> the `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` format (names verbatim). For **each** model
> report the **decision-band count** (categories with μ ∈ `[0.2,0.45]` near the 0.3 gate — MiniLM had
> 876) and a few sanity cases (`Music`, `Optics`, `Thermodynamics`). **Calibrate each model's cosine→μ
> against the 90-node Haiku fixture (linear fit) _before_ counting the band** — the raw band rewards a
> degenerate compressor (e5's raw band is 0 because it pins everything above the gate). The model with
> the *smallest calibrated* band is the budget-optimal pick (fewest Haiku rescores in Prompt C).
> Sanity-check the chosen map feeds
> `gated_ic` / `lin_from_ic` in the Rust core. Deps: `pip install -r prototypes/mu_cosine/requirements.txt`
> + HF egress. Branch off `main`, **open a PR**, report the band sizes there.

**B — train the encoder (the prototype's payoff):**
> Pick up the self-contained ML sub-project in the UnifyWeaver repo: read
> `prototypes/mu_cosine/README.md` (a complete handoff). Branch from `main` and **open your own PR**
> for this work. First verify `pip install -r
> prototypes/mu_cosine/requirements.txt` succeeds and HuggingFace is reachable; if not, report exactly
> what's blocked and stop — don't fake it. Then follow the README's ordered steps: port
> `mu_encoder.py`'s forward to torch, wire MiniLM init into `embed()`, generate+score training pairs
> (`gen_mu_pairs.py` emits candidates; **scoring spends LLM budget — confirm with me before running**),
> train the cosine-μ objective (validated by `train_cosine_mu.py`), validate generalisation (held-out
> μ, and agreement with the graph-side `lin_from_ic`). Heed the integration guards: clamp cosine to
> `[0,1]` and emit names matching `category_parent.tsv` exactly. Keep changes on your own branch;
> do not touch the merged WAM-Rust core.

**C — cascade-refine the dense μ where it changes cutoff decisions (correctness fix for A's output).**
This is the **recommended next step** now that A is realized with e5. Even the chosen e5 prior measures
*relatedness*, not *membership*, so it still rates a few loosely-connected apexes just above the gate
(e.g. `Music` ≈ 0.52, `Politics` ≈ 0.50 — not physics sub-topics), re-polluting the gated cones exactly
where gating decides. The *highest-value* rescore set is **not** the whole uncertainty band — it's the
categories whose μ sits close enough to the **cutoff** that a rescore could **flip the in/out decision**
(the active-learning / uncertainty-sampling principle: a label's value ≈ its probability of changing a
decision; a label far from the cutoff can't change anything). On the e5 calibrated map the decision band
(μ ∈ `[0.2, 0.45]`) is **≈653 cats** — a few hundred Haiku calls, ~tens of KB (measure the exact tight
straddle `[0.25,0.35]` from the emitted `dense_mu_e5.tsv`). Use the model-cascade pattern already in the
merged core (`wikipedia_model_cascade_haiku_then_sonnet`, `wikipedia_fuzzy_gated_hybrid_membership`,
geometric-mean = log-opinion-pool fusion). The Haiku re-scores are *also* the highest-value (boundary)
training labels for Prompt B — so **C feeds B** — and small/expensive, so **commit them as a fixture**
(see the commit-vs-regenerate rule under Persistent storage).
> In the UnifyWeaver repo, refine the chosen e5 dense μ map (`prototypes/mu_cosine/dense_mu_direct.py
> --model e5 --out dense_mu_e5.tsv`) with a model cascade. e5 rates *relatedness*, not domain
> *membership*, so it mis-scores a few categories near the gating cutoff (e.g. `Music` ≈ 0.52,
> `Politics` ≈ 0.50, only loosely physics-related, not physics sub-topics). Rescore only the categories
> that could **change a cutoff decision** — those within a margin of the `0.3` gate, e.g. μ ∈
> `[0.2, 0.45]` (≈650; or the tight straddle `[0.25,0.35]`), weighted toward the just-above-cutoff
> false-positive side. (Don't spend budget far from the cutoff — those can't flip.) Re-score with a
> Haiku subagent asking specifically about **membership** ("Is `<category>` a sub-topic/subfield *of
> physics*? 0..1, 1 = core physics, 0 = unrelated" — not "related to"), batched, same discipline as the
> `wikipedia_physics_*` fixtures. Fuse with the geometric mean `√(prior·haiku)` (the log-opinion-pool
> used in the merged cascade tests — it hard-vetoes a loose connection: `Music` → `√(0.52·0) = 0`). Keep
> categories far from the cutoff on the e5 prior untouched. Emit the refined μ (same format) **and commit
> the Haiku rescores as a fixture** (e.g. `tests/fixtures/wikipedia_physics_boundary_haiku.tsv`,
> `name<TAB>μ`) — it's small, expensive, and the reusable boundary training data for Prompt B. **Spends
> LLM budget, bounded to the decision band — report the band size and confirm with me before scoring.**
> Branch off `main`, open a PR.

### Progress — stream V: cutoff-band boundary labels (Prompt C, realized)

The decision-band Haiku rescore is done and **validates the μ/gating theory on real data**. Tools:
`boundary_band.py` (selects the band + surfaces ambiguous pairs), `fuse_and_validate.py` (geometric-mean
fusion + the three validations). Pipeline:

```
python3 dense_mu_direct.py --model e5 --out dense_mu_e5.tsv     # the calibrated prior (git-ignored)
python3 boundary_band.py   --mu-file dense_mu_e5.tsv           # decision band μ∈[0.2,0.45] → 654 cats
#   → ONE inline Haiku subagent scores MEMBERSHIP for the 654 band names (no file I/O); parent writes:
#     tests/fixtures/wikipedia_physics_boundary_haiku.tsv  (name<TAB>μ, committed — bought once)
python3 fuse_and_validate.py                                    # √(prior·haiku) fuse + validate
python3 check_feeds_rust.py --mu-file dense_mu_e5_fused.tsv     # fused map still feeds the Rust core
```

| result | value |
|---|---|
| decision band μ∈[0.2,0.45] | **654 cats** (360 below-gate, 294 above-gate "leak" side) |
| **decision-flip rate** | **279/654 = 42.7%** flipped across the 0.3 gate (278 IN→OUT leaks vetoed, 1 OUT→IN: `Geological_periods`) |
| just-above-cutoff false-positives removed | **278/294 = 94.6%** (e.g. `Sociology`, `Sex`, `Occupations`, `Team_sports`, `Soccsksargen`, `Internet` → μ=0) |
| node-vs-root Lin | **degenerate** — IC(Physics)≈0.05, all 51 in-domain fixture nodes saturate Lin≥0.999, Pearson(μ, Lin)=0.000 (it tracks specificity, not membership) |
| cosine→μ calibration vs 90-node fixture | **stable** — Pearson **+0.669** / Spearman +0.682 (≈ the +0.665 at emission) |
| Haiku spend | **~34k tokens** (2 inline subagent spawns, 0 tool calls), well under the ~82k/window cap |

This **confirms the active-learning premise**: rescoring only the band (not the whole graph) moved 42.7%
of those labels across the gate, concentrated on the leak side — labels far from the cutoff couldn't
flip and were correctly left on the prior (free). The fusion `√(prior·haiku)` hard-vetoes loose links
(`Fashion` 0.40→0). **Known residual:** the literal ceiling 0.45 excludes ~119 cats in (0.45,0.55]
including the named `Music` (0.523) / `Politics` (0.500) — a hard-veto rescore *can* flip those too, so
0.45 is a budget heuristic, not a correctness bound. A cheap follow-up window can raise the ceiling to
~0.55 (one more batch) to catch them; left out here to honour the [0.2,0.45] band + 2-spawn discipline.

## Current streams (T and V) + Haiku budget discipline

Prompts A–C above are the original handoff (A realized, B/C refined below). The two **active** streams,
to run as parallel HF-egress sessions, are **T (training)** and **V (theory validation)**. T consumes
labels and trains the encoder; V goes slow, identifies the decision band, and produces boundary labels
that feed back into T (V → T). T can start immediately on the merged `mu_pairs_scored.tsv`; V's boundary
fixture is a second training input T folds in on a later pass. Each opens its own PR; coordinate by
cross-referencing them.

**Haiku budget — the metering math (so the cap isn't guessed).** The $100/Max-5× plan's 5-hour window
is unpublished; community data puts it at **~220k tokens** measured in *Sonnet-equivalent* units. Haiku
is **3.75× cheaper** than Sonnet (input $3.00→$0.80 / Mtok, output $15→$4, same ratio), so the **Haiku**
window is ~220k × 3.75 ≈ **825k Haiku tokens**, and a 10%-for-labeling budget is **≈82k Haiku tokens per
window** (conservative — if the window is actually Opus-equivalent, Haiku headroom is ~5× larger again).
A full decision-band pass (~653 cats) is ~30k tokens done efficiently, so it fits one window with room.

**Spend the budget well (the real cost is the spawn, not the items):**
- **Minimize subagent spawns.** A Haiku-subagent's fixed overhead (system prompt + any tool round-trips)
  dominates — empirically ~95% of the cost. Use **one** Haiku subagent for the whole batch (two at most),
  pass the item list **inline in the prompt**, return `name<TAB>μ` lines **inline in the reply**, and do
  **no Read/Write inside the subagent** (file I/O round-trips are what inflate it). The parent parses the
  reply and writes the fixture. (A multi-spawn + file-I/O run cost ~73k for 200 pairs; one inline spawn
  does the same for ~15k.)
- **Batch ≥40 items/call**, score **only** the decision band; far-from-cutoff nodes stay on the prior,
  negatives are μ=0 (free). **Hard stop at ~82k Haiku tokens/window** — checkpoint, commit, continue next
  window if needed.
- **Labels are bought once and committed** as a fixture — never re-score. Even at the conservative
  220k-Sonnet-equiv estimate, a one-time band pass is a small one-window slice you never pay again.

**T — train the dense-μ encoder on the scored labels** (HF-egress session):
> Pick up the μ-cosine ML sub-project in the UnifyWeaver repo. Read `prototypes/mu_cosine/README.md`.
> Branch from `main`, open your OWN PR, don't touch the WAM-Rust core. GATING CHECK first (report+stop
> if it fails): `pip install -r prototypes/mu_cosine/requirements.txt` and confirm HuggingFace is
> reachable (all-MiniLM-L6-v2 downloads + encodes). Then: (1) train `train_cosine_mu_torch.py --mode
> pairs --minilm` on the merged `mu_pairs_scored.tsv` (200 scored positives + 1000 free μ=0 negatives);
> start `n_layers=1, n_experts=1`, Adam, and do NOT re-apply `--dist-bias` (the pairs already encode
> distance via the explicit negatives). (2) Hold out a fraction of positives; report held-out pairwise-μ
> corr + MSE. (3) Re-run `validate_lin_agreement.py`: the single-anchor encoder gave pairwise graph-Lin
> vs semantic-cosine Pearson ≈ −0.13; the varied-anchor labels should move it positive — report the
> delta. (Do NOT validate against node-vs-root Lin — degenerate, saturates at 1.0.) (4) Emit dense μ
> (`emit_dense_mu.py`) and feed `check_feeds_rust.py` (100% coverage, IC general→specific, lin ∈ [0,1]).
> Spend NO Haiku budget here — training is local compute; if you need more/boundary labels, request a
> batch from stream V. Guards: clamp cosine to [0,1]; emit names verbatim. Report held-out corr + the
> lin-agreement delta in the PR.

**V — validate the theory + produce the cutoff-band boundary labels** (HF-egress session, go slow):
> In the UnifyWeaver repo, validate the μ/gating theory on real data and produce the cutoff-band labels.
> Read `prototypes/mu_cosine/README.md` (Prompt C + the budget discipline). Branch from `main`, open your
> OWN PR, don't touch the WAM-Rust core. GATING CHECK first (deps + HuggingFace reachable). Then:
> (1) emit the e5 prior: `dense_mu_direct.py --model e5 --out dense_mu_e5.tsv` (git-ignored); re-confirm
> `check_feeds_rust.py`. (2) Identify the DECISION BAND — μ ∈ [0.2, 0.45] around the 0.3 gate (~653);
> prioritise the just-above-cutoff side (0.30, 0.45] — the loose-relatedness leaks (Music ≈ 0.52,
> Politics ≈ 0.50). Only nodes whose rescore could FLIP the in/out decision are worth labelling. Surface
> the decision-band PAIRS too (band category × Physics-subdomain) where membership is ambiguous.
> (3) Score for MEMBERSHIP, directional — NOT relatedness: "Is `<category>` a sub-topic/subfield OF
> physics? 0..1 (1 = core physics, 0 = unrelated)." Follow the budget discipline above: ONE inline Haiku
> subagent, items in the prompt, scores in the reply, no file I/O, hard stop ~82k Haiku tokens.
> (4) Fuse with the geometric mean √(prior·haiku) (log-opinion-pool — hard-vetoes loose links: Music →
> √(0.52·0) = 0). (5) VALIDATE + document: the decision-flip rate (how many band cats the rescore moved
> across the gate — validates the active-learning premise), confirm node-vs-root Lin is degenerate, and
> confirm the cosine→μ calibration vs the 90-node fixture still holds. (6) COMMIT the rescores as
> `tests/fixtures/wikipedia_physics_boundary_haiku.tsv` (`name<TAB>μ`) — small, expensive, reusable; it
> FEEDS stream T. Report the band size, decision-flip count, and token spend in the PR.

### What to commit vs. host externally

The deciding question is **cheap-and-regenerable vs. expensive-and-irreproducible**, *not* raw size:
- **Commit (in git, as fixtures):** the **LLM-labelled** data — the Haiku boundary rescores (Prompt C),
  the existing `tests/fixtures/wikipedia_physics_*`. These are expensive (LLM budget), *not*
  reproducible, the highest-value asset, and small (the decision band is ~tens of KB). Never re-buy them.
- **Regenerate or host externally (rclone, below):** **model-derived** artifacts — embeddings, the full
  dense-μ map (`dense_mu_e5.tsv` is ~150 KB and regenerable from the model in minutes — git-ignored, with
  only `dense_mu_e5.sample.tsv` committed as an illustrative sample), checkpoints, the 1.5 GB full-graph
  embeddings. Cheap to remake, so don't bloat git; host the big ones.

### Persistent storage (rclone + Dropbox)

The cloud container is **ephemeral** and the ~30 GB disk doesn't survive between sessions, so large
artifacts that outgrow git — MiniLM cache, model checkpoints, full-graph embeddings (~1.5 GB), scored
label sets — should live in external storage and be pulled per session. The agreed approach is
**`rclone` against a Dropbox *app-folder* app** (real VM egress, unlike a connector, which is
context-only). `cloud_setup.sh` is a credential-free template for the environment's setup-script field.

- **Security boundary is the Dropbox app, not rclone.** Create it as *Scoped access — App folder* so
  the token physically can't reach anything outside `/Apps/<YourApp>/`. A config path-prefix is **not**
  a restriction (any command could still type `dropbox:` and reach a full-Dropbox app's whole account).
  Scope permissions to `files.content.read` (+ `.write` only if you persist results).
- **Credentials**: set `DROPBOX_APP_KEY` / `DROPBOX_APP_SECRET` / `DROPBOX_REFRESH_TOKEN` in the
  env-vars field yourself (visible to env editors, no secrets store → app-folder scoping is the safety
  net). Generate the refresh token once via `rclone config` on your own machine. **An agent must not
  enter credentials for you.**
- **Workflow**: `rclone copy "dropbox:datasets" ./data` (down) / `rclone copy ./outputs
  "dropbox:outputs"` (up). `copy` never deletes — prefer it over `sync`. Pre-pull stable files in the
  setup script so they bake into the snapshot; fetch only changing data at runtime.
- **Parallel sessions** (prompts A and B at once) can share one app folder — coordinate writes by
  subfolder (`dropbox:dense-mu/` for A's output, `dropbox:checkpoints/` for B) and by cross-referencing
  their PRs.
- *Alternative:* dedicated object storage (S3/GCS/R2) is arguably a better fit for machine-readable
  blobs (cleaner credential scoping, no token-refresh fragility); `rclone` supports those backends too,
  so the same workflow applies — just a different `[remote]` in `rclone.conf`.
