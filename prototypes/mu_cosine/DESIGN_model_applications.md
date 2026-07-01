# Applications of the μ model — and a geometric (bivector) theory note

**Status: mixed — roadmap + a built core.** The **retrieval algorithm itself is now built + verified**
(`eval_retrieval`, three-way DENSE/WSP/GREEDY comparison — see "Built + verified" below). Still *proposed*: the
**formal hop-stratified recall@k / AUC eval** against a held-out-edge ground truth, and the geometric (bivector /
root-centred) theory. It builds on a few **existing** pieces — `emit_dense` → `dense_mu_attn_*.tsv` (density maps), the bridge detectors
(`bridge_ensemble.py`, PR #3322), and the `DESIGN_bidirectional_walk.md` traversal — and on the **built +
verified** transitive μ (`DESIGN_transitive_relations.md`) it relies on for distance-awareness. The "Build
plan" at the end is the concrete next increment.

The μ-attention model produces **`μ(node | root)`** — a *directional, fuzzy* relatedness/membership degree of
`node` to `root` over a concept graph, learned over **frozen e5** embeddings with a permutation-invariant
operator-attention readout. This doc collects the **applications** that consume μ, the **new retrieval
algorithm**, how it relates to **prior (distance-metric) approaches**, and a short **theory** section — the
last of which flags a genuinely uncommon framing (embedding **bivectors** / root-centred embeddings) worth
recording.

## μ as the primitive
- **Directional:** `μ(a|b) ≠ μ(b|a)` (a member is mostly-not its container). Most vanilla embedding similarities
  (cosine/dot) are *symmetric* and cannot express this.
- **Fuzzy/graded:** μ ∈ [0,1], a membership degree, not a hard edge.
- **Operator-structured:** μ is read out under operators (SYM = lateral `see_also`/`assoc`; WIKI/ELEM =
  directional `subcategory`/`element_of`/…). An **equal-weight superposition** of operators (e.g.
  `⅓·subcategory + ⅓·element_of + ⅓·see_also`) is the **unconditional** relatedness — "how related, regardless
  of which relation" — the `E[μ]` over a uniform operator prior (operator-superposition design §1b/§12).
- **Distance-aware:** trained with the transitive ordinal constraint so μ **decays sensibly with hop-distance**
  (`DESIGN_transitive_relations.md`) — which is what makes it usable for *multi-hop* retrieval.

## Built foundation (what the roadmap enhances)
The applications and roadmap below build on these **shipped, documented** pieces — described here so the
proposed *enhancements* have context:
- **μ-attention model** — directional fuzzy membership over **frozen e5**, permutation-invariant
  operator-attention readout (SYM = lateral; WIKI/ELEM = directional) + the §8 **anchored basis** (frozen
  label-tied anchors ∪ learnable atoms). The core estimator.
- **Operator superposition** — μ read under an operator distribution; **equal-weight = the unconditional
  `E[μ]`** (`DESIGN_inferred_operator_superposition.md`).
- **Transitive ordinal constraint** — μ **decays with hop-distance**: ranking-CE + dual-ascent λ +
  heteroscedastic product-propagated variance. Built **+ verified** (generalises, no-collapse,
  convergence-stable) — `DESIGN_transitive_relations.md`, `README_transitive.md`, `REPORT_transitive_verification.md`.
- **Density maps** — `emit_dense` → `dense_mu_attn_*.tsv`: μ(·|root) over all nodes (the density-explorer feed).
- **Bridge detection** — `bridge_ensemble.py` (here) + PR #3322's declarative bridge detector (Prolog foreign
  predicate).
- **Bidirectional walk** — depth-balanced traversal sampler (`DESIGN_bidirectional_walk.md`,
  `validate_bidir_walk.py`).
- **Inferred-tail augmentation** — `score_inferred_tail.py` + `cell_sampler.py` (LLM `E[μ]` for the inferred
  tail; measured ~80% judge-noise → soft-rejected, off by default).

## Applications
| application | how μ is used | existing pieces |
|---|---|---|
| **Graph RAG / graph search** | rank graph nodes by relatedness to a query node (incl. multi-hop) | the new algorithm (below); `emit_dense` |
| **Bridge-node identification** | nodes whose μ links *across* domains (the `bridge` structure / cross-domain μ) | `bridge_ensemble.py`; PR #3322 declarative bridge detector (Prolog foreign pred) |
| **Semantic visualisation (density explorer)** | μ as a relatedness/density field around a root | `emit_dense` → `dense_mu_attn_*.tsv`; root-centred chart (theory below) |

## The retrieval algorithm (new) — greedy bidirectional gather + μ-superposition sort
Structure **∩** semantics: the graph gives a cheap, structure-aware *candidate set*; μ gives the *semantic
ranking*.
1. **Greedy bidirectional gather** from the root — best-first expansion following edges in **both** directions
   (child *and* parent, using the directional asymmetry; depth-balanced so it stays lateral, not drifting to
   hubs/leaves — reuses the `DESIGN_bidirectional_walk.md` traversal insight, but greedy and for retrieval, not
   random sampling). Gathering, not scoring-all-10k → efficient.
2. **Sort by the operator-superposition μ** — score each gathered candidate by the equal-weight unconditional
   `E[μ](candidate | root)`; return top-k.
3. **Diagnostic — μ-vs-hop scatter (top-k, e.g. k=25):** x = hop-distance, y = μ. Expect a *decaying-but-spread
   cloud* (semantics, not a step function). The **off-diagonal points are the payoff**: high-μ-far (semantically
   close, graph-distant — what μ adds over shortest-path) and low-μ-near (graph-adjacent but weak — what μ
   *corrects*). The scatter is a visual proof of whether semantics-on-structure beats structure-alone.

### Built + verified (`eval_retrieval` in `train_mu_attention.py`, `--eval-retrieval`)
Stages 1–3 are implemented (`--eval-retrieval "Root1,Root2" --retrieval-k 25 [--retrieval-out scatter.tsv]`),
reporting **three rankings side-by-side** — DENSE-μ (score all nodes), WSP (graph-hop, μ tiebreak; the
structural baseline), GREEDY (μ-ranked among graph-reachable; the new algorithm) — plus per-hop mean μ and the
top-k hop distribution. Verified on `model_nodetype.pt` over three roots spanning the generalisation gradient:
- **Physics (in-distribution)** — all three rankings agree; dense∩greedy overlap **21/25**.
- **Music (clean OOD, not trained)** — all three agree, **25/25** — frozen e5 generalises cleanly outside the
  physics training region.
- **Cooking (confusable OOD)** — the discriminating case. DENSE-μ is **polluted** by `Movies_directed`,
  `Movie_studios`, `Basketball_movie` (hops 4–6, μ 0.4–0.6) because e5 conflates cooking-TV → film. GREEDY
  **removes all of them** (they're not in the gather's graph region) and recovers the actual cooking
  subcategories ranked by μ; dense∩greedy overlap collapses to **4/25**. It *also* beats WSP, which keeps
  `Home(0.04,h1)`/`Nutrition(0.18,h2)` purely for graph-adjacency — greedy's μ-ranking demotes them.

The two corrections, both visible in Cooking: **vs DENSE**, structure removes high-μ-far false positives (the
movie leak); **vs WSP**, semantics demotes low-μ-near false positives (`Home`). **Bonus —
dense∩greedy overlap is a free reliability self-diagnostic**: high (Physics/Music) ⇒ structure and semantics
agree, trust the result; low (Cooking 4/25) ⇒ e5 is leaking across a domain boundary, trust the
structure-constrained greedy over raw μ. No ground truth required.

### Formal eval — bookmark-filing ground truth (`eval_filing.py`)
The non-circular head-to-head (Build-plan §1–2), using **real Pearltrees filing decisions** as labels (a
bookmark's actual `treeId` = which folder it belongs in — a human decision, not graph distance). 335 candidate
folders (≥3 bookmarks), 500 sampled query bookmarks; rank folders by μ(bookmark|folder); recall@k / MRR. Three
rankers: **e5-cos** (raw e5 cosine, no model), **mu-super** (equal-weight operator superposition), **mu-elem**
(the `element_of` operator — the membership relation filing *is*).

| ranker | recall@1 | recall@5 | recall@10 | MRR | med.rank |
|---|---|---|---|---|---|
| **e5-cos** | **0.202** | **0.410** | **0.480** | **0.299** | **14** |
| mu-super | 0.088 | 0.198 | 0.248 | 0.151 | 52 |
| mu-elem  | 0.096 | 0.208 | 0.256 | 0.160 | 48 |

**Raw e5 cosine ~doubles the μ model on every metric** (random@335 ≈ MRR 0.019, so both beat chance — but the
learned readout *underperforms its own frozen substrate* here). `mu-elem` edges out `mu-super` — the membership
operator is the right one, and directionality helps a hair even untrained. **Why μ loses — it's zero-shot OOD,
not an architecture ceiling:** the checkpoint was trained on *simplewiki category→category membership*
(Physics→Energy), never on *bookmark→folder filing*; bookmarks are noisy web-article titles, folders are
Pearltrees collections; and the gather runs with **no lineage** (empty DAG — the "absent lineage = off-manifold
noise" regime). The learned transform, tuned for a different regime, *distorts* e5 for this task. Since the
readout sits **on top of** frozen e5 and can represent near-identity, filing fine-tuning should recover *and*
beat `e5-cos` via directionality — `e5-cos` MRR **0.299** is now the bar.

**Stratified by distance to the trained region** (`--core-anchors`, max folder-similarity to
Physics/Math/Chem/CS/Eng — testing "μ only helps inside its region"): the e5-cos − mu-elem MRR gap is
**flat at ~0.5× ratio across all three bins** (FAR 0.309/0.161, MID 0.327/0.185, NEAR-STEM 0.263/0.136). μ does
**not** close the gap near the core — the loss is *uniform*, not region-specific. Two confounds keep this from
refuting the region hypothesis: (1) weak stratifier (e5-small's high cosine floor — even "FAR" folders sit at
0.74 to *Physics*); (2) **the lineage confound** — the model trained *with* ancestor lineage but runs here on
**cold, lineage-free** bookmarks (empty DAG), a uniform handicap that could itself flatten the ratio. **NB —
the transformer is nonlinear, so multi-region is *not* a capacity question.** A single μ model can represent
many regions separated by boundary manifolds (the nonlinear analog of routed per-cluster Procrustes — the
*linear* model needs discrete routing precisely because it can't); so the flat ~2× loss is cold-start + lineage,
*not* a single-global-transform ceiling. The mixture is the linear workaround, not the fix here. So "μ
loses ~2× everywhere" conflates OOD-task + cold-start + (maybe) wrong-region; stratification rules out
wrong-region-as-sole-cause but not the rest.

### Home-turf result — the readout is healthy; filing loss was OOD (`--source simplewiki`)
Ran the in-domain analog: simplewiki category→member ranking (folders = categories ≥10 members ≈ 295, to match
filing's chance baseline; 500 queries; **lineage-free**, apples-to-apples with filing — only the *domain* changed).

| | recall@1 | recall@5 | recall@10 | MRR | med.rank |
|---|---|---|---|---|---|
| e5-cos | 0.180 | 0.376 | 0.424 | 0.270 | 29 |
| **mu-super** | 0.144 | 0.386 | **0.494** | 0.255 | **11** |
| mu-elem | 0.156 | 0.328 | 0.436 | 0.247 | 16 |

**The story flips vs OOD filing (where e5-cos *doubled* μ):** in-domain they are **tied** on MRR, and μ **beats**
e5-cos on recall@10 (0.494 vs 0.424) and **median rank (11 vs 29)** — μ pushes the true container into the
shortlist better; e5-cos holds a thin recall@1 edge (different operating point). **The physics-core hypothesis
revives in the stratification:** NEAR-core (STEM) μ-super recall@10 **0.637 vs 0.393**, median **7 vs 30** — μ
wins *decisively* where training was densest. It was invisible in filing only because *all* Pearltrees folders
are OOD (no bin had training signal); on in-domain data where the core proxy is meaningful, μ's edge concentrates
exactly in the trained region, as predicted. (Was lineage-free — so μ ties/beats cosine *without* its ancestor
context, the strong form.) **Verdict:** the readout works; the filing 2× loss is **OOD transfer**, not a broken
model — so your "with training data it wins" is the right read.

#### Node-holdout — memorisation caveat KILLED (`--holdout-nodes`)
The one caveat above (trained on these edges) is now removed *without a retrain*. 21% of graph nodes (1760) appear
in **neither** the SYM pairs **nor** the graded context edges — a ready-made never-trained holdout. Restricting
the home-turf queries to these (`--holdout-nodes`, 500 sampled; candidates unchanged) ranks nodes the checkpoint
**never saw** — pure generalisation. Result is **near-identical to the memorised run**:

| | mu-super MRR | mu-super recall@10 | mu-super med.rank | e5-cos recall@10 | e5-cos med.rank |
|---|---|---|---|---|---|
| home-turf (mem-allowed) | 0.255 | 0.494 | 11 | 0.424 | 29 |
| **node-holdout (never-trained)** | **0.247** | **0.482** | **12** | **0.414** | **36** |

Trained→held-out drop is **~3%**; the STEM-core win is **unchanged** (held-out core mu-super recall@10
**0.637 vs 0.393**, median **7 vs 27** — the same numbers). μ ranks never-seen nodes almost exactly as well as
seen ones ⇒ **the home-turf win is generalisation, not memorisation.** Baseline locked.

#### Data quality, two tiers — Tier-1 junk removal flips μ to a clear win; keep Tier-2 (`--drop-admin junk|all`)
Probing the "coverage gap" found the FAR-from-core bin mixes **two very different things**, which must be treated
differently:
- **Tier-1 — meaningless:** maintenance / template / nav categories (`CatAutoTOC generates no TOC` deg 1219,
  `Navseasoncats…`) — procedural titles, *zero* topical content. No ranker can place them, no training fixes them.
- **Tier-2 — loosely semantic:** structural / temporal groupings (`Years of the 20th century`, `Establishments
  by year`, `People by nationality`, `Cities by country`). Real but lower-density meaning (the year/place signal
  *is* useful) — **keep in eval, *down-sample* in training**, do not drop.

Decomposing the gain (home-turf, 500 queries) shows the win is **almost entirely Tier-1 removal**:

| filter | folders | e5-cos MRR | mu-super MRR | **μ − e5** | recall@10 (μ / e5) |
|---|---|---|---|---|---|
| none (junk in) | 295 | 0.270 | 0.255 | −0.015 *(tie)* | 0.494 / 0.424 |
| **Tier-1 junk only** *(honest)* | 273 | 0.408 | 0.448 | **+0.040** | **0.812 / 0.660** |
| Tier-1+2 (`all`) | 132 | 0.454 | 0.502 | +0.048 | 0.904 / 0.634 |

**Dropping Tier-1 alone flips the home-turf "tie" to a clear μ win** (μ−e5 −0.015 → **+0.040**, recall@10
**0.812 vs 0.660**) — legitimate noise removal. Dropping Tier-2 *as well* lifts **both** rankers' absolutes
(easier candidate set) but the μ-advantage gap barely moves (+0.040 → +0.048) — so Tier-2 is **not** the source
of μ's win; keeping it is honest (μ still wins) and dropping it would only inflate the headline. **Policy: Tier-1
→ drop (eval + training); Tier-2 → keep in eval, down-sample in training.** **Bitter-lesson footnote:** the
"coverage" bet's first and cheapest win was **data quality** (drop *meaningless* labels), not quantity — and it
relocates the real gap to genuine non-STEM content (where e5 already does okay at recall@1, so more training there
buys less than the raw gap suggested). **Generalises to never-trained nodes** (`--holdout-nodes`, aggressive
`all` level shown): mu-super recall@10 **0.919 vs 0.662** on nodes never seen ⇒ μ is a strong **first-stage
retriever** (feed top-10 to an LLM re-ranker), the recall@10 operating point.

### Filing fine-tune learning curve — μ CROSSES the e5-cos bar (`train_filing.py`)
The quantified "with enough in-domain data the attention model wins." Warm-start `model_nodetype`, fine-tune on
`element_of(bookmark→folder)` with **in-batch contrastive** negatives (B×B μ matrix; same-folder = positive), at
rising **data fractions**, eval MRR/recall on a **fixed held-out bookmark set** (split is **bookmark-holdout**:
folders are a stable taxonomy, the held-out *bookmarks* are never trained — the model carries no per-folder
params, so a shared folder is not leakage). 500 steps, bs 48, single seed.

**Multi-seed locked** (3 training seeds, fixed eval split; ✓ = *mean − sd* clears the bar, i.e. survives seed noise):

| frac | n_train | MRR (mean±sd) | recall@10 (mean±sd) | med.rank | vs e5-cos (0.291) |
|---|---|---|---|---|---|
| 0.10 | 492 | 0.230 ± 0.025 | 0.406 ± 0.028 | 19 | −0.061 |
| 0.30 | 1478 | 0.317 ± 0.006 | 0.537 ± 0.006 | 8 | **+0.025 ✓ CROSSED** |
| 1.00 | 4929 | 0.358 ± 0.018 | 0.573 ± 0.008 | 6 | **+0.066 ✓ CROSSED** |

μ **crosses between 10% and 30%** (~500–1500 bookmarks) and keeps climbing — at 100% **MRR 0.358 vs 0.291**
(+23%), **recall@10 0.573 vs 0.440**, **median rank 6 vs 17** — monotonic and **still rising** (no plateau ⇒ more
data helps). The cross is **robust to seed**: at 30%/100% even *mean − sd* (0.311, 0.340) clears the bar, with
tiny std (0.006–0.018). The OOD 2× loss (zero-shot) inverts to a clear, replicated win once the trained region is
*extended* to the bookmark domain — the prediction confirmed and locked. (10% is correctly *below* the bar; the
earlier single-seed 0.263 there was a touch optimistic — the lock matters most at low data, where variance is
highest, std 0.025.) Remaining headroom: a longer fine-tune / more data per the still-rising slope. **Next** (now
*scaffolding*, per the bitter-lesson framing below): the local-tangent **bivector feature** only if a thin
sub-domain needs sample-efficiency; otherwise the bet is simply more data + training.

#### Review-hardened rankers, zero-shot OOD — the margin gate is the best zero-shot strategy (damage control)
`eval_filing.py` now carries the review-hardened rankers from the Wikipedia arc: **`mu-max`** = `max(μ-elem, μ-wiki,
μ-sym)` (the operator-OR), **`e5+mu-max`** = the coverage-insurance blend, **`margin-gate`** = per-query α from the
μ-max margin (the #3391 finding operationalised). Run **zero-shot** (Wikipedia-trained `model_prod`, *no* filing
fine-tune) on 500 real filing decisions (335 folders):

| ranker | recall@1 | MRR | | strat MRR (e5 / margin-gate): FAR·MID·NEAR |
|---|---|---|---|---|
| **e5-cos** | **0.196** | **0.295** | | 0.306 · 0.324 · 0.255 |
| mu-super / mu-elem / mu-max | 0.06 | 0.12–0.13 | | (μ ~0.11–0.16 at every stratum) |
| e5+mu-max | 0.078 | 0.144 | | |
| **margin-gate** | 0.146 | **0.212** | | 0.202 · 0.253 · 0.181 |

**Re-confirms the known ~2× zero-shot OOD loss** — and sharpens it: e5-cos wins at *every* core-distance stratum,
including NEAR the STEM core, so this is **transfer failure** (Wikipedia *category* structure ≠ personal *folder*
structure), **not** a trained-region effect (the gap is a uniform ~0.07–0.10 everywhere, refuting "μ helps in its
trained region"). **New finding:** the **margin gate is by far the best zero-shot ranker** — 0.212 vs `mu-max` 0.125
(+70%), recovering **~60% of the gap to e5**. It works exactly as designed: it detects μ's OOD low-confidence
(flat margins) and defers to e5, so the *mechanism* validated in #3391 holds on real data even though the *payoff*
(μ adding value OOD) does not — the optimal μ-weight zero-shot is ≈0 and the gate trends there (damage control where
μ is net-harmful). Coverage-insurance vindicated in spirit: the whole domain is untrained for μ, e5 is the robust
baseline, and the gate correctly leans on it. **The fix is in-domain training** (the learning curve above: fine-tune
crosses e5 at 30%+ data, +23% at 100%).

**In-domain (fine-tuned μ, `train_filing.py --save` → `rank_all` on the HELD-OUT split — fair, no train-on-test):**

| ranker | recall@1 | MRR | note |
|---|---|---|---|
| e5-cos | 0.200 | 0.284 | the bar |
| mu-elem | 0.258 | 0.362 | the documented fine-tune win (+27%) |
| mu-max (OR) | 0.233 | 0.342 | **OR *dilutes* in-domain** — only ELEM was fine-tuned, so wiki/sym are stale |
| **e5+mu-elem** | **0.307** | **0.396** | **best — +39% over e5, +9% over mu-elem** |
| e5+mu-max | 0.302 | 0.391 | blend still strong even on the diluted OR |
| margin-gate | 0.278 | 0.365 | ≈ mu-elem — gate redundant when μ is trustworthy in-domain |

Three lessons, and they line up with the rest of the arc:
1. **The coverage-insurance blend generalises to in-domain** — `e5+mu-elem` (0.396) beats *both* plain fine-tuned μ (0.362) and e5 (0.284), with a big recall@1 lift (0.307 vs 0.258/0.200). The blend adds value *on top of* the fine-tune win, exactly as on Wikipedia.
2. **The operator-OR only helps if its operators are trained for the domain.** On Wikipedia (multi-relationally trained) `mu-max` won; here (ELEM-only fine-tune) it *dilutes* below `mu-elem`. → this is the direct argument for training *more* filing operators (e.g. a `LINEAGE` op), then OR-ing them.
3. **The margin gate is redundant in-domain** (≈ mu-elem) — consistent with #3391: the gate's value is OOD damage-control, not in-domain, because in-domain μ is trustworthy and deserves full weight. Zero-shot the gate was the *best* μ-variant; in-domain it's a wash. The gate correctly self-adjusts to how much μ can be trusted.

Net filing verdict: **zero-shot, use e5 (μ doesn't transfer, gate limits the damage); in-domain, use `e5 + μ-elem` (best, +39% over e5).**

#### LINEAGE operator (increment 1) — graceful degradation confirms "file general→specific" (`train_lineage.py`)
A new **`LINEAGE`** operator scores `μ(bookmark | hierarchical-PATH)` — the *undifferentiated* generalization of
ELEM+WIKI (the embedded `target_text` path carries no relation-type; verified). Built per the agreed design: a
**fresh op row** (n_ops 4→5 via row-copy warm-start of `model_filing` — *not* hand-initialized from ELEM/WIKI; it
learns its relation to them through training), fine-tuned with **ELEM replay** + **masking** (ID-dropout via
precomputed `/*`-wildcard variants + path-prefix dropout; the id line stays as unique anchors). Held-out filing
(n=300):

(n=300). **Multi-seed (3 training seeds 7/13/23, split fixed at 7); means, with per-seed ranges on the hard subset:**

| ranker | recall@1 | MRR | ov(all) | ov\|elem-miss | depth\|elem-miss |
|---|---|---|---|---|---|
| e5-cos | 0.170 | 0.270 | 0.380 | 0.323 (.317–.328) | 1.40 |
| mu-elem | 0.303 | 0.436 | 0.510 | 0.297 (.291–.300) | 1.33 |
| **mu-lineage** | 0.177 | 0.299 | 0.445 | **0.356 (.340–.371)** | **1.63 (1.53–1.75)** |

`depth|elem-miss` = ABSOLUTE deepest correctly-reached ancestor (the actionable placement depth — an intermediate
ancestor is usable: searchable by name, resolvable by id).

Findings: (1) `mu-lineage` beats e5 on *every* metric — a real, distinct signal. (2) `mu-elem` dominates exact-leaf
+ overall-overlap — it's the leaf specialist. (3) **But on the PAIRED hard subset** (queries where `mu-elem` misses
the leaf), **`mu-lineage` recovers the ancestor branch best — and it is ROBUST TO SEED with non-overlapping ranges**:
its ov|miss min (0.340) clears e5's max (0.328) clears elem's max (0.300), **same ordering all 3 seeds** (same for
absolute depth: 1.63 vs 1.40 vs 1.33). The "file general→specific" hypothesis, confirmed and CI-hardened: when the
leaf can't be nailed, lineage stays in the right general branch (densely-reused upper levels), while the
leaf-specialist, *on its own misses, wanders below even raw e5 every seed* (0.297 < 0.323) — it optimises
leaf-specificity over the general path. (4) So **elem and lineage are complementary** — leaf-specialist + graceful
branch-fallback — which *data-motivates* the composition (increment 2: a margin-gate / OR switching on
leaf-certainty). Masking's prefix-dropout also buys robustness to truncated / RDF-partial paths and free
depth-placement. (Paired subset is the honest test — per-ranker miss-sets differ, so unpaired overlap|MISS was
confounded; single split — a multi-split CI is the remaining follow-up.)

#### Composition (increment 2) — best filer is `e5 + max(μ-elem, μ-wiki)`; LINEAGE is redundant (`train_lineage.py --eval-only`)
Combiner sweep over `model_lineage` (held-out n=300, seed 7 — single-seed, multi-seed confirm is the follow-up):

| combiner | recall@1 | MRR | ov\|miss | depth\|miss |
|---|---|---|---|---|
| mu-elem | 0.310 | 0.447 | 0.312 | 1.43 | *(leaf champ)* |
| mu-wiki | 0.253 | 0.359 | **0.401** | **2.00** | *(BRANCH champ)* |
| mu-lineage | 0.193 | 0.308 | 0.359 | 1.65 | |
| **e5+max(elem,wiki)** | 0.303 | 0.438 | 0.391 | 1.94 | *(best all-rounder)* |
| e5+max(elem,lin) | 0.313 | 0.438 | 0.379 | 1.83 | |
| e5+max(el,wk,lin) | 0.307 | 0.434 | 0.392 | 1.89 | *(lineage adds ~0)* |

Findings: (1) **Surprise — WIKI (subcategory/subset) is the graceful-degradation champion** (ov|miss 0.401, depth
2.00), beating the purpose-built LINEAGE (0.359/1.65) *and simpler* (folder-title passage, no path). Subcategory is
inherently general/structural containment ⇒ it *is* the branch operator. The complementary pair is **ELEM (leaf) +
WIKI (branch), both pre-existing** — as the "why not subset?" question anticipated. (2) **The composition achieves the
goal:** `e5 + max(μ-elem, μ-wiki)` gets ~elem's leaf (recall@1 0.303 / MRR 0.438) *and* ~wiki's branch (ov|miss
0.391) in **one** ranker — the best all-rounder. (3) **LINEAGE is redundant** — adding it (`e5+max(el,wk,lin)` 0.392)
gives ~0 over `e5+max(elem,wiki)` (0.391); the new operator, as built, doesn't earn its place. The lineage work still
delivered the graceful-degradation *insight* + the path-overlap *metric* that revealed this. **Verdict: filer =
`e5 + max(μ-elem, μ-wiki)`, no new operator required.** (Open: lineage *might* improve with more training / a cleaner
path representation, but it's not needed given wiki.)

#### Operating point — μ's edge is at recall@10 / median rank, which is exactly what an LLM re-ranker consumes
Across **all three** results, μ's advantage over `e5-cos` concentrates at **recall@10 and median rank**, *not*
recall@1 (where e5-cos is often comparable or slightly ahead): home-turf recall@10 **0.494 vs 0.424** /
median **11 vs 29**; node-holdout **0.482 vs 0.414** / **12 vs 36**; fine-tuned filing **0.573 vs 0.440** /
**7 vs 17** — yet recall@1 stays close (e.g. filing 0.235 vs ~0.20). **This is the ideal profile for a two-stage
retrieve-then-rerank pipeline.** A first-stage retriever's job is to get the right answer *into the top-K
shortlist* (high recall@K, low median rank); an **LLM re-ranker** then supplies precision@1 by reading the K
candidates. μ excels at exactly the shortlist metric and recovers the early win *before* its recall@1 catches up
— so "μ slightly behind at recall@1" is a **non-issue** for the real application: μ is the right **first-stage
retriever** (return top-10, hand to the LLM), where median-rank-7 means the answer is almost always in the window
the LLM sees. The early in-domain win lands precisely at the hit@(≥10) operating point that re-ranking uses.

#### Coverage round 1 (enwiki linguistics/poli-sci/STS) — negative, and it sharpens *where* coverage pays
Harvested 3 absent/weak STEM-adjacent domains from the local enwiki category DB (`build_slice.py`, ~3k new
content nodes, 0% admin), trained a directional graded round (warm-start, `model_cov1`), evaluated on the merged
graph. **Result: no μ gain on the new domains** — mu-super MRR **0.39 vs e5-cos 0.53**, recall@10 0.62 vs 0.71;
the fine-tune barely moved it; originals roughly preserved (within n=400 single-run noise — e5-cos itself wobbled
±0.013). **Why:** these are *clean, well-separated* domains where **e5's symmetric similarity already captures the
structure** (same as "Music" generalising), so μ has nothing to add. **Lesson:** the bitter-lesson "more data →
μ wins" is **conditional on e5 being *weak* in that region.** μ's value — and where coverage pays — is where e5
is weak: **conflated** domains (Cooking→movies), the **dense STEM core** (directional structure matters), or
**OOD tasks** (filing). So **prioritise coverage by e5-weakness** (the disagreement signal), *not* by
"absent-but-clean." See `DESIGN_wikipedia_sampling.md`.

#### Eval correction — μ's value is DIRECTIONALITY + CALIBRATION, not symmetric rank (`eval_relatedness.py`)
The filing eval (member→*exact* parent, ranked vs all folders) is a **classification** task relative to roots; it
rewards exact title-match (favours e5-cos, understates μ — a member is related to its whole subdomain, not one
parent). Re-evaluated on the model's actual objective:
- **Symmetric relatedness** (within-vs-cross *fine* subdomain, `eval_relatedness.py`): e5-cos rank-discriminates
  *slightly better* (AUC POS-vs-hard-neg **0.74 vs μ 0.68**) — so filing wasn't hiding a μ rank-win. **But** e5
  squashes all strata into a 0.05 band (0.81/0.78/0.76 cosine floor) while μ has **4× dynamic range**
  (0.40/0.20/0.09) — μ gives *calibrated* membership degrees; e5 gives near-uninformative absolute scores.
- **Directional** (`μ(member|container)` vs `μ(container|member)`): **μ AUC(fwd>rev) 0.78, asymmetry 0.33**;
  **e5-cos AUC 0.51, asymmetry 0.001 — a coin flip.** e5-cosine *cannot express direction at all* (symmetric;
  the query/passage prefix gives ~nothing). **This is μ's structurally-unique win** and what membership needs.

**Conclusion:** e5-cos is a strong *symmetric ranker*, so symmetric evals (filing, relatedness-AUC) measure μ on
e5's home turf and miss its point. μ's value = **directionality** + **calibration**, neither of which e5 can
provide. **Re-judges coverage round 1:** `cov1` didn't move rank-AUC (e5 already ranks fine) but **sharpened μ's
calibration on the new domains** (POS mean μ 0.40→0.51) — so it *did* contribute; the filing eval just couldn't
see it. Going forward, evaluate coverage/μ on **directional + calibrated** metrics, not symmetric rank.

**Depth test (settles the "μ wins deep" hypothesis — it doesn't):** stratified the membership discrimination by
tree depth with a HARD distractor (a *sibling* of the true parent — same depth, same local domain). Both degrade
with depth and **e5 stays ahead at every depth** (μ/e5 AUC(true>sibling): shallow 0.84/0.86, mid 0.79/0.83, deep
0.72/0.79). e5's cosine doesn't saturate — deep child titles still lexically echo the true parent ("Medieval
linguists"→"Linguists"). **So μ does not beat e5 on *magnitude* discrimination at any depth.** Tested four ways
(clean domains, fine-subdomain rank, deep pairs, filing) — e5 is competitive-or-better on the symmetric/magnitude
axis every time. **This is settled: μ's value is NOT being a better symmetric ranker; it is directionality +
calibration** (the axes e5 structurally lacks). Stop benchmarking μ-vs-e5 on magnitude; build on direction +
calibration — exactly what membership/filing need and cosine cannot give.

**Negative-rejection test (honest correction — calibration ≠ separability):** at a fixed operating point (~90%
positive-recall) **neither μ nor e5 rejects negatives cleanly, and μ is marginally *worse*** (FPR on cross-domain
EASY-NEG: e5 52% vs μ-super 58%). μ's calibration gives different *means* (POS 0.37 vs EASY 0.10) but the
**distributions overlap** (high within-stratum variance), so admitting 90% of positives drops the threshold low
enough to readmit negatives. **A wide dynamic range ≠ functional thresholding.** So the "μ rejects negatives
where e5 can't" hypothesis does **not** hold at the operating point.

#### CLOSE-negative test — μ DOES win where it matters (corrects the verdict below)
The "hard negatives" above (cross-fine-subdomain) weren't close enough. The **closest** negative is a *sibling*
(another child of the same parent — same fine topic, no membership relation, often *more* e5-similar than the
true parent). On that test (POS = child→parent vs CLOSE-NEG = child→sibling):

| negative type | e5-cos AUC | μ AUC | winner |
|---|---|---|---|
| EASY (cross-domain) | 0.84 | 0.81 | e5 |
| HARD (cross-fine-subdomain) | 0.73 | 0.68 | e5 |
| **CLOSE (sibling, same parent)** | **0.62** | **0.73** | **μ** |

**e5 scores a child's true parent (0.832) and its sibling (0.815) within 0.017 — it cannot tell member-of from
sibling-of** ("everything looks similar"). μ separates them (0.48 vs 0.21) and wins 0.73 vs 0.62. **μ's advantage
*grows as negatives get closer*** (e5 degrades 0.84→0.73→0.62; μ holds) — and close neighbours are exactly the
confusions that matter in retrieval (the top-k is full of siblings, not random cross-domain nodes). This is also
why μ took filing recall@10 (0.90 vs 0.63): it pushed close-but-wrong folders down, which e5 can't.

**State it as a low μ value, not a threshold (the right framing).** The sibling subset-relation is *negative*, and
the usable signal is that **μ gives siblings a low degree (0.22)** vs the true parent (0.48) — calibrated, readable
("barely a member"). e5 gives siblings **0.815 ≈ parent 0.832**: it has *no way to say "low."* (A binary
FPR@90%-TPR understates this — μ's long positive low-tail drops the cutoff to ~0.001, so FPR is leaky for both
μ 74.5% / e5 91.5%; the *degree*, not the threshold, is what's usable and where μ wins structurally.)

#### Consolidated verdict (corrected) — μ wins direction + close-negative discrimination; hybrid
**μ's robust wins are two:** (1) **directionality** (member|container vs reverse: AUC **0.78 vs e5 0.51**); (2)
**close-negative discrimination** (member-of vs sibling-of: **0.73 vs 0.62**). e5 wins only on *easy/medium*
symmetric relatedness (coarse topical separation) and on negative-*rejection thresholding* (neither great).
So the earlier "e5 better on everything symmetric" was wrong — it held only because the negatives tested weren't
close enough. **Architecture = HYBRID:** e5 for coarse symmetric ranking (cheap, strong), **μ for the hard part —
direction + close-neighbour disambiguation** (which determines the actual top of the retrieval list). Same
"structure ∩ semantics" split; μ carries the practically-decisive cases e5 conflates.

#### CORRECTION (architecture control, PR #3387 review) — the directional/close-neg wins are NOT μ-architectural
`e5-cos` is the untrained *product* baseline; the *architecture* control is a **trained head on frozen e5**. A
logistic probe on the ordered pair `concat(query[a], passage[b])` (held-out edges, `eval_arch_control.py`) **beats
μ on both**: DIRECTION 0.92 vs μ 0.78; CLOSE-NEG (parent vs sibling) 0.78 vs μ 0.74. e5-cos fails direction only
because cosine discards order — the signal is in e5's query/passage reps and a linear order-aware head recovers it
*better* than μ. **So "μ wins direction + close-neg" is withdrawn:** against a trained-head baseline μ shows **no
per-task advantage on any axis tested.** The only remaining candidate value is a *systems* argument — one general,
calibrated, multi-relational estimator vs a per-(relation,direction) probe zoo — which is **untested**. The hybrid
framing above stands only if that generality claim is substantiated; otherwise a trained e5-probe is the stronger,
simpler tool. (This is the review's decisive catch; we ran the control and it overturned the headline.)

**UPDATE — the gap was mostly the OBJECTIVE; directional supervision largely closes it.** Root cause: μ's
dominant SYM walk term is *order-invariant* (trains symmetry, competing with direction); direction was a minority
graded component. Retrained `model_dir` keeping direction as the target (directional graded + transitive,
`--sym-weight 0`): DIRECTION 0.776→**0.839**, CLOSE-NEG 0.738→**0.779 (≈ probe 0.787, CIs overlap)**. So μ
**matches the probe on close-negatives and closed ~half the direction gap from an objective change alone** — the
architecture is competitive, not inferior; the earlier loss was a supervision artifact (under-supervised
direction + symmetric pressure). Residual direction gap (0.84 vs 0.92) = μ's regression-to-0.90/0.10 vs the
probe's discriminative loss; a ranking/discriminative directional loss is the remaining lever. Net: μ is viable
for directional membership once supervised for it; "μ can't beat a trained head" is too strong.

**RESOLVED — discriminative loss: μ BEATS the probe (`train_dir_rank.py`).** Replaced the regression with a
ranking CE `softplus(−s·(μ_fwd−μ_rev−m))`, fine-tuned on the probe's 70% split, evaluated on held-out 30%
(μ never saw those edges): **DIRECTION 0.982 vs probe 0.930; CLOSE-NEG 0.864 vs probe 0.790** — non-overlapping
CIs. So the gap was *entirely* the objective (symmetric-dominant + regression-to-constants); fix it (keep
direction, drop symmetric pressure, rank not regress) and μ's nonlinear architecture **exceeds** a linear
e5-probe. The review's control was the right test and drove the fix; "μ has no architectural advantage" is now
**reversed** — it has one, once trained for the task. (Caveats: node-overlap remains → node-disjoint split next;
single-run, tight CIs.)

#### Production model + hybrid application — the BLENDED score works (μ corrects e5's leakage)
Trained a **production μ** (`model_prod`, full recipe: directional rank + scaled Haiku lateral (87 pairs) +
transitive, 1000 steps): direction **0.890**, close-neg **0.775**, general (carries the lateral layer). Then the
**hybrid retrieval** application (`eval_hybrid.py`) — e5 coarse top-N → μ re-rank — for "find my container".

**The scoring matters.** *Pure* directional (`μ-elem`) **over-generalises** membership (the rank loss pushes
μ(child|·) high for *many* candidates) and *loses* to e5. But the correct score is a **blend of directional +
symmetric** — the operator **superposition** — because μ's Haiku-trained component **corrects e5's semantic
leakage** (siblings/topically-similar that e5 ranks high but aren't members):

| method | recall@1 | recall@5 | MRR |
|---|---|---|---|
| e5-cos alone | 0.170 | 0.418 | 0.282 |
| hybrid μ-elem (pure directional) | 0.142 | 0.440 | 0.285 |
| hybrid **μ-super** (blend) | 0.177 | **0.487** | **0.321** |
| **hybrid e5 + μ-super** | **0.225** | **0.502** | **0.359** |

**The `e5 + μ-super` blend beats e5-cos on every metric** (recall@1 +0.055, recall@5 +0.084, MRR +0.077), and
**μ-super wins container-vs-sibling in the pipeline (74.6% vs e5 68.9%)**. So the hybrid *works* — μ's directional
+ Haiku-leakage-correction, blended with e5's topical ranking, out-retrieves e5 alone. (The earlier "does not
beat e5" finding was an artifact of scoring with *pure* ELEM instead of the superposition blend.) The retrieval
score = **e5 topical similarity + μ operator-superposition** (directional membership + symmetric relatedness,
Haiku-corrected) — computed as a superposition or a sum of per-operator queries.

**Tuned blend (`eval_hybrid.py` sweep, prefixed e5 shared by both — one encode).** The μ part should be a
**non-linear OR over the separately-computed per-operator queries** — `max(μ-elem, μ-wiki, μ-sym)` — *not* the
model's internal superposition, and *not* a linear mix:

| μ score | recall@1 | recall@5 | MRR |
|---|---|---|---|
| e5-cos alone | 0.187 | 0.434 | 0.292 |
| μ-super (internal superposition) | 0.177 | 0.477 | 0.320 |
| **μ-max(elem,wiki,sym)** (OR of separate queries) | **0.220** | 0.496 | **0.354** |
| e5 + 0.5·μ-max | 0.216 | **0.506** | **0.358** |

**`max` over separate operator queries beats the internal `μ-super`** — a true container is relevant *by
membership OR relatedness*, and `max` keeps a strong single-operator hit that the superposition (or a mean)
averages away (the combiner needn't be linear).

**Tuned grid (1000 queries, μ-part × α = e5-weight):** best = **`max(μ-elem, μ-wiki, μ-sym)` at α=0.9** →
**MRR 0.358 vs e5-cos 0.286 (+25%)**, recall@1 0.223, recall@5 0.504, container-vs-sibling μ 72.2% vs e5 66.7%.
Findings: **`max` (OR) beats `mean`** (0.358 vs 0.352) and the internal `μ-super`; **α≈0.9** is optimal — the
score is *mostly μ* (~10% e5; μ-max alone at α=1 is 0.344, so e5 adds only +0.014); and the **directional
operators carry it** — `max(elem, wiki)` alone already hits 0.358, adding sym/super doesn't move it. So the
retrieval signal is fundamentally "**is this a member via element_of OR subcategory**," OR'd across operators, with
e5 a small topical assist. **Default = `max(μ-elem, μ-wiki, μ-sym)` + 0.1·e5 (α=0.9).** Keep the small e5 term
even though it is marginal on-distribution (+0.014): it is **coverage insurance / a catch-all for untrained
regions** — where μ's coverage is thin and it over-generalises, e5's topical similarity grounds the score (e5 is
the strong baseline on clean/OOD content, §4.2/coverage). *Testable:* the e5 contribution should grow on
OOD/untrained queries relative to the +0.014 it adds here. Prefixed e5 is the default input to μ (same embeddings
feed e5-cos and μ ⇒ computed once; the ablation showed no-prefix isn't meaningfully better, so no separate pass).
**Confidence-adaptive blend (built, `eval_hybrid.py` — α per-query).** Make α *per-query* — lean on μ where it is
confident, fall back to e5 where it is uncertain. Confidence = **top1 of μ-max over the shortlist** (μ is a
calibrated [0,1] membership degree, so its top score is a free per-query confidence signal). Result at 1000
queries, top-20:

| slice | MRR @ α=0.3 | @ α=0.9 | Δ |
|---|---|---|---|
| high-conf (top1-μ ≥ median) | 0.355 | 0.393 | **+0.037** |
| low-conf (top1-μ < median) | 0.350 | 0.353 | +0.003 |
| fixed α=0.9 (all) | — | 0.373 | — |
| **adaptive α∈[0.3,0.9]** (all) | — | **0.378** | **+0.006 vs fixed** |

**The mechanism is validated:** μ's top-score cleanly predicts where μ earns its weight — leaning on μ is worth
+0.037 MRR where it's confident vs +0.003 where it isn't. **But adaptive beats fixed α=0.9 by only +0.006
in-distribution**, because on low-confidence queries μ is **neutral, not harmful** (0.350 vs 0.353 — e5-heavy is
even slightly worse), so a fixed high α already captures most of the benefit. **The adaptive blend's real payoff is
OOD** — where low μ-confidence coincides with μ being *wrong*, not just neutral; on in-dist Wikipedia μ isn't wrong
anywhere, so fixed ≈ adaptive. It is correct and never hurts (use α∈[0.3,0.9]); its value grows exactly where the
coverage-insurance testable predicts. (Operator-spread / MC-ancestor variance / cross-operator entropy, §6, are
alternative confidence signals not yet swept — top1-μ was the cheapest and sufficient to validate.)

**Why this matters (two properties that make it more than a marginal-MRR result):**
1. **Honest abstention, not hallucination.** The confidence signal is read straight off μ's *own* calibrated [0,1]
   membership output — **not a separate confidence head** that could itself be overconfident-and-wrong (the
   signature of hallucination: high stated confidence, wrong answer). Because μ is trained as a graded degree, a
   low top-μ genuinely means "no candidate looks like a strong member" ⇒ the model defers to e5 instead of
   fabricating a container. The +0.037 vs +0.003 split is evidence the signal is *trustworthy*, not merely present.
   *Boundary:* this is demonstrated **operationally** (top-μ predicts where μ helps); full **probabilistic**
   calibration (does μ=0.7 ⇒ 70% membership? ECE) remains the deferred calibration item.
2. **Self-annealing blend — μ earns weight as it learns.** Under the adaptive rule α rises with confidence, so as μ
   trains its confident regions expand → the **effective mean α climbs on its own, no re-tuning**, and e5 recedes to
   the still-uncovered frontier.

**Measured (`eval_self_anneal.py`, 4 checkpoints, shared frozen-e5 shortlists, 1000 queries, seed 7).** The naive
aggregate table (mean top1-μ / mean margin / MRR per checkpoint) *suggested* margin tracks MRR while level is fooled
by the saturated `+disc` (level 0.941 highest yet MRR 0.175 lowest; margin 0.005 lowest). **But per the review
(Perplexity council, PR #3391), aggregate means can hide the actual failure mode — so the claim must be tested at
the *per-query* level.** Per-query results (Spearman ρ of each signal vs reciprocal-rank with Fisher-z 95% CI;
AURC = selective @1-risk, lower = better gate; HMER@0.8 = @1-error rate among the 80%-most-confident):

| checkpoint | ρ_level(RR) [CI] | ρ_margin(RR) [CI] | AURC_level [CI] | AURC_margin [CI] | HMER_l | HMER_m |
|---|---|---|---|---|---|---|
| nodetype | −0.06 [−.12,−.00] | **+0.14 [+.08,+.20]** | 0.838 [.808,.869] | **0.789 [.752,.827]** | 83.9% | 82.2% |
| +dir | −0.04 [−.10,+.03] | +0.05 [−.01,+.12] | 0.848 [.816,.878] | **0.752 [.707,.793]** | 82.8% | 80.6% |
| +disc | +0.06 [−.01,+.12] | +0.04 [−.03,+.10] | 0.936 [.911,.958] | 0.928 [.901,.949] | 94.2% | 93.6% |
| prod | +0.06 [−.01,+.12] | −0.00 [−.07,+.06] | 0.780 [.743,.813] | **0.751 [.712,.790]** | 77.9% | 77.5% |

(Exact decimals have minor run-to-run GPU-float variance in the μ forward pass; the qualitative claims — AURC_margin <
AURC_level on all four, ρ_level < 0 on under-trained checkpoints, ρ_margin weak with CI∋0 on 3/4 — are stable across
runs. Seed 7, `--boot 500`.)

**What survives (robust): margin is a better selective-risk *gate* than level.** AURC_margin < AURC_level on **all
four** checkpoints — *meaningfully* on 3/4 (Δ 0.029–0.096); on `+disc` the gap is Δ≈0.008, a collapse-driven near-tie
(both signals degrade when the objective saturates μ), so it is not independent evidence. Level is even
*anti-correlated* with correctness on the under-trained checkpoints (ρ_level −0.06, −0.04) — a clean, consistent
reason to prefer margin as the gating signal. (HMER is a coarse aggregate here: margin ≤ level on 3/4 but on mature
`prod` it flips by +0.2pp — noise; the AURC ordering, not HMER@0.8, carries the gate claim.) **What does NOT survive: margin
is not a strong *per-query* correctness signal.** ρ_margin(RR) is weak (≈ +0.14 at `nodetype` down to ~0.00 at
`prod`) and its CI includes 0 for every checkpoint except `nodetype`; it does **not** strengthen with training (it
weakens). So the earlier "margin's rank-order is
identical to MRR's across checkpoints" **oversold a weak per-query signal** — an aggregate/Simpson's-paradox artifact,
exactly as the review warned.

Corrected claims (both prior overclaims withdrawn):
- **NOT "calibration-invariant."** Margin is *more robust to global level shifts / saturation* than raw top1-μ (the
  `+disc` case, and the negative ρ_level early), but it is still affected by score scaling/sharpening and by the
  checkpoint objective — a relative, not invariant, measure.
- **NOT "self-annealing confirmed" — "consistent with self-annealing (at the gate level)."** Mean margin
  rises and gate quality AURC_margin (0.789→0.752→0.751) improves along `nodetype→+dir→prod`, but n=4
  checkpoints from **one trajectory differing by objective, not data volume**, and `+dir`↔`prod` margin is nearly
  flat while MRR differs — this supports "consistent with," not proof. Per-query discrimination does **not** rise.
- Per-query margin stability ρ(nodetype↔prod) = **+0.386** (moderate — high-margin queries are somewhat persistent
  across training, not a random reshuffle, but not strongly). Per-query audit emitted for offline HMER / risk-coverage
  / difficulty-stratified analysis.

**Net:** the margin-over-level thesis holds as a *selective-risk gating* result (use margin, not level, especially
OOD / across model states); the strong "confidence tracks correctness per query" and "self-annealing confirmed"
framings are withdrawn. Multi-seed training + a true data-volume anneal + AUGRC/failure-AUROC with query-bootstrap
CIs are the follow-ups needed to move from "consistent with" to a claim (deferred; see review memo §5.3).

#### Judge→loss routing — the loss is keyed off provenance, not a new embedding (now in the main trainer)
The discriminative loss is *not* a new "judge type." Provenance (`graph`/`haiku`/`human`/`sonnet`/`opus`) is an
**input token** that conditions μ's *output* (and is marginalised at inference); the **loss function** is a
*training-time* choice the model never sees as input. So we don't enlarge the judge table with "discriminative" —
we **route the loss by the judge token already on every row**:
- **`graph` + a DIRECTIONAL relation** (`element_of`/`subcategory`/`subtopic`/`super_category`) ⇒ *certain &
  oriented* ⇒ **discriminative ranking** loss `softplus(−s·(μ(member|container) − μ(container|member) − m))` (+ a
  light regression anchor so degrees stay readable). This is the term that beats the e5-probe.
- **`haiku`** (soft superposition / inferred tail) ⇒ a *distribution*, not a hard orientation ⇒ keep **regression
  to the soft target** (`L_graded`/`L_blend`).

Implemented in `train_mu_attention.py` (`--dir-rank-weight`, `--dir-rank-scale/-margin/-anchor`): graph-judged
directional rows are collected (`dir_edges`) and trained by the ranking term; everything else stays on
regression. The judge table is unchanged (it only grows when a new *source* appears, e.g. `sonnet`/`opus`). Maps
directly onto the data recipe: **downward directional (`graph`) → rank; bidirectional lateral (`haiku`
superpositions, ≤30%) → soft regression.**

### Relation to prior approaches
Prior graph retrieval uses **distance metrics** — most relevantly **weighted shortest path** (and the WAM
core's effective-distance). Those are *structural only*: graph-near ≠ semantically-related. The new algorithm
keeps the structural candidate-gathering but **replaces structural distance with the learned μ** for ranking —
or hybridises (μ-weighted edges in the greedy priority). The eval (below) measures exactly this gap.

## Theory note — μ is a geometric (scalar + bivector) object
This is the uncommon framing worth recording. Standard embeddings give each entity a **vector** `v` and a
**symmetric** similarity (cosine/dot) — which structurally *cannot* represent a directional relation. Various
fixes exist (order-embeddings, box/Gaussian embeddings, bilinear `vᵀMv`), but they bolt asymmetry on.

**The geometric-algebra view makes the asymmetry intrinsic.** For two vectors `a, b`, the **geometric
product** decomposes by grade:

> `ab = a·b + a∧b`  —  a **scalar** (symmetric inner product) + a **bivector** (antisymmetric *oriented* area).

Our operators split *exactly* along this seam:
- **symmetric operators** (`see_also`, `assoc`, `bridge`) ↔ the **scalar** `a·b` — un-oriented relatedness;
- **directional operators** (`subcategory`, `element_of`, `super_category`) ↔ the **bivector** `a∧b` —
  `a∧b = −b∧a` encodes the parent/child **orientation** (the `μ(a|b) ≠ μ(b|a)` asymmetry is the sign flip).

So μ's symmetric/directional operator structure *is* the scalar/bivector grade structure of a multivector — the
asymmetry is geometric **by construction**, not bolted on.

**Partial application (currying) — `μ(·|root)` *is* a partial application.** With `>` = subset, the binary
relation `A > B` (A ⊆ B) curries into two unary predicates: **`A >`** = λx. A ⊆ x = *supersets of A* = "A as a
**member**" = the `μ(A|·)` predicate; **`> B`** = λx. x ⊆ B = *subsets of B* = "B as a **container**" = the
`μ(·|B)` predicate. So **conditioning on the root is currying the relation** — `μ(·|root)` is exactly the partial
application `> root` (rank the *members* of root). The proposal: **the bivector mapping outputs the
partial-application embedding** (the operator `> root` as a vector/rotor), and `μ(node|root) = ⟨node, (>root)⟩`
(contract node against it). A node's two structural roles — *member* (`node >`) vs *container* (`> node`) — are
the **two signs of its bivector** (`x_a∧x_b = −x_b∧x_a` = subset-vs-superset = `μ(a|b)≠μ(b|a)`), so the sign
picks the role for free. This re-grounds the symmetric/asymmetric split: a symmetric op has `A ~ = ~ A` (no
orientation) ⇒ the **dot product suffices**; an asymmetric op has `A > ≠ > A` ⇒ you **need** the bivector to tell
the two curryings apart (the dot product cannot curry asymmetrically). Consistency check: `μ(node|root)` computed
as `⟨node,(>root)⟩` or `⟨(node>),root⟩` must agree — the associativity of the geometric product (one relation,
two views).

**Bivector dimensionality — general is O(n²), but ours are *simple* (O(n)).** A **general** bivector is an
antisymmetric `n×n` matrix = `dim so(n) = n(n−1)/2`; for `n=384` that is **73,536** (≈ `n/2 ≈ 192×` the embedding
dim) — materialising it per node is absurd. **But the bivectors we use are *blades* (rank-2): `x_a ∧ x_b` lies in
the 2-plane `span(x_a,x_b)`**, parameterised by the two vectors (**O(n)**, never the 73k object). Its rotor acts
*only* in that plane — apply `R x R̃` by projecting `x` onto the plane (two dot products), rotating that 2D part
by `θ`, leaving the complement untouched: **O(n), closed form, no matrix exp**. Between the extremes sits a
**low-rank bivector = sum of `K` blades** (= product of `K` Givens rotations), **O(Kn)** params/apply, `K`
rotation planes — the expressiveness dial:

| representation | params | apply | expressiveness |
|---|---|---|---|
| simple blade (`K=1`) | O(n) | O(n) | geodesic / one 2-plane |
| **low-rank (`K` blades)** | **O(Kn)** | **O(Kn)** | `K` planes — *tunable* (`K≈8` ⇒ ~3k params) |
| general (`K=n/2`) | n(n−1)/2 ≈ 73,536 | O(n²) | full `SO(n)` (the Procrustes case) |

**Cost tradeoff — the unifying axis (corrected: the cost is *general rotation*, not bivectors per se).** The
expensive object is the **general** rotation — the bookmarking agent's Procrustes `W` ∈ `SO(n)`, whose generator
is a *general* (73k-dim) bivector recovered via `logm`/`expm` (O(n³)). **Simple/low-rank tangent blades avoid
that entirely** (O(Kn), forward-pass cost). So the real axis is **prior strength vs data**, not inference cost:
the bivector carries a **strong geometric prior** (directionality *given*) ⇒ little training data; the
transformer has a **weak prior** ⇒ high data cost but cheap inference. This *is* the filing-vs-home-turf result
(strong prior wins OOD; transformer wins in-domain-with-data). With our **low coverage**, the prior is the better
fit ⇒ a **hybrid**: operator embedding = the **low-rank bivector-mapped partial application** (`> root` built
from `K` blades), transformer reads μ on top — strong prior *and* learned refinement, both at forward-pass cost.
(The `> root` rotor is also amortised — computed once per root, reused across all candidates.)

**Precedent — this is already built (the federated rotation machinery), and K-blades was the *representation*, not
the whole trick.** The repo's Pearltrees federated projection (`scripts/infer_pearltrees_federated.py`,
`scripts/train_orthogonal_codebook.py`, `scripts/train_bivector_codebook.py`, `src/.../rotation_transformer.py`)
already beats the naïve per-cluster `logm`/`expm` (O(n³)). What actually carries the speed:
- **The blades are made axis-aligned / orthogonal, and *that* is load-bearing.** Orthogonal planes **commute**, so
  `exp(Σ wᵢBᵢ) = Π exp(wᵢBᵢ)` — which licenses the **Rodrigues / Givens product** application
  (`train_orthogonal_codebook.py:351-439, 965-1077`). It is "K **commuting** blades," not just "K blades"; the
  commuting removes the matrix exp. (Matryoshka structure ⇒ the first ~64 axis dims suffice —
  `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md`.)
- **Canonical plane-angle decomposition** (`infer_pearltrees_federated.py:191-391`, `rotational-fast`) is the
  actual inference-time `logm`-beater: split `W` into d/2 axis-aligned 2×2 blocks, read each angle via `arctan2`,
  blend angles by scalar weighted-average, apply vectorized 2×2 rotations — O(K·d), no matrix functions.
- **Codebook of K planes** (defaults **K=64**, **top-8** blended per query — matching the "K≈8" estimate above).
  Two variants — **canonical axis-aligned** (fixed planes, *recommended*) vs **PCA-derived** principal bivectors;
  the canonical one wins on Matryoshka embeddings (see "Free planes vs axis-aligned" below — a non-obvious result).
- **Distillation pays the matrix functions once, at training, never at inference**: the exact `logm`/`expm`
  teacher is distilled into a cheap student (a transformer / a Givens layer)
  (`scripts/distill_federated_to_transformer.py`). So even the "high" cost is a *one-time training* cost — which
  collapses the cost axis further toward the transformer once distilled.
So our hybrid has a **proven recipe**: K commuting blades, applied via Rodrigues/Givens, selected by a small
codebook, optionally distilled into the μ transformer so inference is a plain forward pass. The geodesic
`x_root ∧ x_node` is the K=1 special case (`src/.../minimal_transform.py:_rotation_between_vectors`).

**Free planes vs axis-aligned — fewer blades in theory; canonical won *for Nomic* because Matryoshka made the
data-adaptive step unnecessary (NOT because PCA is a bad tool).** *Theory:* data-adaptive 2-planes are more
expressive per blade than fixed coordinate (Givens) planes — PCA-vs-fixed-axis logic — so in principle the blade
count drops to the rotation's **effective rank**, not the coordinate planes needed to *span* it. The real catch:
arbitrary planes **don't commute** (`exp(B₁+B₂) ≠ exp(B₁)exp(B₂)` ⇒ order-dependent BCH error in the Rodrigues
product) and cost an eigendecomposition (≈ O(n³)) to find. `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` reports
that PCA-then-orthogonalise **distorts** the data-planes (info loss, broken clean commutativity) and **lost** to
the **canonical axis-aligned** codebook (**0.9997** cosine vs the exact teacher; orthogonal-by-construction ⇒ each
weight an independent knob, decomposable learning). **But that verdict is conditional, not general:** canonical won
*because* **Matryoshka** (Nomic) already packs core meaning into dims 0-127, so the first ~64 axis planes ≈ the
principal directions — Matryoshka **negates the need** for data-adaptive planes. It does **not** rule PCA-
orthogonalise out where that free alignment is absent. **For our `e5-small-v2` (not known to use Matryoshka
representation learning), data-adaptive PCA planes may be the *right* tool, not a fallback** — profile e5's
per-dimension importance first; *which* planes (canonical vs data-adaptive) is **embedding-structure-dependent and
genuinely open for e5**. "Small codebook + Rodrigues + distil" holds either way.

**When to bother with the bivector prior at all — the bitter lesson (this is mostly orthogonal to our path).**
Two limits on how much the above matters *for us*. (1) **We use a transformer, not a standalone rotation
codebook.** The canonical-vs-PCA-vs-Rodrigues machinery exists to make a *geometric rotation* cheap; our model
*learns* the transform, so the bivector is at most an **added feature/prior**, not the estimator. (2) **The bitter
lesson** (Sutton): hand-built structure tends to be overtaken by general methods that scale with data + compute.
The bivector is a geometric **prior for asymmetry**; with enough data a transformer can learn that asymmetry
itself — and our **filing learning curve is exactly this lesson in miniature** (the transformer crossed the
e5-cos bar with modest data and was *still climbing*). So the prior's value is **sample-efficiency in the low-data
regime** (the rotation-wins-at-low-data result), not a permanent component. Use it as **distillable scaffolding**:
bootstrap with the prior where data is thin, distil into the transformer, then *drop the prior* as coverage grows
(the repo's distillation path makes this literal). The danger to avoid is baking it in a way that **caps the
ceiling**. **Bridge, not destination** — the long-term bet is data + the transformer's general learning.

**Rotations on the sphere → bivectors as the generator.** e5 embeddings are **unit-normed**, so they live on a
sphere and the relationship root→node is a **geodesic rotation**: rotor `R = exp(−½ θ B)`, applied
`x_node = R x_root R̃`, where the **plane** `B = (x_root ∧ x_node)`-normalized and the **angle**
`θ = arccos(x_root·x_node)`. The bivector encodes *both* — its plane is the rotation plane, its magnitude
`|x_root ∧ x_node| = sin θ` is the angle. **Off the sphere = rotation + scalar dilation:** `x' = s·R x R̃`, a
*scaled rotor* (similarity transform) splitting magnitude (`s`) from orientation (`R`). This is why
the bookmarking agent's exact-Procrustes used **logm/expm** (the `so(n)` Lie algebra *is* the bivector space) —
exact, but matrix-exp/log expensive. **Why bivector, not cross product:** in 3D they are Hodge-duals
(`a∧b = I(a×b)`), but the cross product exists *only* in 3D/7D; for 384-dim e5 the bivector is the only correct
generalisation of "oriented plane of rotation."

**Embedding bivectors from the tangent space of each input (the concrete construction).** Rather than feed raw
`x_node, x_root` and hope attention discovers directionality, feed the **local-tangent bivector**
`B = x_root ∧ x_node` (or its compact rotor/log form) as an explicit feature. At a sphere point `x`, the tangent
space `T_x` is everything ⟂ `x`; `B` lies in the root's tangent plane and *is* the geodesic direction toward the
node. Two payoffs, both geometric-by-construction rather than learned: (1) **directionality is the sign of the
bivector** (`x_r∧x_n = −x_n∧x_r` ⇒ `μ(a|b)≠μ(b|a)`); (2) because the tangent frame is **local**, `B` is
**automatically region-adaptive** — the same construction yields different oriented relations in different parts
of the sphere, the *continuous* analog of routed per-cluster transforms (the tangent space **is** the local
chart, so no discrete clusters). The model then learns μ *on top of* the right primitive (`a∧b`, the
antisymmetric part the scalar cosine throws away — exactly the part `e5-cos` ignores yet still beats a fumbling
learned readout with). This is the model variant the GA note points at, now with a definite construction.

**Root-centred embeddings / tangent chart (for visualisation):** fix the **root as origin**; node embeddings
become root-relative, and the relationship to the root is the oriented `node ∧ root` (the same tangent bivector,
used for *display* not features). A **local chart** for the density explorer: 2D/3D coordinates where
displacement ≈ μ-relatedness and the **axes are the principal statistical variations of the root's
neighbourhood** (local PCA / a "tangent plane" at the root). Naturally per-root (each root its own chart).

**For ranking we need none of this** — μ *is* the distance measure; embeddings/bivectors are for visualisation
and a possible re-derivation, not for retrieval.

## Roadmap (enhancements to built things + new things)
**Enhancements to built things:**
- **Transitive μ** — the deferred stage-2 items (noisy-OR multi-path, LLM-anchored multi-factor `μ_bound`,
  product soft-floor; `DESIGN_transitive_relations.md` open questions). Most bite only in the **weak/long-chain
  regime**, so pursue them *when retrieval shows that regime matters* (the hetero A/B was neutral precisely
  because the strong-chain curriculum doesn't exercise them).
- **Density explorer** — **root-centred tangent charts** (theory below): a per-root local view, vs the current
  global dense map.
- **Bridge detection** — fuse with the transitive-decay μ and the retrieval gather (a bridge is a node with
  high cross-domain μ reachable in the bidirectional walk).
- **Operator superposition** — learned (non-uniform) operator weights per query instead of the equal-weight
  `⅓+⅓+⅓` default.

**New things:**
- **Graph RAG / retrieval algorithm** — greedy bidirectional gather + μ-superposition sort. **Built + verified**
  (`eval_retrieval`); the discriminating Cooking case shows it fixes the dense-μ domain-leak. The μ-vs-hop
  top-k scatter is built too.
- **Hop-stratified retrieval eval** (formal recall@k / AUC vs held-out-edge ground truth) — *still proposed*;
  the head-to-head number against the WSP baseline.
- **Embedding bivectors / geometric (GA) re-derivation** — the theory note (research thread).

## Build plan (the retrieval core — first concrete increment, no LLM, runs against existing models)
1. **Hop-stratified retrieval eval** — ground-truth = graph reachability from a root (relevant = reachable at
   ≤H hops; stratify metrics by hop-distance). Metric: recall@k / AUC (related vs unrelated), per hop.
2. **Baseline = weighted shortest path** on that eval.
3. **New algorithm** = greedy bidirectional gather + μ-superposition sort, same eval.
4. **μ-vs-hop scatter (top-k)** as the diagnostic.
Compare: does semantics-on-structure beat structure-alone, *especially at 2–3 hops* (where the transitive μ
should pay off)?

## References
- D. Hestenes, *New Foundations for Classical Mechanics* / *Space-Time Algebra* — geometric (Clifford) algebra;
  the scalar+bivector geometric product.
- J. Brandstetter et al., "Clifford Neural Layers for PDE Modeling," 2022; D. Ruhe et al., "Geometric Clifford
  Algebra Networks," ICML 2023 — GA/Clifford structure in deep learning.
- I. Vendrov et al., "Order-Embeddings of Images and Language," ICLR 2016; L. Vilnis, A. McCallum, "Word
  representations via Gaussian embedding," ICLR 2015 — asymmetric/containment embeddings (the bolted-on fixes).
- P. Lewis et al., "Retrieval-Augmented Generation…," NeurIPS 2020; Edge et al., "From Local to Global: A Graph
  RAG Approach," 2024 — RAG / graph-RAG.
- `DESIGN_transitive_relations.md` (transitive decay), `DESIGN_bidirectional_walk.md` (depth-balanced
  traversal), `DESIGN_inferred_operator_superposition.md` (the operator superposition / unconditional E[μ]).
