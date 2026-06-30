# Evaluating μ-attention vs frozen e5 — methodology & results

**Purpose.** A self-contained report of how we evaluate the **μ-attention** model and what we found, written for
external methodology review. The central question: **μ sits *on top of* frozen e5 embeddings — what does the
learned layer add over its own frozen substrate (raw e5 cosine)?** We answer it across retrieval, pair
discrimination, directionality, calibration, and a data-scaling curve, and we are explicit about what *doesn't*
work. (Branch: `claude/mu-coverage-dataquality`. Tools: `eval_filing.py`, `eval_relatedness.py`, `train_filing.py`.)

---

## 1. Model and baseline

- **μ-attention** produces `μ(node | root) ∈ [0,1]` — a **directional, fuzzy** membership/relatedness degree over
  a concept graph. It is a small permutation-invariant transformer over **frozen** e5-small-v2 embeddings (the
  root uses e5's `query:` role, candidates/ancestors use `passage:`, which is what lets `μ(a|b) ≠ μ(b|a)`), read
  out per **operator** (`SYM` lateral; `WIKI`/`ELEM` directional membership). Only the attention/readout/operator
  params are trained; **e5 is frozen.** `mu-super` = equal-weight operator superposition; `mu-elem` = the
  `element_of` operator alone.
- **Baseline: `e5-cos`** = cosine of the frozen e5 vectors (`query:` node vs `passage:` root). Zero trained
  parameters. Because μ is built *on* e5, e5-cos is the honest "does the learned layer earn its keep" baseline.
- **Key asymmetry of the comparison:** e5 is **frozen** (fixed at pretraining — no in-domain data can improve
  it); μ is the **trained** layer (data-scalable). So any improvement from in-domain data accrues to μ, not e5
  (see §6).

## 2. Datasets

| dataset | role | scale |
|---|---|---|
| simplewiki 10k category graph (`category_parent.tsv`) | in-domain (model trained here) | 8,247 category nodes |
| enwiki named category DB (`enwiki_named/category_parent.tsv`) | coverage source (downward slices) | 9.9M title-based edges |
| Pearltrees bookmark filing (`.local`, gitignored) | OOD, **real human labels** (a bookmark's actual folder) | 7,486 bookmarks / 988 folders |

## 3. Metrics (and why each)

- **recall@k, MRR, median rank** — retrieval: is the true target in the top-k / how high. recall@10 + median are
  the **operating point for a two-stage retrieve-then-LLM-rerank pipeline** (first stage needs the answer in the
  shortlist; the LLM does precision@1).
- **AUC(POS vs NEG)** — *threshold-free rank* separability of related vs unrelated pairs (Mann-Whitney).
- **FPR @ fixed TPR** — *thresholding*: at a cutoff admitting X% of positives, what fraction of negatives leak
  through. Tests whether a usable decision boundary exists (distinct from rank).
- **Directional asymmetry** — `AUC(μ(member|container) > μ(container|member))` and `mean fwd − mean rev`. Tests
  the one thing a symmetric metric structurally cannot do.
- **Calibration / dynamic range** — the spread of scores across strata (mean POS vs HARD vs EASY). A *readable*
  degree (`μ=0.2` ⇒ "barely a member") vs a compressed band that can't express "low."

## 4. Results

### 4.1 OOD filing (real labels) — zero-shot, then a data-scaling curve
Rank candidate folders by score; ground truth = the bookmark's actual folder (335 folders ≥3 bookmarks, 500
sampled queries).

**Zero-shot** (`model_nodetype`, never trained on filing):

| ranker | recall@1 | recall@10 | MRR |
|---|---|---|---|
| **e5-cos** | **0.202** | **0.480** | **0.299** |
| mu-super | 0.088 | 0.248 | 0.151 |

e5-cos **doubles** the μ model — but this is **OOD transfer** (μ trained on simplewiki categories, not bookmark
filing), not a verdict on the architecture (see 4.2). 

**Filing fine-tune learning curve** (`train_filing.py`; warm-start, in-batch contrastive `element_of`, **3
training seeds, fixed eval split**, ✓ = mean−sd above the e5-cos bar 0.291):

| data frac | n_train | MRR (mean±sd) | recall@10 | vs e5-cos |
|---|---|---|---|---|
| 0.10 | 492 | 0.230 ± 0.025 | 0.406 | −0.061 |
| 0.30 | 1478 | 0.317 ± 0.006 | 0.537 | **+0.025 ✓** |
| 1.00 | 4929 | 0.358 ± 0.018 | 0.573 | **+0.066 ✓** |

μ **crosses the e5-cos bar between 10–30% of the data and keeps rising** (no plateau); at 100%, recall@10 0.573
vs 0.440, median rank 6 vs 17. The cross survives seed noise (mean−sd clears the bar at 30% and 100%).

### 4.2 In-domain "home turf" (isolates OOD by changing only the domain)
Same task on the simplewiki category graph (folders = categories, members = children); **lineage-free**,
apples-to-apples with filing.

- **With admin categories present:** μ ties e5 on MRR (0.255 vs 0.270) but **beats it on recall@10 (0.494 vs
  0.424) and median rank (11 vs 29)**.
- **Node-holdout (never-trained nodes — kills the memorisation caveat):** near-identical to the trained run
  (mu-super recall@10 0.482 vs 0.494; ~3% drop) ⇒ **generalisation, not memorisation.**
- **Data quality (Tier-1 admin removal) inverts the picture** — see §5: dropping *meaningless* maintenance
  categories (`CatAutoTOC`, `Navseasoncats`) lifts mu-super recall@10 to **0.81 vs e5 0.66** (and to **0.90 vs
  0.63** on the cleanest set). On never-trained clean nodes: mu-super recall@10 **0.919 vs 0.662**.

### 4.3 Symmetric pair discrimination (aligned with the model's training objective)
The sampler trains on *within-subdomain pairs = related, noise = not*. Strata over 6 coarse domains, 800
pairs/stratum: **POS** (same fine subdomain), **HARD-NEG** (different fine subdomain, same coarse), **EASY-NEG**
(different coarse).

| scorer | AUC(POS vs HARD) | AUC(POS vs EASY) | mean POS / HARD / EASY |
|---|---|---|---|
| e5-cos | **0.726** | 0.838 | 0.814 / 0.781 / 0.761 |
| mu-super | 0.671 | 0.814 | **0.367 / 0.214 / 0.095** |

**On rank-AUC, e5 is competitive-or-better.** But note the means: **e5 squashes all strata into a 0.05 band
(0.76–0.81); μ has a 4× dynamic range (0.37/0.21/0.10)** — the calibration point (§5).

**Negative-rejection (thresholding) — FPR at ~90% TPR:**

| scorer | threshold | FPR HARD-NEG | FPR EASY-NEG |
|---|---|---|---|
| e5-cos | 0.759 | 71.6% | 51.8% |
| mu-super | 0.024 | 74.4% | 57.6% |

Neither thresholds cleanly at 90% recall (distributions overlap); μ is marginally worse. **A wide dynamic range
≠ clean separability** — so calibration helps *readability* (§5), not hard thresholding at high recall.

### 4.4 Directionality — μ's decisive, structural win
For 1,500 membership edges: `μ(member|container)` (forward) should be high, `μ(container|member)` (reverse) low.

| scorer | mean fwd | mean rev | asymmetry | AUC(fwd > rev) |
|---|---|---|---|---|
| **mu-elem** | 0.512 | 0.185 | **0.326** | **0.776** |
| e5-cos | 0.838 | 0.837 | 0.001 | 0.509 |

**e5-cos cannot tell which way membership points (AUC 0.509 — a coin flip);** μ gets it right 78% of the time.
The `query:`/`passage:` prefix gives e5 essentially nothing (asymmetry 0.001). This is the property a symmetric
metric *cannot* have.

### 4.5 Close-negative discrimination — μ's advantage grows as negatives get closer
The hardest negative is a **sibling** (another child of the same parent — same fine topic, *no* membership
relation, often *more* e5-similar than the true parent). 2,000 (child, parent, sibling) triples.

| negative type | e5-cos AUC | μ AUC |
|---|---|---|
| EASY (cross-domain) | 0.84 | 0.81 |
| HARD (cross-fine-subdomain) | 0.73 | 0.68 |
| **CLOSE (sibling, same parent)** | **0.62** | **0.73** |

**e5 degrades sharply as negatives tighten and *loses* at the closest negatives; μ holds.** On siblings, e5 scores
the true parent (0.832) and the sibling (0.815) within **0.017** — it cannot distinguish member-of from
sibling-of. μ gives the sibling a **low degree (0.22)** vs the parent (0.48). Depth-stratified with a
sibling-of-parent distractor, both degrade with depth and e5 stays ahead — so μ's win is about negative
*closeness* (sibling), **not** tree depth. **Close neighbours are exactly what fill the top of a retrieval list**,
so this is the practically decisive regime — and it is why μ won filing recall@10.

### 4.6 Architecture control (added per review #5) — the directional/close-neg wins are NOT μ-architectural
`e5-cos` is the *product* baseline (untrained). The *architecture* control is a **trained head on the same frozen
e5 features**: a logistic regression on the **ordered** pair `concat(query[a], passage[b])` (768-d), trained on a
70% edge split, evaluated on the 30% held-out (`eval_arch_control.py`; sibling negatives DAG-filtered to drop
ancestor/descendant "siblings"; bootstrap 95% CIs):

| task | e5-cos (untrained, symmetric) | **e5-probe (trained head on frozen e5)** | mu-elem |
|---|---|---|---|
| DIRECTION (fwd vs rev) | 0.515 | **0.922** [0.911, 0.931] | 0.776 [0.758, 0.793] |
| CLOSE-NEG (parent vs sibling) | 0.635 | **0.783** [0.765, 0.800] | 0.738 [0.718, 0.757] |

**The control beats μ on both.** e5-cos can't do direction only because cosine *discards order*; the directional
signal is present in e5's `query:`/`passage:` representations and a **linear order-aware head recovers it at 0.92
— above μ's 0.78.** So **directionality and close-negative discrimination are NOT μ-architectural wins** — a
trivial trained head on frozen e5 does them *better*. Against this properly-controlled baseline, **μ has not
demonstrated a per-task advantage on any axis we tested**; the earlier "μ's structural win" framing compared μ to
symmetric cosine, not to a trained head. Caveat: the probe is *task-specialised* (trained directly on each task),
while μ is one *general* multi-relational model trained on a different (walk) objective — so this refutes the
*architectural-superiority* claim, not the possibility that a single general calibrated estimator is *useful*
(an untested systems argument, §7).

**Follow-up — the gap was mostly the OBJECTIVE, not the architecture (resolves the caveat).** Root cause: μ's
*dominant* training term (the SYM walk pairs) is **order-invariant** ("feed both orders, same μ") — it actively
trains *symmetry*, competing with direction; direction was only a minority graded component. We retrained
(`model_dir`) keeping the direction as the target (directional graded bulk + transitive multi-hop, **`--sym-weight
0`** to drop the symmetric pressure), warm-started, and re-ran the control:

| task | model_nodetype (symmetric-trained) | **model_dir (directional supervision)** | e5-probe |
|---|---|---|---|
| DIRECTION (fwd vs rev) | 0.776 | **0.839** | 0.919 |
| CLOSE-NEG (parent vs sibling) | 0.738 | **0.779** (≈ probe; CIs overlap) | 0.787 |

**μ now matches the probe on close-negatives and closed ~half the direction gap — from an *objective* change
alone.** So μ's earlier 0.78 was **mostly a supervision artifact** (the symmetric-dominant objective), not an
architectural limit; the architecture is competitive once direction is supervised. The **residual** direction gap
(0.84 vs 0.92) is the last of the objective mismatch — μ still *regresses to fixed 0.90/0.10 targets* (caps
separation) where the probe optimises a *discriminative* loss; a ranking/discriminative directional loss is the
remaining lever. (Single-run, n=1200 held-out; gains exceed the ±0.018 CI.)

## 5. The e5 calibration issue (central methodological point)

e5-small cosine similarities live in a **compressed high band (~0.76–0.84)**: a true parent, a different-fine-
subdomain category, and a *sibling* all score ~0.78–0.83. Consequences:
- **e5 cannot express "low"** — there is no cosine value that reads as "not a member"; everything related-ish is
  ~0.8. A sibling (0.815) is indistinguishable from the true parent (0.832).
- **Rank-AUC hides this** — rank order can survive compression, so AUC makes e5 look fine (§4.3) even though it
  can't *threshold* or *read off a degree*.
- **The usable signal is the calibrated degree, not a threshold.** μ outputs `0.48` for a member and `0.22` for a
  sibling — directly readable. (We deliberately report the *degree*, because the binary FPR@TPR is leaky for both:
  μ's positive low-tail forces a ~0.001 cutoff at 90% TPR.)
This is *why* a learned layer over frozen e5 is worth having: e5 supplies coarse topical similarity; it
structurally cannot supply **direction** or a **calibrated low-end**.

## 6. How more data improves the μ numbers (and not e5's)

- **e5 is frozen** — no amount of in-domain data changes it. The baseline is at its ceiling *by construction*.
- **μ is the trained layer** — the filing learning curve (§4.1) is direct evidence: MRR 0.230 → 0.317 → 0.358
  across 10/30/100% of the data, **monotonic, still rising at 100% (no plateau)**. Extrapolating the slope, more
  in-domain data continues to widen μ's margin over the fixed e5 bar.
- **Coverage sharpens calibration.** A coverage round (enwiki linguistics/poli-sci/STS) did **not** raise rank-AUC
  on those clean domains (e5 already ranks clean content well) but **sharpened μ's positive degree** there
  (mean POS μ 0.40 → 0.51) — i.e. data improves the *calibration/directional* axis even where rank is saturated.
- **Where data pays:** where e5 is *weak* — conflated domains, **close-neighbour disambiguation**, and
  **direction** — not where e5 is already strong (coarse/clean topical separation). So data scaling should target
  e5-weak regions (by disagreement), not "absent-but-clean" domains.

## 7. What works / what doesn't

**Corrected by the architecture control (§4.6) — the earlier "μ wins" claims do NOT survive a trained-head baseline:**
- **Directionality:** signal is in frozen e5; an order-aware **linear probe beats μ (0.92 vs 0.78)**. Not μ-architectural.
- **Close-negative (sibling):** **probe beats μ (0.78 vs 0.74)**. Not μ-architectural.
- **Symmetric rank:** e5 ≥ μ (was already a non-win).
- **Calibrated degrees:** μ has 4× dynamic range, but that is *readability/separation*, not probabilistic
  calibration, and does not give clean high-recall thresholding (FPR leaky). Rename pending (§5).
- **Clean-domain coverage:** no rank gain (e5 already strong).
- **Data scaling:** fine-tuning crossed the **e5-*cos*** bar — but that baseline is uncontrolled; an e5-*probe*
  curve is the needed comparison and is **not yet run**.

**Net (honest, updated after the directional retrain §4.6):** the symmetric-trained checkpoint lost to a trained
e5-probe — but that was **mostly the objective** (μ's dominant SYM term trains *symmetry*). With direction
actually supervised (`model_dir`), **μ matches the probe on close-negatives and closes ~half the direction gap**,
so the **architecture is competitive**, not inferior. μ still does not *beat* the probe on pure direction (0.84
vs 0.92) — the residual is the regression-to-fixed-targets objective vs a discriminative loss. So: μ is a viable
directional/membership estimator once supervised for it; whether it should *replace* a per-task e5-probe rests on
the **systems** argument (one general calibrated multi-relational model vs a probe zoo) plus closing the residual
with a discriminative directional loss. The earlier "μ owns direction structurally" overclaim stays withdrawn,
but "μ can't beat a trained head" is now **too strong** — it matches on close-neg and the gap is shrinking with
the right objective.

## 8. Threats to validity / limitations (for the reviewer)

- **Misaligned first eval.** Our initial filing eval ranks a member against *all folders* (a classification task
  relative to roots) — it rewards exact title-match and favours e5; we treat the relatedness/directional/close-
  negative evals as the aligned ones. (Documented as a correction, not hidden.)
- **"Hard negative" wasn't hard enough** initially (cross-fine-subdomain). The sibling test (§4.5) is the genuine
  close negative; conclusions changed once we used it — a caution about negative *closeness* in any such eval.
- **Sample sizes / seeds.** Filing learning curve is 3-seed; most discrimination evals are single-run, n=400–2000
  (e5-cos wobbled ±0.013 between runs at n=400). Directional/close-negative effects are large enough to clear that
  noise, but rank-AUC differences of ~0.05 are within it.
- **Real labels are noisy.** Pearltrees filing reflects one person's organisation; "the" correct folder is partly
  arbitrary (a bookmark could fit several) — which itself argues for recall@k over recall@1.
- **e5-small specifics.** The compression band and the (non-)Matryoshka structure are model-specific; a different
  frozen encoder could shift the e5 baseline.
- **Memorisation control.** The in-domain win is checked on a node-holdout (nodes in *neither* the SYM pairs nor
  the graded edges), so it is generalisation; but the holdout is from the same graph distribution.

## 9. Questions we'd like the reviewer to scrutinise

1. Is the **directional AUC(fwd>rev)** the right way to credit the one thing e5 can't do, or is there a stronger
   directional metric (e.g., a directional ranking / hierarchy-reconstruction task)?
2. Is reporting the **calibrated degree** (mean μ per stratum) defensible given the FPR@TPR is leaky, or should we
   report a calibration metric directly (ECE, a reliability diagram, d-prime)?
3. For the **close-negative** result, is "sibling" the right hardest negative, and is the closeness *trend*
   (easy→hard→sibling) the right way to show e5's degradation?
4. Is the **data-scaling argument** (frozen-e5-ceiling vs trained-μ-headroom, plus the learning curve) sound, and
   what would make the extrapolation rigorous (e.g., more fractions, a fitted curve + CI, a second OOD domain)?
5. Is the **hybrid** conclusion the right read, or does the evidence support μ-alone for the directional use case?
