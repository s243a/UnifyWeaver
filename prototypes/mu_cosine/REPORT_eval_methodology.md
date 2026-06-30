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

**Works (μ's robust, repeatable wins):**
- **Directionality** — AUC 0.78 vs e5's 0.51 coin-flip. Structural; e5 cannot do it.
- **Close-negative (sibling) discrimination** — AUC 0.73 vs 0.62; the win *grows* as negatives get closer.
- **Calibrated degrees** — 4× dynamic range; readable "low" that e5 lacks.
- **Data scaling** — fine-tuning crosses the e5-cos bar on a real OOD task with modest data and keeps rising.

**Doesn't (honest negatives):**
- μ is **not** a better *symmetric rank* metric than e5 on easy/medium negatives (e5 AUC ≥ μ there).
- μ's calibration does **not** give clean high-recall thresholding (FPR leaky; distributions overlap).
- **Clean-domain coverage gave no rank gain** (e5 already strong on clean content).
- Zero-shot OOD μ loses badly (it needs in-domain data).

**Architecture implied:** a **hybrid** — e5 for cheap coarse symmetric ranking; **μ for direction + close-
neighbour disambiguation + calibrated degree** (the practically decisive top-of-list cases). Not "μ replaces e5."

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
