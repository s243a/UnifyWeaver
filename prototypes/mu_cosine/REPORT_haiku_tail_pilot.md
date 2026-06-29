# Haiku inferred-tail augmentation — pilot (DESIGN §9/§14)

First real run of the §14 Haiku contract over the **inferred tail** (conf<1.0 fused-graph rows), via
`score_inferred_tail.py` + `cell_sampler.py`. Goal: **validate the pipeline end-to-end**, not move the model.

## What ran
- **Extract:** 66 unique untagged pairs (`assoc` 22 / `subtopic` 23 / `element_of` 21) from
  `context/*_edges.tsv`, physics/EE-concentrated (LTI 34, circuit 24, ds/chaos/cyb ~8). Bridge rows
  (title-match identity) excluded — not a "which relation" question.
- **Score:** 3 Haiku subagents, §14 prompt, batched 22/each → strict-JSON cell distributions.
- **Ingest:** §12(5) closure (`applies[named] ++ unknown ++ none` → categorical Σ=1) → per-pair partition +
  `E[μ]`, cached to `haiku_scored_tail.tsv` (gitignored — pearltrees-derived titles).

## Pipeline: works ✅
66/66 parsed and partitioned. **Both §14 rules held in the wild:**
- μ memberships routinely sum **> 1** (overlapping relations both fire) — Haiku did not normalise them.
- `applies[named]` sums **< 1**, leaving honest mass on `none`/`unknown` — Haiku did not pad to 1.

`cell_sampler.expected` produces sane `E[μ]` (0.38–0.62 on these lateral/weak-taxonomic rows); the 9
`cell_sampler` tests + this live run jointly exercise the §12–§14 core.

## Label (conf 0.4) vs Haiku — the augmentation earns its keep
| comparison | count | note |
|---|---|---|
| exact same cell | 15 (22%) | weak guess confirmed |
| same family (taxonomic dir-flip, or see_also↔assoc) | 20 (30%) | right *kind*, e.g. label `subtopic` vs Haiku `super_category` = same edge, opposite direction |
| different family | 28 (42%) | genuine disagreement |
| Haiku says unrelated (`none`≥0.4) | 3 | weak "assoc" guesses Haiku rejects |

**Only 53% label-consistent.** The conf-0.4 structural guesses diverge substantially from Haiku — exactly
the correction signal the augmentation exists to provide. A recurring pattern: pearltrees parent→child edges
labelled `subtopic` read as `super_category` to Haiku (the NODE is the *broader* one) — a **direction-convention
mismatch worth a closer look**, not necessarily a Haiku error.

## Honest caveats
- **This validates the pipeline, not Haiku-as-truth.** Haiku shows apparent tendencies (favours
  `super_category`; generous `see_also`) — a single-scorer pilot can't separate "label was wrong" from
  "Haiku is biased." The §9 plan (eval vs Haiku on a *held-out* tail) still needs the held-out split.
- **Budget overran:** ~**146k tokens** vs the ~30–70k estimate. Causes: Haiku verbose per pair, and
  persisting each batch to file via a **second emission** (the SendMessage re-write doubled output).
  **Lesson for scale:** have the scorer write its file on the *first* pass, trim the prompt, batch larger.
- **83→66 after dedup**; the tail is small. Per §13, 66 distributional rows won't move the model against
  the labelled bulk — **training impact needs the generate-more step** (grow the physics tail).

## Next
1. **Decide generate-more** (expand physics neighbourhoods → grow the tail) before wiring training — the
   pilot proves the pipe; scale proves the signal.
2. When wiring training: forward 70/30 (§13), draw cells from these cached `P(cell)` via `cell_sampler`
   (isolated RNG), self-posterior for the labelled bulk, held-out tail for the §9 eval.
3. Investigate the `subtopic`/`super_category` direction-convention mismatch — it may be a systematic
   labelling-vs-prompt direction issue, cheap to fix and high-leverage.

## Addendum — selective Sonnet escalation of sharp `none`-vs-graph rows
Every tail row is a graph edge (conf<1.0 asserts a relation), so a high Haiku `P[none]` is a direct
graph↔Haiku contradiction. `score_inferred_tail.py flag --none-min 0.3` selected **6** such rows; a single
**Sonnet** subagent re-judged them (the ¼-budget tier; Opus reserved). Sonnet wrote its JSON on the *first*
pass — ~21k tokens, no double-emission.

| node → root | label | Haiku none | Sonnet none | read |
|---|---|---|---|---|
| thermodynamics → Flickr_(Thermodynamics) | element_of | 0.60 | 0.71 | **spurious** (ROOT is a Flickr collection) |
| Z_transform → Zeta_function_regularization | element_of | 0.64 | 0.67 | **spurious** (different math domains) |
| Z_transform → Recommended_for_beginner | element_of | 0.33 | **0.81** | escalation **sharpened** → navigational node |
| control-system-eng → Technicien_automaticien | assoc | 0.30 | 0.45 | leans spurious |
| Gaussian_Linear_Models → MIT…LecNote19 | element_of | 0.30 | 0.23 | both keep some relation (note may cover it) |
| lti-system-theory → Tellegen's_theorem | subtopic | 0.52 | **0.27** | Sonnet **softened** → real-but-loose lateral |

**Takeaways:**
- The selective multi-judge tie-break is cheap and decisive — it confirmed 2 spurious edges, *sharpened* 1
  borderline navigational case (0.33→0.81), and *corrected* Haiku's over-doubt on 2 real-but-loose ones.
- **Many conf-0.4 edges target navigational/meta nodes** (Flickr collections, "recommended for beginners",
  lecture-note PDFs) → the augmentation is also a **graph-cleaning** signal: these edges should likely be
  pruned, not just down-weighted.
- Where Haiku and Sonnet still diverge most (Tellegen, the lecture note), an **Opus** final tie-break is the
  natural next escalation — but only for genuine judge-disagreement, not all `none` rows.

## Update — Engineering expansion + the no-clean decision
**Generate-more, round 1 (no harvesting):** fused `Engineering.smmx` @ hops 3 against the 375 cached pt
trees → 162 new untagged-tail pairs (assoc 92 / subtopic 51 / element_of 19), deduped vs the pilot and
Haiku-scored (3 large batches, concise prompt). **Cumulative distributional rows: 66 → 228.**

**Decision: do NOT clean the `none` cases** (supersedes the "prune spurious edges" suggestion in the
escalation addendum). Rationale (user steer):
- **Disagreement encodes uncertainty.** Graph-asserts-edge vs Haiku-says-`none` leaves mass on *both* the
  relation cell and `none` → an honestly uncertain `E[μ]`. The fuzzy/distributional model is *built* to carry
  that; resolving it away would discard information.
- **Negatives are useful.** The 39 strong-`none` rows (≥0.4) are μ≈0 supervision for the `none` anchor (§9) —
  exactly the negative signal the open-world partition needs.

So: keep all 228 rows (positives + uncertain + negatives). The `flag`/escalation path stays available as an
*optional* uncertainty-refinement on high-stakes rows, **not** a cleaning gate. The "navigational/meta node"
observation is still a true characterisation of where `none` mass concentrates — just retained as data, not pruned.

**Deferred (not yet):** the `none`-disagreement column is *multi-purpose*. Today it's uncertainty + negatives.
Later we can **sort by highest `none`-disagreement to hunt genuine graph errors** — a graph that confidently
asserts an edge a strong judge confidently rejects is a likely *real mistake*, distinct from honest
uncertainty. Same column, different read (keep-as-data now vs sort-descending-to-audit later); `flag` already
selects by `none`, so only a descending sort is needed when we want it.

## Measurement arc — wiring, scale, and the saturation/blind-spot verdict
**Wired** the §9/§14 path into the trainer (`--haiku-tail`: override inferred conf<1.0 graded targets with the
cached LLM E[μ] + judge provenance, fwd/rev by title-match). **Scaled** the distributional set **66 → 228 →
417** rows (pilot + Engineering + System&Control, all from cache, no harvester; 411 haiku + 6 sonnet).

**Fresh from-scratch sweep (3 seeds × base/treat, 500 steps):** UNRELIABLE — seed-3 diverged for *both* arms
(WIKI 36%, μ→0); on the 2 valid seeds treatment gave only a small WIKI-order gain (+0.7, +1.6%); SYM/OOD were
noise (the dramatic first-run drop 23.6→7.9% OOD did NOT replicate). The repeat correctly deflated a fluke.

**Warm-start sweep (2 seeds × base/treat from `model_nodetype.pt`, 400 steps):** STABLE (no divergence; the
3→5 judge-embedding resize auto-handled). Result — **treatment ≈ baseline on every clean metric**:

| metric (warm-start) | base (s1/s2) | treat (s1/s2) |
|---|---|---|
| WIKI order-acc | 99.8 / 99.8 | 99.8 / 99.9 |
| SYM corr (40) | +0.665 / +0.698 | +0.676 / +0.711 |
| WIKI OOD-leak | 28 / 20% | 28 / ~20% |

**Conclusion (honest):** at convergence the base metrics are **saturated and indistinguishable** with/without
augmentation. The fresh-run gain was undertraining, not quality. **Critically, these held-out metrics measure
the BASE simplewiki population — they are structurally blind to the inferred physics/EE tail the augmentation
changes.** "No effect on WIKI/SYM" ≠ "no effect" — it means we measured the wrong population. This is the same
labeled-population-vs-inferred-tail blind spot flagged in the design.

**The missing instrument (next):** a **held-out-tail eval** (§9/§12(3)) — split the 417 scored rows, train
treatment on the train-tail only, and compare each model's μ/E[μ] to the held-out Haiku E[μ] *on the tail*.
That is the only measurement that can see whether augmenting the tail improved the tail. Until it exists, the
data-scaling payoff is unmeasured. (More data is cheap and unblocked; reading it is the gap.)

## Held-out-TAIL eval — the decisive (negative) result
Built `--eval-tail` (§9/§12(3)): model E[μ] under an equal-weight operator superposition (none excluded) vs
the cached Haiku E[μ], on tail rows whose Haiku target the model never trained on. Split 417 → 333 train-tail
/ 84 holdout; warm-start from `model_nodetype.pt`, 2 seeds × base/treat, only difference = the train-tail override.

| seed | arm | TAIL corr | TAIL MSE | model μ̄ |
|---|---|---|---|---|
| 1 | base | +0.209 | 0.062 | 0.253 |
| 1 | treat | +0.231 | 0.082 | 0.190 |
| 2 | base | +0.196 | 0.063 | 0.262 |
| 2 | treat | +0.187 | 0.070 | 0.231 |

**The augmentation gives NO measurable lift on the tail.** Corr is flat (~+0.20 both arms, treat +0.02/−0.01
across seeds = within noise); MSE slightly *worse* for treatment. The **baseline already correlates +0.20 with
Haiku on the tail with zero Haiku training** — the frozen-e5 base extracts ~that much on its own, and the
augmentation can't push past it.

**Two interpretations → different levers:**
1. **Representation ceiling (frozen e5):** the base already captures what e5 affords; better targets add no
   information the embedding lacks. Lever = richer encoder, not more data.
2. **Dilution (§13 imbalance, unsolved):** train-tail is only 333/~5000 graded targets (~7%), far below the
   30% target. At 7% of the gradient even perfect targets can't move the model. Lever = upweight the tail.

**Disambiguating experiment (next, cheap, no new data):** upweight the train-tail to ~30% effective share and
re-run the tail eval. Beats baseline → it was dilution (more/heavier tail data helps). Still flat → it's the
representation ceiling (need a better embedding). This is the experiment that tells us which ceiling we're against.

## RESULT — augmentation helps at ~30% tail-weight (dilution confirmed, 3-seed)
The tail-upweight disambiguation, held-out-tail corr (model E[μ] vs held-out Haiku E[μ]), 3 seeds:

| arm | s1 | s2 | s3 | mean ± sd |
|---|---|---|---|---|
| base (tw1, no aug) | 0.202 | 0.144 | 0.143 | **+0.163 ± 0.028** |
| **treat tw6 (~30%)** | 0.277 | 0.290 | 0.287 | **+0.285 ± 0.006** |
| treat tw12 | 0.318 | 0.179 | 0.325 | +0.274 ± 0.067 |

**Verdict: the distributional augmentation DOES help the inferred tail — a robust, low-variance +0.12 corr
lift — but only at ~30% tail-weight (`--tail-weight 6`).** This is **dilution, confirmed**: at the natural ~7%
share the tail signal was drowned (the earlier flat result); given its §13-intended ~30% share it replicates
tightly (sd 0.006). `tw12` over-weights — same mean, 10× variance — an **inverted-U peaking near 30%**, the
capacity-ceiling shape already located for this 3-layer model. **Frozen e5 is NOT the binding limit** (if it
were, no weighting would help).

**Actionable conclusions:**
1. **Free lever now:** train with `--tail-weight ~6` (the §13 30% operating point). No new data needed.
2. **Scaling the tail is justified:** the tail signal is real and learnable and we're below the representation
   ceiling — so more tail data should compound (it just must carry ~30% weight, not 7%).
3. **Don't over-weight** (tw12): past ~30% the small model destabilizes (no mean gain, high variance).

**Caveats:** still the agree-with-Haiku frame (the independent Sonnet/human check via the now-tagged
provenance is the next rigor tier); 82 holdout rows; equal-weight-superposition probe. But a 3-seed
replication at sd 0.006 is signal, not noise — this is the payoff of measuring the tail + repeating across seeds.

## Independent (Sonnet) check — inconclusive; and the cascade triage
Scored the 84 holdout-tail rows with **Sonnet** (one batch, the independent judge) and re-ran base vs tw6
against that reference. **Inter-judge agreement (Haiku vs Sonnet) on the tail: corr +0.277** — i.e. read as
reliability, **~70–80% of per-row tail variance is judge-noise**. This is largely a *selection effect*: the
tail IS the conf<1.0 region the categoriser couldn't confidently tag — we deliberately scored the
high-disagreement rows; the tagged (conf=1.0) data is far cleaner.

**Model vs Sonnet (independent):** base 0.413/0.324, tw6 0.260/0.491 — **means ~0.37 vs ~0.38, variance ±0.15
swamps any effect → INCONCLUSIVE.** (A seed-1-only read of "tw6 < base" was an artifact; seed 2 flipped it —
the same single-seed trap.) So the tw6 +0.12 lift is **agreement-with-Haiku**; it does **not** demonstrably
transfer to an independent judge at this sample size. Notably **base-vs-Sonnet (~0.37) > Haiku-vs-Sonnet
(+0.28)** → the latent signal IS learnable; **Haiku is a noisy teacher of it**, and the frozen-e5 base already
reads the tail better than Haiku labels it.

**Cascade triage (built — `score_inferred_tail.py flag`):** Haiku as cheap triage → escalate only contentious
rows to a stronger judge/human. Two signals (OR): `P[none] ≥ none_min` (graph↔judge contradiction, the
spurious subset) and normalised `entropy(P(cell)) ≥ ent_min` (judge unsure which relation, the fuzzy subset);
ranked by contention, `--top-k` budget-bounds it. **Finding:** at reasonable thresholds ~**half** the tail
flags (211/417) — the tail is *uniformly* ambiguous, so triage-by-uncertainty doesn't shrink to a tiny set;
`--top-k` (e.g. 42 = one Sonnet query) is the practical form, and the **`none` subset dominates contention**.

**Forward path (revised by these results):**
1. **Don't scale single-judge tail data** — its independent value is unproven and it overfits the teacher.
2. **Denoise the target instead:** ensemble judges (avg Haiku+Sonnet) on the contentious set, or the model's
   **self-posterior** (§13) — empirically motivated since base-vs-Sonnet (0.37) > Haiku-vs-Sonnet (0.28).
3. **Cascade, don't blanket-spend:** Haiku triage → top-K escalation to Sonnet/Opus/human (Anthropic tiers via
   subagents today; a non-Anthropic interface is a future generalisation).
