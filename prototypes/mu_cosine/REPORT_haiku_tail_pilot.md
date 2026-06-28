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
