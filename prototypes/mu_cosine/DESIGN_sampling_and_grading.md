# Sampling, grading, training & evaluation methodology (μ-attention rounds)

How a "round" works end-to-end, the techniques used, and **the rules for when sample data is kept vs
thrown out**. This is the accreted practice behind `REPORT_matheng.md` → `REPORT_engineering_finetune.md`
→ `REPORT_widen_enwiki.md` → `REPORT_math_fields.md` (and the stats / signals-control rounds folded into
the last). Generators: `gen_*_pairs.py`, `gen_boundary_pairs.py`. Trainer: `train_mu_attention.py`.

```
graph (enwiki, name-resolved)              scripts/ingest_enwiki_categories.py (2024 categorylinks fix)
  └─ slice  (depth-bounded closures ∩ μ-coherence, admin/leak filtered, OOM-safe)
       └─ SAMPLE  downward (within-subfield) + bidirectional-coinflip (boundary)
            └─ REJECT junk  (admin / leak / e5-coherence / dedup) ; keep graded-noise
                 └─ INSTRUCT-GRADE with Haiku (6-band rubric, batched, budget-capped)  → mu_pairs_scored_*.tsv
                      └─ FINE-TUNE-WITH-REPLAY  (warm-start, replay buffer, lower LR, corpus tag)
                           └─ EVAL  discrimination (argmax + margin) · per-stratum corr · placebo control · forgetting
```

---

## 1. Sampling techniques

**Downward sampling** — depth-bounded *downward closure* of a seed category (its sub-tree). Captures
*within-subfield* relatedness (both endpoints are the same topic). Depth ≤ 2–3; deeper explodes on the
densely cross-linked enwiki graph. Used for "what is *inside* this field" (`gen_*`, `closure()`).

**Bidirectional-coinflip sampling** — the depth-balanced zero-drift walk (`walk_bidir(..., mode="coinflip")`,
see `DESIGN_bidirectional_walk.md`): at each interior node a fair up/down coin, hub-down-weighted within
the chosen direction. From a seed it reaches *up* to the seed's parents and *across* to sibling subfields,
so it captures the **boundary** — "what this field sits *between*" (Algebraic_geometry ↔ algebra & geometry;
Estimation_theory ↔ signal-processing & econometrics). Mean boundary-μ ranks how *central* a seed's
neighbourhood is (tight siblings → higher; fans out across domains → lower).

**Strata** per round: `pos_*` (within-domain/area/subfield positives, graded high), `cross_*` (cross-area
or cross-domain, graded), free `neg` (domain-node × uniform-random, μ=0, costs no LLM budget).

---

## 2. When we KEEP vs THROW OUT sample data

Rejection is layered; each layer removes a different failure mode. **Order matters** — cheap structural
filters first, the e5 filter next, the LLM grade last.

| filter | what it drops | why / lesson learned |
|---|---|---|
| **Admin / maintenance blocklist** (regex) | `Wikipedia_*`, `Articles_*`, `Hidden_*`, `Commons*`, `*_stubs`, `Template*`, `Automatic_category_TOC`, `Overpopulated_categories`, `*_by_*`, `Award_recipients`, … | enwiki categories are ~14% admin scaffolding; they carry no membership signal and pollute closures/walks. |
| **Per-round leak blocklist** | nodes that sneak in 1–2 hops from a seed but belong elsewhere — e.g. physics-primary nodes (`Mechanics`,`Thermodynamics`) out of the Math pool; `Music`/`Visual_arts`/`Time` out of core-physics; `Physiology`/`Psychiatry` out of Engineering (they enter via `Applied_sciences`→Medicine) | e5's flat baseline cosine can't reject these by margin; a small documented blocklist is the pragmatic fix. Always *documented* in the generator. |
| **Depth bound** | anything beyond depth 2–3 of the seed | the full downward closure of a broad root ≈ the whole graph (verified: simplewiki Physics closure was ~7,800 nodes). Depth-bounding is the primary no-re-explosion guard. |
| **e5 μ-coherence** (`mathy()` / `--coh-keep`) | bidir-walk endpoints whose e5 **argmax over the domain roots** is *not* in the keep-set (default `Mathematics`; `Mathematics,CS,Engineering[,Physics]` for applied rounds), or below a cos floor (~0.74) | the bidir coinflip drifts up through apex hubs and back down into Oceanography/Alchemy/textile-artists; the coherence argmax is what keeps it on-topic. Applied/engineering categories need a *multi-domain* keep-set (stats↔signal↔control) or the genuine boundary is wrongly dropped. |
| **Apex-cap on sibling expansion** | when building a slice, siblings under *high-degree* parents (≤120 children kept; Mathematics/Engineering/Science skipped) | the `+siblings` step blew a stats/control slice from ~1.3k to ~70k nodes by pulling apex hubs' children. Cap by parent degree. |
| **OOM avoidance** | never load the full 9.9M-edge / 2.6M-node graph into Python dicts | it OOM-kills the loader. Work on a **slice**; for categories outside an existing slice, pull their neighbourhood by **streaming BFS** (downward closures + one level up + apex-capped siblings), re-reading the edge file per level. |
| **Dedup vs cumulative** | any pair already in `mu_pairs_scored_cumulative.tsv` (= all prior rounds concatenated) | never re-buy a label; keeps the cumulative set clean for replay. |
| **Graded, NOT dropped** | residual cross-domain noise that survives the above | we do **not** hard-filter the last 5–15% of off-topic pairs — the graded rubric scores them ~0, which is *honest training signal* ("these are unrelated"). Hard-dropping would bias the negatives. The means in each report already reflect this. |

Two honest sub-cases worth calling out:
- **A whole round can be thrown out as a *modelling* input while kept as a *finding*.** The core-physics
  round (`REPORT_phys_discrim.md`) produced clean data but the placebo control showed it didn't move
  discrimination; we committed the data + the negative result rather than pretending it helped.
- **Low scores on "plausible" nodes are correct, not bugs.** Haiku graded `Vision`×`Sound` at 0.15
  because `Vision` is perception, not acoustics; `Statistics`-downward is *diffuse* (0.47) because the top
  category is admin/people-heavy. We keep those — diffuseness is real signal.

---

## 3. Instruct grading (the Haiku graded-rubric scoring)

"Instruct grading" = scoring each candidate pair's membership/relatedness μ ∈ [0,1] with a **graded
instruction rubric** given to a cheap LLM judge (Haiku), rather than a binary in/out label. The rubric is
the same shape every round (tuned examples per round):

| band | meaning | examples |
|---|---|---|
| **1.0** | one nested in / a direct sub-topic of the other | `Number_theory`↔`Prime_numbers`, `Estimation_theory`↔`Estimator`, `Control_theory`↔`Optimal_control` |
| **0.7–0.9** | same narrow subfield, strongly related | `Real_analysis`↔`Analytic_functions`, `Homeomorphisms`↔`Diffeomorphisms` |
| **0.5–0.7** | same broad area **or** a genuine *high-to-both* boundary | `Geometry`↔`Topology`, `Elliptic_curves`↔`Commutative_algebra`, `Mechanics`↔`Engineering` |
| **0.3–0.5** | cross-domain but real (math-of-physics, theory↔application) | `Differential_geometry`↔`General_relativity`, `Category_theory`↔`Type_systems` |
| **0.1–0.3** | weak / tangential; **people & org lists** (`*_theorists`, `Fellows_of…`, `American_*`) relate only modestly | |
| **0.0–0.1** | unrelated | `Topological_spaces`↔`Hot_drinks`, `Stability_theory`↔`Interacting_galaxies` |

**How it's used:**
- Issued to **inline Haiku sub-agents** (one or several in parallel), pairs passed *in the prompt*, scores
  returned *inline* (`a<TAB>b<TAB>score`), **no tools, no file I/O** in the sub-agent. Batches ~150–220
  pairs each; ≥40 minimum to amortise prompt overhead.
- **Budget discipline:** hard cap ≈ typical allocation (~50–70k Haiku tokens / 5-h window); rounds are
  sized to that. The user confirms budget before any scoring spend.
- The rubric prompt is **tailored per round** with in-domain anchor examples and an explicit *"many pairs
  here are random/unrelated — score those near 0, do not inflate"* instruction, because cross strata are
  deliberately noisy.
- Scores are **persisted once** (`mu_pairs_scored_*.tsv`), reconciled against the candidate set (report
  `filled / neg / missing`), and **bought once** — labels are never re-generated (dedup, §2).
- Each pair carries **provenance** (`corpus`, `judge`): current rounds tag `corpus=enwiki` (vs the older
  `simplewiki`), judge = `haiku` for these graded positives, `graph` for the free μ=0 negatives. The model
  has a maskable provenance token for this (see `DESIGN_directional_attention.md` §provenance); it's
  masked-by-default so μ is source-agnostic unless asked.

---

## 4. Training — fine-tune-with-replay (the default since #3324)

Adding a domain is a **continual-learning** step, not a from-scratch retrain:
- `--init-from <checkpoint>` — warm-start; the head is **not** re-initialised.
- `--replay-pairs <cumulative> --replay-frac 0.4` — each SYM batch mixes ~40% OLD (replayed) examples with
  the new domain's pairs, preventing catastrophic forgetting.
- `--lr ~1.5e-4` (≈⅓–⅕ of the from-scratch 5e-4), `--steps ~400–600`.
- `--pairs-corpus enwiki` — tags the new data's provenance corpus (replay stays simplewiki).
- e5 is **frozen** throughout; only tags + the 2-layer attention + per-operator readout learn.

Why this is the default: across seeds it matches a full retrain on retention, **beats** it on the freshly-
added domain, runs at <½ the compute, and is **far more stable** (full retrain *collapsed* at seed 7 in
`REPORT_engineering_finetune.md`; warm-start never does).

---

## 5. Evaluation — and the controls that keep it honest

- **Cross-domain discrimination** (`discrimination_probe`): for clear nodes of each root, is μ(node|own-root)
  the argmax over all roots? Report **both** hard argmax **and** the ranking/margin view (true-root rank,
  signed margin, top-1/top-2). Argmax is seed-sensitive (swings 56–92%); **top-2/margin is the robust
  metric**. Always print the full confusion matrix.
- **Intra-subfield discrimination** (the *non-saturated* axis): μ(node|`Real_analysis`) vs
  μ(node|`Algebra`/`Topology`/…). Coarse (cross-domain) discrimination saturates on frozen e5 (textbook
  fields are 100%); **sibling subfields do not** (~54% on e5 alone) — that's where new data still helps.
- **Per-stratum held-out corr**: `eval_per_stratum.py` reproduces the deterministic split and reports
  Pearson(μ, target) per stratum — the genuine "did the ranking improve" signal.
- **Placebo / churn control** (`drift_control.py`): re-train with replay but **no new data**, measure how
  much the probe moves vs the real round. New-domain *capability* (AI: 0/5→4/5, placebo stays 0/5) is real;
  *discrimination on already-saturated domains* is mostly churn (engineering/math: real ≈ placebo). Run it
  before crediting any gain to data.
- **Forgetting guards**: physics SYM corr ~+0.83 held, WIKI edge order-acc ~99%, gate-leak 0/5 (filtered to
  graph-present probe nodes for science-only slices).

**Meta-finding the controls produced:** μ-discrimination is bounded by *what frozen e5 already knows, and
at what granularity* — new data buys **capability** only for e5's blind spots (genuinely absent domains
like AI) and a **modest gain** for fine intra-field sibling distinctions; it does **not** move the coarse
cross-domain separations e5 already nails (modern physics, engineering, math subfields all 100% at
baseline). Spend labelling budget on blind spots + boundaries, not on re-confirming saturated fields.

---

## 6. Test matrix — roots & subroots, and a subsampling policy

The discrimination probes are the standing test set. As it grows we **subsample** rather than run every
node every time (below).

### Tier 1 — cross-domain roots (the 6-way `DOMAIN_PROBE`, always tested)
| root | probe nodes (representative) |
|---|---|
| Physics | Thermodynamics, Optics, Mechanics, Electromagnetism, Motion_(physics) *(+ modern: Quantum_field_theory, Particle_physics, General_relativity, Condensed_matter_physics)* |
| Chemistry | Periodic_table, Acids, Chemical_compounds, Oxygen, Chemical_reactions |
| Mathematics | Calculus, Differential_equations, Mathematical_analysis, Logic, Fields_of_mathematics *(+ subfields: Number_theory, Group_theory, Topology, Real_analysis, Complex_analysis)* |
| Computer_science | Software, Computer_hardware, Operating_systems, Computer_networking, Computer_architecture |
| Engineering | Mechanical_engineering, Civil_engineering, Engineering_disciplines, Machines, Infrastructure |
| Artificial_intelligence | Machine_learning, Neural_networks, Deep_learning, Computer_vision, Natural_language_processing |

Borderline (no ground-truth, argmax only): Atoms, Electronics, Measurement, Materials, Energy.

### Tier 2 — intra-math subfield roots (the non-saturated axis)
Real_analysis · Complex_analysis · Number_theory · Group_theory · Topology · Set_theory · Combinatorics ·
Probability_theory · Geometry · Linear_algebra · (Mathematical_logic). **Boundary / multi-membership
subfields** explicitly probed: Algebraic_geometry, Algebraic_topology, Arithmetic_geometry,
Geometric_group_theory, Topological_groups, Analytic_number_theory, Algebraic_number_theory.

### Tier 3 — applied-math domains (sampled, candidate test roots — not yet in the standing probe)
Statistics · Estimation_theory · Statistical_theory · Information_theory · Signal_processing ·
Control_theory · Systems_theory. (Promote to a probe tier once a round trains on them; each is a known
*multi-domain boundary*, so test argmax **and** margin, expecting high-to-both.)

### Subsampling policy (as the matrix grows)
- **Always-on:** Tier 1, full (6 roots × 5–10 nodes = ~40 nodes) — cheap, the headline metric.
- **Rotating:** Tier 2/3 — test **k = 4 nodes per root**, drawn by a *per-eval seed* so coverage
  accumulates across runs; a **full sweep** periodically (and whenever a round touches that subtree).
- **Promotion rule:** a domain joins the standing probe once a round has trained on it; a *boundary* node
  is added when its high-to-both behaviour is the thing under test.
- **Always report the robust metric** (top-2 / mean-margin), not just argmax, since argmax is seed-noisy.
- **Always pair a new-domain claim with the placebo control** (§5) before calling a gain "the data".

---

*Reusable tools:* `gen_boundary_pairs.py` (`--down`/`--bidir`/`--coh-keep`), `drift_control.py`
(placebo), `eval_per_stratum.py`, the streaming-BFS neighbourhood extractor (inline in the stats/sysinfo
rounds — promote to a script when next needed).

## 7. Label-data storage

Each round's bought labels are committed as `mu_pairs_scored_<round>.tsv` (one source of truth per round;
~3.6 MB tracked total, **5,039 distinct Haiku-scored positive pairs** + ~59k *free* μ=0 negatives that are
regenerable from the graph). The **cumulative** replay/dedup set (`mu_pairs_scored_cumulative.tsv`) and the
`*_prior.tsv` baseline are **100% derived** — gitignored, and regenerated by **`./make_cumulative.sh`**
(run after a fresh clone or after adding a new round). Do **not** commit them.

**When to move to Hugging Face (not yet):** the corpus is < 4 MB and the schema/composition are still
evolving, so a versioned/citable HF dataset would freeze a moving target with no external consumer.
Trigger the move on **any single scored file > ~10 MB, total > ~50 MB, or a citation/release need** — not
on a positive count (we already have ~5k positives). When it comes: full corpus on HF (the grading rubric
+ provenance schema in this doc make the dataset card), a small curated "golden" subset on GitHub for
repro/CI. If the *free negatives* ever balloon, store positives-only + a regen recipe (they're free and
graph-derived), keeping the bought labels tiny.
