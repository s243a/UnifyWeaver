# Math-deepening round — core subfields from enwiki

Seeds downward from the core mathematics subfields the user named (calculus/analysis, real & complex
analysis, number theory, group theory, topology, set theory, category theory, geometry, …) now that the
enwiki graph has the depth. Adds them by fine-tune-with-replay. Prototype only; no WAM-Rust core changes.

## Data (`gen_math_fields_pairs.py`, corpus=enwiki)

`wide_enwiki_math` slice: `wide_enwiki` ∪ depth-≤2 closures of 17 math subfields, **e5-coherence-filtered**
(enwiki's dense cross-links leak physics/CS/music into raw closures, so keep only nodes whose e5 argmax
over the domain roots is Mathematics) → **15,181 nodes (+813 new math nodes)**: Galois theory, Riemannian
geometry, Banach spaces, functors, etc. Subfields grouped into **areas** (analysis / algebra /
geometry-topology / foundations / discrete) so within-area positives are coherent.

**533 Haiku pairs** scored (~56k tokens, 3 parallel subagents, graded rubric) → `mu_pairs_scored_mathfields_260622-123223.tsv`.
Stratum mean μ shows a clean **within-area > cross-area > cross-domain** gradient — evidence the areas have
real internal structure:

| within-area | μ̄ | cross-area | μ̄ | cross-domain | μ̄ |
|---|---|---|---|---|---|
| pos_geomtop | 0.70 | cross_AA (analysis×algebra) | 0.36 | cross_MP (math×physics) | 0.12 |
| pos_analysis | 0.69 | cross_GA (geom×algebra) | 0.33 | cross_MS (math×CS) | 0.12 |
| pos_foundations | 0.64 | cross_AF (analysis×found.) | 0.25 | | |
| pos_algebra | 0.63 | cross_GTd (geom×discrete) | 0.23 | | |
| pos_discrete | 0.60 | | | | |

## Results (fine-tune-with-replay, seed 1)

The discrimination probe's **Math entry was extended to 10 nodes** — the classic 5 (Calculus,
Differential_equations, Mathematical_analysis, Logic, Fields_of_mathematics) + 5 deepened subfields
(Number_theory, Group_theory, Topology, Real_analysis, Complex_analysis).

| model | Math (10 nodes) | overall | SYM corr |
|---|---|---|---|
| baseline (eng+enwiki, no math deepening) | **10/10, margin +0.29, top-2 100%** | 33/39 (85%) | +0.827 |
| placebo (warm+replay, no new math) | 10/10, margin +0.27 | 34/39 (87%) | — |
| fine-tune+replay (math deepening) | 10/10, margin +0.26 | 34/39 (87%) | +0.827 |

### (1) Math discrimination was ALREADY saturated — e5 knows the textbook fields
All 10 math probe nodes — including the deepened subfields `Number_theory`, `Group_theory`, `Topology`,
`Real_analysis`, `Complex_analysis` — argmax **Mathematics 10/10 on the baseline**, never trained on the
math-deepening data. The deepening leaves it flat (10/10 → 10/10; margin +0.29 → +0.26; the **placebo is
identical**). This is the **same pattern as modern physics** (enwiki round): frozen e5 already
discriminates standard academic fields, so new data cannot move the already-saturated argmax. The old
"Math 3/5" brittleness (`REPORT_matheng.md`) was a *thin-simplewiki-graph + math-of-physics-probe*
artifact, not a real gap — on the enwiki graph with a proper math probe, Math is clean.

### (2) The data's value is the within-subfield RANKING, not discrimination
Held-out per-stratum corr: `pos_discrete` **+0.922**, `pos_foundations` +0.804, `pos_algebra` +0.673,
`cross_GA` +0.732, `cross_AA` +0.570, `pos_analysis` +0.482, `pos_geomtop` +0.178 (weak, n=11);
overall +0.827. The model learned the fine within-area relatedness and the area gradient — that is the
genuine contribution, not a discrimination gain.

### (3) No forgetting
SYM corr held at **+0.827** (= baseline), Physics discrimination even ticked up (6/9 → 7/9), AI held
(2/5). Replay did its job.

### (4) Intra-math SUBFIELD discrimination — the non-saturated axis (the right test)
The cross-domain probe (μ(node|Mathematics) vs the *other domains*) only asks "is this more math than
physics" — which e5 nails 100%. The sharper question is whether the subfields discriminate **against each
other**: μ(node | Real_analysis) vs μ(node | Algebra / Topology / Set_theory / …). Probing clear member
nodes of 10 subfields, argmax over the **subfield** roots:

| | subfield argmax | mean-rank /10 | top-2 |
|---|---|---|---|
| baseline (no math data) | 21/39 (54%) | 2.72 | 64% |
| placebo (churn, no new math) | 23/39 (59%) | 2.46 | 74% |
| fine-tune (math data) | 24/39 (62%) | 2.28 | 79% |

Two things this reveals that the cross-domain probe hid: **(a)** subfield discrimination is genuinely
**not saturated** — frozen e5 separates the math subfields only 54% (vs 100% for math-vs-other-domains),
so there *is* headroom here; **(b)** the math data helps (54%→62%, top-2 64%→79%), but the placebo shows
**most of the gain is churn** from re-optimising on the denser math graph — the data's marginal
contribution *beyond churn* is real but modest (+3 pts argmax, +5 pts top-2, mean-rank −0.18). So the
deepening data does add a little genuine subfield-separation signal, just not a lot.

### (5) Worked example — `Algebraic_geometry`, the algebra∩geometry boundary
A focused micro-round on `Category:Algebraic_geometry` (real parents: `Fields_of_abstract_algebra` **and**
`Fields_of_geometry`) demonstrates the two sampling modes and the boundary structure (`gen` →
`mu_pairs_scored_alggeom_260622-162338.tsv`, 80 Haiku pairs ~16k tok):

- **Downward** (depth-≤2 closure, within-AG): mean μ **0.75** — tight subfield relatedness
  (`AG`↔`Scheme_theory`/`Moduli_theory`/`Morphisms_of_schemes` = 1.0, `Algebraic_curves`↔`Elliptic_curves`
  0.92).
- **Bidirectional-coinflip** (up to the two parents, across to siblings; e5-math-coherence-filtered to
  stop apex drift into Oceanography/Alchemy): mean μ **0.36** — the *graded* boundary
  (`Elliptic_curves`↔`Commutative_algebra` 0.65 = algebra side, `AG`↔`Conic_sections` 0.70 = geometry
  side, `Curves`↔`Classical_geometry` 0.75; junk ~0).

On the trained model `Algebraic_geometry` reads high-to-**both** its parents (top anchors Topology 0.80,
Geometry 0.77, + Linear 0.76, Group 0.51) — correct multi-membership, the intra-math analogue of
`Mechanics×Engineering` / `AI⊂CS`. The over-attraction of the generic `Topology`/`Linear_algebra` anchors
(why AG argmaxes Topology, not "algebraic geometry") is exactly the subfield-anchor mis-calibration that
the 62%/79% subfield separation leaves on the table.

### (6) Reusable boundary sampler + more worked examples
`gen_boundary_pairs.py` generalises the AG round (`--down` / `--bidir` seeds, e5-coherence-filtered).
Three more samples (`mu_pairs_scored_boundary_260622-171513.tsv`, 120 pairs ~17.5k tok) reproduce the
downward > bidir gradient and show the bidir-μ tracks how *central* a seed's neighbourhood is:

| seed | mode | mean μ | reads as |
|---|---|---|---|
| `Algebraic_topology` | downward | **0.76** | tight within-AT (Homology/K-theory/Knot_operations = 1.0) |
| `Topological_methods_of_algebraic_geometry` | bidir | **0.47** | a narrow boundary — all neighbours are tight AG/topology/homotopy siblings |
| `Tensors` | bidir | **0.36** | a wider boundary — reaches linear-algebra, differential-geometry AND mathematical physics |

### (7) Statistics + Estimation Theory round (the flagged gap, now filled)
The coverage gap — Probability solid but **Statistics** only spillover and **Estimation_theory** absent —
is closed with a dedicated round (`mu_pairs_scored_stats_260622-182510.tsv`, 200 pairs ~32k tok). These categories live
*outside* the math slice, so the neighbourhood was pulled from the full 9.9M-edge enwiki graph by a
**streaming BFS** (downward closures of the seeds + one level up + siblings; the naive full-graph load
OOMs). Downward from `Statistics`/`Estimation_theory`/`Statistical_theory`; bidirectional from
`Estimation_theory`/`Statistical_theory` — with **`--coh-keep Mathematics,Computer_science,Engineering`**
so the bidir walk keeps the genuine stats↔signal-processing↔control boundary instead of filtering it to
math-only.

| seed | mode | mean μ | reads as |
|---|---|---|---|
| `Estimation_theory` | downward | **0.77** | tightest — a focused subfield (Estimator/M-estimators/Point_estimation/Bayesian_estimation all 1.0) |
| `Statistical_theory` | downward | 0.67 | coherent theory cluster |
| `Statistics` | downward | **0.47** | most diffuse — the top category is broad/admin-heavy (educators, regions, organizations) |
| `Estimation_theory` | bidir | **0.50** | the multi-domain bridge — signal-estimation, econometrics, decision theory |
| `Statistical_theory` | bidir | 0.39 | wider statistical boundary |

`Estimation_theory` is confirmed as the stats analogue of `Tensors` — a genuine multi-domain boundary
(stats / signal processing / econometrics / decision theory), and the downward-μ ranks subfield
*focus* (Estimation 0.77 tight ≫ Statistics 0.47 diffuse).

### (8) Signals / Systems / Information / Control round
The math↔EE↔CS interface (`mu_pairs_scored_sysinfo_260622-184444.tsv`, 280 pairs ~34.5k tok; streaming-BFS slice,
`--coh-keep Mathematics,Computer_science,Engineering,Physics`). Note `Linear_systems`/`Linear_system_theory`
are not populated enwiki categories — linear-systems theory lives under `Control_theory`/`Systems_theory`
(Classical/Nonlinear/Optimal control, Stability_theory), which is what was seeded.

| seed | mode | mean μ |
|---|---|---|
| `Control_theory` / `Signal_processing` | downward | **0.67** (tight cores) |
| `Information_theory` | downward | 0.60 (some CS-networking spillover) |
| `Systems_theory` | downward | **0.42** (diffuse — bleeds into social/economic/psychology "systems") |
| `Control_theory` bidir | bidir | 0.40 |
| `Information_theory` bidir | bidir | 0.33 |
| `Signal_processing` bidir | bidir | 0.29 (messy EE/imaging/audio reach) |

Same shape as the stats round: focused subfields score tight (~0.67), umbrella categories diffuse (~0.42),
and the bidir-μ ranks how clean the boundary is. These domains' enwiki neighbourhoods are *noisier* (game
theory, social systems, economics, even astronomy tangle in via broad parents) — the graded rubric scored
that junk ~0, but it's a reminder that engineering/applied categories need stricter coherence filtering
than pure-math ones.

### (9) Cybernetics / Systems-theory round
Both modes from `Cybernetics` + `Systems_theory`, plus downward from `Systems_science` /
`Operations_research` / `Dynamical_systems` (`mu_pairs_scored_cyber_260622-223014.tsv`, 280 non-neg pairs ~37k tok;
slice `wide_enwiki_cyber`, 1,467 nodes / 2,369 edges; `--coh-keep Mathematics,Computer_science,Engineering`).

| seed | mode | mean μ |
|---|---|---|
| `Cybernetics` | downward | **0.79** (tight — the automation/biocybernetics subtree is coherent) |
| `Systems_science` | downward | **0.77** |
| `Operations_research` | downward | 0.70 |
| `Dynamical_systems` | downward | 0.66 |
| `Systems_theory` | downward | **0.50** (diffuse — social/economic/psychology "systems" bleed in) |
| `Cybernetics` | bidir | 0.39 |
| `Systems_theory` | bidir | 0.26 (messiest reach — game theory, social systems, politics) |

**On the user's "I'd have thought cybernetics would hit *more*":** it does score the *cleanest* of any
seed here downward (0.79) — but it is **thin**, not broad. Today's `Category:Cybernetics` has only **4
direct children** (Automation, Cyberneticists, Biomedical_cybernetics, Organizational_cybernetics; ~34-node
depth-2 pool). The concept's historical breadth has been *redistributed* by decades of Wikipedia
recategorization into Systems_science / Control_theory / Dynamical_systems / AI — so the breadth shows up
in the **bidirectional reach** (232 math-coherent endpoints) and in cybernetics' role as a *connective*
node, not in a deep downward subtree. Same saturation shape as rounds 7–8: focused cores score tight
(~0.79), umbrella categories diffuse (~0.26–0.50), and the bidir-μ ranks boundary cleanliness.

### (10) System-theory round (mindmap-aligned roots)
Downward from the direct roots of the user's SimpleMind *System Theory* map — `Network_theory`,
`Dynamical_systems`, `Complex_systems_theory`, `Systems_analysis` — plus bidirectional on the broad
`Networks` hub (`mu_pairs_scored_systheory_260622-231618.tsv`, 195 non-neg ~35k tok; slice
`wide_enwiki_systheory`, 599 nodes / 967 edges via the new reusable `build_slice.py`; `--coh-keep
Mathematics,Computer_science,Engineering`). Sampling the *same* roots that anchor the mindmap is
deliberate: it maximises the enwiki∩mindmap node overlap, which is the cheap "bootstrap-from-overlap"
calibration path for the eventual SimpleMind/Pearltrees tie-in (see `DESIGN_calibrated_judges.md` §5).

| seed | mode | mean μ |
|---|---|---|
| `Network_theory` | downward | **0.77** (tight; its 6 enwiki subcats map ~1:1 onto the mindmap's "Network theory" branch) |
| `Complex_systems_theory` | downward | **0.73** (thin pool, n=9) |
| `Systems_analysis` | downward | 0.65 |
| `Dynamical_systems` | downward | 0.62 |
| `Networks` | bidir | **0.075** (≈ hard negatives) |

**Key finding — separate the hub from the theory.** `Networks` is a generic enwiki hub dominated by
telecom / transport / broadcasting / gaming / social-media subcategories with no tie to mathematical
network theory, so its bidirectional boundary walk is almost pure junk (0.075 — it functions as *hard
negatives*, not positives). `Network_theory`, the tight sibling the user's map actually points at, scores
0.77. The down-`Network_theory` / bidir-`Networks` split (the user's call) cleanly isolated the signal from
the hub — confirming that **broad container categories make poor positive seeds; their tight theory-named
children are the ones to sample.**

## Honest verdict — the saturation pattern, confirmed
This round sharpens the meta-finding across the whole arc:

- **AI** (enwiki round) was *absent* from e5's confident repertoire → new data added a genuinely new
  separable domain (0/5 → 4/5). **New capability.**
- **Modern physics, Engineering, and now Math subfields** are textbook fields e5 *already* discriminates
  → new data only refines the **ranking**, never the (saturated) discrimination. **Refinement, not
  capability.**

**Frozen e5 saturates COARSE (cross-domain) discrimination but NOT FINE (intra-field subfield)
discrimination.** Math-vs-other-domains is 100% out of the box, so deepening can't move it; but
subfield-vs-subfield is only 54% on e5 alone, and there fine-tuning helps (→62%, top-2 →79%) — though
the placebo shows churn drives most of it, with the data adding a modest real increment. So the bound is
not "what e5 knows" flatly, but **at what granularity** e5 resolves: it confidently places fields under
their domain, but blurs sibling subfields. The right place to spend future labeling budget is therefore
**e5's blind spots and its blur** — genuinely novel/absent domains (AI: new capability) and *fine
intra-field distinctions* (subfields: a modest but real gain) — not re-confirming the coarse cross-domain
separations e5 already nails. (Credit: the subfield axis was surfaced by a reviewer asking whether nodes
were tested relative to *Real_analysis*, not just *Mathematics*.)

Reproduce: build `wide_enwiki_math` (closures ∩ e5-coherence) → `gen_math_fields_pairs.py` → score the
535 non-neg pairs → `mu_pairs_scored_mathfields_260622-123223.tsv`; `UW_MU_GRAPH=…/wide_enwiki_math/…
UW_E5_CACHE=e5_tables_enwiki_math.pt train_mu_attention.py --pairs mu_pairs_scored_mathfields_260622-123223.tsv
--pairs-corpus enwiki --replay-pairs mu_pairs_scored_prior.tsv --init-from <baseline> --lr 1.5e-4
--steps 500` (+ placebo with `--pairs mu_pairs_scored_prior.tsv`).
