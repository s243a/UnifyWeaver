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

**533 Haiku pairs** scored (~56k tokens, 3 parallel subagents, graded rubric) → `mu_pairs_scored_mathfields.tsv`.
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
535 non-neg pairs → `mu_pairs_scored_mathfields.tsv`; `UW_MU_GRAPH=…/wide_enwiki_math/…
UW_E5_CACHE=e5_tables_enwiki_math.pt train_mu_attention.py --pairs mu_pairs_scored_mathfields.tsv
--pairs-corpus enwiki --replay-pairs mu_pairs_scored_prior.tsv --init-from <baseline> --lr 1.5e-4
--steps 500` (+ placebo with `--pairs mu_pairs_scored_prior.tsv`).
