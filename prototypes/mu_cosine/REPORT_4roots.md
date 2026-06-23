# Four-root multi-domain — Physics / Chemistry / Mathematics / Computer_science (#3310 follow-up)

Extends the two-root multi-domain work (#3310) to **four core science roots**, building on #3310's
μ-coherence pools and #3309's bidirectional-coinflip walk. Goals: keep closing the SYM gap with more
Physics, and test **multi-domain discrimination** (does the same model score a node correctly against
*its own* root vs the other three?). Honest question: which domains discriminate cleanly vs which are
too thin in this graph.

## Pools — depth-bounded closure ∩ μ-coherence (verified graph facts)

The full downward closure from `Physics` is ≈ the whole graph (7811 nodes — densely cross-linked), so an
unbounded closure is useless as a domain pool. `Branches_of_science` is the shared ancestor of all four
roots within ~3 hops. Pools = **depth-≤3 downward closure ∩ argmax-over-roots** (each node joins the root
it is most e5-coherent with), Physics additionally fixture-calibrated. (AI is absent from this graph.)

| root | closure(≤3) | pool | sample |
|---|---|---|---|
| Physics | 74 | 38 | Mechanics, Physicists, Thermodynamics, Matter, Energy |
| Chemistry | 31 | 30 | Chemical_compounds, Chemical_reactions, Periodic_table, Acids |
| Mathematics | 12 | **9** (thin) | Fields_of_mathematics, Calculus, Logic, Differential_equations |
| Computer_science | 41 | 32 | Software, Computer_networking, Computer_architecture, Operating_systems |

All six pairwise pool overlaps **0** — disjoint by argmax. Math is thin (9 nodes), as expected.

## Data (steps 2–3)

885 new pairs Haiku-scored (**two inline subagents, ~43.7k tokens, 0 tool calls**; graded rubric):
within-domain (Physics 300, Chemistry 100, Math 25, CS 100), all six cross strata (50 each), and a
bidirectional-coinflip batch (60, seeded at Physics) for lateral diversity. Committed in
`mu_pairs_scored_4roots_260621-004105.tsv`. Stratum mean μ: pos_phys 0.40, pos_chem 0.84, pos_math 0.64, pos_cs 0.66.

## Retrain (multi-task, keep WIKI/LLM)

`train_mu_attention.py --pairs mu_pairs_scored_4roots_260621-004105.tsv --llm --steps 900`. WIKI held-out order-acc
**98.6%** (target ~99% — preserved). Gate-leak 0/5 every operator; every dense map feeds
`check_feeds_rust`. SECONDARY node-gated lin-agreement **+0.237** (#3310 +0.183, control +0.124).

### (a) per-stratum SYM held-out corr (control +0.726; comparable `pos` stratum was +0.695 in #3310)

| stratum | n | corr | μ̄ target |
|---|---|---|---|
| overall | 467 | **+0.812** | — |
| **`pos`** (comparable to #3310) | 160 | **+0.570** | 0.68 |
| pos_phys | 114 | +0.876 | 0.40 |
| pos_chem | 51 | +0.841 | 0.79 |
| pos_cs | 20 | +0.825 | 0.70 |
| pos_math | 5 | +0.927 | 0.53 |
| cross_ALL | 35 | +0.855 | 0.28 |

**Honest:** the four-root broadening did **not** further close the SYM gap — the comparable `pos`
stratum *regressed* +0.695 → +0.570 (n=160, partly held-out sampling, partly the cost of spreading
capacity across four domains + the noisier cross/bidir strata). The within-domain strata are all strong
(+0.83–0.93) and overall is +0.812, but the headline of this PR is **discrimination**, not a further
SYM gain. (Per-cross-stratum corrs are low-resolution — most cross targets are ≈0, e.g. cross_PS μ̄=0.07
— so a few points dominate the correlation; the discrimination test below is the real cross-domain read.)

### (b) MULTI-domain discrimination — argmax μ over the four roots: **18/20 (90%)**

Confusion (true → argmax):

```
           Phys   Chem   Math   Comp
   Phys       5      0      0      0
   Chem       0      5      0      0
   Math       2      0      3      0
   Comp       0      0      0      5
```

| domain | clean? | detail |
|---|---|---|
| **Physics** | ✅ 5/5 | Thermodynamics/Optics/Mechanics/Electromagnetism/Motion all argmax Physics |
| **Chemistry** | ✅ 5/5 | Acids/Chemical_compounds/Oxygen/Chemical_reactions clean; Periodic_table close (P 0.87 / C 0.88) |
| **Computer_science** | ✅ 5/5 | strongest separation (Software 0.84, Networking 0.79; low μ to other roots) |
| **Mathematics** | ⚠️ 3/5 | `Calculus` (P 0.83 > M 0.78) and `Differential_equations` (P 0.86 > M 0.76) leak to **Physics**; `Logic`/`Fields_of_mathematics`/`Mathematical_analysis` discriminate correctly |

Borderline nodes land sensibly: `Atoms`→Chem (0.86), `Electronics`→CS (0.63), `Measurement`/`Materials`/
`Energy`→Phys.

## Honest verdict — which domains discriminate, which are too thin

- **Physics, Chemistry, and Computer_science discriminate cleanly (5/5 each).** CS is the crispest — its
  vocabulary (`Software`, `Operating_systems`, `Computer_networking`) barely overlaps the other sciences,
  so μ to the other three roots stays low.
- **Mathematics only partially (3/5), and the failures are principled, not random.** Its two
  physics-shared core topics — `Calculus` and `Differential_equations` — are the actual mathematical
  machinery *of* physics, so e5 + the (much larger) Physics training pool pull μ\|Physics above μ\|Math.
  Its non-physics core (`Logic`, `Fields_of_mathematics`, `Mathematical_analysis`) separates correctly.
  Math is **too thin in this graph** (9-node depth-≤3 pool, vs 30–38 for the others) to overcome the
  Physics bias on the shared calculus/analysis nodes — exactly the sparsity the task flagged up front.
- **Implication:** four-root discrimination works at 90% with a single shared model and no new
  architecture; closing the last 2/20 needs either more Math labels (to counter the Physics-pool size
  bias) or a margin that explicitly separates the math-of-physics nodes — not more breadth, which (per
  the `pos`-stratum regression) trades against the original physics-relatedness fidelity.
