# Two-judge posterior — first build (k=2 GLM): joint beats product, but the cross pseudo-judge doesn't add

*First build of `DESIGN_two_judge_posterior.md`. `fit_two_judge_posterior.py` fits the joint `P(D,S | features)`
over the directional (D) and symmetric (S) operators and walks the combiner ladder, measuring held-out (node-
disjoint) log-loss. 880 LLM-scored Wikipedia pairs (gpt-5.5-low, `wiki_rel_scored.tsv`), 20 splits. 2026-07-06.*

## Setup
- **Labels** (fuzzy LLM `mu[]`, so D & S can co-occur): D = max over {subcategory, subtopic, element_of,
  super_category}; S = max over {see_also, assoc}; binarised at 0.5. Joint class ∈ {00,01,10,11}.
- **Features** (the two judges + model): `μ_D` = model directional readout (max of μ_HIER either way),
  `μ_S` = model symmetric readout (μ_SYM), `d` = graph up-walk hit-prob.
- **Pseudo-judges:** `μ_D², μ_S²` (self / confidence rung), `μ_D·μ_S` (cross / correlation rung).
- **Metric:** held-out log-loss of the joint (D,S), mean over 20 node-disjoint splits.

## Result
```
n=880   D+ 45%   S+ 16%   BOTH(1,1) 4%   corr(D,S label) = −0.19
model                         held-out log-loss (20 splits)
product-of-marginals          0.7608 ± 0.110
joint linear (μ_D,μ_S,d)      0.7113 ± 0.114      ← the win
joint +self (μ_D²,μ_S²)       0.7105 ± 0.117
joint +CROSS (μ_D·μ_S)        0.7107 ± 0.117
```

## Two findings
1. **JOINT beats product-of-marginals (0.711 vs 0.761)** — the core claim of the design holds: the operators are
   correlated and modeling them jointly captures what product-of-marginals (which forces `P(D,S)=P(D)P(S)`) throws
   away. This reproduces the `mu_posterior` "joint > PoE" result specifically for the D/S operators.
2. **The explicit cross pseudo-judge (`μ_D·μ_S`) adds nothing** — the three joint rungs are identical within noise.

## What that refines in the design
The design leaned on the cross pseudo-judge as *the* large-n correlation term. Empirically, **the correlation here
is UNCONDITIONAL** — a baseline "a pair is *either* directional *or* symmetric" (`corr(D,S) = −0.19`, co-occurrence
only 4%). The **joint multinomial already captures unconditional correlation through its 4-class structure** (the
class intercepts encode the co-occurrence rate), so the explicit `μ_D·μ_S` feature is redundant. The cross
pseudo-judge earns its keep only for **feature-CONDITIONAL** correlation — where the D↔S coupling *varies with* the
readouts — which this h=1 data doesn't have.

So the essential second-order move is **joint modeling** (a multinomial over the operator outcome), *not* the
interaction feature. Restated on the ladder: product-of-marginals (independent) → **joint (captures unconditional
correlation) = the win** → +cross (captures conditional correlation = data-dependent, not needed here).

## The cross term is HOP-CONDITIONAL — measured (2026-07-06)
The cross pseudo-judge is the QDA/heteroscedastic term: it earns its keep only if `Cov(D,S)` **varies across the
space** (LDA/linear suffices for a constant covariance — a Gaussian's score is linear, its correlation a single
change of basis absorbed into linear weights). So the deciding check is whether `corr(D,S)` is a function of hops.
It is (`corr(D,S)` on the LLM labels, threshold 0.3):

| hop | SimpleMind corr / BOTH% | Wikipedia corr / BOTH% |
|---|---|---|
| 1 | +0.24 / 68% | nan / 24% |
| 2 | +0.22 / 55% | −0.31 / 28% |
| 3 | +0.42 / 60% | −0.10 / 26% |
| 4 | +0.26 / 38% | −0.18 / 6% |
| 5 | +0.20 / 30% | +0.18 / 20% |

- **Heteroscedastic in both** — `corr(D,S)` varies with hop, so the covariance is not constant ⇒ the cross term is
  warranted *once hops are included* (the h=1-only fit above was a single, constant slice → LDA sufficed).
- **Opposite SIGN by corpus** — SimpleMind: D & S **co-occur** (+corr, BOTH 30–68%; a concept is nested *and*
  associated). Wikipedia: they **compete** (−corr, BOTH ≤28%). The first build measured Wikipedia h=1 (−0.19) — the
  worst place to find a co-occurrence term.
- **⇒ build the cross pseudo-judge on SimpleMind / a deliberate concept graph**, with the term **hop-conditional**
  (`μ_D·μ_S` coupled to `d`), where the co-occurrence is strong and the correlation swings — the QDA regime the
  h=1 linear fit could not exercise.

## Caveats & next
- **h=1 Wikipedia is the wrong regime for the cross term.** D↔S *anti*-correlate here (compete); the *positive
  co-occurrence* that would exercise a conditional cross-term appears at **deep hops** / concept graphs
  (`REPORT_multihop_direction`, SimpleMind ~2× symmetric mass). The cross pseudo-judge should be retested where D
  and S genuinely co-occur.
- **`μ_rev=0` clamp** (hard constraint) is not exercised by this D/S build; it enters the directional sub-model.
- **`LLM_op` is the label here, not a feature** — this is the *student/distillation* direction (predict the LLM
  from free features `μ_D, μ_S, d`). The teacher (LLM as a feature toward an independent op) needs a third label
  source; deferred.
- Next: (a) retest the cross term on the deep-hop / SimpleMind co-occurrence regime; (b) move `d` to a continuous
  walk on the Pearltrees multi-parent DAG (on h=1 categories `d` is near-binary and contributes little).

Repro: `fit_two_judge_posterior.py --scored wiki_rel_scored.tsv --e5-cache wiki_rel_e5.pt --graph 100k_cats/... --seeds 20`.
