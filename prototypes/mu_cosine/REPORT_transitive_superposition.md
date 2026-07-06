# Transitive cross-judge superposition — graph-walk (IS) ⊕ LLM-direct (OUGHT)

*Implements the two-part transitive architecture (REPORT_multihop_direction §e/f, the IS/OUGHT principle). The
non-graph side is the deferred piece now built: the LLM scores multi-hop pairs **directly** (`score_with_codex`),
supplying element/subcategory transitive membership; the graph side is the walk hit-prob. 2026-07-05, branch
`claude/transitive-superposition`.*

> **⚠ Read the numbers with these caveats (review):** all magnitudes are **single-seed** on **250 LLM-scored
> pairs (h≤5)**. The LLM OUGHT signal exists **only to h=5** — so every h≥6 row in §2 is an **extrapolation
> regime** where one blend component was trained to zero. The walk-vs-superpos *margins* are single-seed and want
> multi-seed before quantitative use; the qualitative direction of each effect is the finding.

## (1) VALIDATION — does the LLM's direct transitive judgment match the graph models?
250 multi-hop pairs (50 chains × h=1..5) scored by gpt-5.5-low (element_of / subcategory, mu_fwd & mu_rev):

| h | LLM subcat_fwd | LLM element_fwd | LLM **rev** (e+s)/2 | walk hit-prob | 0.9^h |
|---|---|---|---|---|---|
| 1 | 0.768 | 0.258 | 0.007 | 0.517 | 0.900 |
| 2 | 0.600 | 0.114 | 0.005 | 0.292 | 0.810 |
| 3 | 0.500 | 0.096 | 0.002 | 0.223 | 0.729 |
| 4 | 0.360 | 0.121 | 0.000 | 0.185 | 0.656 |
| 5 | 0.263 | 0.109 | 0.000 | 0.185 | 0.590 |

*(The walk hit-prob plateaus at 0.185 for h=4 and h=5 — a branching-factor floor in this 50-chain sample, not a
bug: at depth the up-walk from these descendants converges onto the same near-root ancestors.)*

- **Transitive membership is real and continuous** — the LLM rates a grandchild ~0.60 a *subcategory* of the
  grandparent, decaying smoothly 0.77→0.26. Judging multi-hop pairs directly works (no math composition needed).
- **The LLM decay is a DISTINCT third signal** — it sits *between* the walk (0.29 @h2) and `p^h` (0.81 @h2), so
  it's neither, which is exactly what makes superposing it worthwhile.
- **`μ_rev ≈ 0` at every hop (0.007→0.000)** — the semantic **OUGHT** judge **agrees with the graph's structural
  `μ_rev=0`** and flatly refutes `1−p^h` (which wanted 0.41 @h5). On the reverse, IS and OUGHT *concur*; the
  interesting blend is entirely in the forward. (element_of is low/noisy — these are category-category pairs, so
  *subcategory* is the right relation; element contributes a low view.)

## (2) SUPERPOSITION — graph-walk ⊕ LLM, trained and eval'd vs walk-only
`emit_transitive_superposition.py`: `μ_fwd = w_g·hit_prob + w_e·llm_elem_fwd + w_s·llm_subcat_fwd`,
`μ_rev = w_g·0 + w_e·llm_elem_rev + w_s·llm_subcat_rev` (Dirichlet mix), judge=dir-blend. Trained from `model_prod`
(+ walk transitive + h=1 anchor). Eval on deep held chains (h=1..8):

| h | walk μ / dir% | superpos μ / dir% |
|---|---|---|
| 2 | 0.284 / 99% | 0.522 / 96% |
| 3 | 0.153 / 99% | 0.253 / 92% |
| 5 | 0.054 / 95% | 0.061 / 70% |
| *— beyond h=5: LLM OUGHT signal = 0 (extrapolation regime) —* | | |
| 8 | 0.028 / 81% | 0.018 / **36%\*** |

*\* h≥6 is OUT of the LLM's scoring horizon (h≤5) — the superposition's OUGHT component is extrapolated to zero
there, so the 36% is partly "no OUGHT signal," not purely the blend failing. Do not cite it as the blend's h=8
performance without that qualifier.*

- **The OUGHT raises forward magnitude** at mid-hops (0.52 vs 0.28 @h2) — the blend is more semantically calibrated
  (the LLM thinks membership is stronger than the branch-diluted walk does).
- **…but it costs deep-hop DIRECTION** — superpos falls to 36% @h8 (below chance) vs walk's 81%. Two causes: the
  LLM blend dilutes the clean structural signal the walk relies on for direction, and the **LLM was only scored to
  h=5** so there is no OUGHT signal at h≥6.

## Interpretation — the naive blend is a trade, not a free lunch
The IS/OUGHT split is even sharper than expected: the **graph walk (IS) owns *direction at depth*** (crisp,
root-converging, robust past where the LLM was scored), while the **LLM (OUGHT) owns *magnitude calibration***
(a semantically-truer "how much" of membership). A single Dirichlet-blended target mixes them and **loses the
graph's direction edge** to gain magnitude. So they shouldn't be flattened into one target naively.

**Next (deferred):** (a) score deeper (h≥6) so the OUGHT doesn't vanish at depth; (b) **hop-dependent weighting** —
lean graph for direction/at-depth, LLM for magnitude/near — instead of a uniform mix; (c) keep them as *separate
operators* the model selects contextually rather than a pre-blended target (which is closer to the original
"teach the model to separate the inputs" motivation). The clean win here is the **validation** (§1): the LLM
confirms continuous transitive membership and `μ_rev≈0`, grounding the graph model and refuting `1−p^h`.

Repro: `score_with_codex.py` on multi-hop pairs → `emit_transitive_superposition.py` → fine-tune → deep-hop eval.
Caveat: single-seed; 250 LLM-scored pairs (h≤5); walk-vs-superpos margins want multi-seed before quantitative use.
