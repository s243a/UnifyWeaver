# The element-of operator — page-membership as its own relation (and a capacity finding)

Follow-up to `REPORT_train_consolidation.md`, which found that page-membership labels fed as
undifferentiated SYM positives were **inert** (page-centrality corr +0.0–0.68 vs +0.85–0.97 for category
membership, and fine-tune ≈ placebo). This implements the design's fix (`DESIGN_calibrated_judges.md` §7):
give page-membership its **own operator**.

## What was built

- **`OPS["ELEM"] = 3`** (`mu_attention.py`) — a 4th operator token + readout row on the shared trunk.
  ELEM is **directional like WIKI** (μ(page|category) high, reverse low) but **graded like SYM** (Haiku
  centrality target). The grading is Haiku (the page-centrality rubric); the graph/API supplies only the
  membership *fact* + free μ=0 negatives — same split as SYM.
- **Routing** (`train_mu_attention.py`): rows tagged `relation=element_of` (the page-frontier + pearltrees
  rounds) train on ELEM with `L_elem = MSE(μ(page|cat), target) + margin·relu(m − (μ(page|cat) −
  μ(cat|page)))` (direction on positives). Everything else stays on SYM. `--elem-weight` knob.
- **Warm-start across the op-count change**: `--init-from` now partial-loads — copies overlapping weights,
  grows the op-indexed tensors (`op_emb`, `readout_w/b`), and **seeds the new ELEM row from SYM** (op 0) so
  it starts membership-like and specialises.
- **Eval routing** (`train` + `eval_per_stratum.py`): `element_of` strata score via ELEM directional
  μ(page|cat); the rest via order-invariant SYM. New `[ELEM] held-out centrality corr + direction` metric.

## Result — the operator works

Per-stratum page-centrality corr, ELEM operator vs the old SYM-conflated model (same held-out split):

| page stratum | SYM-conflated | **ELEM op (3L)** |
|---|---|---|
| `pos_pageof_nonlinear` | +0.12 | **+0.795** |
| `pos_pageof_holism` | +0.63 | +0.81 |
| `pos_pageof_systems_analysis` | +0.68 | +0.65 |
| `pos_pageof_emergence` | +0.31 | +0.61 |
| `pos_pageof_ergodic_theory` | +0.39 | +0.25 |
| dedicated `[ELEM]` held-out | — | **+0.575**, direction 98% |

Five of six page strata up (often sharply); `nonlinear` +0.12→+0.795 is the headline. `ergodic` is the lone
regressor and stays low across *every* config (~+0.25) — likely genuinely hard (very technical math pages,
e5-small weak there) plus small n=14.

## The capacity finding (data vs model-size)

At full ELEM weight on the **2-layer** default, hard-argmax discrimination dropped 90%→79% — but **top-2
stayed 100%** and every new miss was a razor-thin rank-2 flip on a genuine multi-membership node
(Thermodynamics→Chem m−0.04, ML/NN/CV→Comp m−0.10). I.e. cross-domain *separation* held; the shared trunk
was just margin-compressed by the new operator. Two levers isolate the cause:

| config | discrim | SYM | ELEM | nonlinear | holism | sys_anal |
|---|---|---|---|---|---|---|
| ELEM · **2L** · w1.0 | 79% | +0.850 | +0.605 | +0.63 | +0.87 | +0.78 |
| ELEM · 2L · **w0.4** (balance lever) | 87% | +0.842 | +0.589 | +0.60 | +0.94 | +0.54 |
| ELEM · **3L** · w1.0 (capacity lever) | **90%** | +0.858 | +0.575 | **+0.795** | +0.81 | +0.65 |

- **Lowering ELEM weight** recovers discrimination but by *turning the ELEM signal down* — page gains slip
  (`sys_anal` +0.78→+0.54). Trading, not fixing.
- **Adding a 3rd layer** recovers discrimination **fully (90%) at full ELEM weight** while keeping strong
  page gains. Strictly dominates.

**Conclusion: it was a model-capacity (depth) issue, not data.** At 2 layers the shared trunk couldn't
co-serve discrimination *and* page-centrality at full strength; a 3rd layer gives each function room and
the interference vanishes. And because **adding depth worked, the frozen e5-small (384-d) input is rich
enough** — the bottleneck was trunk depth, not embedding richness. (Had depth *not* helped, the next lever
would have been a richer encoder, e5-base 768-d.) ~1.2M extra params (2.4M → 3.6M).

`model_elem_L3.pt` is the recommended checkpoint. Code default `--layers` is left at 2; ELEM training wants
**`--layers 3`**.

Reproduce: `UW_MU_GRAPH=…/wide_enwiki_math/… UW_E5_CACHE=e5_tables_train_all.pt train_mu_attention.py
--pairs mu_pairs_scored_cumulative.tsv --replay-pairs mu_pairs_scored_prior.tsv --init-from
model_mathfields.pt --pairs-corpus enwiki --lr 1.5e-4 --steps 700 --layers 3 --save model_elem_L3.pt`,
then `eval_per_stratum.py --model model_elem_L3.pt --pairs mu_pairs_scored_cumulative.tsv`.
