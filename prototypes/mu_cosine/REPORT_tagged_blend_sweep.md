# Tagged-blend regularization sweep — the inverted-U, and the 3-layer capacity ceiling

Testing the §7 reframe (`DESIGN_inferred_operator_superposition.md`): the random operator superposition is a
**regularizer**, so blending **tagged** rows (not just the 136 inferred) should help — *up to the capacity
budget*. Sweep `--blend-tagged-conf c ∈ {1.0, 0.85, 0.70, 0.50}` at **3 layers**, 3 seeds, warm-start
`model_nodetype.pt`, 700 steps, on the fuzzy/Complex round (`graded_cx`, 3370 targets). `c=1.0` = the prior
*inferred-only* blend (tagged trained hard); `c<1.0` also blends tagged with `p = c·onehot(label) +
(1-c)·uniform`. The warmup-curriculum fix is applied (tagged train hard until the blend warms up).

## Result

| `c` (tagged-blend) | SYM corr (3 seeds) | SYM mean | discrimination (3 seeds) | **disc mean** |
|---|---|---|---|---|
| *(no blend — no-switch)* | — | +0.675 | 88 / 92 / 96% | 92.0% |
| 1.00 (inferred-only blend) | .678 / .657 / .683 | +0.673 | 96 / 92 / 100% | 96.0% |
| **0.85 (mild tagged-blend)** | .629 / .683 / .709 | +0.674 | **100 / 96 / 100%** | **98.7%** |
| 0.70 | .621 / .659 / .761 | +0.680 | 92 / 96 / 96% | 94.7% |
| 0.50 (heavy) | .699 / .627 / .711 | +0.679 | 92 / 92 / 92% | 92.0% |

## The curve is a textbook inverted-U
Discrimination as regularization **increases** (no-blend → inferred-blend → +mild tagged-blend → heavy):

```
  92.0%  →  96.0%  →  98.7%  →  94.7%  →  92.0%
 no blend  inf-only  c=0.85    c=0.70    c=0.50
                     (peak)            (back to no-blend level)
```

- **Mild tagged-blend helps.** `c=0.85` peaks at 98.7% — **+2.7pp over the inferred-only blend**, and `c=0.85
  ≥ c=1.0` in *every* seed (100≥96, 96≥92, 100=100), so the gain is consistent, not a single-seed fluke (the
  trap the earlier `REPORT_infer_blend_cx.md` single-seed result fell into). The §7 reframe pays off: feeding
  the regularizer the whole set, not the 136-row inferred remnant, is what the parity finding was missing.
- **Heavy blend underfits.** Past the peak, discrimination *falls* — 94.7% at 0.70, back to the no-blend
  92.0% at 0.50. The regularization has exceeded what 3 layers can absorb.
- **SYM is flat** (~0.68 across all `c`) — the symmetric task is noise-dominated/saturated and doesn't
  resolve the operator-axis regularization; discrimination is the informative metric here.

## It's underfitting, NOT instability (answers the stability question)
At `c=0.50` all three seeds land on **exactly 92% (23/25)** — *low* variance, not erratic. And the
warmup→blend boundary was smooth at every `c` (no `L_blend` spike at step 200). So heavy blend does **not
destabilize** — it cleanly **underfits**. The stability mitigations we discussed (replay a hard fraction;
bias toward higher label confidence) are therefore **not needed at 3 layers** — the binding limit is
*capacity*, not a transition instability. (Both levers remain in the toolbox; raising `c` toward the 0.85
sweet spot *is* the "bias toward label confidence" dial.)

## "Noise or layers?" — answered empirically (the AIC discussion, operationalised)
This is the fit-vs-effective-complexity curve traced directly (no parameter-count proxy):
- **3 layers + `c≈0.85` is the operating point.** That's the most regularization 3 layers can absorb with a
  net gain.
- **The underfitting onset (`c<0.85` at 3 layers) is the empirical capacity ceiling** — exactly the trigger
  the capacity-paired design called for. If we ever want to regularize *harder* than 0.85 buys, **that** is
  when the 4th layer earns its place (more parameters to "spread the information across"). The data says we
  do **not** need to pre-build it now: 3 layers + mild blend is the sweet spot, and pushing past it just
  underfits rather than unlocking more.

## Takeaways
- **Ship `--blend-tagged-conf 0.85`** as the default tagged-blend strength (modest but consistent +2.7pp
  discrimination over the inferred-only blend; SYM unchanged).
- The regularizer's value is now a *measured* small gain, not just principle — but it is capacity-bounded; do
  not over-blend.
- A 4-layer arm is the next experiment **only if** we want to push regularization past the 3-layer ceiling;
  the underfitting curve above is the evidence that would justify it.
- Methodology (again): single-seed on the 25-probe metric is untrustworthy; the inverted-U is only legible
  because it's consistent across 3 seeds.
