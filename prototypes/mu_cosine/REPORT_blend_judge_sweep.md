# Does the constructed blend judge teach generality? — multi-seed test

*Tests the hypothesis (user, 2026-07-05): training on judge *superpositions* teaches generality, so base
held-out (never touched by the blend pairs) should rise. 3 arms × 3 seeds, fine-tuned from `model_prod` (800→600
steps, full recipe, `--pairs mu_pairs_scored_cumulative` for the base SYM held-out, `UW_MU_GRAPH=100k_cats`).
Metric: **base simplewiki SYM held-out corr** (1074 positives, disjoint from the Wikipedia blend pairs).*

## Arms
- **A `LLM only`** — graded round = the 880 Wikipedia pairs under `judge=gpt-5.5-low` (no constructed blend).
- **B `fixed-λ`** — A + 880 `judge=blend` SYM rows, `blend = 0.5·μ_e5_sym ⊕ 0.5·P(SYM|1/d,asym-ops)`.
- **C `random-λ`** — A + 880 `judge=blend` rows with `λ ~ U(0,1)` per pair (the blend regulariser).

## Result

| arm | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| A `LLM only` (control) | +0.792 | +0.788 | +0.789 | **+0.790** |
| B `fixed-λ blend` | +0.790 | +0.755 | +0.815 | +0.787 |
| C `random-λ blend` | +0.785 | +0.782 | **+0.092** | +0.553 |

*(model_prod, no fine-tune: +0.41. Within-seed blend−control: seed1 −0.002/−0.007, seed2 −0.033/−0.006, seed3
+0.026 / **−0.697 collapse**.)*

## Verdict — the generality-via-constructed-blend hypothesis is NOT supported

1. **The base-SYM lift is real and multi-seed-stable (+0.41 → ~+0.79) — but it is the DATA + fine-tune, not the
   blend.** The control (`A`, no constructed blend) matches the blend arms (+0.790 vs +0.787). Fine-tuning on the
   LLM Wikipedia round — which itself spans multiple judge tags (`gpt-5.5-low`, and the base pairs' `haiku`/
   `graph`) — is what generalises the base SYM.
2. **The constructed blend judge adds nothing over the LLM data** (B−A ≈ 0 across seeds).
3. **Random-λ is risky** — seed 3 collapsed to +0.09. Forcing a per-pair random blend can destabilise training.

## Takeaways
- **Multi-seed caught a wrong single-seed story:** the +0.781 first-run lift was real *as a lift* but *not*
  attributable to the superposition. This is exactly why single-seed deltas aren't believed.
- **Data/judge diversity may buy the generality for free** — you may not need a hand-constructed blend judge;
  training across several real judge tags already spans views.
- **If you do use a constructed blend, it must beat a no-blend control across seeds** before you credit it, and
  **random-λ needs a stability guard** (warmup, λ schedule, or a floor).
- Not overturned: the *estimator* win from earlier (joint head > PoE).

## ⚠ The +0.79 is likely an LLM-ALIGNMENT artifact, not a general SYM gain (user, 2026-07-05)

The base SYM held-out **eval targets are haiku-scored (LLM)**, and the fine-tune added **LLM data** (gpt-5.5-low)
+ more passes over the base haiku pairs. So "+0.41 → +0.79" measures the model getting better at **the LLM's
notion of SYM** — which is exactly what an LLM-scored eval rewards. This is a **train/eval-share-a-judge
confound**: more LLM training data → higher LLM-based eval, almost by construction (and why *all* arms rose
together). It does **not** establish a *general* SYM improvement. To claim that, evaluate against an
**independent, non-LLM** target — graph-structural relatedness on held-out pairs, a human check, or a downstream
task (retrieval / filing recall). Until then, read +0.79 as "better aligned to the LLM SYM judge," scoped, not
"better SYM."

Data: `gen_wiki_relation_pairs.py` → `score_with_codex.py` → `convert_scored_to_graded.py` + `emit_blend_judge.py`.
Sweep: `/tmp/mu_data/blend_sweep.sh`.

## ✅ On the RIGHT metric — predict the judge SUPERPOSITION on held-out — the blend DOES earn its keep

The LLM-scored SYM eval above was the wrong yardstick (it rewards LLM-alignment). The relevant measure (user):
**how well does the model predict the judge superposition `T = (1−λ)·e5_ref ⊕ λ·graph_ref` on 360 held-out pairs
never trained on** (`eval_blend_prediction.py`; `e5_ref` = raw-e5 cosine, `graph_ref` = conf-weighted
`1/d + μ_HIER/μ_ELEM` from model_prod — the graph half is *not* LLM, so no confound). corr(SYM readout, T):

| model | judge input | corr(SYM, T) | corr(SYM, graph_ref) |
|---|---|---|---|
| prod | agnostic | +0.746 | +0.714 |
| A (LLM-only) | blend | +0.675 | +0.641 |
| **B (blend)** | **blend** | **+0.847** | **+0.823** |

**Verdict flips on the right target:** B (blend-trained, read under `judge=blend`) predicts the held-out
superposition **best (+0.847)**; **LLM-only training *drifts away* from the graph half** (A graph_ref +0.641 <
prod +0.714 — the LLM-alignment confound made visible), while the **blend judge recovers+improves it** (B
graph_ref +0.823). And the **judge input matters** — B under `blend` +0.847 vs agnostic +0.787. So the blend
adds exactly the structural signal LLM-only training discards. *Caveats:* single-seed checkpoints (the 3×3 sweep
above didn't `--save`); `T` is graph-dominated here (`e5_ref` mean 0.907, low variance on this Wikipedia sample),
so "predict T" leans on the graph half — which is the point, but a held-out set with more e5 spread would
sharpen the e5 side.

### Multi-seed — CONFIRMED (checkpoints A/B × seeds 1-3, `--save`, `eval_blend_prediction.py`)

`corr(SYM readout, T)` under `judge=blend`, and the discriminating `graph_ref` half:

| seed | A (LLM-only) | B (blend) | B−A | A graph_ref | B graph_ref |
|---|---|---|---|---|---|
| 1 | +0.675 | **+0.847** | +0.172 | +0.641 | +0.823 |
| 2 | +0.753 | **+0.832** | +0.079 | +0.729 | +0.805 |
| 3 | +0.664 | **+0.845** | +0.181 | +0.632 | +0.817 |
| **mean** | **+0.697** | **+0.841** | **+0.144** | +0.667 | **+0.815** |

*(prod baseline +0.746 / graph_ref +0.714.)* **B beats A on every seed** (mean +0.841 vs +0.697, vs prod +0.746);
B robustly **recovers the graph half** (+0.815) that **LLM-only training drifts from** (+0.667); and reading
under `judge=blend` beats agnostic for B on all three seeds. So the blend-judge advantage on the *right* metric
is not a single-seed fluke — it's consistent. (Scope: `T` is graph-dominated on this Wikipedia held-out set, so
this confirms the *structural* half most strongly; a higher-e5-variance held-out would test the e5 side.)
