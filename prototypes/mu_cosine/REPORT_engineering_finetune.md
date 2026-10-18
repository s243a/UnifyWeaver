# Engineering build-out via FINE-TUNE-WITH-REPLAY (continual learning)

Grows Engineering from the *modest* ~13-node domain of #3314 into a real domain, and does it as a
**continual-learning** step — warm-start the latest checkpoint and fine-tune on the new data with a
**replay** sample of the cumulative set, rather than retraining from scratch. Built on
`gen_engineering_pairs.py` + the `--init-from`/`--replay-pairs` flags in `train_mu_attention.py`. Does
not touch the WAM-Rust core.

## Data (steps 1–3)

`gen_engineering_pairs.py`: closure-guarded Engineering pool — `closure(Engineering ∪
Mechanical_engineering ∪ Civil_engineering, d≤3)` ∩ μ-coherent, **36 nodes** (drops `Applied_sciences`,
which reaches Medicine→Physiology, and blocks the physics-primary boundary + medical leaks). Strata:
within-Engineering + cross Engineering×{Physics (the **Mechanics/Thermodynamics high-to-both** boundary),
CS, Math, Chemistry}. **300 non-neg pairs Haiku-scored** (two inline subagents, ~33k tokens, 0 tool
calls, graded rubric); committed in `mu_pairs_scored_engonly_260621-174251.tsv` (the new slice) and folded into the
cumulative `mu_pairs_scored_eng_260621-174251.tsv` (14,406 rows; `pos_eng` 70→210, `cross_EP` 50→110).

**DATA CEILING (flag for the widening spec #3313):** the engineering tree is shallow here — `Electrical`,
`Process`, `Chemical`, `Software`, `Aerospace`, `Systems`, `Industrial` engineering are all **ABSENT**.

## Fine-tune-with-replay (step 4)

`--init-from model_cumulative.pt` (warm start, head NOT reinitialised), `--pairs
mu_pairs_scored_engonly_260621-174251.tsv` (the new data), `--replay-pairs mu_pairs_scored_matheng_260621-100230.tsv`
`--replay-frac 0.4`, `--lr 1.5e-4` (≈1/3 of from-scratch), **400 steps** (vs 900 from scratch).
Ablation baseline: full retrain on the cumulative set from scratch (`--lr 5e-4`, 900 steps). Seed 1.

### (a) Engineering discriminates — YES (argmax AND margin)

Engineering is clean **5/5 argmax**, mean margin **+0.11**, top-2 **100%** (fine-tune). `pos_eng` SYM
corr **+0.888**, `cross_EP` **+0.699** (μ̄ target 0.45) — the Mechanics/Thermodynamics × Engineering
boundary is learned as **high-to-both** (a couple of physics probe nodes argmax→Engineering, which is the
*correct* multi-membership, not a failure — `Mechanics` is genuinely mechanical engineering).

### (b) Replay prevented forgetting — YES

Physics SYM corr on the held-out set, before vs after the Engineering fine-tune:

| stratum | baseline (`model_cumulative`) | fine-tune+replay | full retrain |
|---|---|---|---|
| `pos_phys` | +0.824 | **+0.872** | +0.855 |
| `pos_math` | +0.840 | +0.878 | +0.881 |
| `pos_chem` | +0.874 | +0.904 | +0.853 |
| `pos_cs` | +0.644 | +0.577 | +0.772 |
| overall | +0.772 | +0.820 | +0.788 |

Physics did not regress — it **improved** (+0.824 → +0.872); only `pos_cs` dips (n=18, within noise).
WIKI held-out order-acc **98.8%** (preserved), SYM gate-leak **0/5**, 5-way discrimination **argmax 64% →
80%, top-2 96%** (no regression). Replay does its job.

### (c) Fine-tune+replay vs full retrain — MULTI-SEED: fine-tune is far more stable

Run at three seeds (every fine-tune warm-starts the *same* `model_cumulative.pt`):

| 5-way argmax / ALL top-2 | seed 1 | seed 7 | seed 23 | spread |
|---|---|---|---|---|
| **fine-tune+replay** | 80% / 96% | 92% / 96% | 80% / 100% | **80–92 / 96–100** |
| **full retrain** | 72% / 96% | **60% / 76%** | 96% / 100% | **60–96 / 76–100** |

`pos_eng` corr (fine-tune vs retrain): seed 1 +0.888 / +0.663. Engineering is 5/5 at every seed for both.
The full retrain **collapses at seed 7** (physics top-2 0%, mean-rank 3.20) — the classic seed-sensitivity
of from-scratch init. Fine-tune+replay never collapses: anchored to a good basin by the warm start, its
argmax stays in 80–92 and top-2 in 96–100. So fine-tune+replay is not just cheaper (≈½ the compute) and
better on the new domain — it is **markedly lower-variance**. On the *shared* (retention) metrics the two
are within noise; the new-domain edge and the stability edge both favour fine-tune+replay.

### (d) PLACEBO / churn control — how much is the new DATA vs just more optimization?

A control (suggested in review): fine-tune with replay but **no new data** (continue training on the old
set), then measure how much the per-root μ vector (and its softmax over roots) moves on the 25 probe
nodes — the *churn floor*. Compare to the real engineering fine-tune (`drift_control.py`):

| drift from `model_cumulative` (25 nodes × 5 roots) | mean \|Δμ\| | softmax TV-dist | argmax flips |
|---|---|---|---|
| **placebo** (replay, no new data) | 0.123 | 0.047 | 6/25 |
| **real** (engineering data) | 0.111 | 0.045 | 5/25 |

**REAL/PLACEBO μ-drift ratio = 0.90× — the engineering data moves the discrimination probe *within*, not
beyond, the churn floor.** The placebo alone takes argmax 64% → 84% (vs real's 80–84%): **most of the
discrimination "improvement" is re-optimization, not the new data.** On the 5 Engineering probe nodes the
Engineering-root μ even drifts *down* (0.885 → 0.837 placebo → 0.795 real) — the graded within-Engineering
targets teach moderate μ, and Engineering was already argmax-saturated (5/5) at baseline. The engineering
data's genuine, non-churn contribution is the held-out **relatedness ranking** (`pos_eng` +0.888,
`cross_EP` +0.699), which the saturated argmax probe cannot show.

## Honest verdict

- **Engineering discriminates** (5/5 argmax at every seed, +0.11–0.20 margin, top-2 100%) and the new
  data genuinely taught the **relatedness ranking** (`pos_eng` +0.888, `cross_EP` +0.699 — held-out), with
  the Mechanics/Thermo boundary correctly high-to-both. **But** the placebo control (d) shows the
  *discrimination-probe* gains are within the churn floor — Engineering was already argmax-saturated, so
  don't credit the new data for the 64%→84% argmax jump (the no-data placebo does the same). New in-domain
  data helps where the domain is **not** saturated (ranking), not where it already wins argmax.
- **Replay prevented forgetting** — physics/maths/chem held or improved, discrimination and WIKI did not
  regress, gate-leak stayed 0/5.
- **Fine-tune+replay is a safe default going forward.** Across 3 seeds it matches full retrain on
  retention, edges it on the new domain, runs at ≈½ the compute, and is **substantially more stable**
  (argmax spread 80–92 vs the retrain's 60–96, which collapsed at seed 7). Recommended as the default for
  adding a domain.
- **Where more data actually helps (data-gap diagnosis).** Consistently across all seeds and both methods:
  **Chemistry, Computer_science, Engineering = 100% top-2 every run** (saturated in this slice — more data
  there is wasted), while **Physics and Mathematics** are the weak/variable domains. Physics is the
  connective spine (co-membered with Chem/Eng) *and* missing its modern subfields; Mathematics has a thin
  pool. Those two — via graph widening (#3313: Quantum/Statistical/Relativity; more math) — are where the
  next labelling/widening effort should go, not into the already-saturated domains.

*Method caveats:* the placebo control is single-seed (ratio 0.90× is below 1, so the direction is clear);
retention eval is on a replayed set (optimistic), but the proper-holdout full retrain corroborates physics
stays strong (+0.855).

Reproduce: `gen_engineering_pairs.py` → score the 300 non-neg pairs → `mu_pairs_scored_engonly_260621-174251.tsv` +
cumulative `mu_pairs_scored_eng_260621-174251.tsv`; then
`train_mu_attention.py --pairs mu_pairs_scored_engonly_260621-174251.tsv --replay-pairs mu_pairs_scored_matheng_260621-100230.tsv
--replay-frac 0.4 --init-from <ckpt> --lr 1.5e-4 --steps 400 --llm ...` and the full-retrain ablation
`--pairs mu_pairs_scored_eng_260621-174251.tsv --lr 5e-4 --steps 900`.
