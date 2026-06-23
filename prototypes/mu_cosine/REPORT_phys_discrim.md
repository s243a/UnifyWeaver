# Core-physics discrimination — honest re-measure + a targeted sampling run

Follow-up to #3314's finding that *physics argmax is brittle while its ranking stays strongest*. Two
steps: **(0)** re-measure discrimination with a ranking/margin metric (no budget); **(1–2)** a small
targeted core-physics sampling run (budget-bounded, user-approved). Builds on
`gen_multidomain_pairs.py` / `gen_math_eng_pairs.py` / `mu_attention.py`; does not touch the WAM-Rust core.

## Step 0 — re-measure with a ranking/margin metric (no budget)

`discrimination_probe` now reports, alongside hard argmax: the true root's **rank** among the 5, the
**signed margin** μ(true) − max-other (>0 ⇔ argmax-correct), and **top-1/top-2** rates. Re-run on the
merged math-eng set (`mu_pairs_scored_matheng_260621-100230.tsv`) across the same 3 seeds as #3314:

| seed | argmax acc | Phys mean-rank | Phys margin | Phys top-2 | ALL mean-rank | ALL top-2 | SYM corr |
|---|---|---|---|---|---|---|---|
| 1 | 64% | 2.00 | −0.02 | 80% | 1.44 | 92% | +0.736 |
| 7 | 92% | 1.60 | +0.01 | 80% | 1.12 | 96% | +0.800 |
| 23 | 88% | 1.20 | +0.08 | 100% | 1.12 | 100% | +0.769 |

**The hard argmax swings 64–92%, but the ranking is robustly strong** — overall mean-rank 1.12–1.44,
top-2 92–100%. Physics specifically sits right at the #1/#2 boundary (mean margin −0.02…+0.08, top-2
80–100%); when it "loses" argmax it loses by **0.01–0.07** to a genuine co-member (Thermodynamics↔Chem,
Mechanics↔Engineering, Optics↔Engineering). So **#3314's "brittleness" is largely a metric artifact —
correct multi-membership of the connective spine domain**, not a modelling failure. (Math behaves the
same way: `Calculus`/`Differential_equations` are the maths *of* physics/engineering.)

## Step 1 — targeted sampling (`gen_core_physics_pairs.py`), and the data ceiling

Bidirectional-coinflip walks seeded at the **classical** physics subfields present in the slice —
{Electromagnetism, Classical_mechanics, Mechanics, Optics, Thermodynamics, Acoustics} — collecting a
μ-coherent core. **Sampling is itself ceiling-limited**, two ways:

1. **e5's "Physics" baseline cosine is flat and high** (`Popular_culture`=0.82, `Music`=0.84), so
   argmax-Physics alone admits junk once a bidir walk drifts up through the apex hubs and back down.
   Clean membership requires the depth-bounded **downward closure** (the #3312 guard) + a strict margin
   + a small documented blocklist of recurring philosophy/arts leaks. Even then the clean classical core
   is only **~23 nodes**, many of them ambiguous (`Vision`, `Physical_objects`, `Energy_in_transport`,
   `Physicists_by_nationality`).
2. **The modern subfields are ABSENT from the graph** — `Quantum_mechanics`, `Statistical_mechanics`,
   `Fluid_dynamics`, `Mathematical_physics`, `Relativity` (and QFT / condensed-matter) are all missing.
   This is the data-ceiling the widening spec (#3313) targets.

**152 non-neg pairs** scored (one inline Haiku subagent, ~15.6k tokens, 0 tool calls, graded rubric):
80 within-core-physics + 24 each core×{Math, Chemistry, Engineering}. Committed in
`mu_pairs_scored_corephys_260621-125413.tsv`. Haiku honestly graded the ambiguous nodes low (e.g. `Vision`×`Sound`
0.15 — "Vision" is perception, not acoustics), so the within-physics stratum mean μ is only ~0.50.

## Step 2 — retrain + verdict: it did NOT help (slightly regressed)

`train_mu_attention.py --pairs mu_pairs_scored_corephys_260621-125413.tsv` (same 3 seeds):

| seed | argmax acc | Phys top-2 | ALL top-2 | SYM corr |
|---|---|---|---|---|
| 1 | 68% | 60% | 92% | +0.762 |
| 7 | 64% | 80% | 84% | +0.660 |
| 23 | 80% | 80% | 96% | +0.716 |
| **mean** | **71%** (matheng 81%) | **73%** (matheng 87%) | **91%** (matheng 96%) | **+0.713** (matheng +0.768) |

Per-stratum (seed 1): **`pos_phys` +0.823** (held vs the +0.838 target), `pos_corephys` +0.811,
`pos_eng` +0.830, `pos_math` +0.727, overall +0.774; WIKI order-acc 98.4%; SYM gate-leak 0/5. So
**physics SYM holds and no domain regresses** — but **discrimination did not improve by either metric**;
on average it slightly *regressed*. The sharper-but-ambiguous classical pool taught the Physics anchor
*lower* μ on its own borderline nodes (Vision/Physical_objects scored low), while the core×Engineering
cross pairs nudged the Engineering anchor *up* — so Engineering, not Physics, gained pull.

## Honest verdict

**Physics brittleness is mostly structural (it is the connective spine — high-μ to several roots, so
argmax is ill-posed where ranking is correct) + a missing-subfield (graph) problem — NOT a
data-sharpness problem solvable in this slice.** Step 0 already showed the ranking is strong (top-2
92–100%); step 2 confirmed that adding sharper *classical* data — the only physics this graph has — does
not move discrimination, because the clean core is small and ambiguous and the modern subfields that
would actually separate physics are absent. **The real lever is graph widening (#3313)**, to bring in
Quantum/Statistical/Fluid/Relativity; the margin metric added here is the right yardstick to re-measure
against once that data exists.

Reproduce: `gen_core_physics_pairs.py` → score the 152 non-neg pairs → merge into
`mu_pairs_scored_corephys_260621-125413.tsv`; `train_mu_attention.py --pairs mu_pairs_scored_{matheng,corephys}.tsv
--llm --steps 900 ... --seed {1,7,23} --quick-val` (the `[DISCRIM]` block prints both metrics).
