# Multi-domain data — more Physics + a Chemistry probe (#3305 follow-up)

Two goals: (1) **more Physics** within-domain pairs to keep closing the SYM gap (#3305 reached +0.479
vs the MiniLM control's +0.726); (2) a **modest Chemistry** set + a **cross-domain** stratum to test
whether the model can *discriminate* Physics from Chemistry (μ(node|Physics) vs μ(node|Chemistry)).

## Data generation (the child-only sampler + μ-coherent pools)

- **`--child-only` downward walks** added to the sampler (`gen_more_sym_pairs.py`): a children-only
  adjacency so walks descend instead of wandering up across domains. But downward walks alone still
  drift (the Wikipedia DAG nests `Physics → Cosmology → Creation_myths → Hinduism`), and raw e5 cosine
  to the word "Physics" barely separates (`Politics` 0.84 ≈ `Thermodynamics` 0.88). So
  `gen_multidomain_pairs.py` builds **clean domain pools = graph downward-closure ∩ μ-coherence**
  (calibrated e5 μ for Physics; argmax chem-vs-phys for Chemistry) — the lightweight version of #3307's
  bridge-guided "μ-coherent neighbourhood" seeding. Physics pool 44, Chemistry pool 47, overlap 0.
- **650 new pairs Haiku-scored** (graded sameness; within-domain high, physics×chemistry moderate,
  unrelated ~0) under the budget discipline: **one inline subagent, ~26k Haiku tokens, 0 tool calls**.
  Strata means: pos_phys 0.36, pos_chem 0.67, cross 0.29. Committed in `mu_pairs_scored_multidomain.tsv`
  (= the 800-pos large set + 350 pos_phys + 150 pos_chem + 150 cross + 2600 free negatives).

## Retrain (multi-task, keep WIKI/LLM)

`train_mu_attention.py --pairs mu_pairs_scored_multidomain.tsv --llm --steps 900` (same recipe as #3305).

### (a) SYM held-out corr — did more Physics close the gap further? **Yes.**

| run | data | SYM held-out corr | WIKI order-acc |
|---|---|---|---|
| control (#3287 MiniLM symmetric) | — | +0.726 | 50% (structural) |
| #3302 multi-task | 200 pos | +0.335 | 99.1% |
| #3305 multi-task | 800 pos | +0.479 | 97.7% |
| **this — multi-domain** | **1450 pos** | **+0.822** (overall) | **99.7%** |

The overall +0.822 *exceeds* the control, but the held-out is now multi-domain, so it is **not** a pure
"more Physics" measure — chemistry pairs are tightly clustered (easy). The honest, apples-to-apples read
is the **per-stratum** held-out corr (`eval_per_stratum.py`):

| held-out stratum | n | corr | μ̄ target |
|---|---|---|---|
| **`pos`** (same distribution as #3305) | 156 | **+0.695** | 0.68 |
| `pos_phys` (tight physics pool) | 63 | +0.873 | 0.37 |
| `pos_chem` (tight chemistry pool) | 36 | +0.913 | 0.66 |
| `cross` (physics×chemistry) | 35 | +0.882 | 0.23 |

On the **comparable `pos` stratum**, more data raised corr **+0.479 → +0.695** — the gap to the control
shrank from 0.247 to **0.031** (nearly closed). The easy chem/cross strata lift the overall to +0.822.

### (b) Physics-vs-Chemistry discrimination — **10/10 correct**

μ(node|Physics) vs μ(node|Chemistry) via the SYM operator:

| class | node | μ\|Physics | μ\|Chemistry | ✓ |
|---|---|---|---|---|
| physics | Thermodynamics | 0.91 | 0.85 | P>C ✓ |
| physics | Optics | 0.88 | 0.73 | P>C ✓ |
| physics | Mechanics | 0.87 | 0.63 | P>C ✓ |
| physics | Electromagnetism | 0.92 | 0.76 | P>C ✓ |
| physics | Motion_(physics) | 0.91 | 0.69 | P>C ✓ |
| chemistry | Periodic_table | 0.88 | 0.95 | C>P ✓ |
| chemistry | Acids | 0.57 | 0.92 | C>P ✓ |
| chemistry | Chemical_compounds | 0.78 | 0.94 | C>P ✓ |
| chemistry | Oxygen | 0.72 | 0.86 | C>P ✓ |
| chemistry | Chemical_reactions | 0.69 | 0.90 | C>P ✓ |
| borderline | Atoms | 0.78 | 0.84 | ~ (chem-leaning) |
| borderline | States_of_matter | 0.78 | 0.81 | ~ (balanced) |
| borderline | Materials | 0.73 | 0.51 | ~ (phys-leaning) |
| borderline | Energy | 0.91 | 0.77 | ~ (phys-leaning) |
| borderline | Chemical_elements | 0.78 | 0.94 | ~ (chem-leaning) |

**Every clear-physics node scores higher to Physics, every clear-chemistry node higher to Chemistry**,
and borderline nodes are sensibly graded (Atoms/Chemical_elements lean chemistry, Energy/Materials lean
physics, States_of_matter balanced). The cross-domain stratum taught the model to separate the domains.

### Other metrics (all clean)
- gate-leak 5-probe **0/5** on every operator; OOD leak SYM 2.3% / WIKI 0.2% / LLM 0.1% (control 1.1%).
- Every operator's dense map feeds `check_feeds_rust` (100% resolution, IC general→specific).
- SECONDARY node-gated lin-agreement: **+0.183** (non-sat +0.181) — up from #3305's +0.070 and now
  **above** the control's +0.124.

## Honest verdict

- **More Physics data closed the SYM gap** on the apples-to-apples `pos` stratum (+0.479 → +0.695,
  control +0.726; gap 0.247 → 0.031). Not yet saturated — more positives would likely close the last
  0.03. The headline +0.822 is real but partly reflects the easier multi-domain held-out (chemistry
  pairs cluster tightly), so it is reported alongside the per-stratum breakdown rather than on its own.
- **Discrimination works: 10/10.** A modest Chemistry set + 150 cross-domain pairs were enough to make
  the *same* model score nodes correctly against *either* root — Physics-vs-Chemistry is now directional
  and graded, with no new architecture.
- The directional WIKI capability is untouched (**99.7%**), gate-leak stays clean, and the structural
  lin-agreement improved past the control. The child-only + μ-coherent-pool sampler (the #3307 idea,
  approximated by the e5 prior) was the enabler for clean in-domain and cross-domain strata.
