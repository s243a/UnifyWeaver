# Model selection under regularization — why AIC inverts, what maps, and how we measure it instead

Background notes for the capacity/regularization work (`DESIGN_inferred_operator_superposition.md` §7, the
tagged-blend sweep, the 3L-vs-4L comparison). Captured mainly so the **right theory has a pointer** — see the
reading list at the end. The question that prompted it: *"when we hit underfitting, is the answer less noise
or more layers — could something like AIC tell us?"*

## Why AIC inverts the logic in our regime

**AIC** `= 2k − 2·ln L̂` (Akaike 1974): penalise the parameter count `k` to trade fit against complexity.
Two of its assumptions break for a regularised neural net:

1. **`k` (raw parameter count) is the wrong complexity measure for over-parameterised nets.** AIC's
   derivation assumes the asymptotic regime `n ≫ k` and that `k` counts *effective* degrees of freedom. We
   are in `k ≫ n` (millions of params, thousands of examples), where the `2k` penalty is astronomical and
   meaningless — yet the model generalises. The *effective* complexity is far below `k`.
2. **Regularisation changes the effective DoF, not `k`.** Dropout / noise / the tagged-blend leave the
   parameter count identical but shrink effective capacity. AIC is blind to this — it cannot distinguish
   "3 layers + heavy blend" from "3 layers + no blend" (same `k`). So it literally cannot answer the
   noise-vs-layers question, which is *entirely* about effective complexity.

And the deeper inversion, specific to the capacity-as-headroom framing: **AIC says more parameters = more
complexity = penalise.** But in the regularisation-bounded regime, adding a layer can *decrease* effective
complexity — it is *room to spread the same information more softly*. A 4th layer's value here is not extra
fitting power; it is extra **noise-absorbing headroom** (cf. the 3L/0.7 → 4L/0.7 recovery from underfitting
in `REPORT_capacity_conf_tradeoff.md`). AIC has the sign backwards for exactly that effect.

## What *does* map — Watanabe's Singular Learning Theory (WAIC / WBIC)

The principled analog of AIC/BIC for neural nets is **Sumio Watanabe's Singular Learning Theory (SLT)**. It
exists precisely because a neural net is a **singular** statistical model: its Fisher information matrix is
*degenerate* (parameters are non-identifiable — many settings give the same function), so the regular-model
asymptotics behind AIC/BIC (which assume a positive-definite Fisher info) simply do not hold, and `k` is the
wrong object.

- In SLT, the role played by `k/2` in the asymptotic free energy is taken over by the **Real Log Canonical
  Threshold (RLCT)**, a.k.a. the **learning coefficient `λ`** — a birational invariant from algebraic
  geometry that is the *effective* parameter count. Crucially, **`λ ≤ k/2`, with equality only for regular
  models**, and regularisation/structure is exactly what *lowers* `λ`. That is the theoretically-correct
  version of "regularisation shrinks effective complexity."
- **WAIC** (Widely Applicable / Watanabe–Akaike Information Criterion) is the singular-model generalisation
  of AIC — asymptotically equal to Bayes leave-one-out cross-validation, valid for singular models.
- **WBIC** (Widely Applicable Bayesian Information Criterion) is the singular generalisation of BIC —
  estimates the Bayes free energy / marginal likelihood with the RLCT in place of `k/2`, computable from a
  *single* MCMC run at inverse temperature `β = 1/ln n`.

So the intuition "we can only regularise to the degree capacity allows, and a layer buys headroom" is, in
SLT terms, a statement about how architecture + regularisation move the learning coefficient `λ`.

## But practically — we measure it directly

Estimating `λ` for a real net is itself hard (an active research area). We don't need it: **the train-vs-
held-out gap as a function of regularisation strength *is* the empirical fit-vs-effective-complexity curve.**
That is what the tagged-blend sweep traces — the inverted-U in `REPORT_tagged_blend_sweep.md` is the measured
version of the AIC trade-off, with the underfitting onset marking the capacity ceiling. We replaced the
analytic complexity penalty (which AIC gets wrong for nets) with a *measurement*. Related practical anchors:

- **The one-standard-error rule** (Breiman et al. 1984; Hastie–Tibshirani–Friedman, *ESL*): pick the
  *most-regularised* model within ~1 SE of the best CV score, not the argmax — the parsimony choice that
  motivated preferring `c≈0.7` over the noisy `c=0.85` peak.
- **Double descent** (Belkin et al. 2019; Nakkiran et al. 2019): test error vs capacity is *non-monotonic*
  past the interpolation threshold — concrete evidence that raw parameter count is not a monotone complexity
  penalty, exactly AIC's failure.

## Reading list (the point of this note)

Singular Learning Theory / WAIC / WBIC:
- Sumio Watanabe, *Algebraic Geometry and Statistical Learning Theory*, Cambridge Univ. Press, 2009 — the
  foundational text (defines singular models, the RLCT/learning coefficient, the free-energy asymptotics).
- Watanabe (2010), "Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information
  Criterion in Singular Learning Theory," *JMLR* 11 — **WAIC**.
- Watanabe (2013), "A Widely Applicable Bayesian Information Criterion," *JMLR* 14 — **WBIC**.
- Sumio Watanabe, *Mathematical Theory of Bayesian Statistics*, CRC Press, 2018 — a more accessible
  book-length treatment.
- Practical Bayesian use of WAIC: Vehtari, Gelman & Gabry (2017), "Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC," *Statistics and Computing* 27 (the `loo`/WAIC workflow).
- Accessible ML-oriented entry points: Liam Carroll's "Distilling Singular Learning Theory" online series,
  and the Timaeus / "developmental interpretability" group's writing — these connect RLCT estimation to deep
  nets (search those terms; the field is moving, so prefer recent surveys).

Classical criteria (for contrast):
- Akaike (1974), "A new look at the statistical model identification," *IEEE TAC* 19 — **AIC**.
- Schwarz (1978), "Estimating the dimension of a model," *Annals of Statistics* 6 — **BIC**.

Effective-complexity / over-parameterised regime:
- Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine-learning practice and the bias–variance
  trade-off," *PNAS* — double descent.
- Nakkiran et al. (2019/2021), "Deep Double Descent: Where Bigger Models and More Data Hurt."
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2009) — the 1-SE rule, AIC/BIC,
  effective degrees of freedom.

> One-line takeaway: AIC penalises `k`; for a singular model the right object is the **learning coefficient
> `λ` (RLCT)** that regularisation actually moves (Watanabe's WAIC/WBIC) — but we sidestep estimating `λ` and
> read the **train-vs-held-out gap vs regularisation strength** directly.
