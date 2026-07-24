# Process-Expression Language — Philosophy

## Why declare the data-generating process

Every training target is the output of some function: a judge, a fusion, a routing policy, a decay
schedule. The model currently learns per-process calibrations through FLAT tokens (judge_emb,
corpus_emb, op_emb) — and the roster already smuggles in compositions as opaque names:
"kalman-fused" IS kalman(D,S); "blend" IS blend(j₁,j₂); "dir-blend" IS a 3-estimator
superposition. Flattening throws away structure the model could use:

- **Transfer:** `routing(e5, sonnet, t=0.02)` and `routing(e5, sonnet, t=0.03)` are near-identical
  processes; flat tokens make them strangers, expressions make them neighbors.
- **Zero-shot conditioning:** an unseen composition (swap haiku→sonnet, add lineage context) gets
  a meaningful embedding by construction instead of a cold token.
- **Honest provenance:** mixing corpora produced by different pipelines without declaring the
  pipeline invites silent distribution confusion — the expression is the antidote, machine-readable.

This extends the name-function principle (NameFunctionCond: "the name is a function pointer, the
card is its documentation") from atoms to programs.

## The information objective

The language must balance CONCISENESS against SPECIFICITY. Operationalized: an expression
distinction earns its tokens iff it buys held-out predictive gain —

    the GAIN-PER-TOKEN EFFICIENCY CURVE:  Δ(held-out NLL)  per  expression token

(Naming amended per review #3974-r1 finding 8: this is NOT MDL — a true MDL objective would also
charge for the grammar/registry/model code length and require a pinned tokenizer so "token" is
well-defined. We adopt the efficiency-curve framing, pin the tokenizer (the e5 tokenizer at a
recorded revision) so lengths are comparable, and leave a genuine two-part code-length objective
to the Codex/theory lane if the curve proves insufficient.) Too coarse (all judges = "llm")
discards distinctions that change the label distribution; too fine (temperature, prompt hash,
date on every row) adds description length without gain and fragments the data into unlearnable
slivers. The optimal grammar level is an EMPIRICAL quantity — swept on an INNER validation split,
never on the set later used for reported inference. Control: shuffled/mismatched cards must
DEGRADE the conditioned arm toward (not necessarily below) the unconditional arm — when judges
often agree, a wrong card still carries mostly-right information, so the prespecified check is
"shuffling loses a significant fraction of the conditioning gain", not "shuffling goes negative".

## Multi-verbosity training (owner's design point)

Train on SEVERAL verbosity renderings of the same expression, biased toward concise:

- Each row's card is sampled per epoch from {V0, V1, V2, V3} with a concise-heavy distribution
  (indicative: V1 60%, V2 25%, V3 5%, V0 10%).
- WHY BOTH DIRECTIONS: concise cards teach the equivalence classes (all sonnet-routing variants
  share a V1 rendering → shared statistical strength); verbose cards teach the distinctions that
  matter within a class (t, N, judge modifier). The model learns the abstraction HIERARCHY, so at
  inference you can condition at whichever specificity you actually know.
- WHY BIAS CONCISE: (1) most distinctions don't survive the MDL test — spending most gradient on
  V1 matches where the information is; (2) concise cards are what inference will usually supply;
  (3) V3-heavy training would fragment 922 judged rows into near-singleton conditions.
- V0 (empty card) rows keep an unconditional readout alive — the agnostic-anchor precedent — and
  double as the conditioning-dropout that makes the verbosity ladder robust.

## Relation to the program

- **Label factory economics:** judge routing is an inference-time COST but a training-time ASSET.
  The end-state filer is μ (self-contained, free at inference); judges generate labels and
  occasionally re-rank. The expression `distill(e5(routing(...)))` names precisely this asset.
- **Target-factory policy:** the sqrt-KF-in-training decision (Kalman as target factory, not
  in-loop layer) generalizes: EVERY target factory gets an expression; the expression is the
  factory's name. Codex's theory lane owns the grammar's formal semantics; the application lane
  owns the empirical MDL ladder.
- **Certainty re-entry:** once expression-conditioned μ exceeds e5 in some region, the bias-state
  info_ratio gate becomes meaningful again — certainty-weighting returns exactly when the
  precondition (μ ≥ e5 somewhere) is met, per the owner's framing.
