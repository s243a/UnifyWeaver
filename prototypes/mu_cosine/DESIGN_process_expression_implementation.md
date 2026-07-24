# Process-Expression Language — Implementation Plan

Phased, each phase independently shippable with a measurable exit criterion. Reuse-first: the
card→e5→NameFunctionCond pathway already exists (judge onboarding at r=0); phase 1 adds no
architecture.

## P0 — expression module (`process_cards.py`)

- AST for the v0 grammar (spec doc), canonicalizer, verbosity renderer V0–V3, e5 embedding via
  the existing card path (cache keyed on canonical string).
- Registry of CURRENT processes as expressions: e5-auto, haiku@N10 routing, sonnet.lineage@N10,
  sonnet.lineage@N20, kalman(luna.D, luna.S), lineage(graph, decay=0.85), the blend judges.
- Unit tests: canonicalization idempotence, default elision, verbosity monotonicity (V1 ⊂ V2 ⊂ V3
  informationally), embedding cache determinism.
- Exit: every existing judge/op token has an expression rendering; round-trip stable.

## P1 — minimal experiment: expression-conditioned distillation (the data already exists)

Corpus (no new API spend): the four differently-generated label sets on manifest fcf5e1d6 —
auto rows (margin ≥ 0.03, e5 top-1), haiku@N10 picks (695), sonnet.lineage@N10 picks (695),
sonnet.lineage@N20 picks (227).

- Model: e5-residual head — score = e5_cos + correction(μ | card), correction zero-init (starts
  exactly at e5: the owner's "at least as good as e5" floor is mechanical, warm-start lesson).
- Per-row card sampled per epoch: V1 60% / V2 25% / V3 5% / V0 10% (concise bias).
- Split: node-disjoint over judged queries (#3845 rule); recorded placements are EVAL-ONLY.
- Arms: (a) expression cards, (b) flat judge tokens, (c) merged unlabeled pile, (d) SHUFFLED cards
  (specificity control — must hurt). Report held R@1/MRR vs the e5 floor + paired bootstrap.
- Exit: (a) ≥ e5 on held (floor holds) AND (a) > (b),(c) with (d) degraded ⇒ the language earns
  its keep on owned data.

## P2 — MDL verbosity sweep

- Retrain P1-arm-(a) at fixed single verbosities V1/V2/V3 and at 3 mixture profiles; measure
  held ΔNLL (and ΔMRR) per mean expression token.
- Deliverable: the gain-per-token curve — the empirical answer to "how specific should the
  language be", and the tuned training mixture (replacing the indicative 60/25/5/10).
- Exit: a chosen default profile with the curve as justification, recorded in the report.

## P3 — compositional embedding + zero-shot

- Tree encoder over the AST (superposition first — random_operator_embedding precedent — then a
  small learned encoder if superposition plateaus).
- Zero-shot test: hold out ONE process entirely (e.g. sonnet.lineage@N20), condition on its
  expression at inference, measure whether structure transfers from the seen siblings vs a cold
  flat token. This is the test flat tokens cannot pass by construction.
- Exit: zero-shot conditioned > unconditioned on the held process.

## P4 — program integration

- Expressions as target-factory descriptors everywhere targets are minted (fine_tune_*,
  meta-judge, campaign emitters): every scored TSV/pick file carries its canonical expression in
  the header; loaders fail closed on missing/unknown expressions (provenance contract).
- Judge-card registry entries for composite judges become expression cards (kalman-fused, blend,
  dir-blend) — delete the flattening.
- Hand the grammar's formal semantics + the information objective to the Codex lane (their kind
  of object); keep the empirical ladder here.

## Risks / open questions

- 922 judged rows is small for 4 conditions — P1 leans on the e5-residual floor and the shared-
  strength effect of concise cards; if underpowered, the label factory has a costed refill path
  (the routing loop, now with measured judge value).
- Card embedding via e5 of a formal string is a leap of faith at V2/V3 (bracketed kwargs are not
  natural language); if e5 collapses them, fall back to template rendering ("sonnet judge with
  lineage context, menus of 10") — canonicalization keeps the mapping deterministic either way.
- Verbosity mixture is per-row augmentation: verify no train/held row shares (query, folder) under
  different cards across the split (the node-disjoint rule already covers this; test it anyway).
