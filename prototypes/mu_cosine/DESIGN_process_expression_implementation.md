# Process-Expression Language — Implementation Plan

Phased, each phase independently shippable with a measurable exit criterion. Reuse-first: the
card→e5→NameFunctionCond pathway already exists (judge onboarding at r=0); phase 1 adds no
architecture.

## P0 — expression module (`process_cards.py`)

- AST for the v0 grammar (spec doc), the TYPED OPERATOR-SIGNATURE REGISTRY (versioned; arity,
  param types + defaults, output type, atom/operator dual roles), canonicalizer, verbosity
  renderer V0–V3, e5 embedding via table-expansion onboarding (cache keyed on canonical-AST hash
  + verbosity + renderer version + embedding revision + prefix — never the rendered string alone).
- Lossless process identity (canonical AST + factory/manifest fingerprint) split from lossy cards
  per the amended spec; acceptance test: the registry parses every example in the spec doc.
- Registry of CURRENT processes as expressions: e5-auto, haiku@N10 routing, sonnet.lineage@N10,
  sonnet.lineage@N20, kalman(luna.D, luna.S), lineage(graph, decay=0.85), the blend judges.
- Unit tests: canonicalization idempotence, default elision, verbosity monotonicity (V1 ⊂ V2 ⊂ V3
  informationally), embedding cache determinism.
- Exit: every existing judge/op token has an expression rendering; round-trip stable.

## P1 — minimal experiment: expression-conditioned distillation (the data already exists)

Corpus (no new API spend): the four differently-generated label sets on manifest fcf5e1d6 —
auto rows (margin ≥ 0.03, e5 top-1), haiku@N10 picks (695), sonnet.lineage@N10 picks (695),
sonnet.lineage@N20 picks (227).

**Status: EXPLORATORY and TRANSDUCTIVE (review finding 5).** The 1,200-query manifest's outcomes
already informed the routing ceilings and the selection of t, N, and Sonnet — no partition of it
can serve as an untouched confirmatory test. Confirmation is reserved for a future bookmark
cohort with a structurally frozen catalog, collected AFTER this design is fixed.

- **Frozen row ledger (finding 4), built and hashed before any training:** 1,200 unique queries;
  922 unique judged queries; 1,617 judge pick-records (695 haiku@N10 + 695 sonnet.lineage@N10 +
  227 sonnet.lineage@N20; 695 queries appear under BOTH haiku and sonnet processes); + 278
  auto-rows (margin ≥ 0.03, process = e5-auto) → 1,895 process-target records. The ledger file
  records (query, folder-menu, process AST hash, pick|null, card renderings) per row.
- **Estimand + targets:** the trained object scores (query, folder) pairs conditioned on a card;
  the training target of a judge row is its picked folder (one positive vs the other menu
  folders as in-menu negatives); NULL picks contribute no positive — they train only the
  abstention-consistent negatives (all menu folders down-weighted ×0.5, recorded knob).
  Per-process row weights normalize so no process dominates by count (recorded in the ledger).
- **Split grouping:** node-disjoint over (bookmark, true-folder) identities AND process-complete —
  every process rendering of a query travels with it to the same side. Uncertainty: resample
  typed query-blocks and folder-blocks (two-endpoint node-block bootstrap, the #3845 machinery),
  never individual process rows.
- Model: e5-residual head — score = e5_cos + correction(μ | card), correction zero-init. **Zero
  init is an equality START, not a floor (finding 6):** training can degrade held metrics. The
  guarantee is procedural — e5 is an explicit ROLLBACK CANDIDATE selected on inner validation
  (if no checkpoint beats e5 there, ship e5), then ONE evaluation on the outer held set with a
  prespecified noninferiority margin (held MRR ≥ MRR_e5 − 0.01).
- Per-row card sampled per epoch: V1 60% / V2 25% / V3 5% / V0 10% (concise bias; P2 tunes this).
- Arms: (a) expression cards, (b) flat process tokens, (c) merged pile (no conditioning),
  (d) SHUFFLED cards — permutation defined as: within each split side, permute the card column
  across rows UNIFORMLY AT RANDOM among rows with distinct process ASTs (so a shuffled row's card
  is always wrong), 5 permutation draws, mean reported.
- **Frozen primary decision (finding 7):** primary metric = held MRR of arm (a) vs arm (c),
  paired two-endpoint node-block bootstrap, 3 training seeds pooled by mean-per-query; success =
  CI excludes zero AND point gain ≥ +0.01 MRR. Everything else (R@1, arm (b), shuffled fraction
  lost, per-process breakdowns) is secondary/descriptive — reported, no multiplicity claims.
- Exit: primary success ⇒ proceed to P2; primary failure but rollback floor held ⇒ the language
  is not earning conditioning gain on this corpus — stop and hand the grammar to the theory lane
  before spending more.

## P2 — gain-per-token verbosity sweep (naming per finding 8: efficiency curve, not MDL)

- Retrain P1-arm-(a) at fixed single verbosities V1/V2/V3 and at 3 mixture profiles; SELECT on an
  inner validation split, report once on held (never select on the reported set). Token counts
  under the pinned e5 tokenizer revision.
- Deliverable: the gain-per-token curve — the empirical answer to "how specific should the
  language be", and the tuned training mixture (replacing the indicative 60/25/5/10).
- Exit: a chosen default profile with the curve as justification, recorded in the report.

## P3 — compositional embedding + zero-shot

- Tree encoder over the AST (superposition first — random_operator_embedding precedent — then a
  small learned encoder if superposition plateaus).
- Zero-shot test (finding 9 — one held process proves nothing): PREREGISTERED
  leave-one-composition-out over SEVERAL held processes — at minimum {sonnet.lineage@N20,
  haiku@N10, kalman(luna.D, luna.S), one lineage-decay variant} — each evaluated against four
  controls: frozen-string e5 card, additive bag-of-nodes embedding, cold flat token, and
  unconditional. Report per-held-process and pooled.
- Exit: pooled zero-shot conditioned > unconditional AND > cold flat token across the LOCO set
  (not just one favorable sibling).

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
