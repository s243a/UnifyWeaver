# Process-Expression Language — Implementation Plan

Phased, each phase independently shippable with a measurable exit criterion. Reuse-first: the
card→e5→NameFunctionCond pathway already exists (judge onboarding at r=0); phase 1 adds no
architecture.

## Reader's key (external-review shorthand + glossary)

"Finding N" = gpt-5.6-sol's review of #3974 (amended in #3978): 1 grammar self-parse, 2 identity
vs cards, 3 NameFunctionCond honesty, 4 row ledger/estimand, 5 exploratory-only manifest,
6 zero-init ≠ floor, 7 frozen primary, 8 not-MDL naming, 9 multi-process LOCO. "r2 item N" =
sol's follow-up: 1 lexical gaps, 2 flat-token primary, 3 deterministic composition.

Glossary — **legacy_unbound**: pre-v2 task/pick artifacts without content-bound hashes;
descriptive evidence only, never upgradeable. **v2 bundle**: content-bound task+pick envelope
(routed_policy.py) binding catalog, privacy index, policy tier, judge contract, and byte hashes.
**node-block bootstrap**: resampling typed (bookmark, folder) identity blocks — never individual
process rows — so dependence between rows sharing a node is respected (#3845 machinery).
**process-complete split**: every process rendering of a query lives on the same split side.

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
- Exit: every existing judge/op token has an expression rendering; V3 and canonical identities
  round-trip stably. Lower-verbosity cards are intentionally lossy and are never identifiers.

## P1 — minimal experiment: expression-conditioned distillation

The prospective, machine-checked contract is
`PROTOCOL_process_expression_p1.md` + `PROCESS_EXPRESSION_P1_PREREG.json`; it is authoritative
where this implementation sketch is less specific.

The old manifest `fcf5e1d6` contains the four useful process families, but its judge tasks/picks
are `legacy_unbound` and included nonpublic candidates. They cannot be upgraded retroactively to
the v2 execution/privacy contract and therefore cannot enter the primary ledger. They remain a
local-only descriptive sensitivity. Primary P1 either uses newly verified public-only v2 bundles
or records `blocked_no_eligible_v2_labels`; producing new labels is a separate spending decision.

**Status: EXPLORATORY and TRANSDUCTIVE (review finding 5).** The 1,200-query manifest's outcomes
already informed the routing ceilings and the selection of t, N, and Sonnet — no partition of it
can serve as an untouched confirmatory test. Confirmation is reserved for a future bookmark
cohort with a structurally frozen catalog, collected AFTER this design is fixed.

- **Frozen row ledger (finding 4), built and hashed before any training:** the historical
  1,200-query population had 922 unique judged queries, 1,617 judge pick-records, 278 auto rows,
  and 1,895 process-target records. Those are descriptive counts, not a required P1 sample size.
  The primary ledger records whatever survives prospective v2 re-derivation, privacy,
  execution-bundle, population, and join gates.
- **Estimand + targets:** the trained object scores (query, folder) pairs conditioned on a card;
  the training target of a judge row is its picked folder (one positive vs the other menu
  folders as in-menu alternatives); NULL picks contribute no primary ranking loss because they
  identify no positive. A fractional-negative treatment is inner-training sensitivity only.
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
- **Frozen primary decision (finding 7; amended per review r2 item 2):** the claim is
  "STRUCTURED EXPRESSIONS earn their keep over flat tokens", so the primary is arm (a) vs
  arm (b) — held MRR, paired two-endpoint node-block bootstrap, 3 training seeds pooled by
  mean-per-query; success = superiority (CI lower endpoint > 0 and point gain ≥ +0.01 MRR).
  A CI lower bound ≥ −0.005 is classified as in-distribution noninferiority only, never P1
  success. A later P3 zero-shot LOCO result cannot retroactively change that classification.
  Secondary: (a) vs (c) ("conditioning helps at all"), R@1, shuffled fraction lost, per-process
  breakdowns — reported, no multiplicity claims.
- Exit: superiority authorizes P2. Noninferiority may authorize a separately preregistered P3
  mechanism study, but not P2; inferiority or failure of the e5 safety margin stops the ladder.

## P2 — gain-per-token verbosity sweep (naming per finding 8: efficiency curve, not MDL)

- Retrain P1-arm-(a) at fixed single verbosities V1/V2/V3 and at 3 mixture profiles; SELECT on an
  inner validation split, report once on held (never select on the reported set). Token counts
  under the pinned e5 tokenizer revision.
- **NL-template arm (2026-07-24 review):** e5 embedding a formal bracketed string is a leap of
  faith — e5 was trained on natural language. Add one arm rendering the SAME canonical AST as a
  natural-language template ("sonnet judge with lineage context, menus of 10 and 20, thresholds
  0.02/0.03") and compare against the formal V2 card at matched information content. This
  promotes the risk-section fallback to a tested primary design choice.
- Deliverable: the gain-per-token curve — the empirical answer to "how specific should the
  language be", and the tuned training mixture (replacing the indicative 60/25/5/10).
- Exit: a chosen default profile with the curve as justification, recorded in the report.

## P3 — compositional embedding + zero-shot

- Tree encoder over the AST — DETERMINISTIC composition (review r2 item 3): node embedding =
  card-e5(operator) + Σ position-weighted child embeddings with fixed per-position weights; any
  stochastic element (e.g. dropout-style card sampling) is seeded by SHA-256 of the canonical AST,
  so identical expressions always embed identically. A small learned tree encoder follows only if
  the deterministic superposition plateaus. The learned follow-on is specified separately in
  `DESIGN_expression_encoder_future.md`; its complete typed-role-path position contract and
  mathematically corrected ablation set are in `DESIGN_tree_position_encoding_theory.md`. Those
  documents are implementation handoffs, not permission to bypass this phase's activation gate.
- Zero-shot test (finding 9 — one held process proves nothing): PREREGISTERED
  leave-one-composition-out over SEVERAL held processes — at minimum {sonnet.lineage@N20,
  haiku@N10, kalman(luna.D, luna.S), one lineage-decay variant} — each evaluated against four
  controls: frozen-string e5 card, additive bag-of-nodes embedding, cold flat token, and
  unconditional. Report per-held-process and pooled.
- **Cross-corpus LOCO arm (OPENQ-012, preregistered here):** composition-level LOCO answers
  generalization across PROCESSES; the program's forest-level question is generalization across
  CORPORA. Hold out a corpus (Pearltrees / SimpleMind / wiki — the three task-matched harnesses)
  under fixed expressions and measure whether expression conditioning transfers better than flat
  tokens. Additive arm; does not modify the frozen P1 primary.
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

- The historical 922 judged-query count was small for four conditions, and the eligible
  public-only v2 population may be smaller. P1 uses e5 as an explicit rollback candidate rather
  than pretending zero initialization guarantees a floor; if underpowered, the label factory has
  a costed refill path that requires a separate spending decision.
- Card embedding via e5 of a formal string is a leap of faith at V2/V3 (bracketed kwargs are not
  natural language); if e5 collapses them, fall back to template rendering ("sonnet judge with
  lineage context, menus of 10") — canonicalization keeps the mapping deterministic either way.
- Verbosity mixture is per-row augmentation: verify no train/held row shares (query, folder) under
  different cards across the split (the node-disjoint rule already covers this; test it anyway).
