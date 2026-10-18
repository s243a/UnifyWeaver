# Report: ruling 4(b) — `interpretation/3` and `representation/4` as a checked-in module

*§7 of [`DESIGN_process_expression_patterns.md`](DESIGN_process_expression_patterns.md) says of
its own clauses: "These clauses illustrate the relation; they are not claims about a checked-in
Prolog module." This report covers making them one, in
[`pattern_stache/pe_interpret.pl`](pattern_stache/pe_interpret.pl) — the
specified-but-never-built half of the patterns doc.*

**Nothing here mints an identity.** The §0 ladder runs ground → semantically explicit →
representation-resolved → factory-verified, and this module stops at the third rung. The one
digest it computes is a *ruleset-dependency* digest (§7.1 requires it in the receipt), the same
category as the registry mirror's source hash.

## What is implemented, against each subsection

### §7.1 — the dispatcher pre-pass

All three branches, with the refusal as a first-class outcome rather than a fallthrough:

| spec clause | implemented as |
|---|---|
| "explicit `estimand=...` goes directly to the matching clauses" | `dispatch/2` → `explicit([Request])`, options unchanged |
| "absent estimand plus a content-bound `family_spec` expands to the finite, fully defaulted explicit requests listed by that specification" | `family(SpecId, Digest, Requests)`; the `family_spec` field is consumed by the expansion (leaving it would fail `consume_only` downstream — tested) |
| "absent estimand and absent `family_spec` is `underconstrained`, not an implicit enumeration" | `underconstrained(no_estimand_and_no_family_spec)`, and it **survives all the way out through `resolve/3`** rather than degrading into "here are all the interpretations" (tested for both selectors) |
| "the family-spec content digest is a transitive ruleset dependency and appears in the receipt" | sha256 of the whole ruleset file, in `receipt.dispatch.family_spec_sha256` |

### §7.1 — `interpretation/3`

Both illustrative rules transcribed, with all five helpers the spec names:
`option_exact/2`, `option_or_default/4`, `required_options/2`, `extract_options/3`,
`consume_only/3`. The four `hop_decay` defaults are exactly the spec's
(`real_value("0.85")`, `int_value("1")`, `ancestor`, `unbounded_depth`); `structural_score`'s
nine fields are required with no defaults. `extract_options/3` **transfers** `impl` rather than
discarding it — tested by observing it arrive in the representation options.

`consume_only/3` implements the exact partition as four conditions, one per word of §7.3's
rejection list: no key twice (*duplicate*), every transferred option was supplied and is
disjoint from the semantic keys (*exactly once*), and every supplied key lands in one part or
the other (*unknown / misspelled / inapplicable*).

### §7.2 — `representation/4`

The spec's three precise forms for `hop_decay_targets`. `impl` is consumed here exactly once.
Absent `impl` leaves **all three** compatible candidates — and the test that matters is the
negative one: three candidates means `unique` returns `not_unique(3)`, so absence "never creates
an implicit deployable default" is verified by the *absence of a default*, not just by the
presence of candidates. Unknown `impl` admits no rule → `no_candidates` → `unique` fails.

### §7.3 — every option consumed

**Both worked examples from §7.3 are tests, verbatim**: `structural_score` + `decay` (not
consumed) and `hop_decay` + `decay` twice (duplicate). Plus the cases §7.3 names but does not
work: unknown field, misspelled field (`decy`), inapplicable field, duplicate estimand, and a
duplicate with an *identical* value — that last one because "exactly once" is about occurrences,
not about disagreement, and a map-shaped options representation would have made it
unrepresentable. That is why options are a **list of occurrences** here.

### §8.1 / §8.2 (needed to make §7 observable)

Candidate collection with grouping and sorted deriving rule IDs, a finite resource cap that
errors rather than truncating, explicit `no_candidates`, and `unique` distinguishing zero from
many (`no_candidates` vs `not_unique(N)`) rather than both being bare failure.

## Test counts and refusal coverage

`test_pe_interpret.pl`: **51 tests** in seven units.

| unit | tests | of which refusals |
|---|---:|---:|
| `dispatch_prepass` | 10 | 4 |
| `interpretation_rules` | 6 | 1 |
| `option_consumption_refusals` | 7 | **7** |
| `option_helpers` | 6 | 4 (embedded negative cases) |
| `representation_rules` | 8 | 4 |
| `collection_and_unique` | 4 | 1 |
| `receipts` | 4 | 1 (mints-no-identity) |
| `walk_shape_rider` | 4 | — |

**Roughly half the suite is refusals**, which matches where §7's content actually is: §7.3 is
entirely about what must be rejected, so a suite weighted toward acceptance would have been
testing the easy half.

Full regression: **547 tests across thirteen suites**, 21/21 tutorial blocks, mirror hash green.

## Underdetermination (stop-clause: recorded, not ruled)

*Five places where §7 did not determine behaviour. **Two have since been ruled** (items 2–3
below, marked RULED and encoded); three remain parked with their fail-closed handling
endorsed.*

1. **No registry backs the §7.2 equivalence check.** §7.2 says the three precise forms "enter
   one candidate set only if the registry proves identical" on seven listed properties — but
   `hop_decay_targets`, `graph_walk_hop_decay` and the rest are **not in the v0.5 registry at
   all**; they are the patterns doc's own illustrative names. There is nothing to consult. The
   module admits all three into one set on the spec's say-so and marks that as a stand-in. **A
   registration question for the coordinator**, and the largest gap here.
2. **`lineage_interpretations_v1`'s contents are never listed.** — **RULED**: v1 contains
   exactly the bare `hop_decay` entry. The ruleset is collapsed to that; `structural_score`
   enters only via a future spec *version* grounding all nine fields, which is a new content
   digest. (The *entry* remains a prototype construction; what is no longer open is the rule it
   must satisfy.)
3. **"Fully defaulted" vs `required_options`.** — **RULED**: "fully defaulted" means fully
   grounded by the spec's own content, not by registry defaults alone; a spec entry must expand
   to a complete explicit request. Encoded as a ruleset **well-formedness property** rather than
   only prose: every entry of every spec must expand to a request that actually interprets, so
   an entry naming a required-field estimand without its values fails a test instead of
   producing a quietly empty candidate set. Verified non-vacuous — a deliberately violating
   entry fails it.
4. **No canonical encoding exists for semantic-request ASTs.** §8.1 step 7 sorts groups "by
   canonical typed-AST bytes" and step 8 hashes that; §9's receipt wants four typed-AST digests.
   Neither is computable: `pe-typed-ast-v1` has not frozen an encoding for these terms. The
   module sorts by standard term order as an explicit stand-in, computes **no** hash, and the
   receipt **enumerates the fields it omits and why** — so a consumer sees the gap rather than
   receiving something that looks complete. Inventing those digests would mint identities at a
   rung the ladder says this stage must not reach.
5. **The typed-value wrapper vocabulary is not closed.** §7.1 names `ref/1`, `atom/1`, `str/1`
   and uses `real_value/1`, `int_value/1`, but never enumerates the set. The module accepts what
   §7 uses and rejects `str/1` for `impl` (enumerated configuration is not free text) — a
   reading, flagged as one.

One smaller judgement, flagged rather than buried: §7.3 forbids rules depending on "cuts, clause
order". The rules here contain no cuts and are order-independent (all derivations collected,
then grouped); helper predicates use if-then-else, which commits *within* a helper and never
across rule selection. That is the reading taken, stated so a reviewer can disagree explicitly.

## The cowalk delta

The patterns doc predates `cowalk`, so **no coarse form or interpretation rule was invented for
it**. What one would need, as a registration question:

| piece | what is missing |
|---|---|
| a coarse family | there is no `cowalk_op` analogous to `lineage_op` — the doc's coarse/precise split has no cowalk entry |
| estimands | `lineage_op` dispatches on `estimand(hop_decay \| structural_score)`. Cowalk's registered estimands are the general `path`/`ancestry` set; whether a cowalk coarse family dispatches on those or on new cowalk-specific estimands is unanswered |
| the walk/weight axis | `walk` and `weight` are *registered kwargs with enumerated values*, not estimands. Whether an interpretation rule consumes them as semantic fields (like `decay`) or transfers them as representation constraints (like `impl`) is exactly the §7.1/§7.2 boundary question, and the doc cannot answer it because it predates them |
| the declared shape | `WALKS` declares each walk's shape (`sibling`/`cousin` → `palindromic`; `WALK_SHAPES` also admits `non_palindromic`). If a coarse family dispatches on the *family* rather than the walk, it must **read** the declared shape — never infer it from the name |
| precise forms | no representation candidates exist for a cowalk semantic; the §7.2 equivalence list would need cowalk-specific criteria (the seven properties mention direction and hop convention, which a palindromic walk may satisfy differently) |

The cheapest first question for the coordinator was the third row — and it has been **RULED:
`walk` and `weight` are SEMANTIC fields; only `impl` is a representation constraint.** The
criterion is §7.2's own: representations may share a candidate set only if they prove identical
observable behaviour, and changing the walk changes which targets are in the set while changing
the weight changes the scores — both change the estimand's value, unlike `impl`, which changes
only the route to the same value. So a future cowalk coarse family maps `walk`/`weight` into the
semantic request and transfers only `impl`, the same partition `lineage_op` uses. Recorded in
`pe_emit.pl`'s header beside the walk-shape rule so that work starts from the ruling. The
remaining rows stay open: **the coarse family is not built**, and what §7.2's equivalence
criteria contain for cowalk is still a registration question.

## Rider: declared walk shapes reach the mirror

`gen_registry_mirror.py` now emits `pe_walk_shape/2`, `pe_walk_shape_kind/1` and
`pe_weight_value/1` from `pc.WALKS` / `pc.WALK_SHAPES` / `pc.WEIGHTS`. Regeneration was one
command, the hash check stayed green, and four tests pin the declarations — including that every
declared walk's shape is a declared shape *kind*, which is what makes reading the declaration
meaningful rather than decorative. The first consumer of the classification no longer has to
touch the generator.
