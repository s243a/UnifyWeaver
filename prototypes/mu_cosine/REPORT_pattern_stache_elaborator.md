# Report: the Prolog-side elaborator prototype — the ruled design under implementation contact

*Sixth report in the series
([prototype](REPORT_pattern_stache_prototype.md) →
[second consumer](REPORT_pattern_stache_second_consumer.md) →
[graduation](REPORT_pattern_stache_graduation.md) →
[third](REPORT_pattern_stache_third_consumer.md) /
[fourth consumer](REPORT_pattern_stache_fourth_consumer.md) → this). Implements
[`DESIGN_prolog_elaborator.md`](DESIGN_prolog_elaborator.md) to its own ruled design — all
seven §5 rulings decided in PR #4093 — in `prototypes/mu_cosine/pattern_stache/` per ruling
2(a). Where the note fenced something, the fence was treated as the spec.*

## Verdict first: the ruled design survived contact — with four findings, none silently repaired

The architecture map held: the fixpoint loop really is consumer 2's mode split iterated,
`pe_where`'s validation machinery really did transplant, and the ground path really does end at
the sealed bytes with nothing new between. **Zero pressure on the pattern grammar** — the
elaborator adds no template cases and never touches the dialect; the stop-clause did not fire.
The four places where implementation contact taught something the note didn't say are reported
below as findings, per the task's discipline.

## What was built

| file | contents |
|---|---|
| `gen_registry_mirror.py` | ruling 5(b): generates the closed-constraint mirror from the sealed `process_cards.py` REGISTRY, embedding the source's sha256 |
| `pe_registry_mirror.pl` | GENERATED — 14 atoms, 13 operators, 32 kwspecs, plus `pe_output/2`, `pe_modifier/2`, `pe_required/2`; re-checks the embedded hash at load and **refuses to load on drift** |
| `pe_elaborate.pl` | the elaborator: fixpoint discharge loop, `ground/1` vs `pattern/2` states, canonical residual store (ruling 1), origins alongside (ruling 3), scope per ruling 4(a) |
| `test_pe_elaborate.pl` | 88 tests: 25×2 golden byte rows, 25 structural-oracle rows, fixpoint, residual behavior, canonical store, mirror drift + cross-checks, fail-closed battery |

Modified under the ruled exception: `pe_emit.pl` (hand mirror deleted) and `pe_where.pl`
(consumes the generated mirror; validation helpers exported for reuse). All other constraints
held: `template_system.pl`, `templates/`, sealed grammar/bundles, production engine, and
witnesses 1–2 untouched; vNext Python never imported.

```bash
swipl -g run_tests -t halt test_pe_elaborate.pl     # 88 tests
# full regression: 88 + 50 + 60 + 38 + 12 + 5 (probe) + 65 (engine) — all green
```

## The mirror retirement (ruling 5(b)), diff summary

**Deleted from `pe_emit.pl`**: the entire hand-maintained registry block — 14 `pe_atom/1`,
3 `pe_variadic/1`, 32 `pe_kwspec/4`, 13 `pe_operator/1` facts (~86 lines) — replaced by one
`use_module(pe_registry_mirror, ...)`. **Deleted from `pe_where.pl`**: every module-qualified
`pe_emit:pe_*` read (the flagged consumer-3 liberty), replaced by direct imports from the
generated mirror. Two deliberate vocabulary corrections came with generation, because the
mirror now carries the registry's kinds **verbatim**:

- `estimand` and `impl` are their own kinds (the hand copy had flattened them to `string`);
  both drivers treat them string-like for rendering and legality;
- `mu`'s kind is its declared type `judge` (the hand copy invented `expr`); node-valued kwargs
  are now recognized as "declared kind is an output type".

**The drift test is exactly the golden bundle**: all 110 pre-existing byte tests (consumers 3–4)
pass unchanged over the swapped mirror — the swap changed nothing observable. The load-time
check is real and tested: a tampered mirror (zeroed hash) beside a copy of the sealed source
refuses to load in a subprocess.

## The fixpoint loop, and its termination argument

One pass attempts every goal: a binding with a ground value discharges by substitution (after
`pe_where`-style per-position legality over the term, including the ratified pin refusals); a
ground `has_type/2` discharges by the closed registry check (`pe_output/2`, modifier-aware via
`pe_modifier/2`) or **errors** in the author's vocabulary; anything under-instantiated
residuates; anything outside ruling 4(a)'s scope errors at first sight. The loop re-runs
because discharging a binding can make a residual checkable — the note's own observation, now a
test (`binding_discharge_unlocks_residual`).

**Termination, in one paragraph:** every pass either discharges at least one goal or is the
last pass. Discharge strictly removes a goal from the store and nothing ever adds one, so the
store size is a strictly decreasing measure across progressing passes; substitution only
instantiates variables (monotone — nothing unbinds), so no goal ever regresses from discharged
to pending. With N goals the loop therefore runs at most N passes, each linear in the store,
and termination needs no further hypothesis.

## Verification per the note's oracle inventory

- **Ground path, sealed strings**: all 25 golden rows through `elaborate/3` with an empty
  store, byte-equal on both surfaces; the five repeated-atom rows again through binding stores
  (the §12 form, now via the loop).
- **Ground path, sealed structure — the note's find, delivered**: the bundle's `resolved_ast`
  verified for **all 25 rows**. The elaborated ground term is converted by the same
  introspection discipline (mirror as Config) into the sealed JSON's shape — `kind`/`name`/
  `output`/`args`/`kwargs` (value kwargs as `{key, lexical, value_type}` with registry-verbatim
  value types, node kwargs as `{key, node, value_type}`), numeric positional literals, mods,
  pins — and compared after normalizing both sides to plain terms (measured: dicts with
  anonymous tags are never `==`; normalization is required, not optional).
- **Residual path, behavioral** (no sealed oracle — the note says so honestly): discharge
  order-independence, fixpoint termination on chains (worst-case ordering included), the sort
  key's probe measurements re-asserted at the API level, pattern-state shape (including
  ground-term-with-residuals and open-term-with-empty-store), and origins riding alongside a
  pure store.
- **Cross-check, data only**: `frontend_registry_fixture.json` from vNext's testdata — for
  every fixture *reference* entry whose name the v0.4 mirror registers, the fixture's type maps
  to the mirror's output under the documented overlap (`corpus`→`substrate`, `judge`→`judge`);
  the fixture's required `margin.t` matches `pe_required(margin, t)`. No Python imported,
  wrapped, or ported.
- **Fixtures** honor ruling 7: every pattern fixture is named under a mandatory
  `no_identity/1` marker, checked by a meta-test; no digests, no canonical bytes, no
  persistent names.

## Findings — where contact taught something the note didn't say

1. **Cross-call `==` of live residual stores is impossible *by design*, and the note's
   order-independence phrasing didn't say at which level it holds.** `elaborate/3` copies its
   input (the copy-don't-mutate discipline the note itself carries over from `pe_where`), so
   two calls never share variables and their live stores can only be variants (`=@=`). The
   `==`-identity the ruled sort key promises holds exactly at the **numbered-projection**
   level — which is also the only level `peid-v1` will ever hash. Tests assert both: live
   states `=@=`, numbered projections `==`. Within one call, `canonical_store/3` returns the
   caller's own goal instances reordered, where live `==` is observable and tested.
2. **The §3 mode table has a corner the note's §1 example glossed:** discharging `C=simplemind`
   grounds `has_type(_X, substrate(C))`'s *type argument* while `_X` stays free — the goal
   correctly **residuates partially instantiated** rather than becoming checkable. Caught as a
   wrong test expectation; the code was right and the test now documents the corner. Follow-on
   design question for the AST lane, recorded not decided: should a `has_type` whose *type*
   side is fully ground discharge its registry half early, leaving a narrower residual? v1
   does not — discharge is all-or-nothing per goal, matching the note.
3. **`pe_where` needed an export-list addition** (`check_binding_shapes/1`,
   `check_no_duplicates/1`, `occurrences/3`, `check_value_at/2`). The note ruled the machinery
   "reused as-is" without saying how; the alternatives were a new module-qualified liberty
   (the exact pattern ruling 5(b) just retired) or exporting. Exported, behavior unchanged
   (witness suites green). Consequence: reused validation raises `pe_where(...)`-tagged errors
   inside elaborator flows (`bad_binding`, `duplicate_binding_for_one_variable`) while
   elaborator-native checks raise `pe_elaborate(...)` — a cosmetic seam, kept visible rather
   than papered over with re-wrapping.
4. **A load-time `initialization/1` error does not set swipl's exit status.** The mirror's
   fail-closed drift check throws during load, but a process that merely *loads* the tampered
   mirror still exits 0; the subprocess test must invoke `pe_mirror_verify` as the `-g` goal.
   Recorded because anyone wiring the mirror check into CI will hit the same trap.

One smaller note in the same spirit: any `=/2` goal is claimed by the binding channel
(including malformed ones with a bound left-hand side), so `foo = bar` is refused as
`bad_binding` rather than surfacing as a misleading unknown-constraint or already-bound error.

## Ruling compliance

| ruling | status |
|---|---|
| 1 — sort key | numbervars-by-traversal projection, `==`-dedup; term vars numbered first, store-only vars by projection-least goal; ordering device only, no identity claim |
| 2(a) — location | all files in `prototypes/mu_cosine/pattern_stache/` |
| 3 — origins | alongside, never inside: `elaborate/4` returns `Goal-Origin` pairs aligned with the canonical store; `pattern/2` holds pure goals (tested) |
| 4(a) — scope | bindings + `has_type/2` + closed registry checks; everything else `unknown_constraint`, fail closed |
| 5(b) — mirror | generated, hash-embedded, load-checked, drift-refusing; both hand copies retired; golden bytes prove the swap inert |
| 6 — deferred | untouched: nothing here encodes residuals for any encoder |
| 7 — fixtures | `no_identity/1` marker mandatory, meta-tested |

The identity fence held throughout: no digest, no canonical bytes, and no persistent name
exists anywhere for a pattern state.
