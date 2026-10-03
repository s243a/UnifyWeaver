# Report: pattern_stache graduation — the calls the evidence did not make

*Third report in the series
([prototype](REPORT_pattern_stache_prototype.md) →
[second consumer](REPORT_pattern_stache_second_consumer.md) → this). Two witnessed consumers
was the graduation threshold; the dialect now has a specification
([`docs/design/SPEC_pattern_stache.md`](../../docs/design/SPEC_pattern_stache.md)) and a
production engine (`src/unifyweaver/core/pattern_stache.pl`, tests in
`tests/core/test_pattern_stache.pl` — 65: the 38 dispatcher tests and 12 consumer tests ported,
plus 10 migration-checklist and 5 linearity tests). The prototype directory is untouched, as the
historical witness.*

**This report exists to flag the places where two consumers' evidence UNDER-determined a
decision and a call had to be made.** Each is stated with what was chosen, why, and what
reversing it would cost — so the owner reviews decisions, not code.

## Calls made, in decreasing order of reviewability-worthiness

### 1. Non-linear patterns are a load ERROR, not silently permitted

Neither consumer wrote `f(X, X)`; unification would give it natural semantics for free. The spec
excludes it, and the engine **enforces** the exclusion: `nonlinear_pattern` at load. Rationale:
fail closed on the never-witnessed (PROJECT_PHILOSOPHY §5) — permitting it silently would ship
unspecified behaviour that a later spec version could not change without breaking files.
Reversal cost: delete one check (`pattern_is_linear/1` call in `read_case_pattern/3`) and
specify the semantics. **This is the strongest call in the set** — it refuses a capability the
machinery has.

### 2. Nested `{{match}}` is characterized but not promised — the opposite treatment

The scanner is depth-aware (inherited from the copied core scanner), the prototype's nested
tests are ported and pass, and the spec scopes nesting **outside the v1 contract** with a
revisit condition. So: non-linearity is *refused*, nesting is *permitted-but-unpromised*. The
asymmetry is deliberate and worth reviewing: nesting has tested, working semantics (child scope
composition) that the prototype witnessed even though neither real consumer template used it;
non-linearity was never exercised by anything. Refusing nesting would have meant surgery on the
scanner; refusing non-linearity cost eight lines. If the owner prefers symmetry, the cheap
direction is promoting nesting into the contract (the tests already exist), not refusing it.

### 3. The engine ships no consumer driver; the 12 consumer tests carry a test-local copy

The second consumer's phase protocol (`elaboration | obligation | runtime | no_checker`), mode
split, and closed-fact execution are **consumer contract, not engine behaviour** — so
`constraint_dispatch.pl` was not graduated into core. The ported consumer tests inline a ~40-line
copy of the driver instead. Cost: that copy can drift from the prototype's. Judged acceptable
because the driver is itself prototype-grade; when a real elaborator consumes the dialect, it
brings its own driver and these tests keep witnessing the engine half.

### 4. `.stache` is not wired into `render_named_template/4`

The prototype's Q5 answer sketched how core *would* dispatch: record the dialect at load, branch
on the tagged term in the render path. Doing that means touching `template_system.pl`, which is
frozen. So in production, `.stache` files are used through the `pattern_stache` module's own API
(`load_stache_file/2`, `render_stache_file/3`), not through the named-template resolution chain.
Deferred with a condition: wire it when a consumer needs to load `.stache` templates *by name
through the existing config machinery* — none does today.

### 5. Order-dependent overlap stays a WARNING

The trichotomy's middle row (unifiable, neither subsumes) warns and applies first-match-wins.
Neither consumer produced such a pair in a real template — the row is witnessed only by
synthetic tests — so whether it should eventually be an error is genuinely open. Production
change from the prototype: the warning goes through `print_message/2` with a proper
`prolog:message//1` clause rather than raw `format/3` to stderr, so downstream tooling can
intercept it. Escalating to an error later is a one-line change but a *contract* change.

### 6. `{{q:Key}}` syntax is frozen by a single consumer's demand

Quoted interpolation earned its place (consumer 2 reads rendered output back as terms), but the
*spelling* — a `q:` prefix inside the braces — was chosen once, without alternatives being
tried. Anything downstream that parses `{{...}}` tags now meets a colon form. Flagged because
syntax is the hardest thing to change after files exist.

### 7. Missing-key behaviour and whitespace stay as the string dialect has them

Unknown placeholders are left verbatim (matching `template_system.pl`, and matching what the
characterization tests recorded); the engine never trims case-body whitespace — both witnessed
consumers trimmed per-rendering with `normalize_space/2`, so the spec assigns whitespace to the
caller. Both are recorded as spec exclusions with revisit conditions rather than design: real
templates larger than the worked examples may want mustache "standalone line" semantics, and a
fail-on-unbound mode would be a new dialect version since it changes output.

### 8. The module name collides with the prototype's, by design

Both the production module and the prototype module are named `pattern_stache`, so they cannot
be loaded into one process. Accepted: the prototype is a sealed witness, not a library; its test
suites run in their own processes and still pass untouched (verified in this change). Renaming
the prototype would falsify the historical record; renaming production would put the wrong name
on the artifact going forward.

### 9. "Four shapes" is stated as one rule with five witnessed members

The task language says four; the second consumer's census found a fifth member (mixed
ground-leaf/variable, `has_type(X, substrate(pearltrees))`) inside the same rule. The spec
presents the **rule** (first-order, linear, guard-free terms) as normative and the five shapes
as its evidence base, which is the honest framing — the alternative (enumerating exactly four)
would have excluded a witnessed, load-bearing pattern.

## What did NOT happen

- No third consumer shape turned up that the witnessed grammar cannot express — the
  stop-and-report clause was never triggered.
- `template_system.pl` and `templates/` are untouched; the freeze held by construction (new
  extension → new parser; the loader refuses `.mustache` paths).
- No grammar extension was made mid-implementation. The only engine additions beyond the
  prototype are enforcement (linearity) and plumbing (message hooks), both subtractive of
  ambiguity rather than additive of capability.

## Verification snapshot

| suite | result |
|---|---|
| `tests/core/test_pattern_stache.pl` (production) | 65/65 pass |
| `prototypes/.../test_pattern_stache.pl` (witness, untouched) | 38/38 pass |
| `prototypes/.../test_constraint_dispatch.pl` (witness, untouched) | 12/12 pass |
| `template_system.pl` self-test (`test_template_system/0`) | passes, unchanged |
