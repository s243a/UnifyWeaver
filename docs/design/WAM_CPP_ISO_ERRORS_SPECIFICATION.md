# WAM C++: ISO Error Terms — Specification

Concrete shape of the ISO-error dispatch we're implementing in
`wam_cpp_target.pl`. For *why* each decision, see
`WAM_CPP_ISO_ERRORS_PHILOSOPHY.md`.

## 1. Config schema

A config is a Prolog file that asserts two kinds of facts using a
dedicated module:

```prolog
% iso_errors.pl

:- module(iso_errors_config, [
       iso_errors_default/1,
       iso_errors_override/2
   ]).

% Global default for predicates not otherwise listed.
iso_errors_default(true).

% Per-predicate overrides. Either direction is allowed.
iso_errors_override(legacy_lookup/3, false).
iso_errors_override(unsafe_div/3,    false).
iso_errors_override(experimental:my_pred/2, true).
```

**Fact shapes:**

| Fact | Arity | Meaning |
|---|---|---|
| `iso_errors_default(Mode)` | 1 | Default for all predicates. `Mode` is `true` or `false`. Optional; omitted means `false`. |
| `iso_errors_override(PI, Mode)` | 2 | Mode for one predicate. `PI` is a predicate indicator (`Name/Arity` or `Module:Name/Arity`). |

**Module qualifier:** if `PI` is `Module:Name/Arity`, the override
applies only to that module. Bare `Name/Arity` matches the predicate
in whichever module the generator is compiling — the default is to
match across all modules (this is the common case in single-module
projects). We'll likely need to revisit this rule when multi-module
projects appear.

**Conflict resolution:** later `iso_errors_override/2` facts in the
file override earlier ones for the same `PI`. The
`iso_errors_default/1` fact is read once; multiple definitions are
an error.

## 2. Loading the config

`write_wam_cpp_project/3` gains an option:

```prolog
write_wam_cpp_project(Preds, [
    iso_errors_config('config/iso_errors.pl')
], Dir).
```

The loader:

1. `consult/1`s the file in a fresh sub-namespace (so its
   `iso_errors_*` facts don't bleed into user code).
2. Reads `iso_errors_default/1` (if present) into a generator-local
   default. Absent → `false`.
3. Builds a dict (or association list) of `PI → Mode` from
   `iso_errors_override/2` facts.

Inline overrides on the `write_wam_cpp_project/3` call line still
work and merge with the config, with inline winning. So:

```prolog
% Config says default false.
write_wam_cpp_project(Preds, [
    iso_errors_config('config/iso_errors.pl'),
    iso_errors(true),                          % override the default
    iso_errors(legacy_lookup/3, false)         % override one predicate
], Dir).
```

## 3. Per-predicate dispatch in the generator

The generator already walks predicates one at a time inside
`emit_setup_function/3`. We hook into the per-predicate phase:

```prolog
emit_setup_function(Predicates, Options, SetupCpp) :-
    iso_errors_resolve_options(Options, IsoConfig),
    foreign_pred_keys_from_options(Options, _ForeignKeys),
    findall(Items, (
        member(PI, Predicates),
        catch(
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true)], WamText),
              parse_pred_blocks(WamText, Items0),
              iso_errors_rewrite(IsoConfig, PI, Items0, Items)  % <-- new
            ),
            _, fail)
    ), PerPredItems),
    ...
```

`iso_errors_rewrite/4` walks the items for one predicate and rewrites
`builtin_call(Key, N)` lines according to the predicate's mode:

```prolog
iso_errors_rewrite(IsoConfig, PI, Items0, Items) :-
    iso_errors_mode_for(IsoConfig, PI, Mode),
    maplist(iso_errors_rewrite_item(Mode), Items0, Items).

iso_errors_rewrite_item(true,  builtin_call(Key, N), builtin_call(IsoKey, N)) :-
    iso_errors_swap_key(Key, IsoKey), !.
iso_errors_rewrite_item(_, Item, Item).
```

`iso_errors_swap_key/2` is a simple lookup table:

| Lax key | ISO key |
|---|---|
| `is/2` | `is_iso/2` |
| `>/2` | `>_iso/2` |
| `</2` | `<_iso/2` |
| `>=/2` | `>=_iso/2` |
| `=</2` | `=<_iso/2` |
| `=:=/2` | `=:=_iso/2` |
| `=\\=/2` | `=\\=_iso/2` |
| `succ/2` | `succ_iso/2` |

(Operator-name keys with `/2` suffix follow the existing convention.
Builtins not in the table are left unchanged — they're considered
ISO-equivalent or not yet audited.)

## 4. Runtime registrations

Each ISO-flavored builtin lives next to its lax counterpart in
`builtin()`. Example for `is/2`:

```cpp
if (op == "is/2") {
    bool ok = true;
    Value rhs = eval_arith(get_cell("A2"), ok);
    if (!ok) return false;                    // lax: silent failure
    if (!unify_cells(get_cell("A1"),
                     std::make_shared<Cell>(rhs))) return false;
    pc += 1; return true;
}
if (op == "is_iso/2") {
    bool ok = true;
    Value rhs = eval_arith(get_cell("A2"), ok);
    if (!ok) {
        // Build error(type_error(evaluable, <term>), _) and throw.
        throw_iso_error(make_type_error("evaluable",
                                        deref(*get_cell("A2"))));
        return false;
    }
    if (!unify_cells(get_cell("A1"),
                     std::make_shared<Cell>(rhs))) return false;
    pc += 1; return true;
}
```

The two share zero code apart from `eval_arith` — that's
intentional: the lax version's hot path stays untouched.

## 5. Error-term constructors

A small set of helpers in the runtime:

```cpp
struct WamState {
    ...
    // Construct error(<ErrTerm>, _) and dispatch via execute_throw.
    void throw_iso_error(Value err_term);

    // Convenience builders.
    Value make_type_error(const std::string& expected, Value culprit);
    Value make_instantiation_error();
    Value make_domain_error(const std::string& domain, Value culprit);
    Value make_evaluation_error(const std::string& kind);
};
```

`throw_iso_error` sets `A1` to the constructed `error/2` term, then
calls `execute_throw()`. Existing catch/3 machinery handles unwind
and recovery dispatch without changes.

The `Context` slot of `error(ErrType, Context)` is left as a fresh
unbound variable for the v1. Future revisions could carry the
predicate indicator that threw — that's an enhancement, not a
correctness issue.

## 6. Error-term shapes

These are the structures `throw_iso_error` builds for the v1
builtins. The list will grow as more builtins are audited.

| Builtin | Trigger | Thrown term |
|---|---|---|
| `is_iso/2` | RHS contains unbound | `error(instantiation_error, _)` |
| `is_iso/2` | RHS non-evaluable atom `foo` | `error(type_error(evaluable, foo/0), _)` |
| `is_iso/2` | Divide by zero | `error(evaluation_error(zero_divisor), _)` |
| `>_iso/2` etc. | Either arg unbound | `error(instantiation_error, _)` |
| `>_iso/2` etc. | Either arg non-evaluable | `error(type_error(evaluable, X/N), _)` |
| `succ_iso/2` | Both args unbound | `error(instantiation_error, _)` |
| `succ_iso/2` | X negative | `error(type_error(not_less_than_zero, X), _)` |
| `succ_iso/2` | Y zero or negative | `error(domain_error(not_less_than_zero, Y), _)` |

(Atom-form predicate indicators are constructed as `Atom/Arity`
compound terms, matching ISO's recommendation.)

## 7. Migration path

For each existing ISO-relevant builtin:

1. Decide what the ISO error should be (see §6).
2. Add an `if (op == "<key>_iso/N")` branch alongside the existing
   lax `if (op == "<key>/N")`.
3. Add the key to `iso_errors_swap_key/2`.
4. Add e2e tests that:
   - Verify lax behavior is unchanged (existing tests, no edits).
   - Verify ISO behavior with `catch(Goal, error(Pattern, _), ...)`.

The same pattern applies when adding a new builtin: ship both
flavors from the start, or document that the lax flavor is the only
one and gets the lax key.

## 8. Test surface

Two new test categories in `test_wam_cpp_generator.pl`:

```
cpp_e2e_iso_*    — verifies ISO-mode predicates throw the right
                   error terms (catch with specific patterns).
cpp_e2e_lax_*    — verifies lax-mode predicates still fail silently
                   (regression guard against the rewrite accidentally
                   touching lax call sites).
```

Plus a unit test for the config loader:

```
test_iso_errors_config_loader — parses a sample config, verifies
                                 iso_errors_mode_for/3 returns the
                                 right Mode for several predicates
                                 (with and without overrides,
                                 module-qualified and bare).
```

## 9. Files touched (estimated)

- `src/unifyweaver/targets/wam_cpp_target.pl`
  - `iso_errors_*` config-loading + rewrite predicates (~80 lines).
  - Runtime `builtin()` gains `_iso/N` cases (~20 lines per builtin).
  - Runtime gains `throw_iso_error` + `make_*_error` helpers (~60
    lines).
- `tests/test_wam_cpp_generator.pl`
  - 1 config-loader unit test + 1 test per ISO-enabled builtin.
- `docs/design/WAM_CPP_ISO_ERRORS_PHILOSOPHY.md` — this design.
- `docs/design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md` — this design.

## 10. Implementation phasing

This is sized for **three small PRs** so each is reviewable:

1. **Plumbing PR**: config loader, `iso_errors_rewrite/4` machinery,
   key-swap table, `throw_iso_error` runtime helper. No behavior
   change yet — `iso_errors_swap_key/2` returns empty so nothing
   gets rewritten. Smoke-tested by a config-loader unit test.
2. **First builtin PR**: add `is_iso/2` registration and one entry
   to the swap table. Tests cover lax-still-works + ISO-throws.
3. **Sweep PR**: add the remaining arith compares + `succ_iso/2`
   and their swap-table entries. Tests for each.

Future builtins follow the §7 pattern incrementally.
