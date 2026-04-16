# PR Title

`feat(wam-go): add CallForeign parity for hybrid WAM execution`

# PR Description

## Summary

Adds the first complete Go-side `CallForeign` path for the hybrid WAM target.

This closes the biggest remaining architectural gap after the earlier Go hybrid
WAM parity work: Go can now route selected WAM calls through native foreign
execution instead of forcing everything through ordinary interpreted WAM
dispatch.

## What Changed

### `CallForeign` instruction support

Added a Go WAM instruction variant for foreign dispatch:

- `CallForeign`

The runtime step loop now recognizes this instruction and executes the
registered foreign predicate handler directly.

### Foreign registration/runtime plumbing

Added Go runtime support for registering foreign predicate metadata:

- native kind
- result layout
- result mode
- string config
- usize config
- indexed atom fact pairs

This gives the Go WAM runtime the same basic setup model already used by the
hybrid Rust/Haskell implementations.

### Foreign wrapper/codegen support

The Go target can now compile explicit `foreign_predicate(...)` specs into:

- wrapper-level foreign setup code
- direct `executeForeignPredicate(...)` entry paths
- `call` / `execute` rewrite to `CallForeign` when the target matches a
  rewrite target in the foreign spec

Foreign-lowered WAM predicates are compiled separately from the shared WAM
table path, matching the same architectural split used in Rust.

### Result handling and backtracking

Added Go foreign result handling for:

- deterministic results
- stream results with choice-point-backed backtracking resume

Choice points now carry enough foreign state to resume multi-result foreign
execution correctly on backtrack.

### Initial Go foreign kernel coverage

This PR adds the first Go-native foreign kernels for the hybrid WAM runtime:

- `countdown_sum2`
- `list_suffix2`
- `list_suffixes2`
- `transitive_closure2`

This is intentionally a scoped first slice, not full Rust-kernel parity.

## Tests

Added focused tests covering:

- foreign wrapper generation from explicit specs
- `call` / `execute` rewrite to `CallForeign`
- deterministic and stream foreign execution behavior in generated Go projects

Verified locally with:

```sh
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
```

## Why This Matters

The earlier Go hybrid parity work fixed shared WAM assembly, compile-time
control-flow resolution, and stronger backtracking state restoration. The next
missing piece was foreign dispatch parity.

Without this change, Go still had to interpret work that the Rust/Haskell
hybrid WAM targets already hand off to native kernels. This PR establishes that
same execution model for Go.

## Scope Boundaries

This PR does **not** yet add full Go foreign-lowering parity with Rust.

Still missing:

- `foreign_lowering(true)` auto-detection parity
- broader foreign kernel coverage
- grouped/tuple-heavy result shaping beyond this first slice
- weighted / A* / distance kernel parity

## Files Changed

- `src/unifyweaver/targets/wam_go_target.pl`
- `templates/targets/go_wam/instructions.go.mustache`
- `templates/targets/go_wam/state.go.mustache`
- `tests/test_wam_go_foreign_lowering.pl`
