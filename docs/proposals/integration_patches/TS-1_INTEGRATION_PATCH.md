<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# TS-1 Integration Patch (for INT-0 to apply centrally)

**Task:** TS-1 — TypeScript hardening (bindings count + matrix row, test coverage).
**Worktree:** `agent-a3b21199cdd47f72a`
**Shared-file rule:** this agent did NOT edit `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `core/advanced/test_advanced.pl`, or `glue/js_glue.pl`.
All changes those files need are captured below.

---

## 1. `docs/BINDING_MATRIX.md` — add the TypeScript row

There is currently **no TypeScript row** in the Summary table. Add this row to
the `## Summary` table (near the top, e.g. above the Python row since it is now
the richest target):

```markdown
| **TypeScript** | 179 | 13 (Built-ins, String, Math, Array, Object, JSON, Console/I-O, Node fs, Node path, Promise/Async, Date, Collections (Map/Set), Number) |
```

**Count is registry-verified:** `test_typescript_bindings/0` reports
`Total bindings registered: 179` (the `.pl` file has 180 `declare_binding(typescript, …)`
lines; one is the `term_expansion/2` directive-support template, not a real binding).

- **Before TS-1:** 154 bindings, 11 categories.
- **After TS-1:** 179 bindings, 13 categories (+25 bindings; +2 categories).

New in TS-1 (`bindings/typescript_bindings.pl`):
- **Collections (Map/Set)** — 18: `map_new/1`, `map_from/2`, `map_get/3`,
  `map_has/3`, `map_size/2`, `map_keys/2`, `map_values/2`, `map_entries/2`,
  `map_set/4`, `map_delete/3`, `map_clear/1`, `set_new/1`, `set_from/2`,
  `set_has/3`, `set_size/2`, `set_add/3`, `set_delete/3`, `set_clear/1`.
  (Constructors + reads are `pure`; `set`/`add`/`delete`/`clear` are `effect(mutation)`.)
- **Number formatting** — 7: `number_to_fixed/3`, `number_to_precision/3`,
  `number_to_string_radix/3`, `number_is_integer/2`, `number_is_safe_integer/2`,
  `number_parse_int/3`, `number_parse_float/2`. (All `pure, deterministic, total`.)

(Optional, if a TS column is later added to the detailed per-binding tables:
TypeScript implements `abs/2`, `sqrt/2`, `round/2`, `floor/2`, `ceil/2`, `pow/3`,
`log/2`, `log10/2`, `sin/2`, `cos/2`, `tan/2`, `min/3`, `max/3`, the full string
method set, `json_parse/2`/`json_stringify/2`, `path_*`, `read_file*`/`write_file*`,
etc. — see the category table in `docs/TYPESCRIPT_TARGET.md#bindings`.)

---

## 2. `core/advanced/test_advanced.pl` — no change required

`typescript_target` is **already** loaded there (line 39:
`:- use_module('../../targets/typescript_target', [])`), which registers its
multifile recursion-pattern dispatch clauses. The suite tests the compiler
generically rather than iterating a per-target codegen list, so no TS-specific
matrix entry is needed. No patch.

---

## 3. `core/target_registry.pl` — no change required

`typescript` is already registered (js family). TS-1 adds no new target. No patch.

---

## 4. `glue/js_glue.pl` — no change required

TS-1 does not change runtime selection. No patch.

---

## 5. PAR-1 arm (when the parity harness lands)

Register `typescript` as an arm in the future `PAR-1`
(`tests/test_js_pattern_cross_target_conformance.pl`). TS emits a Node-runnable
CLI entry point (`if (process.argv[2]) …`) for the numeric recursion patterns,
so the oracle diff can invoke `node` on the generated `.ts`/transpiled output.
No file exists yet — this is a forward note, not a patch.

---

## 6. Findings for INT-0 (not fixed here — out of TS-1 scope)

Two pre-existing correctness issues in `targets/typescript_target.pl` surfaced
while writing the tests. TS-1 does **not** change codegen, so tests assert only
against current behavior; flagging for a follow-up card:

1. **`compile_facts/3` emits an empty fact array.** `format_ts_tuple/2` calls
   `generate_field_names(length(Args, L), L, FieldNames)` — `L` is never bound by
   an actual `length/2` call, so the `generate_field_names(_, 0, [])` clause
   unifies `L=0` and yields `FieldNames=[]`; the subsequent `maplist/3` against
   the (non-empty) `ArgStrs` then fails, so every fact row is dropped and
   `parentFacts`/`colorFacts` come out `[]`. The interface, query helper, and
   membership helper are still generated. Fix: bind arity before generating field
   names (e.g. `length(Args, L), generate_field_names(L, FieldNames)`).
2. **`isXxx` membership helper embeds a Prolog term.** In `compile_facts/3` the
   final format argument is `generate_match_expr(FieldNames)` (a compound term,
   not a call), so `~w` prints `generate_match_expr([arg1,arg2])` verbatim into
   the generated `.some(f => …)` body. A `generate_match_expr/2` variant already
   exists (unused) that returns the real `f.argN === argN && …` expression;
   call that instead.
3. **`transitive_closure` has no dedicated generator in `compile_recursion/3`.**
   It is declared in `target_info/1.recursion_patterns` but falls through to the
   `tail_recursion` branch. Genuine transitive-closure lowering only happens via
   the general-recursive multifile hook. Consider either wiring
   `pattern(transitive_closure)` to that hook or dropping it from `target_info/1`.

---

## Files changed by TS-1 (this worktree only)

- `src/unifyweaver/bindings/typescript_bindings.pl` — +`register_collection_bindings`,
  +`register_number_bindings` (wired into `init_typescript_bindings/0`); +25 bindings.
- `tests/core/test_typescript_target.pl` — **new** plunit suite (14 tests).
- `docs/TYPESCRIPT_TARGET.md` — new Bindings + Test Coverage sections.
- `docs/proposals/integration_patches/TS-1_INTEGRATION_PATCH.md` — this file.

## Acceptance (verified in-worktree with SWI-Prolog)

```
$ swipl -q -g test_typescript_target -t halt tests/core/test_typescript_target.pl
..............
```

14 tests, all pass, exit 0, no warnings. The pre-existing
`tests/core/test_typescript_native_lowering.pl` and
`test_typescript_bindings/0` also still pass (self-test reports 179 bindings).
