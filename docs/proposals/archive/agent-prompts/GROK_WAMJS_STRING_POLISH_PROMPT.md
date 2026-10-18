<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — string polish for wam_javascript: `string_length/2` + `writeq`/`~q` quoting (D34 residuals)

> Branch from `grok/wamjs-string-tag` (latest JS-WAM: interpreter + bagof/setof +
> indexing + lowered emitter + builtin breadth + term parser + fact sources +
> term-meta + op/3 + **real string tag**) and EXTEND — do not rebuild. This is
> pure runtime polish on top of the new string tag; keep 48/48 conformance green.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-string-tag`** (the latest JS-WAM state, which just added a
distinct `V.String` term tag) as `grok/wamjs-string-polish`. Prolog dialect is
SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v18+).

### The gap (D34 residuals)
The new string tag left two small, self-contained residuals (both noted in
`docs/WAM_JAVASCRIPT_STATUS.md`):
1. **No `string_length/2`.** SWI's `string_length(+String, -Length)` returns the
   character length of a string (also accepts an atom/number, coercing to text).
2. **`writeq`/`~q` does not quote strings (or atoms that need quoting).** `writeq/1`
   is not registered, and `format('~q', [T])` does not quote nested strings inside a
   term — e.g. `format('~q', ["ab", foo])` should render the string double-quoted and
   quote an atom only when it needs it.

### Goal (match SWI-Prolog exactly)
1. **`string_length/2`** — add the builtin. `string_length("abc", N)` → `N = 3`;
   accepts a string, atom, or number (coerce to its text form) as SWI does.
2. **`writeq/1` + quoted rendering** — register `writeq/1` and make the quoted
   writer (the one `~q` already uses) quote correctly:
   - a **string** renders double-quoted: `writeq("ab")` → `"ab"` (and nested inside a
     term/list: `writeq(["ab", foo])` → `["ab",foo]`).
   - an **atom** is single-quoted **only when it needs it** (contains spaces/special
     chars, is empty, starts with uppercase, is a non-symbolic-atom that wouldn't
     re-read as itself) — e.g. `writeq('hello world')` → `'hello world'`, `writeq(foo)`
     → `foo`, `writeq([])` → `[]`. Reuse SWI's needs-quoting rule as the oracle; a
     pragmatic subset is fine (document what you cover).
   - plain `write/1` is unchanged (string prints its text unquoted, atom bare).
   - `~q` in `format/2,3` uses the same quoted writer, so nested strings in a term now
     quote.

### Reference
- `templates/targets/javascript_wam/runtime.js.mustache` — the term rep (the new
  `V.String` tag, `write`, the existing `~q`/quoted path, the string builtins, the
  builtin dispatch). Extend the writer you already have; don't add a second one.
- SWI-Prolog `string_length/2`, `writeq/1`, and atom-quoting rules are the oracle.
- A mature WAM/runtime's quoted writer (grep the rust/haskell WAM runtimes for how
  `writeq`/quoted output decides atom quoting and renders strings) for the rule set.
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — register `string_length/2`
  and `writeq/1`.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache`: `string_length/2`;
  register `writeq/1`; upgrade the quoted writer so strings render `"…"` and atoms
  quote only when needed, reused by `~q`. `write/1` unchanged.
- Register the two builtins in `src/unifyweaver/bindings/javascript_wam_bindings.pl`.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: move `string_length/2` and `writeq`/`~q`
  string quoting from residual to implemented; note the exact atom-quoting subset you
  cover and any remaining residual.
- Extend `tests/test_wam_javascript_builtins.pl` with probes run under `node`, matched
  to SWI: `string_length("abc", N)` → 3; `writeq(["ab", foo])` → `["ab",foo]`;
  `writeq('hello world')` → `'hello world'`; `write("ab")` still → `ab` (unquoted);
  `format('~q', ["x", y])` quotes the string.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, `wam_target.pl`, or the cross-target conformance harness. (The
  compiled-`"foo"`-literal-collapse residual needs `wam_target.pl` and is OUT OF
  SCOPE here — leave it as a documented residual.)

### Constraints
- Do NOT break the 48/48 conformance, the builtin/parser/fact-source/lowered/term-meta
  probes — re-run all. Interpreter tier; no new runtime dependency. `write/1` output for
  existing conformance queries must not change (they don't use `writeq`).

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green incl. the new probes.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` and
   `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. Show `string_length`, `writeq` (string + quoted atom + bare atom), plain `write` on a
   string, and `~q` nested-string probes' Node output next to SWI's and confirm they match.

### Handoff format
Return: the changed runtime/bindings files, the extended test file, the updated status doc,
`INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-string-tag` and EXTENDED it (net additions), the atom-quoting subset you
implemented, and any residual.

## ↑↑↑ Copy to here
