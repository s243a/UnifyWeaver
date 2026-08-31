<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — compiled string literals reach the JS WAM runtime as real strings (D34/D35 residual)

> Branch from `grok/wamjs-string-polish` (latest JS-WAM: interpreter + string tag +
> string_length/writeq) and EXTEND — do not rebuild. **This one edits TWO SHARED
> files** (`wam_target.pl`, `wam_text_parser.pl`) that every WAM target uses, so the
> safety rule below (default-to-atom + full-fleet conformance) is the whole point.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-string-polish`** (the latest JS-WAM state) as
`grok/wamjs-string-literal`. Prolog dialect is SWI-Prolog (`swipl`); the JS runtime
targets Node (`node`, v18+).

### The gap (D34/D35 residual)
The JS WAM runtime now has a distinct `V.String` tag and string builtins, BUT a
**source-level string literal** written in a compiled clause — e.g. `greet("hi").` or
`p(X) :- X = "hi".` — still reaches the runtime as the **atom** `hi`. Reason:
`wam_target:quote_wam_constant/2` lowers a Prolog `string` value to the *same* textual
token as an atom (no string/atom distinction survives into the bytecode), and the shared
classifier `wam_text_parser:wam_classify_constant_token/2` only ever returns
`atom | integer | float`. So `string("hi")` is false for a compiled `"hi"` literal even
though `atom_string(hi, S)` at runtime correctly yields a string.

### The design — DO IT THIS SAFE WAY (do not invent a `string(_)` Class)
`wam_classify_constant_token/2` is consumed by ~10 targets (python/go/haskell/r/lua/
scala/javascript/elixir emitters + the text parser) across ~19 runtime families. If you
add a new `string(Name)` return Class, every consumer that pattern-matches
`atom/integer/float` without a catch-all breaks. So instead:

1. **`wam_target:quote_wam_constant/2`** — when `string(Value)` (and NOT atom/number),
   emit a **double-quoted** token `"..."` (escape `\\` and `\"`); atoms stay bare/single-
   quoted exactly as today, numbers unchanged. This is the *only* change to how constants
   are spelled, and it only affects values that are Prolog strings.
2. **`wam_text_parser:wam_classify_constant_token/2`** — recognize an outer double-quoted
   token, **strip the quotes, and STILL return `atom(Name)`**. This keeps every existing
   consumer byte-for-byte unchanged (a `"hi"` literal degrades to `atom(hi)` in every
   runtime that isn't string-aware — exactly today's behavior). Add and **export** a
   companion semidet `wam_constant_token_is_string(+Token)` that is true iff the raw token
   has outer double-quotes — the *only* new signal.
3. **JS path only** (`wam_javascript_target.pl` and/or the JS lowered emitter + the
   `put_constant`/`get_constant`/`set_constant` handling in
   `templates/targets/javascript_wam/runtime.js.mustache`) — where a constant token
   satisfies `wam_constant_token_is_string/1`, build a `V.String` instead of interning an
   atom. Thread the raw token (or a per-constant `is_string` flag) through to the JS
   emitter; no other runtime consults the new signal, so no other runtime changes.

### Reference
- `src/unifyweaver/targets/wam_target.pl` — `quote_wam_constant/2` (line ~1215) and the
  quote-state contract comment above it.
- `src/unifyweaver/targets/wam_text_parser.pl` — `wam_classify_constant_token/2` (the
  shared classifier; the single-quote→atom precedent shows exactly where to add the
  double-quote branch and the companion predicate).
- `templates/targets/javascript_wam/runtime.js.mustache` — the `V.String` tag and the
  constant-building path.
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — dispatch if needed.
- SWI-Prolog (`double_quotes=string`, so `"hi"` reads as a string) is the oracle.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- `wam_target.pl` — double-quoted spelling for `string` constants only.
- `wam_text_parser.pl` — double-quote branch returns `atom(Name)` (quotes stripped);
  new exported `wam_constant_token_is_string/1`.
- The JS emitter/runtime — build `V.String` for string-tokened constants.
- `docs/WAM_JAVASCRIPT_STATUS.md` — move the "compiled `"foo"` literals collapse to
  atom" residual to implemented; note the shared-file design.
- `tests/test_wam_javascript_builtins.pl` — a probe compiling a clause with a `"hi"`
  literal, run under `node`, asserting `string(X)` true and value `hi` (vs SWI).
- `INTEGRATION_PATCH.md` — **REQUIRED** here (you edited two shared files). List the
  exact `quote_wam_constant/2` and `wam_classify_constant_token/2` changes, state that
  the classifier still returns `atom(_)` for double-quoted tokens (no new Class), and
  record the full-fleet conformance result below.

### Constraints — CRITICAL (shared files)
- The classifier MUST keep returning `atom | integer | float` (double-quoted ⇒ `atom`).
  Do NOT add a `string(_)` Class. `wam_constant_token_is_string/1` is the only new signal
  and only the JS path reads it.
- No other runtime's emitted output may change. Interpreter tier; no new dependency.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green incl. the new literal probe.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` and
   `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` → green.
3. **FULL-FLEET conformance, not just JS:** run the cross-target conformance with EVERY
   registered target, both with and without the JS arm —
   `swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` (all
   default targets) AND
   `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl`
   — BOTH still fully green, no xfail/skip/regression on ANY target. This is the guardrail
   that proves the shared-file change didn't break rust/haskell/lua/elixir/go/python/etc.
4. Show a compiled `"hi"` literal under `node`: `string(X)` true, `atom(X)` false, value
   `hi`, next to SWI, and confirm they match.

### Handoff format
Return: the two changed shared files, the JS emitter/runtime changes, the extended test
file, the updated status doc, the REQUIRED `INTEGRATION_PATCH.md`, and a note stating you
branched from `grok/wamjs-string-polish` and EXTENDED it (net additions), that the
classifier still returns `atom(_)` for double-quoted tokens (no consumer breakage), the
full-fleet conformance result, and any residual.

## ↑↑↑ Copy to here
