<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P7 follow-up: regex `match/2[,3]` guard codegen (for INT-0)

**Task:** complete the G-P7 guard story by adding the explicitly-deferred regex
`match/2,3` guard to the TypeScript and Clojure pattern targets (negation +
type-checks landed as D24). `match/2,3` is UnifyWeaver's regex-match predicate.

**Worktree:** `agent-a4d6175fdde69b302`

**Shared-file rule:** this agent did **NOT** edit
`src/unifyweaver/core/clause_body_analysis.pl`, any `wam_*` file, the
`annotated_js`/`vanilla_js` targets, `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, or `glue/js_glue.pl`.

## `match` semantics found (from `docs/MATCH_PREDICATE.md` / `docs/AWK_MATCH_PREDICATE.md`)

- **Argument order: subject FIRST, pattern SECOND** — `match(String, Pattern)`.
  (The task prompt's illustrative `match('^a.*', X)` was reversed; the docs and
  `python_target.pl:translate_match/6` both use `match(Var, Pattern)`.)
- **`match/3` third arg = regex TYPE** (`auto | ere | bre | awk | python | pcre`),
  a dialect hint — **not** flags and **not** a capture list. Captures are the 4th
  arg of `match/4` (not implemented here).
- **Truthiness = unanchored search.** The Python reference renders `match/2,3` as
  `re.search(pattern, str(x))` (boolean). Anchoring is expressed *in the pattern*
  (`^`, `$`). match/4 capture handling is out of scope (Python itself only does
  boolean `re.search` for match/2,3 today).

## What landed

Both `match/2` **and** `match/3` render. The `match/3` regex-type argument is
treated as **advisory**: the generated code uses the host's native regex engine
(JS ECMAScript `RegExp`; JVM `java.util.regex` / CLJS `js/RegExp`), exactly as
Python's translator uses `re` regardless of the declared type. Dialect
translation (e.g. POSIX `bre`/`awk` → ECMAScript) is **not** performed — a
`match/4` + dialect-translation follow-up can layer on top. This matches the
existing cross-target reality documented in `MATCH_PREDICATE.md` (regex flavors
differ per target; `auto` is the portable choice).

| Prolog guard | TypeScript | Clojure / CLJS |
|---|---|---|
| `match(X, Pat)` | `new RegExp("<Pat>").test(x)` | `(re-find (re-pattern "<Pat>") x)` |
| `match(X, Pat, Type)` | same (Type advisory) | same (Type advisory) |
| `\+ match(X, Pat)` | `!(new RegExp("<Pat>").test(x))` | `(not (re-find (re-pattern "<Pat>") x))` |

`re-find` / `RegExp.test` are both unanchored searches returning a truthy value,
matching Python's `re.search`. Backslash regex escapes are preserved into the
host string literal (`\d+` → `"\\d+"`); double-quotes are escaped.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/typescript_target.pl` | Added `ts_guard_condition/3` clauses for `match/2` and `match/3`; helpers `ts_match_condition/4`, `ts_regex_pattern_string/2` + `ts_regex_escape_chars/2`, and `ts_body_match_subject/2`. Introduced `RegExp` cleanly (this file had none before). Made the arity-1 streaming filter **input-type aware**: `ts_stream_input_ts_type/3` types the input line as `string` (and coerces via `trimmed`, not `Number(trimmed)`) when the body regex-matches it — otherwise the numeric default is unchanged. `ts_streaming_filter_module/5` gained the `InType` parameter. |
| `src/unifyweaver/targets/clojure_target.pl` | Symmetric `clojure_guard_condition/3` clauses for `match/2,3`; helpers `clojure_match_condition/4`, `clojure_regex_pattern_string/2` + `clojure_regex_escape_chars/2`, and `clojure_body_match_subject/2`. Extended `clojure_native_cli_entry/4` with a `clojure_pred_string_input/2` case so a predicate that regex-matches its first arg receives the raw argv **string** (not `Integer/parseInt`), enabling text matching under bb/nbb. |
| `tests/core/test_typescript_target.pl` | Added a `G-P7 follow-up` section: structural tests (RegExp emitted, string input typing, backslash escaping, `\+` composition, `match/3`) and two `condition(node_available)` node-execution tests vs a `library(pcre)` `re_match/2` oracle. Added `:- use_module(library(pcre))`. |
| `tests/core/test_clojurescript_target.pl` | Added a `G-P7 follow-up` section: structural tests plus two `condition(nbb_available)` nbb-execution tests vs the same pcre oracle. Added `:- use_module(library(pcre))`. |

The `annotated_js` and `vanilla_js` targets inherit the TS scaffold change (they
delegate to TS then type-strip / JSDoc-rewrite); ClojureScript inherits the
Clojure clauses (`re-find`/`re-pattern` need no interop rewrite). All four
inheriting suites stay green.

## Routing: how positive `match` reaches the guard renderer

`match/2,3` is **not** classified as a guard by
`clause_body_analysis:is_guard_goal/2` (verified: `is_guard_goal(match(_,_),[])`
fails). So it reaches the per-target guard renderer through the two paths that do
**not** consult `is_guard_goal`:

1. **TS arity-1 streaming filter** (`mode(pipeline)`/`mode(generator)`) — passes
   every body goal straight to `ts_guard_condition/3`.
2. **If-then-else condition position** (both targets) — the condition of
   `(Cond -> Then ; Else)` is rendered via the guard renderer regardless of
   classification.

`\+ match(...)` additionally composes in **every** path (including batch native
lowering), because the D24 negation clause `is_guard_goal(\+(_), _)` already
classifies any negated goal as a guard, and the renderer's `\+` clause recurses
into the new `match` clause. **No change to the D24 negation clause was needed.**

## OPTIONAL shared-file change (INT-0 decision — not required for this slice)

A **bare positive** `match/2,3` guard in the **default batch native-clause
lowering** path (e.g. `p(X) :- match(X, '^a').` compiled with `[]`) is currently
**not** rendered: `is_guard_goal/2` does not recognize `match`, so the batch path
treats it as an ordinary call and aborts with
`existence_error(procedure, match/2)`. This mirrors the D24 status of positive
`member` (only reached via negation / control flow).

To make bare positive `match` a first-class batch guard, add to
`src/unifyweaver/core/clause_body_analysis.pl` (shared file — intentionally left
untouched here):

```prolog
%% Regex match/2,3 as a boolean guard (subject FIRST, pattern SECOND).
is_guard_goal(match(_, _), _) :- !.
is_guard_goal(match(_, _, _), _) :- !.
```

placed among the other `is_guard_goal/2` clauses (after the type-check clause,
before the implicit failure). This is purely additive and behavior-preserving for
non-`match` clauses. With it, `match` also flows through the batch path and
`clause_guard_output_split/4`. The acceptance criteria for this slice are met
without it via the two routing paths above, so it is offered as a follow-up, not
applied.
