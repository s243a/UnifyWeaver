<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# A3 — Pushing a real Prolog program through the pattern targets

**Step A3 of the UnifyWeaver transpilation maturity demonstration.**
Subject: `examples/cli_args/cli_args.pl` — the oracle-verified Prolog
reimplementation of peerhailer's `src/cliArgs.js` produced in step A1
(17/17 corpus tests, 5067-line differential, 0 divergences).
Targets: `typescript_target.pl` and its two inheritors, `annotated_js_target.pl`
and `vanilla_js_target.pl`.

The question this step answers is **not** "can we ship a transpiled CLI parser".
It is "where exactly is the boundary of the pattern compilers", stated precisely
enough to be a punchlist. The catalogue in §4 is the deliverable.

---

> **STATUS UPDATE (post-A3 follow-up run).** A second run closed five of the open
> gaps; the sections below are the original A3 snapshot and are NOT rewritten.
> Now **CLOSED** (fixes + regression tests in `test_typescript_cli_args_shapes.pl`,
> 55 tests): **G-A3-8** (fact fallback no longer executes the predicate — genuine
> ground-fact predicates only, everything else refuses fast (≤12 ms, was 20 s/1.5 GB)
> with an actionable `unsupported_lowering` error; guard inside `compile_facts/3`,
> inherited by AJS/VJS), **G-A3-13** (`true`/`false` emit as JS booleans),
> **G-A3-14** (mid-sequence ITE bindings get stable `v<N>` names via let+assign;
> unmapped variables refuse loudly instead of leaking `_G…`), **G-A3-15**
> (reversible builtins honour the head output slot; both-known renders a check),
> **G-A3-11.1/.2/.4** (arity-1 semidet signature `('arg1: any') → boolean` with
> fall-through `return false`; CLI entry passes every argument with per-token
> coercion; `compile_module/3` refuses an all-unsupported module and emits a
> WARNING banner for partial ones).
> Still OPEN: **G-A3-6** (guard hoisting, M), **G-A3-9** (multi-output loops, L),
> **G-A3-10** (ITE in recursive bodies, M), **G-A3-12** (compound terms, M),
> **G-A3-16** (list/pair head patterns, M), and NEW **G-A3-11.3** (parameter/CLI
> types are hardcoded guesses; needs body-driven type inference — probe
> `gap_g_a3_11_3_cli_entry_cannot_pass_a_numeric_looking_character` pins it).
> Priority order stands: G-A3-10 → G-A3-9 (then all four parse mechanisms become
> expressible and A4's differential becomes a live gate).

---

## 0. Headline

| | |
| --- | --- |
| Predicates in `cli_args.pl` | **43** |
| Lower **correctly** (verified running under node against the SWI oracle) | **2** |
| Lower to code that is **wrong or unusable** | **3** |
| Reach a lowering path but **refuse loudly** (explicit `throw`, after this step's fixes) | **21** |
| Reach **no** lowering path — the dispatcher falls back to `compile_facts/3`, which *runs* the predicate | **17** (4 of them yield a wrong fact table; 13 error or run away) |
| Named gaps catalogued | **17** (`G-A3-1` … `G-A3-17`) |
| Gaps closed in this step | **7** (G-A3-1, -2, -3, -4, -5, -7, -17) |
| Fraction of the parser transpilable today | **≈5 %** by predicate count; **0 %** by mechanism — none of the four engines runs |

Before this step the numbers were 2 predicates on a lowering path and 41 on
none; of those 2, **both emitted syntactically invalid TypeScript** that node
refuses to parse. The step's fixes moved 21 predicates from "silently produces
nothing usable" to "says out loud which goal it cannot lower", and made the two
substring helpers genuinely correct.

**The honest summary: the pattern targets cannot transpile this program, and are
not close.** The distance is not a missing builtin or two — it is that
`typescript_target.pl` has no general clause-body compiler. What it has is a set
of shape recognisers (facts, guard-only clauses, six recursion patterns,
aggregates, data sources) plus a best-effort expression renderer, and
`cli_args.pl` is a program of ordinary first-order Prolog that sits outside all
of them.

---

## 1. Inventory of `cli_args.pl`

43 predicates. Grouped by the shape that decides how a pattern target sees them.

### 1.1 Data (ordered assoc lists of compound terms)

| predicate | shape |
| --- | --- |
| `global_options/1` | one fact; body is a list of `Name-Kind` pairs |
| `default_registry/1` | one fact; a 7-entry list of `Name-schema(Options,Positionals)` / `Name-group(Actions)`, nested two deep |
| `js_object_prototype_keys/1` | one fact; a 12-element string list |

These are *facts*, but their single argument is a nested term, not a scalar
tuple — the shape `compile_facts/3` is built for.

### 1.2 Character logic (the two regexes, re-expressed)

| predicate | shape |
| --- | --- |
| `is_long_flag/1` | `string_chars/2` + list pattern `['-','-',First|Rest]` + helper calls; semidet |
| `looks_like_legacy_flag/1` | same |
| `long_flag_tail/1`, `legacy_flag_tail/1` | list recursion, semidet, no output argument |
| `js_alpha/1` | `char_code/2` + an if-then-else chain over code-point ranges; semidet |
| `js_flag_char/1` | same, 4 branches |

### 1.3 String / list helpers

| predicate | shape |
| --- | --- |
| `starts_with/2` | 4 string builtins + 2 guards; **semidet, no output argument** |
| `substring_from/3` | `string_length` → `is` → `sub_string`; det, last arg is the output |
| `substring_range/4` | `is` → `sub_string`; det |
| `first_equals_index/2` | `string_chars` + a helper walk |
| `first_char_index/4` | list recursion with an index accumulator and an if-then-else |
| `split_flag_token/3` | if-then-else producing **two** outputs, one of them `some(V)`/`none` |
| `string_member/2` | list recursion, semidet, `==`-based |
| `pair_lookup/3` | list recursion over `K-V` pairs, if-then-else, semidet |
| `nth0_default/4` | single clause, nested if-then-else, integer first argument |
| `last_element/2` | list recursion, if-then-else |
| `strip_brackets/2` | `string_chars` both directions around a helper |
| `drop_brackets/2` | list recursion building a list, if-then-else, disjunctive test |

### 1.4 Flag maps (JS object-assignment semantics)

| predicate | shape |
| --- | --- |
| `flags_set/4` | if-then-else guarding a helper |
| `flags_put/4` | list recursion over `K-V`, **builds a list**, if-then-else |
| `merge_flags/3`, `merge_flags_/3` | accumulator loop over `K-V` pairs |

### 1.5 Schema resolution

| predicate | shape |
| --- | --- |
| `schema_for/5` | if-then-else, **two outputs**, compound terms `group(_)` / `schema(_,_)` |
| `registry_entry/3`, `action_entry/3` | if-then-else, compound output |
| `option_kind/3` | 3-way if-then-else chain |
| `js_object_prototype_key/1` | helper + membership |
| `is_global_key/1` | if-then-else over two helper calls; semidet |

### 1.6 The four engines

| predicate | shape |
| --- | --- |
| `lenient_loop/5` | **tail-recursive loop, 2 accumulators, 2 outputs**, body is a 3-way if-then-else with `\+` |
| `strict_loop/8` | **tail-recursive loop, 2 accumulators + a status output, 3 outputs**, 4-way if-then-else chain |
| `strict_option/11` | nested if-then-else, 11 arguments, calls back into `strict_loop/8` |
| `scan_leading_globals/4` | tail-recursive loop, 1 accumulator, 2 outputs, nested if-then-else |
| `check_arity/3` | `length/2` ×2, if-then-else chain, `string_concat`, tagged output |
| `count_required/3` | accumulator loop with an if-then-else chain |
| `parse_lenient/3` | wrapper: loop + `reverse/2` |
| `parse_strict/4` | wrapper: loop + `reverse/2` + `append/3` + tagged outcome |
| `parse_args/2`, `parse_args/3` | top-level; deeply nested if-then-else, tagged results |
| `next_value/2`, `lenient_result/2` | small if-then-else wrappers |

### 1.7 Construct census

Every predicate is det or semidet; there are no cuts, no exceptions, no
`assert`/`retract`, no `library(pcre)`, no `library(apply)`.

| construct | predicates using it |
| --- | --- |
| if-then-else (`->`/`;`) | 24 |
| list/pair head patterns (`[H|T]`, `K-V`) | 11 |
| string builtins (`string_length`, `sub_string`, `string_concat`, `string_chars`, `char_code`) | 12 |
| compound terms as values (`ok/2`, `err/1`, `some/1`, `none`, `schema/2`, `group/1`) | 9 |
| more than one output argument | 7 |
| no output argument at all (semidet test) | 9 |
| tail recursion with ≥1 accumulator | 8 |
| `length/2`, `reverse/2`, `append/3` | 3 |

---

## 2. Method

`cli_args.pl` is a module; the compilers read clauses via `user:clause/2`, so a
module-declaration-stripped copy was consulted into `user` and each predicate
pushed through `typescript_target:compile_predicate_to_typescript/3`.

**One methodological finding shaped the rest of the work.** Running the full
dispatcher on every predicate is *not safe*: its last resort is
`compile_facts/3`, which enumerates "facts" by **calling the goal**. On
`flags_put/4` the resulting `findall/3` grew without bound — the process was at
540 MB and climbing when it was killed. All subsequent probing therefore calls
the individual lowering paths (`native_ts_structural/3`,
`native_ts_clause_body/3`, `ts_aggregate_predicate/3`) directly, and treats
"reaches `compile_facts/3`" as its own outcome. This is G-A3-8.

`annotated_js_target` and `vanilla_js_target` were checked separately per
predicate rather than assumed to inherit — which is how G-A3-7 was found.

---

## 3. Compile matrix

`path` is the lowering path that claims the predicate; `result` grades the
output. Measured after this step's fixes.

| # | predicate | path | result |
| --- | --- | --- | --- |
| 1 | `global_options/1` | facts fallback | **wrong** — one row, `arg1: "[state-string,name-string]"` |
| 2 | `default_registry/1` | facts fallback | **wrong** — the whole registry flattened into one string |
| 3 | `js_object_prototype_keys/1` | facts fallback | **wrong** — one stringified row |
| 4 | `js_object_prototype_key/1` | native clause | loud refusal |
| 5 | `is_long_flag/1` | native clause | loud refusal (`=/2`, `js_alpha/1`, `long_flag_tail/1`) |
| 6 | `long_flag_tail/1` | facts fallback | **instantiation_error** |
| 7 | `looks_like_legacy_flag/1` | native clause | loud refusal |
| 8 | `legacy_flag_tail/1` | facts fallback | **instantiation_error** |
| 9 | `js_alpha/1` | native clause | **wrong** — guard is correct, but the function takes 0 parameters and returns the char code |
| 10 | `js_flag_char/1` | native clause | **wrong** — same |
| 11 | `starts_with/2` | native clause | **wrong** — guard hoisted above the `const`s it reads (G-A3-6) |
| 12 | `substring_from/3` | native clause | **CORRECT** — verified under node |
| 13 | `substring_range/4` | native clause | **CORRECT** — verified under node |
| 14 | `first_equals_index/2` | native clause | loud refusal (`first_char_index/4`) |
| 15 | `first_char_index/4` | facts fallback | **instantiation_error** |
| 16 | `split_flag_token/3` | native clause | loud refusal |
| 17 | `string_member/2` | facts fallback | **resource_error(stack)** |
| 18 | `pair_lookup/3` | native clause | loud refusal + stringified head pattern (G-A3-16) |
| 19 | `nth0_default/4` | native clause | loud refusal |
| 20 | `last_element/2` | native clause | loud refusal |
| 21 | `flags_set/4` | native clause | loud refusal |
| 22 | `flags_put/4` | facts fallback | **unbounded findall — process killed** |
| 23 | `merge_flags/3` | native clause | loud refusal |
| 24 | `merge_flags_/3` | facts fallback | **unbounded findall — process killed** |
| 25 | `schema_for/5` | native clause | loud refusal |
| 26 | `registry_entry/3` | native clause | loud refusal |
| 27 | `action_entry/3` | native clause | loud refusal |
| 28 | `parse_lenient/3` | native clause | loud refusal |
| 29 | `lenient_loop/5` | facts fallback | **instantiation_error** |
| 30 | `parse_strict/4` | native clause | loud refusal |
| 31 | `strict_loop/8` | facts fallback | **instantiation_error** |
| 32 | `strict_option/11` | native clause | loud refusal |
| 33 | `next_value/2` | facts fallback | **instantiation_error** |
| 34 | `option_kind/3` | native clause | loud refusal |
| 35 | `check_arity/3` | native clause | loud refusal |
| 36 | `count_required/3` | facts fallback | **instantiation_error** |
| 37 | `strip_brackets/2` | native clause | loud refusal (`drop_brackets/2`) |
| 38 | `drop_brackets/2` | facts fallback | **unbounded findall — process killed** |
| 39 | `parse_args/2` | native clause | loud refusal |
| 40 | `parse_args/3` | native clause | loud refusal |
| 41 | `lenient_result/2` | facts fallback | **instantiation_error** |
| 42 | `scan_leading_globals/4` | facts fallback | **instantiation_error** |
| 43 | `is_global_key/1` | facts fallback | **wrong** — an EMPTY fact table, i.e. the predicate compiles to "never true" |

**Totals by lowering path (43 = 26 + 17):**

* **26 reach a lowering path** (all of them the native clause-body path; the
  structural, aggregate and streaming paths claim nothing in this program):
  **2 correct**, **3 wrong**, **21 loud refusal**.
* **17 reach no lowering path** and fall through to `compile_facts/3`:
  **4 produce a wrong fact table**, **10 raise `instantiation_error` /
  `resource_error(stack)`**, **3 run an unbounded `findall/3` until killed**.

**Wrong output, all sources: 7 of 43** — `js_alpha/1`, `js_flag_char/1`,
`starts_with/2` from the clause-body path, and `global_options/1`,
`default_registry/1`, `js_object_prototype_keys/1`, `is_global_key/1` from the
fact fallback.

### 3.1 Inheritance

`vanilla_js_target` inherits **identically** for every predicate: the same
lowering, with TypeScript type syntax stripped. Byte-comparison of the function
bodies confirmed.

`annotated_js_target` inherited identically **after** G-A3-7 was fixed. Before
the fix it *refused* `substring_from/3` and `substring_range/4` outright — the
two predicates that lower correctly — because its TS→JSDoc line walker mistook
the generated line `const v5 = (v4 - arg2);` for the opening line of a
multi-line arrow-function signature and swallowed the rest of the file.

### 3.2 `compile_module/3`

`compile_module/3` does **not** compile clauses at all. It dispatches on a
declared pattern type (`tail_recursion`, `list_fold`, `linear_recursion`,
`factorial`) via `generate_pred_code_ts/4`; any other type — including `facts` —
matches no clause and is silently dropped by the enclosing `findall/3`. Asking
it for the `cli_args` module produces:

```
// Generated by UnifyWeaver TypeScript Target
// Module: cliArgs


```

An empty module, with no error. That is G-A3-11's sibling and is folded into
the catalogue as **G-A3-11**.

---

## 4. Gap catalogue

Sizes: **S** = a table entry or one clause; **M** = a contained restructuring of
one renderer; **L** = new machinery.

Each entry names the machinery it extends. Reproductions live in
`tests/core/test_typescript_cli_args_shapes.pl` — the closed gaps as
regression tests, the open ones as probes that pin the current behaviour and
will FAIL when the gap is closed.

---

### G-A3-1 — No lowering for deterministic string/char builtins · **S** · CLOSED

**Extends:** the expression/output renderer (`ts_output_goal/4`,
`ts_output_goal_last/3`, `ts_branch_value/3`).

**Trigger**
```prolog
p(S, L) :- string_length(S, L).
```

**Was:** `string_length/2`, `sub_string/5`, `string_concat/3`, `string_chars/2`,
`char_code/2` had no rendering anywhere. The goal was silently deleted and the
surrounding code went on to reference a variable nothing assigned.

**Correct lowering:** `const arg2 = arg1.length;`

**Fix landed:** `ts_string_builtin/4` plus a `ts_sb_rule/5` table covering
`string_length`/`atom_length`, `string_concat`/`atom_concat`,
`sub_string`/`sub_atom` (indexed mode), `string_chars`/`atom_chars` **both
directions**, `string_codes`/`atom_codes` both directions, `char_code/2` both
directions, `number_string/2` both directions, `atom_string`/`string_to_atom`,
`string_lower`/`string_upper`/`downcase_atom`/`upcase_atom`. Mode selection is
driven by the VarMap: a rule applies only when every input term is already
resolvable. Two passes — `strict` (output must be a fresh variable) then `loose`
(output may already be mapped, the normal case for a goal writing into the
clause head's output argument).

---

### G-A3-2 — Statement blocks wrapped in `return …;` · **S** · CLOSED

**Extends:** `native_ts_clause_body/3` and the multi-clause if-chain emitters.

**Trigger** — the simplest transform predicate there is:
```prolog
p(X, Y) :- Y is X * 2.
```

**Was:**
```ts
function p(arg1: number): string {
    return const arg2 = (arg1 * 2);
  return arg2;;
}
```
Not valid TypeScript, not valid JavaScript. node rejects the module at parse
time. This was the *flagship batch path* of the target and it had never emitted
a runnable function for a two-goal clause.

**Correct lowering:**
```ts
    const arg2 = (arg1 * 2);
    return arg2;
```

**Fix landed:** `ts_clause_code_form/2` classifies the clause code as an
expression (never ends in `;`, never spans lines) or a statement block;
`ts_clause_body_text/3` emits `return <expr>;` for the former and the re-indented
block for the latter. A block that renders **no** `return` means goals were
dropped, so an explicit `throw new Error("incomplete lowering: …")` is appended
rather than letting the function fall off its end returning `undefined`.

---

### G-A3-3 — `,` / `;` / `->` in a guard position had no rendering · **S** · CLOSED

**Extends:** the guard renderer (`ts_guard_condition/3`).

**Trigger** — `js_flag_char/1`, verbatim:
```prolog
js_flag_char(C) :- char_code(C, X),
    ( X >= 0'a, X =< 0'z -> true ; X >= 0'A, X =< 0'Z -> true
    ; X >= 0'0, X =< 0'9 -> true ; X =:= 0'- ).
```

**Was:** `ts_guard_condition/3` handled only binary comparison operators, `\+`,
`member/2`, `match/2,3` and type checks. A conjunction inside the condition made
the whole if-then-else unrenderable, and it was dropped.

**Correct lowering:**
`((v >= 97 && v <= 122) ? (true) : (…))`

**Fix landed:** clauses for `true`/`fail`/`false`, `(A, B)` → `&&`,
`(A ; B)` → `||`, and a guard-position if-then-else → `(c) ? (t) : (e)`. Each
sub-condition still recurses through `ts_guard_condition/3`, so an unrenderable
inner goal makes the whole render fail cleanly instead of emitting wrong code.

---

### G-A3-4 — Unrenderable goals were silently deleted · **S** · CLOSED

**Extends:** `ts_render_classified_mid/5` / `ts_render_classified_last/4`.

**Trigger** — `strip_brackets/2`, verbatim:
```prolog
strip_brackets(String, Stripped) :-
    string_chars(String, Chars), drop_brackets(Chars, Kept),
    string_chars(Stripped, Kept).
```

**Was:** the catch-all clauses `ts_render_classified_mid(_, VM, [], [], VM)` and
`ts_render_classified_last(_, _, [], [])` erased any goal the target could not
render — here the entire `drop_brackets/2` call. The emitted function read an
undefined variable and returned it. Silent wrong answers, no diagnostic.

**Correct lowering:** either lower the call, or refuse. Refusing is what the
target can honestly do today.

**Fix landed:** both renderers became deterministic dispatchers
(`ts_render_classified_mid_/5`, `ts_render_classified_last_/4` behind an
if-then-else). The fallback now emits
`throw new Error("incomplete lowering: unrendered goal drop_brackets/2");`.
Only the functor and arity are embedded, so nothing from the Prolog term can
escape into the generated string literal. The `->` commit is load-bearing twice
over: it keeps a guard-only sequence yielding zero lines (so the guard/output
split path still runs) and it makes the fallback reachable only when the real
renderer genuinely has nothing.

---

### G-A3-5 — The "guarded tail" renderer discarded the rest of the clause · **S** · CLOSED

**Extends:** `ts_render_classified_goals/4`.

**Trigger** — `starts_with/2`, verbatim:
```prolog
starts_with(String, Prefix) :-
    string_length(String, L), string_length(Prefix, N), L >= N,
    sub_string(String, 0, N, _, Sub), Sub == Prefix.
```

**Was:** the output-followed-by-guards clause rendered the guards as the exit
test and returned — **throwing away every classified goal after the guard run**.
`starts_with/2` compiled to a function that returned the prefix length and never
looked at the substring at all.

**Fix landed:** the clause now requires `Remaining == []`; when something
follows the guard run, control falls through to the general sequence clause so
the remaining goals are rendered. (What that exposes next is G-A3-6.)

---

### G-A3-6 — Guards are hoisted above the assignments that define them · **M** · OPEN

**Extends:** `native_ts_clause/5` — the split between "clause condition" and
"clause body".

**Trigger** — `starts_with/2` again. Today:
```ts
function starts_with(arg1) {
    if (v3 >= v4) {                       // <-- v3/v4 read here ...
        const v3 = arg1.length;           // <-- ... declared here
        const v4 = arg2.length;
        const v5 = arg1.slice(0, 0 + v4);
        if (v5 === arg2) { return v5; }
    }
    ...
}
```

`native_ts_clause/5` collects **all** guard conditions into a single `Condition`
that the emitters place in the clause's `if (…)` header, ahead of the block.
That is correct only for guards over head arguments; a guard over a body-local
variable becomes a temporal-dead-zone `ReferenceError` under node and a
"used before its declaration" error under `tsc`.

**Correct lowering:** guards must be emitted in clause order, interleaved with
the assignments — head-argument guards in the header, body-local guards as
in-block `if (!(…)) …` tests or as a nested `if`.

**Size M.** It is a restructuring of one predicate's contract (return a list of
`(condition, position)` rather than one conjunction), not new machinery, but it
touches every emitter that consumes `Condition`.

Probe: `gap_g_a3_6_guard_hoisted_above_its_own_definitions`.

---

### G-A3-7 — annotated_js mistook `const x = (expr);` for an arrow signature · **S** · CLOSED

**Extends:** `annotated_js_target:signature_start/1`.

**Trigger:** any generated line whose right-hand side is parenthesised —
e.g. the `const v5 = (v4 - arg2);` that G-A3-1/G-A3-2 now emit routinely.

**Was:** `signature_start/1` accepted a line starting with `const ` that
contained `" = ("`. The line walker then treated it as the opening line of a
multi-line arrow signature, consumed the entire rest of the file looking for a
`=> {` that never came, and the whole TS→JSDoc rewrite **failed** — so
annotated_js refused predicates vanilla_js compiled fine. A real inheritance
break.

**Fix landed:** the `" = ("` alternative now additionally requires `"=>"` on the
same line. Every arrow signature `typescript_target` emits satisfies that;
generic ones are still caught by the `" = <"` alternative.

---

### G-A3-8 — The last-resort fallback EXECUTES the predicate · **M** · OPEN

**Extends:** the dispatcher in `compile_predicate_to_typescript/3`.

**Trigger:** any rule predicate no native path claims.
```prolog
flags_put([], K, V, [K-V]).
flags_put([K0-V0|R], K, V, Out) :- ( K0 == K -> Out = [K0-V|R] ; Out = [K0-V0|R1], flags_put(R, K, V, R1) ).
```

**Is:** the final clause falls through to `compile_facts/3`, which builds the
fact array with `findall(…, (functor(Goal,…), call(Goal), …), Facts)` — it
**runs the predicate with every argument unbound**. Measured outcomes over the
17 `cli_args` predicates that reach it:

* 4 → a syntactically valid but semantically **wrong** fact table
  (`is_global_key/1` compiles to an *empty* table, i.e. "never true");
* 10 → `instantiation_error` or `resource_error(stack)`;
* 3 → an unbounded `findall/3`; killed at a 20 s / 1.5 GB cap.

**Correct behaviour:** take the facts path only when *every* clause body is
`true`; otherwise refuse with a diagnostic naming the predicate. The fact
enumeration itself is fine for real fact predicates — the bug is the guard, not
`compile_facts/3`.

**Size M** because the same fallthrough exists for `type(recursion)` and
`type(module)` and the refusal has to be threaded back through the callers that
today assume `compile_predicate/3` always succeeds.

Probe: `gap_g_a3_8_fact_fallback_is_offered_a_rule_predicate`. *Do not write a
probe that calls `compile_predicate/3` on such a shape — it will eat the test
runner's memory.*

---

### G-A3-9 — Loops keep exactly one output · **L** · OPEN

**Extends:** the structural-recursion lowering (`native_ts_structural/3`,
`ts_struct_detect/5`, `ts_struct_inputs/3`).

**Trigger:**
```prolog
loop([], A, B, A, B).
loop([X|Xs], A0, B0, A, B) :- A1 is A0 + X, B1 is B0 + 1, loop(Xs, A1, B1, A, B).
```

**Is:** `ts_struct_detect/5` accepts it and sets `Mode = function(Arity)` — the
**last** argument is the output, full stop. Everything else becomes a parameter.
So argument 4, the *other* output, is emitted as a required input and then
compared against the accumulator:

```ts
export function loop(a1: any[], a2: any, a3: any, a4: any): any {
  if (a1.length === 0 && a4 === a2) { return a3; }
  ...
}
```

The caller has to already know half the answer. This compiles, runs, and is
wrong.

**Correct lowering:** detect the set of output positions (arguments that are
free in the head of the base clause and threaded unchanged through the recursive
call), and return a tuple/object — `return { a4: …, a5: … }`.

**This is the gap that blocks the parser.** `lenient_loop/5` has 2 outputs,
`strict_loop/8` has 3, `scan_leading_globals/4` has 2. No amount of builtin-table
work reaches them.

**Size L**: a new output-mode analysis plus a calling convention for
multi-output predicates, propagated through every call site the target emits.

Probe: `gap_g_a3_9_second_output_becomes_an_input`.

---

### G-A3-10 — if-then-else in a recursive body defeats the structural path · **M** · OPEN

**Extends:** `ts_struct_goal/13`.

**Trigger:**
```prolog
loop([], Acc, Acc).
loop([X|Xs], Acc0, Acc) :- ( X > 0 -> Acc1 is Acc0 + X ; Acc1 = Acc0 ), loop(Xs, Acc1, Acc).
```

**Is:** `ts_struct_goal/13` has clauses for comparisons, `is/2`, `=/2` and
recursive calls — and nothing for `;`/`->`. One if-then-else anywhere in the body
makes `ts_struct_goals/12` fail, the structural path refuses the whole
predicate, and the dispatcher drops to G-A3-8.

**Correct lowering:** an if-then-else whose branches assign the same variables
becomes `let acc1; if (cond) { acc1 = …; } else { acc1 = …; }`.

**Every one of `cli_args`' loops has this shape.** Closing G-A3-10 without
G-A3-9 gets none of them; closing both gets all four engines.

**Size M**: the branch machinery already exists in `clause_body_analysis`
(`output_ite`, `shared_output_vars`) and in the clause-body renderer; it has to
be wired into the structural path with `let`-then-assign instead of `const`.

Probe: `gap_g_a3_10_ite_in_a_recursive_body_refuses`.

---

### G-A3-11 — Generated scaffolding does not match the predicate · **S** · OPEN

**Extends:** `native_ts_clause_body/3`'s module template and
`compile_module/3`'s `generate_pred_code_ts/4`.

Three separate defects in the wrapper around an otherwise-correct body:

1. **Arity-1 predicates get a zero-parameter function.**
   `build_ts_arg_list(Arity-1)` assumes the last argument is an output, so a
   semidet arity-1 predicate compiles to `function js_alpha(): string { … arg1 … }`
   — the body references a parameter that does not exist. This is why
   `js_alpha/1` and `js_flag_char/1` are graded *wrong* even though their guard
   expressions are exactly right.
2. **The CLI entry point passes one argument.**
   `console.log(pred(parseInt(process.argv[2])))` regardless of arity, and always
   `parseInt`, so a string-argument predicate cannot be driven at all. This is
   why the end-to-end check in §5 needs a hand-written driver.
3. **Types are hardcoded** `arg<N>: number` / `: string`. Harmless under
   `node --experimental-strip-types` and erased entirely by vanilla_js, but wrong
   under `tsc` and wrong in the annotated_js JSDoc.
4. **`compile_module/3` silently emits an empty module** for any predicate whose
   declared type is not one of its four canned patterns (see §3.2).

**Size S** each, but (1) changes every generated signature, so it needs its own
regression pass.

---

### G-A3-12 — Compound terms as values become string literals · **M** · OPEN

**Extends:** `ts_literal/2`, `ts_expr/3`, `ts_term_expr/3`.

**Trigger:** `schema_for/5`, `parse_strict/4`, `check_arity/3`, `next_value/2`,
`split_flag_token/3` — everything that returns `ok(P, F)` / `err(Msg)` /
`some(V)` / `none` / `schema(O, P)`.

**Is:** `ts_literal/2`'s last clause is `term_string(Value, S)` wrapped in
quotes. `ok([a], [b-c])` becomes the JavaScript string `"ok([a],[b-c])"`. The
tag is gone, the payload is gone, and nothing downstream can destructure it.

**Correct lowering:** a tagged object — `{ tag: "ok", args: [ …, … ] }` — with
matching destructuring on the read side, or JS-native shapes for the well-known
tags. The A1 README calls this out explicitly: *"targets that flatten compound
terms need to keep the tag; this is the pattern the whole 'no exceptions in the
compiled core' design rests on."*

**Size M**: a value representation decision plus renderers on both the
construct and the match side.

Probe: `gap_g_a3_12_compound_term_becomes_a_string_literal`.

---

### G-A3-13 — The atoms `true`/`false` are lowered to the strings `"true"`/`"false"` · **S** · OPEN

**Extends:** `ts_literal/2`.

```prolog
ts_literal(true, '"true"').
ts_literal(false, '"false"').
```

**Is:** in this target every Prolog atom is a JS string, so the boolean atoms
collapse into their own names.

**Why it matters here:** `cli_args` flag values are *strings or the atoms
`true`/`false`*, and the oracle corpus asserts
`flags["include-key"] === true` — a boolean, distinct from the string `"true"`
that `--x=true` would produce. Any transpile that stringifies them fails the
corpus.

**Correct lowering:** `true` → `true`, `false` → `false`. **Careful:** this is a
representation change, not a typo fix — the surrounding code (`compile_facts`'s
`arg<N>: string` interfaces, the `is<Pred>(...args: string[])` helper) assumes
string-valued arguments throughout. **S** to change, **M** to make consistent.

Probe: `gap_g_a3_13_boolean_atoms_are_stringified`.

---

### G-A3-14 — Unmapped variables leak their internal `_G` names into the output · **S** · OPEN

**Extends:** `ts_expr/3`.

```prolog
ts_expr(Var, VarMap, TExpr) :- var(Var), !,
    ( lookup_var(Var, VarMap, Name) -> TExpr = Name ; term_string(Var, TExpr) ).
```

**Is:** when a variable is not in the VarMap the generated JavaScript gets
`_41598` — an identifier that is declared nowhere and differs run to run. Seen
in the wild on an if-then-else chain that binds in its branches and then
continues:

```prolog
p(X, Y) :- ( X > 10 -> T = big ; X > 5 -> T = mid ; T = small ), Y = T.
```
→ `const arg2 = _42774;`

**Correct behaviour:** fail. A variable with no binding means the clause was not
fully analysed, which is exactly the situation G-A3-4 now reports out loud.

**Size S** to change; the risk is that failing here turns some currently-"working"
lowering into a refusal, so it wants its own regression pass.

Probe: `gap_g_a3_14_unmapped_variable_leaks_into_output`.

---

### G-A3-15 — A reversible text builtin picks the wrong direction when both arguments are mapped · **S** · OPEN

**Extends:** `ts_string_builtin/4` (added by G-A3-1).

**Trigger:**
```prolog
p(Cs, S) :- string_chars(S, Cs).      % should BUILD S from Cs
```

**Is:** both variables are head arguments, so both are in the VarMap, the
`strict` pass (output must be a fresh variable) finds nothing, and the `loose`
pass takes the first matching rule — decompose. Emits `Array.from(arg2)` where
`arg1.join("")` was meant.

**Correct lowering:** prefer the direction whose output is the clause head's own
output argument. That means threading the head's output slot into
`ts_string_builtin/4`.

**Where it bites in `cli_args`:** `strip_brackets/2`'s closing
`string_chars(Stripped, Kept)`. (That predicate refuses for other reasons today,
so this gap is currently masked.)

Probe: `gap_g_a3_15_reversible_builtin_picks_decompose_when_ambiguous`.

---

### G-A3-16 — A compound/list head argument becomes a string comparison · **M** · OPEN

**Extends:** `ts_head_conditions/4` (via `ts_literal/2`).

**Trigger:**
```prolog
pair_lookup([K-V|Rest], Key, Value) :- ( K == Key -> Value = V ; pair_lookup(Rest, Key, Value) ).
```

**Is:** every non-variable head argument goes through `ts_literal/2`, which
stringifies a compound term. The generated head test is:

```ts
if (arg1 === "[_61230-_61232|_61226]") { … }
```

— a comparison against the Prolog *source text* of the pattern, complete with
internal variable names. Always false. Eleven `cli_args` predicates are
first-argument-indexed list or pair walks.

**Correct lowering:** destructure, exactly as the structural path already does in
`ts_match/6`: `arg1.length > 0`, `const k = arg1[0][0]`, `const rest = arg1.slice(1)`.
The machinery exists — `native_ts_structural/3` does this correctly — it is just
not reachable from the clause-body path, which is where these predicates land
because their bodies contain if-then-else (G-A3-10).

**Size M**: reuse `ts_match/6`'s pattern binder from `native_ts_clause/5`.

Probe: `gap_g_a3_16_list_head_pattern_becomes_a_string_literal`.

---

### G-A3-17 — The guard/output split path threw away intermediate assignments · **S** · CLOSED

**Extends:** `ts_output_goals/3`.

**Trigger:**
```prolog
p(Cs, Out) :- string_chars(S, Cs), Out = S.
```

**Was:** `ts_output_goals/3` threaded only the VarMap through every non-final
output goal and **discarded its `const …;` line**. The clause compiled to
`return v3;` with `v3` declared nowhere. A silent ReferenceError, distinct from
G-A3-4 (this path never reached the classified-goal renderers at all).

**Fix landed:** the intermediate lines are kept; several output goals now yield a
statement block ending in `return <expr>;`, which G-A3-2's emitter handles.

---

### Not-a-gap notes

* String equality is fine. `==`/`\==` map through `expr_op/2` → `ts_op/2` to
  `===`/`!==`, which is the right JS reading for `cli_args`' string comparisons.
* Character-code literals (`0'a`, `0'-`) are read by SWI as integers before the
  compiler ever sees them, so `js_alpha/1`'s ranges lower to `>= 97 && <= 122`
  correctly. The A1 README flagged this as a silent-breakage risk; it is not one
  for these targets.

---

## 5. What actually runs

### 5.1 The two predicates that work

`substring_from/3` and `substring_range/4` compile to correct TypeScript and
correct vanilla JavaScript:

```js
function substring_from(arg1, arg2) {
    const v4 = arg1.length;
    const v5 = (v4 - arg2);
    const arg3 = arg1.slice(arg2, arg2 + v5);
      return arg3;
}
```

Nine cases (including `""`-producing and `--`-prefixed inputs) run under node and
match the SWI oracle exactly:

```
node          swipl
"state=alpha" "state=alpha"
"hello"       "hello"
""            ""
""            ""
"def"         "def"
"key"         "key"
"abcdef"      "abcdef"
""            ""
"a"           "a"
```

**Disclosure:** the *driver* that calls these functions is hand-written, because
the compiler's own emitted CLI entry point passes only `process.argv[2]` and
always through `parseInt` (G-A3-11.2) — it cannot drive a 3-argument
string predicate. Every line of the two **functions** is compiler output; only
the `console.log(...)` calls around them are not. This is pinned as a test:
`compiled_substring_from_runs_under_node`.

### 5.2 The four mechanisms: none of them runs

| mechanism | status |
| --- | --- |
| lenient loop (`parse_lenient/3` + `lenient_loop/5`) | **no.** `lenient_loop/5` reaches no lowering path (G-A3-9, G-A3-10); the wrapper refuses loudly. |
| strict loop (`parse_strict/4` + `strict_loop/8` + `strict_option/11`) | **no.** Same, plus 3 outputs and a tagged status (G-A3-12). |
| leading-globals scan (`scan_leading_globals/4`) | **no.** 2 outputs, nested if-then-else. |
| `schemaFor` (`schema_for/5` + `registry_entry/3` + `action_entry/3`) | **no.** Depends on `pair_lookup/3` (G-A3-16), compound terms (G-A3-12) and 2 outputs (G-A3-9). |

**A hand-written JS shim around the compiled pieces would be cheating, and this
report will not claim otherwise.** The compiled pieces are two substring
helpers. Everything that makes `cli_args.pl` a *parser* — the loops, the assoc
lookups, the tagged results — is either refused or absent. Writing the loops in
JavaScript by hand and calling `substring_from` from them would produce a
working parser that demonstrates nothing about the compiler.

For the same reason, **`run_differential.sh` was not re-pointed at a transpiled
build**: there is no transpiled build to point it at. The bar the A1 README
sets — same seed, 0 divergences — is not reachable and pretending otherwise by
hand-filling the gaps would defeat the measurement. The differential harness
becomes the acceptance gate the moment G-A3-9 + G-A3-10 + G-A3-12 land; until
then the meaningful gate is the shape suite in
`tests/core/test_typescript_cli_args_shapes.pl`.

---

## 6. Maturity assessment

**Where the pattern targets actually are.** `typescript_target.pl` is a
*template engine with shape recognisers*, not a compiler. Its strongest paths —
facts, the six recursion patterns, aggregates, data sources, streaming filters —
each recognise a canonical shape and emit a good, purpose-built template. Inside
that envelope the output is genuinely good. Outside it, the target does not
refuse: it renders whatever fragments it recognises and drops the rest. That is
the single most important finding of this step, and it is not about `cli_args`
at all — before this step, `p(X,Y) :- Y is X*2.` compiled to unparseable
TypeScript, and no test noticed, because the tests assert on substrings of the
output rather than on whether node can load it.

**What that means for the demo.** A3 cannot show a transpiled parser. What it
can show, and now does, is a precise, executable account of the distance:
17 named gaps, 7 fixes landed, and a suite that fails the day any of them is
closed or re-broken. That is a more useful artifact than a shim-assisted
success.

**Two structural recommendations beyond the individual gaps.**

1. **Add an "is this valid JavaScript" gate to the target's test suite.** Every
   `has(Code, "…")` assertion in `test_typescript_target.pl` passed while the
   target emitted `return const arg2 = …;;`. A single `node --check`
   (or `new Function(src)`) over the generated module for each compile test
   would have caught G-A3-2 the day it was introduced. This is cheap and it is
   the highest-leverage change in this report.
2. **Make refusal a first-class outcome.** The dispatcher's "always succeed,
   fall back to facts" contract is what turns every unsupported shape into
   either a wrong answer or a runaway process (G-A3-8). A compiler that can say
   *"I cannot lower `strict_loop/8`: multi-output accumulator loop"* is more
   useful than one that always returns a string.

## 7. Recommended priority order

| rank | gap | size | why first |
| --- | --- | --- | --- |
| **1** | **G-A3-8** — fallback executes the predicate | M | It is a *hazard*, not just a gap: it hangs the compiler and burns memory on ordinary input, and it makes every other gap harder to investigate (this report's tooling exists only to route around it). Cheapest large safety win. |
| **2** | **G-A3-10** — if-then-else in a recursive body | M | Unblocks the structural path for all four `cli_args` engines and for the general class of "Prolog loop with a conditional step". The branch machinery already exists in `clause_body_analysis`; this is wiring, not invention. |
| **3** | **G-A3-9** — multi-output loops | L | The other half of the same unlock. With 10 and 9 landed, `lenient_loop/5`, `strict_loop/8` and `scan_leading_globals/4` all become expressible and the differential harness becomes a meaningful gate. Larger, so second in the pair. |

Then, in rough order: **G-A3-16** (list/pair head destructuring in the
clause-body path — reuses `ts_match/6`, unblocks 11 predicates), **G-A3-6**
(guard placement — required before anything with body-local guards can run),
**G-A3-12** (tagged compound values — required for `ok/err` results), and the
S-sized cleanups **G-A3-11**, **G-A3-13**, **G-A3-14**, **G-A3-15**.

---

## 8. Changes landed in this step

| file | change |
| --- | --- |
| `src/unifyweaver/targets/typescript_target.pl` | G-A3-1 string/char builtin table (`ts_string_builtin/4`, `ts_sb_rule/5`); G-A3-2 statement-vs-expression clause bodies (`ts_clause_code_form/2`, `ts_clause_body_text/3`); G-A3-3 `,`/`;`/`->` in guard position; G-A3-4 deterministic classified-goal dispatchers with a loud unrendered-goal fallback; G-A3-5 guarded-tail remainder check; G-A3-17 `ts_output_goals/3` keeps intermediate assignments |
| `src/unifyweaver/targets/annotated_js_target.pl` | G-A3-7 `signature_start/1` no longer treats `const x = (expr);` as an arrow signature |
| `tests/core/test_typescript_cli_args_shapes.pl` | **new** — 30 tests: regressions for the 7 fixes, an end-to-end node run of the compiled `substring_from/3`, and 8 executable gap probes |

`vanilla_js_target.pl` needed no change; it inherits all of the above.

### Regression results (verbatim)

```
$ swipl -q -g test_typescript_target_core -t halt tests/core/test_typescript_target.pl
Registered source type: csv -> csv_source
Registered source type: json -> json_source
................  Compiling multi-call linear recursion: comb/2
.  Compiling direct multi-call recursion: comb/2
  Recursive calls: 2
........[TypeScript Target] Initialized with bindings
.[TypeScript Target] Initialized with bindings
..........................................

$ swipl -q -g test_annotated_js_target -t halt tests/core/test_annotated_js_target.pl
Registered source type: csv -> csv_source
Registered source type: json -> json_source
.................

$ swipl -q -g test_vanilla_js_target -t halt tests/core/test_vanilla_js_target.pl
......................

[test_vanilla_js_target] ALL TESTS PASSED

$ swipl -q -g test_typescript_source -t halt tests/core/test_typescript_source.pl
.  Compiling CSV source: ts_pipe/3
.  Compiling CSV source: ts_pipe/3
.  Compiling JSON source: ts_product/3
.  Compiling JSON source: ts_product/3
.  Compiling JSON source: ts_product/1
.

$ swipl -q -g test_typescript_cli_args_shapes -t halt tests/core/test_typescript_cli_args_shapes.pl
Registered source type: csv -> csv_source
Registered source type: json -> json_source
..............................
```

All five green (no `!` failure markers, no `ERROR` lines).
