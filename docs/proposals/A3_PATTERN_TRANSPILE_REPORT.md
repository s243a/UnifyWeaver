<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# A3 — Pushing a real Prolog program through the pattern targets

> ## STATUS UPDATE (seventh run — the CLOJURESCRIPT port). **ClojureScript is now the third pattern target carrying the entire real-world program.**
>
> `examples/cli_args/cli_args.pl` — all 43 predicates, `parse_args/2` included —
> compiles into ONE ClojureScript namespace through `clojure_target`'s new A3
> whole-program lowering and `clojurescript_target`'s JVM→JS interop rewrite.
> Measured against the same oracle, with the same generator and the same seed:
>
> | | |
> | --- | --- |
> | nbb loads the generated namespace (the `node --check` analogue) | **clean** |
> | predicates lowering with no dropped goal | **40 of 40** rule predicates; the other 3 are ground-fact CONSTANT TABLES, inlined at their call sites |
> | peerhailer's contract corpus | **17 / 17** contract points, **25 / 25** argv-lines, error messages included |
> | differential vs the JS oracle | **5067 lines, 0 divergences, 0 message mismatches, 0 crashes** |
>
> Deliverables live in `examples/cli_args/cljs/`; the shape suite is
> `tests/core/test_clojurescript_cli_args_shapes.pl` (66 tests).
>
> ### What transferred UNCHANGED
>
> Every *design* in §G-A3-6/9/10/12/16/18/19/20 transferred as-is. The two
> call-graph fixpoints are ports of the TypeScript ones down to the clause
> structure — the GREATEST fixpoint for outputs (`clj_out_table/2`, optimistic
> start, `clj_out_meet/3` keeping the shorter suffix) and the LEAST fixpoint for
> fallibility (`clj_fail_table/2`, head-coverage gap plus a fallible call in
> body-goal position, condition position exempt). Both reproduce the TS lane's
> answers on the real program *exactly*, which is the strongest evidence the port
> is a port and not an approximation, and it is pinned as a test: the output
> analysis finds `lenient_loop/5` → {4,5}, `strict_loop/8` → {6,7,8},
> `scan_leading_globals/4` → {3,4}, `split_flag_token/3` → {2,3},
> `starts_with/2` → {}; and the sentinel set is the *same nine* predicates
> (`endgame_sentinel_set_is_the_same_nine_as_the_ts_lane`). Arity mangling,
> head-built outputs, the reversible-text-builtin direction rule, the
> constant-table inline and the dependency closure all came across unchanged.
>
> ### What Clojure made EASIER — the expression language pays off four times
>
> TypeScript is a statement language, so the TS lowering accumulates statement
> strings, threads a `return` through every branch, and needs `ts_assemble/3` to
> re-nest in-block guards plus a manual indentation walker. Clojure is an
> expression language, so a clause body is ONE expression built by folding an
> ordered item list from the inside out — `bind(N,E)` → `(let [N E] …)`,
> `gopen(C)` → `(if C … Fall)`. That single change collapses four TS mechanisms:
>
> * **G-A3-10's VALUE form.** TS declares `let _s0, _s1;` ahead of the block and
>   assigns at the end of each branch because a JS `if` is a statement. Here the
>   `if` IS the value: `(let [_s0 (if C then else)] …)`, and with several shared
>   outputs `(let [[_s0 _s1] (if C [t0 t1] [e0 e1])] …)`.
> * **G-A3-18 in CONDITION position.** TS needs a mutable `let _t0;` assigned
>   *inside* the condition to smuggle a value out. Here the call is an ordinary
>   `bind` item and the condition reads the name.
> * **`ts_assemble/3` and the indentation walker** have no analogue: nesting IS
>   the structure.
> * **Structural equality.** Clojure's `=` is already structural, so the emitted
>   `_uwEq` helper is simply **not ported**. Verified against every distinction
>   the program depends on — `(= true "true")` is false (G-A3-13 survives),
>   nested tagged maps compare structurally, and vectors compare equal to seqs,
>   which is what lets `first`/`rest`/`cons` be used without a `vec` round-trip.
>
> Clojure also forced one thing to be MORE explicit, and better for it: in TS a
> failing in-block guard reaches no `return` and drops off the end of the block.
> An expression has no such escape, so the alternative must be named — the fold
> threads a `Fall` expression, which is the next clause's chain or the
> predicate's exit. Where a clause body actually uses it, the successor is bound
> to a zero-argument thunk so it is written once; where it does not — every
> multi-clause predicate in `cli_args`, all first-argument-indexed — the chain
> stays a plain nest of `if`s.
>
> ### What needed a DIFFERENT design
>
> * **Tuple** — a Clojure **vector** `[out1 out2]`, destructured with
>   `(let [[a b] (f …)])`. Same positional convention, native syntax.
> * **Compound** — a **map** `{:$ "f" :args […]}`. The TS lane rejected a tagged
>   array because a Prolog list is already a JS array; in Clojure a list is
>   already a vector, so the same argument rejects a tagged vector and the map is
>   the analogous safe choice.
> * **Sentinel** — `(def ^:private uw-fail (Object.))`, rewritten to
>   `(js/Object.)` for the JS host, tested with `identical?`. A namespaced keyword
>   would work *today* (nothing this target lowers a term to is a keyword) but
>   that is a property of the current term renderer, not of the convention; a
>   freshly allocated object has reference identity no term can produce and no
>   data crossing the module edge can forge. Same forgery argument as the TS
>   `Symbol`, same strength.
> * **Forward declarations.** The one place Clojure needs something JavaScript
>   gets free: a JS `function` declaration is hoisted, so the TS lane may emit a
>   caller before its callee and a mutually recursive pair in either order. A
>   Clojure var must exist before it is referenced, so a multi-predicate module
>   opens with `(declare …)` naming every function it defines.
> * **Chars.** A Prolog char is a ONE-CHARACTER STRING on both hosts, so
>   `string_chars/2` decomposes with `(mapv str (seq s))`, not `(vec s)` — `vec`
>   answers JVM characters on the JVM and one-character strings under CLJS, and
>   only one of those compares equal to the atom `'='` the program tests against.
>   Exactly three runtime helper lines are host-specific and they go through the
>   existing interop-rewrite mechanism.
> * **Recursion: direct calls, not `recur`.** The faithful analogue of the TS
>   lane's `return pred(…)`, and the same risk — neither JS nor nbb eliminates
>   tail calls. `recur` would give real TCO but only for calls the generator can
>   *prove* are in tail position with a matching binding vector. The risk is
>   measured rather than asserted (`examples/cli_args/cljs/probe_depth.cljs`):
>   the character walks run to a ~2000-character token against a 26-character
>   bound (~77×) and the argv walks to ~700 tokens against a 7-token bound
>   (~100×), and the differential records 0 crashes. Pointed at an input whose
>   size is not bounded by a command line, `recur` would stop being optional —
>   that is the stated limit.
>
> ### Routing, and the byte discipline
>
> The A3 paths are a RESCUE, exactly as `native_ts_general/3` is: they claim only
> a predicate whose historical `clojure_target` answer would be defective — it
> failed, it leaked an internal `_NNN` variable name, it stringified a compound,
> it emitted a call to a Prolog builtin Clojure does not have (`(string_length s)`
> — G-A3-1 in its Clojure incarnation), it is arity-overloaded, it has two or
> more outputs (G-A3-9's hazard), or one of its callees speaks a different
> calling convention. **26 shapes × 2 emitters = 46 outputs diffed against the
> pre-port compiler: 45 byte-identical, 1 changed, and the change is required** —
> a multi-predicate module gains its `(declare …)` line.
>
> Three things the port had to get right that the TS lane never faces, each found
> by a gate rather than by reading:
>
> * **ONE MODULE, ONE SET OF NAMES.** A module can mix the two lowerings and they
>   spell a function differently (`merge-flags-` vs `merge_flags_`). The A3 set is
>   now fixed ONCE per module and published, so a call site, a `(declare …)` entry
>   and a `(defn …)` header cannot disagree; the closure runs in BOTH directions,
>   because a historical-path predicate calling an A3 one is the same bug
>   mirrored. Found by nbb refusing the namespace with "Unable to resolve symbol:
>   `merge_flags_`".
> * **A fact clause is NOT a defect here.** `ts_clause_body_defective/2` treats one
>   as such because the TS clause-body path is guarded against fact clauses
>   upstream; Clojure's historical path lowers a mixed fact+rule predicate
>   correctly. Porting the branch verbatim moved four passing runtime-smoke tests
>   onto the new path.
> * **An empty output set has two causes.** A genuine semidet TEST, and a CONSTANT
>   ANSWER (`positive(X, yes) :- X > 0.`) where the last head argument is ground
>   in every clause. The historical path renders the constant answer correctly, so
>   only *two or more* outputs force the A3 path.
>
> Also fixed on the way: `clj_arith/3` distinguishes `/` from `//` (the old
> `clojure_op/2` mapped `/` to `quot`, which silently turns a division into a
> truncation), and the runtime block is emitted once rather than once per
> occurrence of its own name (`sub_string/5` is nondeterministic inside
> `findall/3`).
>
> ### Still open
>
> * **`recur`** — see above. Direct calls, bounded by input size, measured.
> * **The reversible-text-builtin direction between two HEAD arguments.**
>   `p(Cs, S) :- string_chars(S, Cs).` compiles as a semidet test that computes a
>   value and answers `true`, because the direction rule requires the known side to
>   come from a PRECEDING GOAL. This is the TypeScript lane's behaviour for the
>   same clause, reproduced exactly, and it is probe-pinned in both suites
>   (`gap_reversible_builtin_direction_is_ambiguous_between_head_arguments`).
> * **The `bb` (JVM host) runtime for the A3 path.** The three host-specific
>   runtime lines are written in JVM form and rewritten for the JS host, so the
>   JVM spelling exists — but the A3 lowering has only been exercised under nbb.
>   Every path `bb` already served is untouched.
> * **The corpus gate is an equivalence, not an import swap** — see
>   `examples/cli_args/cljs/README.md`, which states the difference rather than
>   glossing it.


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
> **G-A3-10 is now also CLOSED** (third run): if-then-else composes with the
> structural-recursion path via two lowerings chosen by position — TAIL (each
> branch gets its own `return`: recursive branch = loop continuation, exit
> branch = value; nested else-if composes; no dead trailing return) and VALUE
> (`let _sN` + per-branch assignment, read by the following goals). Conditions
> render via `ts_guard_condition/3`; classification reuses
> `clause_body_analysis`'s `if_then_else_goal/4` + shared-output-vars (core
> unmodified). `ts_term_expr/3`/`ts_arith/3` now REFUSE on unbound variables
> (were emitting `undefined`/`_G…`). `string_member/2` and `first_char_index/4`
> from cli_args now compile oracle-correct (lowers-correctly count 2 → 4).
> Shapes suite 55 → 67 tests.
> **G-A3-9 is now also CLOSED** (fourth run): multi-output structural loops
> return a positional tuple `[out1,…,outN]` — output positions found by a
> both-halves accumulator-discipline analysis (recursive clauses only thread,
> base clauses produce), tail calls stay `return pred(...)` (the tuple flows
> through), non-tail calls destructure `const [_s0,_s1] = pred(...)`, ITE
> branches compose. Single-output path proven BYTE-IDENTICAL over a 12-shape
> harness. Multi-output modules get a JSON CLI entry. Shapes suite 67 → 81.
> Payoff: `lenient_loop/5`'s and `scan_leading_globals/4`'s exact loop skeletons
> now compile and match SWI; G-A3-9 is no longer the blocker for ANY mechanism.
> **G-A3-6, G-A3-12 and G-A3-16 are now also CLOSED** (fifth run), and with them
> the step's milestone: **peerhailer's LENIENT parse compiles whole and matches
> the SWI oracle under node.** `parse_lenient/3` plus the 13 predicates it
> transitively calls become ONE module through `compile_module/3`
> (`include_dependencies(true)`), `node --check`s clean, and agrees with SWI on
> every argv line tried — including the greedy legacy read
> (`tunnels --a --b=c` → `flags.a === "--b=c"`), the bare flag
> (`--x` → the BOOLEAN `true`), the inline value, the empty argv and the
> `__proto__` assignment JavaScript silently drops.
> * **G-A3-6** — two halves. (1) GUARD PLACEMENT: a guard that reads a value a
>   preceding statement declared is no longer hoisted into the clause header
>   (a TDZ `ReferenceError`); it becomes a nested in-block `if`, so a failing
>   guard falls through to the next clause. (2) CROSS-PREDICATE CALLS, lowered by
>   the CALLEE's output count, which `ts_pred_outputs/3` reads from its clauses:
>   0 outputs → the callee returns a boolean and the call IS a condition
>   (`starts_with(t, "--")`, and `\+` composes to `!`); 1 output →
>   `const _sN = q(ins);`; N outputs → `const [_sN, _sN1] = q(ins);`, G-A3-9's
>   tuple destructured. FAILURE SEMANTICS: only the 0-output form carries failure,
>   as `false`; a callee with outputs is a det function that THROWS when no clause
>   matches. Carried by a new **general clause lowering** (`native_ts_general/3`)
>   built on the structural path's machinery, whose MODE comes from
>   `ts_pred_outputs/3` instead of a decomposition argument — so a predicate needs
>   neither recursion nor a cons pattern, and an arity > 1 SEMIDET test
>   (`starts_with/2`) finally has a signature (`(a1, a2) => boolean`). It is a
>   RESCUE path: it runs only where the clause-body path's own answer would be
>   defective, so everything that compiled correctly still compiles there.
> * **G-A3-12** — `f(A1..An)` → `{$: "f", args: [...]}`. Not a tagged array: a
>   Prolog list is already a JS array, so `["f", e1]` could not be told from the
>   list `[f, e1]`, and that distinction is the gap. Construction, clause-head
>   matching, matching in an if-then-else CONDITION (which BINDS the payload into
>   the then-branch), and structural `==`/`\==` via an emitted `_uwEq` all use it.
>   Atoms stay strings, `true`/`false` stay booleans, lists stay arrays; the four
>   are pairwise distinguishable at run time and no test throws.
> * **G-A3-16** falls out of G-A3-12: `[K-V|Rest]` is a cons test plus a `-`/2 tag
>   test plus a destructure, so `flags_put/4` and `pair_lookup/3` compile.
> Byte-identity of every previously-working shape re-proven over a 26-shape ×
> 3-target harness (78 outputs, 0 diffs). Shapes suite 81 → 103 tests.
> Still OPEN — the one shared blocker left is a **semidet callee WITH outputs in
> if-then-else CONDITION position** (`( pair_lookup(L,K,V) -> ... ; ... )`,
> `( option_kind(O,K,Kind) -> ... ; ... )`): it must both fail and bind, which
> needs a failure sentinel in the calling convention rather than the current
> throw. That single shape is what still stops `strict_loop/8`, `option_kind/3`,
> `registry_entry/3`, `action_entry/3` and `is_global_key/1` — see the revised
> §5.2. Also open: a ground-fact predicate used as a CONSTANT TABLE
> (`js_object_prototype_keys/1` — no clause has a variable in its head, so the
> output analysis sees no output), list-BUILDING recursion (`drop_brackets/2`),
> **G-A3-11.3** (type inference, probe-pinned) and variable-to-variable `==`,
> which is still JS identity (probe-pinned). By-design refusal: a branch
> containing a bare failable test.

---

> ## STATUS UPDATE (sixth run — the ENDGAME). **The pattern lane now transpiles the entire real-world program.**
>
> `examples/cli_args/cli_args.pl` — all 43 predicates, `parse_args/2` included —
> compiles into ONE JavaScript module through `compile_module/3` with
> `include_dependencies(true)`. `node --check` clean, no WARNING banner, no
> dropped goal. Measured against the same oracle the Prolog reference was
> measured against in step A1:
>
> | | |
> | --- | --- |
> | peerhailer's contract corpus (`oracle/cliArgs.test.mjs`, import swapped) | **17 / 17** |
> | differential vs the JS oracle — the harness's own generator, same seed | **5067 lines, 0 divergences, 0 message mismatches** |
> | mechanisms end-to-end | **4 of 4** (lenient, strict, leading-globals scan, schemaFor) |
> | predicates lowering with no dropped goal | **40 of 40** rule predicates; the other 3 are ground-fact CONSTANT TABLES, inlined at their call sites |
>
> The deliverables live in `examples/cli_args/patternjs/` — `build.sh` (one
> command), the generated module, a term↔JS edge shim carrying no parse logic,
> the corpus with only its import changed, and the differential runner.
>
> Four new gaps were closed, and two pre-existing correctness bugs were found and
> fixed on the way.
>
> * **G-A3-18 — a semidet callee WITH outputs.** The shape that stopped three
>   mechanisms. THE CONVENTION: a predicate that has outputs AND can fail returns
>   its answer (or G-A3-9's tuple) or the module-private sentinel `_uwFail`;
>   callers test `x !== _uwFail`. In CONDITION position the call is made inside
>   the condition, into a `let` the block declares just above the `if` — so `&&`
>   still short-circuits and the payload is bound into the THEN branch only,
>   which is Prolog's scope rule. In BODY-GOAL position it is a `const` plus an
>   in-block test, so a failing call falls through to the caller's next clause.
>   WHY A `Symbol` AND NOT `null`: nothing this target lowers a Prolog term to is
>   ever null or undefined, so `null` would work *today* — but that is a property
>   of the current term renderer, not of the convention, and a JS value crossing
>   the module edge (a registry handed to `parseArgs(argv, registry)`) is outside
>   it entirely. A module-private Symbol cannot be produced by any term and
>   cannot be forged by data.
>   WHICH predicates get it: `ts_pred_can_fail/2`, a LEAST fixpoint over the call
>   graph. A predicate can fail when (1) some INPUT position's clause heads do not
>   cover the value space — no clause has a variable there and the shapes are not
>   exactly `{[], [_|_]}` (`pair_lookup([K-V|Rest], ...)` has no clause for `[]`;
>   `parse_strict(_, schema(O,P), _, _)` matches only a `schema/2` term) — or
>   (2) a fallible call, a `\+`, or a bare comparison sits in BODY-GOAL position.
>   Condition position does not count: a condition is *allowed* to fail.
>   In `cli_args` that names exactly nine predicates: `pair_lookup/3`,
>   `last_element/2`, `option_kind/3`, `registry_entry/3`, `action_entry/3`,
>   `schema_for/5`, `parse_strict/4`, `parse_args/3`, `parse_args/2`. Every other
>   predicate keeps the exit line it had, so the whole lenient mechanism is
>   byte-identical.
>   THE STATED LIMIT, probe-pinned rather than papered over
>   (`gap_g_a3_18_bare_body_match_does_not_make_a_predicate_semidet`): a bare
>   `=` MATCH in body-goal position is NOT taken as evidence of fallibility.
>   Counting it would make `strict_option/11` semidet on the strength of
>   `Rest = [_Consumed|Rest1]` — a match the preceding `next_value/2` has already
>   guaranteed — and cascade the sentinel through `strict_loop/8`,
>   `parse_strict/4` and `parse_args/3`, wrapping every call site in a test that
>   can never be true. The cost is bounded and LOUD: such a predicate keeps the
>   det exit, so a match that does fail throws `no matching clause for p/n` — a
>   crash naming the predicate, never a wrong answer. The 5067-line differential
>   is the evidence that it never fires.
>   G-A3-18 also carries two things the fixpoint forced: the OUTPUT analysis is
>   now a GREATEST fixpoint over the same graph (the single visited-set walk
>   declined on every cycle, so `nth0_default/4` self-recursion answered "no
>   outputs" and the mutual `strict_loop/8` ⇄ `strict_option/11` pair settled for
>   one output where there are three), and an output BUILT IN THE HEAD
>   (`lenient_result(Argv, ok(Positional, Flags))`) is recognised. Plus ARITY
>   MANGLING: JavaScript has no overloading, so `parse_args/2` and `parse_args/3`
>   become `parse_args_2` and `parse_args_3`; an un-overloaded name is untouched.
> * **G-A3-19 — a ground-fact predicate used as a CONSTANT TABLE.**
>   `global_options/1`, `default_registry/1`, `js_object_prototype_keys/1`: every
>   clause is a ground fact, so the output analysis sees no variable in any head
>   and answered "no outputs" — and the cross-call lowering then read the goal as
>   a boolean test and emitted `global_options([...])`, a call to a function
>   nothing declares. The lowering is a MATCH against the table, not a call: with
>   one fact each argument is matched against the constant (an unbound argument
>   BINDS to it, a bound one becomes an equality test), with several facts and all
>   arguments known it is a membership test over the emitted rows. Such a
>   predicate is not a module member at all, so `ts_pred_callees/2` keeps it out of
>   the closure. Calling a multi-row table with an unbound argument would be an
>   enumeration, which this target has no form for; that is refused.
> * **G-A3-20 — list-BUILDING recursion, and a continuation duplicated into
>   branches.** Two lowerings, both for an if-then-else the VALUE form cannot take.
>   (a) DEFERRAL: `drop_brackets([C|Cs], Kept) :- (... -> Kept = Kept1 ; Kept =
>   [C|Kept1]), drop_brackets(Cs, Kept1).` describes its output in terms of a
>   value the call AFTER it produces. JavaScript has no such hole, so the
>   if-then-else is rendered AFTER the rest of the sequence, against the bindings
>   that come out of it. Guarded twice — the in-place lowering must have failed,
>   and the CONDITION must be renderable against the bindings available before the
>   rest runs — so nothing that reads a value the rest produces can be hoisted over
>   it. (b) CONTINUATION DUPLICATION: a tail-context if-then-else whose branch
>   opens with a failable test has no `let` form, because a slot cannot express
>   "and if that test fails, the clause fails". Prolog COMMITS to a branch, so the
>   continuation is appended to both branches and the whole thing becomes a TAIL
>   if-then-else, where a failing branch reaches no return and falls through to the
>   clause's exit. That closes the by-design refusal the fifth run recorded ("a
>   branch containing a bare failable test") and is what lets `parse_args/3`
>   compile. A branch may now also open with a failable test inside
>   `ts_struct_branch_return/8`, nested as an `if` around the branch.
> * **Two pre-existing correctness bugs, found while probing and fixed.**
>   (1) A `V = Term` goal where the clause ALREADY holds a value for `V` was a
>   silent re-binding, dropping the test: `string_concat(S,"!",T), T = "hi!"`
>   compiled to the concatenation with no comparison at all. It is now the
>   comparison it is. (2) `vanilla_js_target`'s tuple-annotation strip rule matched
>   `args: [` inside an OBJECT LITERAL — G-A3-12's compound representation — and
>   ate it, so `{$: "-", args, ["name"]]}` reached node: a syntax error inside an
>   otherwise-correct module. The rule now matches the bracket contents as a TYPE
>   LIST rather than as "anything up to the first `]`".
>   Also hardened: `ts_pred_outputs/3` now FAILS for a predicate with no visible
>   clauses. The reachability walk keeps such a predicate in the table with an
>   empty output set, which reads identically to a genuine semidet test — so an
>   UNDEFINED callee was briefly lowered as a boolean call to a function no module
>   declares. That failure is load-bearing: it is what makes an unknown callee
>   refuse out loud.
> * **Byte discipline.** 26 shapes × 3 targets = 78 outputs, diffed against the
>   fifth-run compiler. **72 byte-identical; 6 changed, all required**, and all in
>   the two shapes G-A3-18 exists for: `a3_wrap(V, some(V))` (an answer built in
>   the head — was compiled as a 2-parameter boolean TEST, so the caller had to
>   supply the answer; now `a3_wrap(a1) -> {$: "some", args: [a1]}`) and
>   `a3_unwrap(some(V), V)` (semidet by head coverage — was a throw, now the
>   sentinel). Both changes are the gap closing, not a side effect.
> * Shapes suite 103 → 116 tests, including two ENDGAME tests that read the real
>   `examples/cli_args/cli_args.pl` from the repository, compile it whole, and run
>   the compiled `parse_args/2` under node against SWI running the same clauses.

---

## 0. Headline

> **Fifth-run census** (the table below is the original A3 snapshot and is not
> rewritten). Of the 43 predicates: **30 now compile with no dropped goal**
> (was 2), 9 compile partially, 4 refuse loudly. The 14 that make up the lenient
> mechanism are verified RUNNING under node against the SWI oracle, as one module.
> Fraction of the parser transpilable: **≈70 %** by predicate count; **1 of 4**
> mechanisms end-to-end, with a second (the leading-globals scan) needing one
> more predicate and a third (strict) needing two.

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

### G-A3-6 — Guards hoisted above their definitions; no cross-predicate calls · **M** · CLOSED (fifth run)

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

**Fix landed (fifth run), and it is wider than the entry above.** The gap turned
out to have a second half — a compiled clause body could not CALL another
compiled predicate at all — and the two were fixed together; see the STATUS
UPDATE at the top for the design. `native_ts_clause/5` was left alone: instead a
general clause lowering (`native_ts_general/3`) picks up any predicate whose
clause-body answer would be defective, which is exactly the set with a hoisted
guard, a dropped goal, or a read of the unassigned output slot. Guards inside it
are placed in clause order — header while nothing has been emitted, nested
in-block `if` afterwards — so a failing guard falls through to the next clause.

Regressions: `g_a3_6_guard_follows_the_assignments_it_reads`,
`g_a3_6_starts_with_matches_swi_under_node`,
`g_a3_6_semidet_callee_is_a_boolean_condition`,
`g_a3_6_negated_semidet_callee_composes`,
`g_a3_6_semidet_callee_as_a_body_goal_nests`,
`g_a3_6_det_single_output_callee_is_a_const`,
`g_a3_6_multi_output_callee_is_destructured`,
`g_a3_6_compile_module_pulls_in_the_dependency_closure`,
`g_a3_6_mutual_recursion_runs_through_declaration_hoisting`, plus the refusals
`g_a3_6_unknown_callee_still_refuses` and
`g_a3_6_output_of_a_call_may_not_be_a_bound_variable`.

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

### G-A3-12 — Compound terms as values become string literals · **M** · CLOSED (fifth run)

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

**Fix landed (fifth run).** `f(A1..An)` → `{$: "f", args: [...]}`, chosen over a
tagged array because a Prolog list is already a JS array (`["f", e1]` could not
be told from the list `[f, e1]`, and that distinction is the gap). Emitted on
construction (`ts_compound_expr/3`), in clause heads and `=`/2 matches
(`ts_match/6` — a tag test plus a positional destructure of `args`), in
if-then-else CONDITIONS (`ts_cond/4`, which binds the payload into the
then-branch only, as Prolog's scope rule requires), and in `==`/`\==` through an
emitted `_uwEq` that compares structurally. Atoms stay strings, `true`/`false`
stay booleans (G-A3-13) and lists stay arrays; `x != null && x.$ === "f"` tells
all four apart and never throws.

Known limit, pinned rather than papered over: the structural comparison is chosen
from the SOURCE, so `p(A, B) :- A == B.` still emits `===`
(`gap_g_a3_12_variable_to_variable_equality_is_identity`).

Regressions: `g_a3_12_compound_is_constructed_as_a_tagged_object`,
`g_a3_12_compound_head_pattern_is_a_tag_test_and_destructure`,
`g_a3_12_compound_in_an_ite_condition_binds_its_payload`,
`g_a3_12_equality_on_compounds_is_structural`,
`g_a3_12_compound_atom_list_boolean_are_distinguishable`,
`g_a3_12_compound_survives_a_cross_predicate_call`,
`g_a3_12_pair_walk_compiles_and_matches_swi` (which is G-A3-16 falling out of
this one).

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

### G-A3-16 — A compound/list head argument becomes a string comparison · **M** · CLOSED at the dispatcher (fifth run)

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

**Fix landed (fifth run), by routing rather than by rewriting.** `ts_match/6` was
already right; what was missing was a compound clause (G-A3-12) and a path that
reaches it. Both landed with G-A3-6/G-A3-12: a predicate with a compound or list
head pattern is now claimed by the structural path or by the new general clause
lowering, both of which destructure. `pair_lookup/3`, `flags_put/4`,
`string_member/2` and `last_element/2` all compile and match SWI
(`g_a3_12_pair_walk_compiles_and_matches_swi`).

**Not fixed:** `ts_head_conditions/4` itself, inside `native_ts_clause/5`, still
sends a non-variable head argument through `ts_literal/2`. That is now
unreachable for anything the dispatcher routes elsewhere, but it is still the
wrong rendering in isolation, so the probe is kept — pointed at the clause-body
path directly — rather than deleted:
`gap_g_a3_16_clause_body_path_still_stringifies_a_list_head_pattern`.

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

### 5.2 The four mechanisms

*(Original A3 snapshot — every row read "no". Revised after the fifth run; the
original wording is preserved beneath the table.)*

*(Revised again after the SIXTH run. All four rows are now "yes", and so is the
whole program: `parse_args/2` compiles into one module of 40 functions. The
fifth-run wording is kept beneath.)*

| mechanism | status after the sixth run | evidence |
| --- | --- | --- |
| **lenient loop** (`parse_lenient/3` + 13) | **YES** — 14 predicates, one module | byte-identical to the fifth run |
| **strict parse** (`parse_strict/4` + `strict_loop/8` + `strict_option/11` + `option_kind/3` + `check_arity/3` + `count_required/3` + `strip_brackets/2` + `drop_brackets/2` + `nth0_default/4` + `last_element/2` + `next_value/2` + `is_long_flag/1` …) | **YES** — 27 predicates, one module, no banner, no dropped goal | G-A3-18 (`option_kind/3`, `last_element/2` in condition position), G-A3-19 (`global_options/1`), G-A3-20 (`drop_brackets/2`) |
| **leading-globals scan** (`scan_leading_globals/4` + `is_global_key/1` + …) | **YES** — 18 predicates | G-A3-18 + G-A3-19: `( global_options(G), pair_lookup(G, Key, _) -> true ; … )` is a constant-table read followed by a semidet-with-outputs test |
| **`schemaFor`** (`schema_for/5` + `registry_entry/3` + `action_entry/3` + `js_object_prototype_key/1`) | **YES** — 6 predicates; `schema_for/5` is 2 outputs behind a sentinel test | G-A3-18 (all three lookups) + G-A3-19 (`js_object_prototype_keys/1`) |
| **`parse_args/2` — the whole program** | **YES** — 40 functions in one module, `node --check` clean, 17/17 corpus, 0 divergences over 5067 lines | `examples/cli_args/patternjs/` |

The fifth-run table, kept for the record:

| mechanism | status then | what each still needed |
| --- | --- | --- |
| **lenient loop** (`parse_lenient/3` + `lenient_loop/5` + `starts_with/2`, `split_flag_token/3`, `flags_set/4`, `flags_put/4`, `looks_like_legacy_flag/1`, `legacy_flag_tail/1`, `js_alpha/1`, `js_flag_char/1`, `first_equals_index/2`, `first_char_index/4`, `substring_from/3`, `substring_range/4`) | **YES — compiles whole and matches SWI under node.** 14 predicates, one module, `node --check` clean. | nothing |
| strict loop (`parse_strict/4` + `strict_loop/8` + `strict_option/11`) | **partial.** `strict_option/11` (the 11-argument, 3-output half), `next_value/2`, `is_long_flag/1`, `long_flag_tail/1`, `pair_lookup/3` all compile. | `strict_loop/8` and `option_kind/3` need a **semidet callee with outputs in condition position** — `( option_kind(Options, Key, Kind) -> ... ; ... )` must both fail and bind. `check_arity/3` needs the same, plus `last_element/2` in a condition. |
| leading-globals scan (`scan_leading_globals/4`) | **near.** `scan_leading_globals/4` itself compiles — 2 outputs, four nested if-then-elses, `some(V)` matching, three cross-calls — and so does `next_value/2`. | only `is_global_key/1`, which is `( pair_lookup(Globals, Key, _) -> true ; ... )`: the same semidet-with-outputs condition. |
| `schemaFor` (`schema_for/5` + `registry_entry/3` + `action_entry/3`) | **partial.** `pair_lookup/3`, `string_member/2`, `default_registry/1` and `js_object_prototype_keys/1` compile. | `registry_entry/3` and `action_entry/3` are the same semidet-with-outputs condition. `js_object_prototype_key/1` additionally needs a **ground-fact predicate used as a constant table** to be recognised as having an output (no clause of `js_object_prototype_keys/1` has a variable in its head, so the output analysis sees none). `schema_for/5` needs both. |

One shape accounted for every fifth-run "partial": a call that is **semidet AND
produces a value**, used as an if-then-else condition. That shape is G-A3-18 and
it is closed; see the STATUS UPDATE at the top for the sentinel convention and
the determinacy analysis that decides which callees get it.

### 5.3 The endgame run (sixth), and what it is honestly claiming

`examples/cli_args/patternjs/` holds the transpiled build. The claim is narrow
and checkable:

* `build.sh` is one command. It reads the FROZEN `cli_args.pl`, compiles
  `parse_args/2` with `include_dependencies(true)`, and `node --check`s the
  result. 40 functions, no WARNING banner, no `incomplete lowering`.
* `cliArgs.patternjs.test.mjs` is `oracle/cliArgs.test.mjs` with **one line
  changed** — the import. **17 / 17 under `node --test`.**
* `run_differential_patternjs.sh` is `run_differential.sh` with the Prolog
  reference replaced by the transpiled parser: the same `gen_cases.mjs`, the same
  seed, the same `compare_jsonl.mjs`. **5067 argv-lines, 0 divergences, 0 message
  mismatches** (4150 ok results, 917 errors — the error MESSAGES match too).
* The only hand-written JavaScript in the path is `cliArgs.mjs`, and it does
  three things: argv array in, `ok(P, F)` → `{positional, flags}` out (unwrapping
  `{$: "-", args: [k, v]}` pairs in order), `err(M)` → `throw new CliError(M)`.
  It carries no branch that depends on an argv token.

The original A3 wording, kept for the record:

**A hand-written JS shim around the compiled pieces would be cheating, and this
report will not claim otherwise.** The compiled pieces are two substring
helpers.<!-- (as of the fifth run: the lenient mechanism in full, with no shim —
the only hand-written JavaScript in the milestone test is the `console.log` that
prints what the compiled `parse_lenient` returned. As of the SIXTH run: the whole
program, with a term↔JS conversion shim at the edge and no parse logic in it.) -->
Everything that makes `cli_args.pl` a *parser* — the loops, the assoc
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
<!-- Sixth run: G-A3-9, -10, -12, -18, -19 and -20 all landed, the build exists,
and the harness IS now the acceptance gate — see §5.3. -->

---

## 5.4 Census after the sixth run

| | |
| --- | --- |
| Predicates in `cli_args.pl` | **43** |
| Lower to a correct FUNCTION with no dropped goal | **40** |
| Ground-fact CONSTANT TABLES, inlined at their call sites (G-A3-19) — not module members | **3** (`global_options/1`, `default_registry/1`, `js_object_prototype_keys/1`) |
| Refuse | **0** |
| Mechanisms end-to-end | **4 of 4**, plus `parse_args/2` itself |
| Fraction of the parser transpilable | **100 %** by predicate count and by mechanism |

One honest footnote on the three constant tables: asked to compile one of them
*standalone*, the dispatcher still routes it to `compile_facts/3`, which
stringifies a nested-term argument into one row
(`{ arg1: "[state-string,name-string]" }`) — the original §3 grading, unchanged.
That output is never reached from the module, because G-A3-19 inlines the table
at its call sites and `ts_pred_callees/2` keeps it out of the dependency closure.
It is a defect of `compile_facts/3`'s row renderer for non-scalar arguments, not
of the program's lowering, and it is left open.

### Still open after the sixth run

* **`compile_facts/3` stringifies a non-scalar fact argument** (above). Harmless
  for this program; wrong in isolation.
* **G-A3-11.3** — parameter and return types are still hardcoded on the
  clause-body path (`arg<N>: number`, `: string`). Probe-pinned.
* **Variable-to-variable `==`** is JS identity: `p(A, B) :- A == B.` emits `===`
  because the structural comparison is chosen from the SOURCE shape, not from a
  runtime type test. Probe-pinned
  (`gap_g_a3_12_variable_to_variable_equality_is_identity`).
* **A bare `=` MATCH in body-goal position does not mark a predicate semidet**
  (G-A3-18's stated limit). Such a predicate keeps the det exit, so a match that
  does fail throws by name instead of failing. Probe-pinned
  (`gap_g_a3_18_bare_body_match_does_not_make_a_predicate_semidet`); the
  5067-line differential is the evidence it never fires here.
* **A DET callee with outputs in CONDITION position** still refuses. It is a call
  that always succeeds and produces a value, used where a test is expected; no
  `cli_args` shape needs it, so it was not built.
* **A multi-row ground-fact table called with an unbound argument** is an
  enumeration — nondeterminism this target has no form for — and is refused.
* **`ts_head_conditions/4`** inside `native_ts_clause/5` still stringifies a
  non-variable head argument. Unreachable for anything the dispatcher routes
  elsewhere; probe kept, pointed at that path directly.

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

### Changes landed in the SIXTH (endgame) run

| file | change |
| --- | --- |
| `src/unifyweaver/targets/typescript_target.pl` | G-A3-18: the output analysis becomes a GREATEST FIXPOINT over the call graph (`ts_out_table/2`, `ts_out_graph/5`, `ts_out_iterate/3`, `ts_out_meet/3`, cached on `variant_sha1/2` of the reachable clause set); an output BUILT IN THE HEAD; the reversible-text-builtin direction rule; `ts_pred_can_fail/2`, a LEAST fixpoint, and the `_uwFail` sentinel it drives (`ts_struct_fail_line/1`, the semidet forms in `ts_struct_goal/13` and `ts_cond/7`, `ts_fail_out_exprs/3`, a second on-demand runtime block); `ts_js_name/3` arity mangling; `ts_is_self/2` so a self-call is matched by name AND arity. G-A3-19: `ts_fact_pred/2`, `ts_fact_call/5`, and their goal / condition clauses. G-A3-20: `ts_struct_seq/15`'s deferral clause and its continuation-duplication clause; `ts_struct_branch_return/8` accepts a branch-opening guard. Two correctness fixes: a re-binding `=`/2 no longer drops its test; `ts_pred_outputs/3` fails for a predicate with no visible clauses; `ts_clause_body_defective/2` also routes a block that declares the same name twice |
| `src/unifyweaver/targets/vanilla_js_target.pl` | the tuple-annotation strip rule matches its bracket contents as a TYPE LIST, so it no longer eats `args: [` inside G-A3-12's compound object literal |
| `tests/core/test_typescript_cli_args_shapes.pl` | 103 → 116 tests: regressions for G-A3-18/-19/-20, the two stated-limit probes, and two ENDGAME tests that read the real `examples/cli_args/cli_args.pl`, compile it whole, and run the compiled `parse_args/2` under node against SWI |
| `examples/cli_args/patternjs/` | **new** — `build.sh`, `build.pl`, `cliArgs.generated.mjs`, the edge shim `cliArgs.mjs`, the corpus with its import swapped, `diff_runner_patternjs.mjs`, `run_differential_patternjs.sh`, `README.md` |

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
