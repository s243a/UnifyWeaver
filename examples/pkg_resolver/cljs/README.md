<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# `resolver.pl` + the `pkg` CLI → ClojureScript

**The whole of uw-resolve P0.5 and the whole of the `pkg` command line,
transpiled from Prolog to ClojureScript and measured against the SWI-Prolog
oracle and the JavaScript CLI they were written from.**

`examples/pkg_resolver/resolver.pl` — all 79 predicates, every one of the ten
queries — compiles into a pair of ClojureScript namespaces that nbb loads
clean. `pkg.cljs` drives it with the *same* `cli/generated/pkg_registry.json`
through the *same* D40 ClojureScript argparser the `cli_args` lane ships, and
prints byte-for-byte what `cli/pkg.mjs` prints.

| gate | result |
| --- | --- |
| nbb loads the generated namespaces (the `node --check` analogue) | **clean** — 2536 WAM instructions, 79 predicates |
| the P0.5 contract corpus (`dump_corpus.pl`, SWI expected) | **39 / 39** matched SWI |
| the `pkg` CLI contract corpus, CLJS CLI vs the JS lane's own expectations | **153 / 153** assertions (the JS suite's 86, plus 66 byte-for-byte cross-checks and the wrapper) |
| differential vs SWI — the harness's own generator, same seed | **2400 cases, 0 divergences, 0 crashes**, all ten queries |

This is the **first backtracking program** the ClojureScript lane has carried.
The D40 argparser is deterministic from end to end; pointing a program with real
choice points at this target found **six defects**, listed below. Five are pinned
by a probe in `tests/core/test_clojurescript_wam_backtracking.pl`, which compiles
a minimal Prolog program and runs it under nbb; the sixth is a scale property and
is measured by B3.

---

## Run it

```bash
bash examples/pkg_resolver/cljs/build.sh            # compile + nbb load check
bash examples/pkg_resolver/cljs/run_corpus_cljs.sh  # the 39-scenario gate vs SWI
node --test examples/pkg_resolver/cljs/test_pkg_cli_cljs.mjs   # the CLI gate
bash examples/pkg_resolver/run_differential_cljs.sh # the 2400-case gate
bash examples/pkg_resolver/cljs/bench_scale.sh      # B3, the 5k-package catalog

# the two runtime probe suites (compile a minimal program, run it under nbb)
swipl -g 'test_clojurescript_wam_backtracking,halt' tests/core/test_clojurescript_wam_backtracking.pl
swipl -g 'test_clojurescript_wam_indexing,halt'     tests/core/test_clojurescript_wam_indexing.pl

nbb --classpath examples/cli_args/cljs:examples/pkg_resolver/cljs \
    examples/pkg_resolver/cljs/pkg.cljs resolve editor \
    --catalog examples/pkg_resolver/cli/generated/catalogs/teaching.json
```

Requires `swipl`, `node` and `nbb` on `PATH`.

## What is in this directory

| file | what it is |
| --- | --- |
| `build.pl` / `build.sh` | ONE command: reads the FROZEN `../resolver.pl`, compiles all 79 predicates, and nbb-loads the result |
| `generated/resolver/core.cljs` | **compiler output** — the WAM instruction table, the intern tables, the lowered prefixes and the predicate wrappers |
| `generated/resolver/runtime.cljs` | **compiler output** — the WAM runtime, rendered from `templates/targets/clojure_wam/runtime.clj.mustache` with the JVM→JS interop rewritten |
| `resolver.cljs` | the EDGE: JSON ⇄ WAM terms, and driving the generated wrappers. No resolver logic |
| `pkg.cljs` | the CLI. No parse logic, no resolve logic |
| `run_corpus.cljs` / `run_corpus_cljs.sh` | the 39-scenario contract gate against SWI |
| `test_pkg_cli_cljs.mjs` | `../cli/test_pkg_cli.mjs` with `runPkg` pointed at `pkg.cljs`, plus the byte-for-byte cross-check |
| `diff_runner_cljs.cljs` | the harness's stdin/JSONL protocol, under nbb |
| `../run_differential_cljs.sh` | `../run_differential.sh` with the CLJS leg swapped in (the original is untouched) |
| `bench_scale.cljs` / `bench_scale.sh` / `scale_to_catalog.mjs` | B3: one `resolve_layered` on the seeded 5000-package catalog |
| `load_check.cljs` | the `node --check` analogue |

`generated/resolver/` rather than a flat file because that is where nbb's
classpath resolver looks for `generated.resolver.core` and
`generated.resolver.runtime`. Every runner here passes `--classpath` pointing at
this directory; `pkg.cljs` additionally needs `examples/cli_args/cljs` on the
classpath, because it *imports* the frozen D40 argparser rather than copying it.

---

## Why the pattern lane stops here, and what carries the program instead

The D40 argparser reaches ClojureScript through the **pattern lane**:
`clojure_target`'s A3 whole-program lowering, then `clojurescript_target`'s
JVM→JS interop rewrite. A3 emits **one deterministic function per predicate** —
a value, or the failure sentinel. That is exactly right for a parser, and it
cannot express `resolver.pl`.

This is not a guess. Compiling `resolve/3` and `resolve_layered/3` through
`clojurescript_target:compile_module/3` with `include_dependencies(true)` emits
28 `defn` forms, of which **7 are `;; TODO: Implement` stubs** — and they are
precisely the predicates that backtrack or enumerate:

| stubbed by A3 | why |
| --- | --- |
| `candidates_high_first/4` | `member(Ver, Desc)` — *enumerates* versions highest-first; the whole search rests on retrying it |
| `resolve_pending/5` | retries `pick` when `no_acc_conflicts/4` or a later request fails |
| `resolve/3`, `resolve_layered/3` | drive `resolve_pending` and cut its first solution |
| `conflicts_in/4` | `member(conflicts(...), Cs)` — a *test* by enumeration |
| `map_requests/3`, `request_to_req/3` | build `req/2` terms whose head is an unbound output compound |

A3's answer for a stub is a function that returns nothing useful. Emitting it
and calling it "compiled" would have been the wrong answer, so this lane is not
used for the resolver.

The lane that *does* carry it is the **WAM lane of the same ClojureScript
target**: `wam_clojure_target:write_wam_clojurescript_files/3`, which compiles
each predicate to WAM instructions, emits them as Clojure data plus a runtime
that executes them, and then hands the whole thing to
`clojurescript_target:clojurescript_interop_rewrite/2` — the same rewrite the
pattern lane uses — to turn the JVM host calls into JS ones. It is the exact
analogue of the frozen `../wamjs/` lane for JavaScript, and it is still
transpilation: nothing in `resolver.cljs` or `pkg.cljs` decides a version, a
constraint or a layer.

(`write_wam_clojurescript_files/3` was exported but had **no caller** anywhere in
the repository before this. Everything below is what that first use found.)

## How backtracking is realized in the emitted code

There is no `mapcat`, no lazy sequence and no continuation-passing anywhere in
the emitted resolver. Backtracking is a **WAM choice-point stack living in an
immutable Clojure map**:

```clojure
{:code   [...]                 ; the shared instruction table, 2536 entries
 :pc     1517                  ; program counter
 :regs   {"A1" ... "X6" ...}   ; argument and temporary registers
 :env-stack   [{...}]          ; permanent (Y) variables, one frame per activation
 :bindings    {...}            ; variable id -> term
 :trail       [{:var .. :old .. :had-old? ..} ...]
 :choice-points [{:pc .. :regs .. :trail-len .. :cut-bar .. :ite true?} ...]
 :cut-bar 3  :cut-bars {2 0, 3 1}   ; this activation's B0, and the callers'
 :unify-queue [...]  :unify-stack [[...]]  ; read-mode cursor, and its stack
 :status :running}
```

* `try_me_else` / `retry_me_else` push a **snapshot** of that map onto
  `:choice-points`. Because the state is persistent, the snapshot is a handful of
  pointer copies, not a deep copy — the one place where Clojure's data model
  makes a WAM *cheaper* than a mutable one.
* Failure calls `backtrack`, which unwinds `:trail` back to the recorded length
  and restores the snapshot. `resolve_pending/5` retrying a version, and
  `close_moving/3` retrying a repair, are both just this.
* `!` prunes `:choice-points` back to `:cut-bar`, the choice-point count recorded
  when the *current activation* was entered (`enter-call-barrier` on `call`,
  `enter-execute-barrier` on `execute`, restored by `proceed`). `\+ G` compiles
  to `try_me_else ELSE / call G / ! / fail`, so this barrier is what makes
  negation work at all.
* `(Cond -> Then ; Else)` compiles to `try_me_else L_ite_else_N / Cond /
  cut_ite`, and `cut_ite` drops choice points down to and including the one that
  `try_me_else` pushed — committing over anything `Cond` left standing.

Each predicate also gets a **lowered prefix** — a straight-line Clojure function
that executes the head and as much of the body as stays in the straight line —
and a `<pred>-state` wrapper that runs the prefix and then hands the state to
`run-wam-state`. The prefix is purely an optimisation: it stops at the first
control transfer (and, for a clause whose if-then-else contains a call, it is the
identity function — see fix #5), and the interpreter takes the program from
wherever it stops. Every instruction is also present in the shared table, so the
two paths cannot disagree about what the program *is*.

## ClojureScript idioms

The WAM lane's representation is data, not the pattern lane's four-way encoding,
so the D40 conventions (`{:$ tag :args [...]}` compounds, vector tuples,
structural `=`) appear **only where `pkg.cljs` talks to the argparser** — the
registry it hands to `parse-args-3` is built in exactly that shape, and the
`ok(Positional, Flags)` it gets back is destructured with it.

Everywhere else:

| | Clojure |
| --- | --- |
| Prolog atom | `{:tag :atom :id <interned id>}`; a plain string in an input term is interned on the way in |
| integer | a JS number |
| compound `f(A…)` | `{:tag :struct :functor <atom> :args [...]}` |
| variable | `{:var <id>}` — the edge uses **string** ids (`"uw-out"`), which the runtime's own integer ids can never collide with, so an answer is always readable back |
| a machine step | `state -> state`, pure; `run-wam-state` is a `loop`/`recur` over `step` |
| an answer | the SUCCEEDING state; `deref-value` over `(:bindings state)` reads the output cell |

`pkg.cljs` builds its output documents as **JS objects with keys asserted in
`pkg.mjs`'s order**, so `--json` is byte-identical rather than merely deeply
equal; and it reproduces JS truthiness where `pkg.mjs` depends on it
(`flags.catalog || process.env.PKG_CATALOG` — the empty string is falsy there,
and `js-truthy?` makes it falsy here).

---

## Target fixes

Six defects, all in the ClojureScript/Clojure WAM lane, all found by pointing the
resolver at it. The first five are pinned by a probe in
`tests/core/test_clojurescript_wam_backtracking.pl`, which compiles a minimal
Prolog program and **runs it under nbb** — every one of these is invisible in the
emitted text, which is why the probes execute rather than grep.

| # | symptom | cause | fix | probe |
| --- | --- | --- | --- | --- |
| 1 | every list walk stopped after one element; `dependents/3` answered `[]`, `candidates_high_first/4` answered the LOWEST version | `get_structure` matching a compound *inside* a list cell called `enter-unify-mode`, which **overwrote** the enclosing cell's read cursor — so the list TAIL, the cell's second argument, was never read and stayed unbound | read mode is a **stack**: `:unify-stack` parks the enclosing cursor and `pop-unify-item` restores it when the inner one is exhausted (what the JS WAM runtime's `read_stack` does) | `nested_read_mode` |
| 2 | `map_requests/3` failed outright; any predicate whose output argument is a compound failed in its own head | `get_structure` had **no write mode** — an unbound register simply backtracked. `get_list` had one; the two had been written separately | one `op-get-structure` with both modes, called by the interpreter **and** by the lowered emitter (`op-get-list` is it with the list functor), so they cannot drift apart again | `get_structure_write` |
| 3 | `satisfies(v(1,0,0), gte(v(2,0,0)))` was **true**; every layered query silently ignored its frozen-base ceiling | `\+ G` lowers to `try_me_else/call/!/fail`, and `!` pruned back to whatever `allocate` last left in `:cut-bar` — a value belonging to some other predicate. Every negation whose goal SUCCEEDED reported success | a per-activation barrier: `enter-call-barrier` on `call`, `enter-execute-barrier` on `execute`, restored by `proceed`, keyed by call-stack depth so a lowered call that does not go through it cannot desynchronise anything | `negation_cut_barrier` |
| 4 | `pick(layered, …)` took the catalog branch for a package the frozen base already pins; `resolve_layered` drifted onto the classic answer | `cut_ite` popped **exactly one** choice point, which was one the *condition* had left — so `->` did not commit and a failing `Then` fell through into `Else` | the ITE's `try_me_else` (recognisable by wam_target's own `L_ite_else_` label, the same signal `wam_rust_target` uses) tags its choice point; `cut_ite` drops down to and including it | `ite_commits` |
| 5 | `upgrade_set_result/4` answered **true with its output argument unbound** | the structured-ITE lowering kept emitting straight-line steps after a `call`, executing them against the CALLEE's program counter and running on to its own `proceed` | a clause containing a control transfer no longer uses that lowering at all — it falls back to the sound stub and `run-wam` interprets it. Stopping at the call is not enough: the structurer has already folded away the `try_me_else` that pushes the ELSE alternative | `call_then_ite` |
| 6 | a 5000-package catalog blew the JS stack before the query started, and backtracking was quadratic | `normalize-term*` recursed into the last argument — one frame per list element. `unwind-trail` copied the WHOLE trail twice per backtrack | the deep axis (the last argument, i.e. the list tail) is walked iteratively with a frame vector; `unwind-trail` truncates with `subvec` and touches only the entries it undoes | (measured by B3; no unit probe — it is a scale property) |

Two further changes are additive API rather than bug fixes, and both were needed
before any of the above could even be observed:

* **`runtime/run-wam-state`** — `run-wam` answers only `true`/`false`. The state
  is immutable, so a caller that passed an unbound variable in an argument
  register had **no way to read the answer back**: the Clojure WAM lane could
  answer ground queries and nothing else. `run-wam-state` returns the succeeding
  state (or `nil`).
* **`<pred>-state` wrappers** emitted alongside the existing boolean `<pred>`,
  which keeps its old text and behaviour exactly.

One existing test (`tests/test_wam_clojure_lowered_emitter.pl`) asserted on the
*spelling* of the inlined `get_structure`/`get_list` code; its two assertions now
name `op-get-structure`/`op-get-list`. Nothing else in the suite changed.

---

## First-argument indexing

`wam_target.pl` has always emitted the standard-WAM switch family ahead of
every multi-clause chain. The Clojure runtime used to **skip all of it**: the
instructions arrived as `:raw` text, `step` matched `^switch_on_` and fell
through, so every clause chain ran unindexed. That was this lane's recorded top
residual. It is now implemented.

What the resolver's own WAM text actually contains (grep `generated/resolver/core.cljs`):

| instruction | count | where |
| --- | ---: | --- |
| `switch_on_term CLen C… SLen S… ListLabel` | 12 | every `p([], …) / p([H\|T], …)` list walk |
| `switch_on_term_a2` | 5 | dispatch on A2 (`satisfies/2` on its constraint, `map_requests/3`) |
| `switch_on_structure F/N:L …` | 10 | `catalog/6` vs `catalog/9`, `base/2` vs `layer/2` |
| `switch_on_structure_a2` | 2 | `req/2` in A2 |
| `switch_on_constant_a2_fallthrough` | 1 | `[]` in A2 |
| `try L` / `trust L` | 1 + 1 | the dedicated dispatch **chain** for `roots_to_pairs/3`, whose *two* list clauses share one index entry |

No plain `switch_on_constant` and no `switch_on_constant_fallthrough` on A1 —
`resolver.pl` has no fact tables keyed on a first-argument atom; its 5000
packages live in **lists inside a `catalog/9` term**, not in 5000 clauses. That
matters for what indexing could buy, and is the first half of the B3 reading
below.

Where it landed:

* `src/unifyweaver/targets/wam_clojure_target.pl` — the switch family and the
  `try`/`retry`/`trust` chain instructions now become Clojure data
  (`:switch-on-constant` / `:switch-on-structure` / `:switch-on-term` with a
  `:reg` of `"A1"` or `"A2"`, and `:try` / `:retry` / `:trust`) instead of
  falling through to the `:raw` catch-all. `switch_on_term`'s argument text is
  parsed **by its own two count fields** — with zero structure entries the
  counts sit next to each other and a positional split misreads them. Dispatch
  keys are added to the intern seeds, atoms and functors both: a key interned
  in a different table than the executing code compares against a freshly
  minted id and never matches.
* `templates/targets/clojure_wam/runtime.clj.mustache` — `resolve-instruction`
  turns each entry list into a map from dispatch key to target address, once,
  at load time; `switch-target` / `switch-structure-target` /
  `switch-term-target` do a hash lookup and jump.

Two properties carry the correctness:

* **First-match-wins.** `wam_target` emits repeated keys —
  `switch_on_structure_a2 req/2:default req/2:L_blocked_acc_5_2` is real output,
  and so is `a:default b:L_…_2 a:L_…_3` for a predicate with two `a` clauses.
  The entry list is a *scan*, and the first match is the answer. Building the
  map last-wins instead silently drops the earlier clause: the probe
  `order_preserved` answers `[3,9]` where SWI answers `[1,3,9]`.
* **Fall through on anything unknown.** An unbound dispatch register, and any
  key the table does not mention, run the full chain. Indexing may only ever
  remove work that could not have produced an answer.

A dispatch hit jumps straight at a clause body and pushes **no** choice point.
That is sound here without the `indexed_entry` flag the JS runtime carries,
because this runtime's `backtrack` *pops* the choice point it resumes, so
`retry_me_else` pushes the next alternative rather than rewriting the top one
and `trust_me` is already a no-op — a chain entered part-way is balanced either
way. The new `try L / retry L / trust L` chain instructions follow the same
protocol.

Pinned by `tests/core/test_clojurescript_wam_indexing.pl` — five probes,
compiled and executed under nbb: order preservation across a repeated key,
dispatch on a key built at run time rather than seeded from the switch text,
an unbound first argument still enumerating every clause, the list/structure
labels, and a pruning if-then-else run inside a backtracking enumeration of the
indexed predicate.

## Benchmarks

One machine, `nbb v1.5.212` on `node v22.22.2`, SWI-Prolog 9.0.4 for x86_64-linux.
Every figure is the harness's own timer, not a hand measurement. **before** and
**after** are the same box, same catalogs, same seeds — the box is slower than
the one the D49 figures were taken on, so compare the two columns to each other,
not to the older absolute numbers.

| | before | after | SWI (reference) |
| --- | --- | --- | --- |
| **B1** contract corpus (39 scenarios), one nbb process, `run_corpus_cljs.sh` | **1.611 s** | **1.58 s** (1.581 / 1.616 / 1.782 over three runs) | — |
| **B2** differential, **2400 cases**, one process per leg, `run_differential_cljs.sh` | **111.01 s** | **78.96 s** (85.05 s on an earlier run) | **2.16 s** |
| **B3** one `resolve_layered` on the seeded 5000-package catalog, `bench_scale.sh` | load 0.263 s, resolve **58.22 s** | load 0.27 s, resolve **38.59 s** (38.59 / 39.66 / 39.77) | load 0.68 s, resolve **0.014 s** |

B1 is unchanged because it is nbb's own start-up (~1.3 s of the 1.6 s); the
39 scenarios are small and their per-case cost is well under 10 ms either way.
B2 improves **1.3–1.4×** and B3 **1.47×**. Both answer exactly what they
answered before — B2 is still 2400 cases / 0 divergences / 0 crashes, and B3
still returns the same ten packages at the same ten versions as SWI.

### Why indexing did not buy an order of magnitude — measured, not argued

`bench_scale.cljs` was re-run with `run-wam-state` replaced by a counting loop
over `runtime/step` (same state, same program), which reports what the machine
actually executes:

```
status :succeeded  steps 7,559,597  backtracks 285,668  max choice points 81
  :put-value       1,998,237     :try-me-else-pc    285,753   (the ITE, not a clause chain)
  :unify-variable  1,562,125     :switch-on-term    285,196
  :get-variable      856,860     :execute-pc matching_deps/4    210,014
  :builtin-call      570,802     :execute-pc matching_versions/4  75,140
```

Two readings, both of which had to be measured rather than assumed:

1. **The indexing works, and the choice points it removes were never the
   cost.** `L_matching_deps_4_2_body` is reached by the switch 285,196 times
   and the clause chain's own `try_me_else` at that predicate never runs — the
   list walk is now deterministic. But 210,014 of those iterations are simply
   *fourteen full scans of the 15,000-row `depends` list*, which is what
   `resolver.pl` asks for and what SWI also does. There was no redundant search
   to index away; the catalog is a term, not a fact table.

2. **The cost is per-instruction interpretation.** 7.56 M WAM instructions in
   ~40 s is ~5 µs each. nbb runs ClojureScript through SCI, and each WAM
   instruction is a fetch, a `case`, several helper calls and a handful of
   persistent-map updates. Measured on this box, one interpreted `assoc` is
   ~0.2 µs and one `get` ~0.15 µs, so ~25 such operations per instruction *is*
   the 5 µs. SWI answers the same query in 15 ms because its WAM instruction
   costs nanoseconds, not because it searches less.

The four runtime changes shipped alongside the indexing all come from that
measurement, and each was verified in isolation before it was made:

| change | measured before → after |
| --- | --- |
| a choice point is the state map with four fields written over it, not a fresh thirteen-entry map literal (`choice-snapshot`) | 5.48 µs → 2.42 µs per snapshot; `try_me_else` 11.4 µs → ~5 µs |
| `deref-value` answers non-variables without entering its loop; `logic-var?` is one keyword lookup, not `map?` + `contains?` | 0.66 µs → ~0.2 µs, on ~3 M calls |
| the read-mode cursor advances with `subvec`, not `(vec (rest q))` | 1.56 M `unify_variable`s stop rebuilding the queue |
| `reg-get-raw` / `reg-set-raw` use `get`/`assoc` on the extracted `:regs` map rather than `get-in`/`assoc-in`; `advance` assocs instead of `update`; `==/2` on two atoms skips building the recursive closure | 0.54 → 0.39 µs, 0.28 → 0.15 µs, `==/2` 8.1 → ~4 µs |

**What is left, and what it is worth.** The floor this shape can reach is the
fixed per-instruction cost: fetch + `case` + one state update measures 1.45 µs,
so 7.56 M instructions cannot go below ~11 s however good each instruction
gets. Beating that needs *fewer interpreted steps per WAM instruction*:
compile each instruction into a closure at load time (no `case`, no
per-instruction field lookups) and fuse straight-line runs so a block updates
`:regs` and `:pc` once instead of once per instruction. That is the recorded
next lever; it is real work and it is not done here.

B2 is the semantics gate, not only a timing: **2400 cases, 0 divergences,
0 crashes**, across all ten queries, against the same seeded generator
(`gen_catalogs.mjs`, unchanged seed) and the same comparator
(`compare_jsonl.mjs`) the JS lane uses.

B3's two legs return **the same selection**, package for package and version for
version (`p1 1.0.0, p10 0.0.0, p12 0.0.0, p2 1.0.0, p3 0.0.0, p30 0.0.0,
p4 0.0.0, p5 0.0.0, p6 1.0.0, p7 0.0.0`) — the ClojureScript answer is right, it
is just slow. Note the shape: the ClojureScript **load** is nearly three times
*faster* than SWI's (a `JSON.parse` and a term build, against SWI's
`atom_json_dict` per line), and the **search** is four thousand times slower.
That is the interpreted WAM over persistent maps, and it is the honest cost of
this lane at that scale. Both legs read the same seeded catalog
(`store/gen_scale_catalog.mjs`, seed `0xc0ffee01`) and the same `probe.json`
request and frozen base.

---

## Residuals — the honest list

1. **Speed at scale.** B3 above: 58.2 s → 38.6 s, which is 1.47×, not the order
   of magnitude first-argument indexing was expected to buy. The step census in
   the benchmark section is why: `resolver.pl` scans the whole package list per
   candidate pick, that scan is what SWI does too, and this lane executes it as
   7.56 M interpreted WAM instructions over persistent Clojure maps at ~5 µs
   each. Nothing is wrong with the answer; this is a correctness-and-portability
   demonstration, not a production resolver on a 5k catalog. The corpus and the
   differential catalogs are small and finish in seconds.

2. **The structured-ITE lowering is now off for any clause containing a call** —
   which is most of `resolver.pl`. That is fix #5, and it is a *retreat to
   correctness*: the fast path was producing wrong answers. Making it work would
   mean threading `lowered-step-advanced?`-style continuation flags through the
   structured emitter *and* keeping the program counter aligned across the
   `try_me_else`/`cut_ite`/`jump` the structurer folds away. Real work, not done.

3. **First-argument indexing is implemented for the interpreted path only.**
   The lowered T4 path (`clojure_strip_switch_prefix`) still drops the switch
   prefix and tries every clause inline — that is unchanged, and sound, but it
   is not indexed. Nor does the runtime index anything the *shared* WAM text
   does not already carry a switch for: `wam_target.pl` decides that, and it is
   frozen. What it emits for `resolver.pl` is listed above; it does not, for
   instance, emit a switch for `satisfies/2` on A1 (every clause takes a
   variable there, so it indexes on A2 instead) — that is a shared-emitter
   question, not a Clojure one.
   The per-instruction interpretation cost is now the lane's top residual, and
   the benchmark section records the measurements and the next lever.

4. **The probes pin the fixed behaviour; they do not reproduce the old wrong
   answers on demand.** Before `run-wam-state` existed there was no way to read
   a compiled predicate's output argument at all, so on the pristine tree the
   probe driver cannot even resolve `<pred>-state` — it errors rather than
   answering wrongly. The wrong answers in the table above were each observed
   directly during the port (`dependents/3` → `[]`, `candidates_high_first/4` →
   the lowest version, `satisfies(v(1,0,0), gte(v(2,0,0)))` → true,
   `resolve_layered` → the classic answer, `upgrade_set_result/4` → true with an
   unbound output) but that is testimony, not a re-runnable red bar.

5. **`pkg deps` is a catalog projection**, not a resolver query — the same single
   documented exception `pkg.mjs` carries, for the same reason.

6. **`catalog/6` vs `catalog/9`.** The edge builds `catalog/6` when layers,
   excluded and aliases are all empty (the rule `diff_runner.pl` applies) while
   `store/scale_demo.pl` always builds `catalog/9`. `resolver.pl`'s accessors
   agree for empty lists and B3's answers match, but the two are not literally
   the same term.

7. **Environment, not this port:** `tests/test_wam_clojure_generator.pl` and
   `tests/test_wam_clojure_benchmark_generator.pl` fail here because `lmdb.h` is
   absent. Verified identical on the pristine files.
   `tests/test_wam_clojure_runtime_smoke.pl` also never terminates in an
   environment without the Clojure JVM jars it looks for: `find_clojure_classpath/1`
   fails, and the script backtracks into `write_wam_clojure_project/3` and
   regenerates the project forever (8,350 regenerations in 900 s). Verified on
   the **pristine** sources at the same commit, with the same count — it is not
   this change, and it is not a Clojure-lane defect this work introduced.
