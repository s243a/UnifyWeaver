<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# `cli_args.pl` → ClojureScript

**The whole of peerhailer's argument parser, transpiled from Prolog to
ClojureScript and measured against the JavaScript oracle it was written from.**

`examples/cli_args/cli_args.pl` — all 43 predicates, `parse_args/2` included —
compiles into ONE ClojureScript namespace through UnifyWeaver's pattern lane
(`clojure_target`'s A3 whole-program lowering, then `clojurescript_target`'s
JVM→JS interop rewrite). nbb loads it clean, and it agrees with the oracle on
every line either gate puts in front of it.

| gate | result |
| --- | --- |
| nbb loads the generated namespace (the `node --check` analogue) | **clean** |
| predicates lowering with no dropped goal | **40 of 40** rule predicates; the other 3 are ground-fact CONSTANT TABLES, inlined at their call sites |
| peerhailer's contract corpus (`oracle/cliArgs.test.mjs`) | **17 / 17** contract points, **25 / 25** argv-lines, error messages included |
| differential vs the JS oracle — the harness's own generator, same seed | **5067 lines, 0 divergences, 0 message mismatches, 0 crashes** |

This is the **third** pattern target to carry the whole program, after
`typescript_target` and its `vanilla_js` inheritor (see
`../patternjs/`).

---

## Run it

```bash
bash examples/cli_args/cljs/build.sh                  # compile + nbb load check
bash examples/cli_args/cljs/run_corpus_cljs.sh        # the 17-point contract gate
bash examples/cli_args/cljs/run_differential_cljs.sh  # the 5067-line gate
nbb --classpath examples/cli_args/cljs \
    examples/cli_args/cljs/probe_depth.cljs           # recursion headroom
```

Requires `swipl`, `node` and `nbb` on `PATH`.

## What is in this directory

| file | what it is |
| --- | --- |
| `build.pl` / `build.sh` | ONE command: reads the FROZEN `../cli_args.pl`, compiles `parse_args/2` with `include_dependencies(true)`, and nbb-loads the result |
| `generated/cli_args.cljs` | **compiler output**, checked in for inspection |
| `diff_runner_cljs.cljs` | the harness's stdin/JSONL protocol, under nbb |
| `run_differential_cljs.sh` | `../run_differential.sh` with the CLJS leg swapped in |
| `extract_corpus.mjs` | pulls the corpus argv-lines out of `../oracle/cliArgs.test.mjs` |
| `compare_corpus.mjs` | compares them per contract point, error messages included |
| `run_corpus_cljs.sh` | the corpus gate |
| `probe_depth.cljs` | measures the recursion headroom the no-`recur` choice leaves |

The only hand-written ClojureScript in the parsing path is
`diff_runner_cljs.cljs`, and it does exactly three things: argv line in,
`ok(P, F)` → `{"ok":{"positional":…,"flags":…}}` out, `error(M)` →
`{"error":M}`. It carries no branch that depends on an argv token. Everything
that makes `cli_args.pl` a *parser* — the two flag regexes, the strict/lenient
split, the schema lookup, the arity check, the exact wording of every error
message — is compiler output.

`generated/` rather than a flat file because that is where nbb's classpath
resolver looks for the namespace `generated.cli-args` (Clojure munges `-` to `_`
in file names). Every runner here passes `--classpath` pointing at this
directory.

---

## The corpus gate is NOT an import swap, and this is why

The `patternjs` lane runs peerhailer's corpus by changing **one line** of the
vendored test file — its `import` — and letting `node --test` do the rest. That
is the strongest possible gate and it is **impossible here**: the corpus is an
ESM module driven by node's test runner, and the transpiled parser is a
ClojureScript namespace living in a different runtime. There is no import to
swap, and pretending otherwise would be the kind of claim this whole exercise
exists to avoid.

The honest equivalent, which `run_corpus_cljs.sh` performs:

0. **run the vendored corpus against the oracle it is written for**, so the
   17 contract points are known green (`17 pass, 0 fail`);
1. **extract the argv-lines those 17 `test()` blocks exercise, from the corpus
   SOURCE** — `extract_corpus.mjs`, which fails loudly if the corpus stops
   having 17 blocks or 25 argv-lines, so the gate cannot silently shrink;
2. **run every line through both parsers** over the harness's own stdin/JSONL
   protocol;
3. **compare per contract point, including the exact `CliError` message** — the
   thing the corpus's own `assert.throws(..., /regex/)` checks and the thing a
   class-only comparison would let slide.

What this does not prove that the import swap would: that the corpus's
*assertions* hold. It proves the transpiled parser is **indistinguishable from
the parser those assertions are written against**, on every line they exercise —
which is the same conclusion by transitivity, given step 0.

(The 17 tests exercise 25 distinct argv-lines; several assert two or three
spellings of one contract point.)

---

## The four representation decisions

Ported from the TypeScript lane, with the rendering adapted to Clojure. The
reasoning is recorded in full in
`docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md`.

| | Clojure | why |
| --- | --- | --- |
| **multi-output tuple** | a positional vector `[out1 out2]` | Prolog's outputs are positional; destructured by callers with `(let [[a b] (f …)])` |
| **compound `f(A…)`** | the map `{:$ "f" :args […]}` | *not* a tagged vector: a Prolog list is already a vector here, so `["f" e1]` could not be told from the list `[f, e1]` — and telling them apart IS the gap |
| **equality** | `=`, with **no emitted helper** | Clojure's `=` is already structural, so the TS lane's `_uwEq` has no analogue. Verified on every distinction the program needs: `(= true "true")` is false, nested maps compare structurally, and vectors compare equal to seqs |
| **failure sentinel** | `(def ^:private uw-fail (js/Object.))`, tested with `identical?` | a freshly allocated host object has reference identity no term can produce and no data crossing the module edge can forge |

The four runtime representations are pairwise distinguishable and no test throws:
`string?` / `boolean?` / `sequential?` / `map?`, and `(:$ x)` answers `nil` rather
than throwing on any non-map.

## Recursion: direct calls, not `recur`

Self-calls are emitted as **direct calls** `(f …)` — the faithful analogue of the
TypeScript lane's `return pred(…)`. `recur` would give real tail-call
elimination, but only for calls the generator can *prove* are in tail position
with a matching binding vector; emitting it anywhere else is a Clojure compile
error. So the same risk the TS lane takes is taken here, and measured rather
than asserted (`probe_depth.cljs`):

| | the gates' bound | where nbb actually gives out | headroom |
| --- | --- | --- | --- |
| character walk (`long_flag_tail/1`, `first_char_index/4`, `drop_brackets/2`) | 26-char token | ~2000-char token | **~77×** |
| argv walk (`lenient_loop/5`, `strict_loop/8`) | 7 tokens | ~700 tokens | **~100×** |

Both are far outside anything a command line produces, and the 5067-line
differential records **0 crashes**. If this lowering were ever pointed at an
input whose size is not bounded by a command line, `recur` would stop being
optional — that is the honest limit, not a claim that it never matters.
