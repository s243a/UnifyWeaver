<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# `patternjs` — peerhailer's CLI parser, transpiled

This directory is the **output** of UnifyWeaver's pattern lane applied to
`../cli_args.pl`, plus the two harnesses that hold it to the same bar the Prolog
reference was held to in step A1.

`../cli_args.pl` is a Prolog reimplementation of peerhailer's `src/cliArgs.js`,
verified against the vendored JavaScript oracle in `../oracle/`. What is here is
that Prolog **compiled back into JavaScript by UnifyWeaver** — the whole program,
all 43 predicates, `parse_args/2` included — and then measured against the same
oracle.

| | |
| --- | --- |
| contract corpus (`../oracle/cliArgs.test.mjs`, import swapped) | **17 / 17** |
| differential vs the JS oracle, same generator and seed | **5067 lines, 0 divergences, 0 message mismatches** |
| `node --check` on the generated module | clean |
| predicates omitted / goals dropped | none |

## Files

| file | what it is |
| --- | --- |
| `build.sh` | **the one command.** Compiles `../cli_args.pl` into `cliArgs.generated.mjs` and `node --check`s it. |
| `build.pl` | the compile itself: read the frozen source's clauses into `user`, then `compile_module/3` with `include_dependencies(true)` through `vanilla_js_target`. |
| `cliArgs.generated.mjs` | **compiler output.** 40 functions — every predicate `parse_args/2` transitively calls — plus the compound-term runtime. Not edited by hand. |
| `cliArgs.mjs` | the **edge shim**: `parseArgs(argv, registry?)` and `class CliError`. Term↔JS conversion only — no parse logic (see the header comment for the exhaustive list of what it does). |
| `cliArgs.patternjs.test.mjs` | peerhailer's own 17-test contract corpus, with **one** line changed: the import. |
| `diff_runner_patternjs.mjs` | the transpiled side of the differential harness; same stdin/JSONL protocol as `../diff_runner.mjs` and `../diff_runner.pl`. |
| `run_differential_patternjs.sh` | the A4 run for this lane: the harness's own generator and comparator, with the transpiled parser opposite the oracle. |

## Reproducing

```bash
bash examples/cli_args/patternjs/build.sh
node --test examples/cli_args/patternjs/cliArgs.patternjs.test.mjs
bash examples/cli_args/patternjs/run_differential_patternjs.sh
```

## What is hand-written and what is not

The only hand-written JavaScript that the parser's behaviour passes through is
`cliArgs.mjs`, and it does exactly three things: it turns a JS argv array into the
token list the compiled entry point takes, it turns the compiled `ok(P, F)` term
into `{ positional, flags }` (unwrapping `{$: "-", args: [k, v]}` pairs into a
plain object, in pair order), and it turns `error(M)` into
`throw new CliError(M)`. Every decision about what an argv line *means* — both
flag regexes, the strict/lenient split, the schema lookup, the arity check, the
exact wording of every error message — is in `cliArgs.generated.mjs`, which is
compiler output.

The `A1` README's standard for this demo was: *a hand-written JS shim around the
compiled pieces would be cheating.* The shim here carries no branch that depends
on an argv token.

## How to read the generated module

Four representation choices, all documented in
`docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md`:

* a Prolog atom or string is a JS string; `true`/`false` are JS booleans; a list
  is a JS array; a compound `f(A1..An)` is `{$: "f", args: [...]}` (G-A3-12/13);
* a predicate with more than one output argument returns a positional tuple
  (G-A3-9);
* a predicate that is **semidet and has outputs** returns its answer or the
  module-private `_uwFail` Symbol; callers test `x !== _uwFail` (G-A3-18);
* `parse_args/2` and `parse_args/3` become `parse_args_2` and `parse_args_3`,
  because JavaScript has no arity overloading (G-A3-18).
