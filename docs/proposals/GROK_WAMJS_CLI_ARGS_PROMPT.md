<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — A2: transpile the peerhailer argparser via wam_javascript

> Branch from `grok/wamjs-profiling` (latest JS-WAM) and EXTEND. This is the
> **maturity demonstration**: a REAL ~400-line Prolog program — peerhailer's CLI
> argument parser, oracle-verified against the production JS — compiled by
> UnifyWeaver's `wam_javascript` target and passing the production test corpus
> with only the import swapped. Everything the compiler/runtime can't handle is
> a real bug or gap: fix it in the target/runtime, never by editing the parser's logic.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-profiling`** (the latest JS-WAM state) as `grok/wamjs-cli-args`.
Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v18+).

### Get the program under test
The A1 deliverables live on branch **`claude/peerhailer-exploratory-docs-aodas5`**
(not on your parent). Bring over ONLY the `examples/cli_args/` directory:
`git checkout origin/claude/peerhailer-exploratory-docs-aodas5 -- examples/cli_args/`
It contains:
- `cli_args.pl` — the Prolog reference parser (module `cli_args`, `parse_args/2,3`),
  **oracle-verified**: 17/17 on the corpus + 5067-line differential vs the real JS
  with 0 divergences. Pure, det, tail-recursive, no cuts/throws/pcre. Its README
  documents every construct used and the JS-quirk modelling (prototype-chain keys,
  `__proto__` no-op, the two flag regexes, lenient `--` empty-key).
- `oracle/cliArgs.js` + `oracle/cliArgs.test.mjs` — the production oracle + its
  17-test corpus (the bar), `oracle/package.json` (ESM scoping).
- `diff_runner.mjs` / `diff_runner.pl` / `gen_cases.mjs` / `compare_jsonl.mjs` /
  `run_differential.sh` — the seeded differential harness (stdin argv-lines →
  JSONL; README documents the protocol).

### Goal (A2)
1. **Compile `cli_args.pl` with the `wam_javascript` target** (interpreter tier is
   fine; use `mixed`/lowered where it works) into a Node module.
2. **A thin hand-written ESM shim** — `examples/cli_args/wamjs/cliArgs.mjs` — that
   exports the production interface over the compiled module:
   `export function parseArgs(argv, registry?)` and `export class CliError extends Error`.
   The shim ONLY: converts the JS `argv` string array into the WAM call, runs the
   compiled `parse_args/2` (or `/3` when a registry is given — if threading a custom
   registry through is disproportionate, ship `/2` and state it: the corpus only
   uses the default), converts the result term back (`ok(Positional,Flags)` →
   `{positional, flags}` with flags as a plain object in pair order, `true`/`false`
   atoms → JS booleans, strings → strings; `err(M)` → `throw new CliError(M)`).
   The PARSING LOGIC must all be the compiled Prolog — a shim that re-implements any
   parse rule is cheating and voids the demonstration.
3. **Pass the corpus**: a copy `examples/cli_args/wamjs/cliArgs.wamjs.test.mjs` of
   `oracle/cliArgs.test.mjs` with ONLY the import line changed to the shim. All 17
   green under `node --test`.
4. **Pass the differential**: re-point the harness's Prolog leg at your build (a
   `diff_runner_wamjs.mjs` speaking the same stdin/JSONL protocol via the shim), same
   seed: **0 divergences, 0 message mismatches** vs the oracle across the full sample.
5. **Build script**: `examples/cli_args/wamjs/build.sh` — one command that invokes
   swipl + the wam_javascript target to regenerate the compiled module from
   `cli_args.pl`, so the build is reproducible (no checked-in artifacts that can't
   be regenerated; checking in the generated module TOO is fine for inspection).

### Expected reality (read this)
`cli_args.pl` exercises: multi-accumulator tail loops, deep if-then-else chains,
`sub_string/5`, `string_chars/2` both directions, `char_code/2`, `string_concat/3`,
`string_length/2`, `append/3`, `reverse/2`, `length/2`, pair lists (`K-V`), string
`==`/`\==` (with the D34 string tag semantics live: `""` distinct, `true` atom vs
`"true"` string distinct), integer arithmetic and char literals (`0'a` etc.), and
compiled string literals (D37). If any of it miscompiles or misruns, that's a
target/runtime bug — fix it on this branch (runtime template, emitter,
`wam_javascript_target.pl`) with a minimal probe added to
`tests/test_wam_javascript_builtins.pl`, and list each such fix in the handoff.
If you hit a genuinely L-sized compiler gap you cannot close, document it precisely
(minimal reproduction) and say how far you got — but the expectation is that the
JS WAM tier, being a full interpreter, runs this program whole.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers on new files; oracle files stay verbatim)
- `examples/cli_args/wamjs/`: `build.sh`, the generated module, `cliArgs.mjs` shim,
  `cliArgs.wamjs.test.mjs`, `diff_runner_wamjs.mjs`, a short `README.md` (build, run,
  what the shim does and does NOT do).
- Any runtime/emitter/target fixes the program forced, each with a probe in
  `tests/test_wam_javascript_builtins.pl`.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: "runs the peerhailer argparser (A2)" with
  the corpus + differential numbers.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, `wam_target.pl`, `wam_text_parser.pl`, the conformance harness,
  or — CRITICAL — `examples/cli_args/cli_args.pl` and everything else at the
  `examples/cli_args/` top level (the reference and harness are frozen; your work
  lives under `examples/cli_args/wamjs/`).

### Constraints
- Do NOT break the 48/48 conformance or the builtin/parser/fact-source/lowered/
  term-meta/string/profiling suites — re-run all. No new runtime dependency; the
  shim + compiled module must run under plain `node` (ESM ok).

### Acceptance (must pass before handoff)
1. `bash examples/cli_args/wamjs/build.sh` regenerates the module from `cli_args.pl`.
2. `node --test examples/cli_args/wamjs/cliArgs.wamjs.test.mjs` → **17/17**.
3. The differential with the wamjs leg → sample ≥ 5000, **0 divergences, 0 message
   mismatches** (paste the summary block).
4. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` and the
   lowered/fact-sources suites → green.
5. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still green.

### Handoff format
Return: the wamjs/ tree, every target/runtime fix the program forced (each: symptom →
root cause → fix → probe), the corpus + differential numbers, shim scope (what it
converts, confirmation it implements no parse rule, whether `/3` registries are
supported), rough performance note (time to parse the 5000-line sample vs the JS
oracle), and any residual.

## ↑↑↑ Copy to here
