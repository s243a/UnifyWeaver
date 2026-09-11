<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — performance: close the interpreter gap on the argparser (item 1, GP-PERF)

> Branch from `grok/wamjs-cli-args` (latest JS-WAM: everything + the A2 argparser
> build) and EXTEND. The wamjs argparser runs the 5067-line differential ~50–100×
> slower than the JS oracle (interpreter tier). Make it substantially faster with
> ZERO semantic change — the corpus, the differential, and conformance are the
> unchanged acceptance gates, now doubling as performance regression tests.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-cli-args`** as `grok/wamjs-perf`. Prolog dialect is
SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v18+).

### The gap (GP-PERF, item 1 of the post-demonstration roadmap)
`examples/cli_args/wamjs/` runs the 5067-line differential in ~4.6–8.9 s vs the JS
oracle's ~0.05–0.18 s. A `UW_PROFILE=1` run over 200 differential lines says exactly
where the time goes (measured on the merged branch — reproduce it yourself first):

```
pred                             calls       instr     CPs   maxCP            ns
first_char_index/4                3910       96193    3910       8     166740376
starts_with/2                     1110       36218     458       9      55990731
string_member/2                   1804       27219    1821       5      50829152
pair_lookup/3                     1245       22842    1117       8      37777446
lenient_loop/5                     588       15217    1113      10      34116698
default_registry/1                 200       96200       0       0      25593512
split_flag_token/3                 363        9306     363       9      22465859
```

Three shapes of waste, in priority order:
1. **Det/semidet helpers pay a choice point per step** — `first_char_index/4` creates
   3910 CPs for 3910 calls; `string_member/2`, `pair_lookup/3`, `flags_*` similar.
   These are exactly the T4/T5 shapes the Tier-2 lowered emitter
   (`wam_javascript_lowered_emitter.pl`) exists for — but the A2 build compiles
   interpreter-only. Get the argparser building in **`mixed` emit mode** and extend
   the lowered emitter wherever these hot predicates currently fall back (the
   fallback is loud — instrument it to LIST which predicates fell back and why).
2. **`default_registry/1` rebuilds a huge ground term every call** — 96,200
   instructions for 200 calls, zero CPs. A ground fact's term is immutable: build it
   ONCE and reuse (intern/memoize the constructed term per predicate where every
   clause is ground — semantics unchanged because ground terms cannot alias into
   bindings; justify trail-safety in a comment).
3. **Interpreter dispatch overhead generally** — after 1+2, re-profile; if the loop
   itself is now the cost, take the cheap wins only (avoid re-allocating hot
   objects, tighten `deref`) — no rewrites.

### Goal
≥5× wall-time improvement on the full 5067-line differential (report before/after
timings from `run_differential_wamjs.sh`; stretch: ≥10×), with:
- `node --test examples/cli_args/wamjs/cliArgs.wamjs.test.mjs` → still **17/17**;
- the differential → still **0 divergences, 0 message mismatches**;
- `CONFORMANCE_TARGETS=javascript` harness → still green;
- all builtin/parser/fact-source/lowered/term-meta/string/profiling suites → green.

### Reference
- `src/unifyweaver/targets/wam_javascript_lowered_emitter.pl` — the T4/T5/T6/ITE
  fast paths + fallback machinery. Note it already preserves the Y-snapshot
  convention (Allocate/Deallocate through `Runtime.step`).
- `examples/cli_args/wamjs/build.pl` — switch/extend to `emit_mode(mixed)`;
  regenerate via `build.sh`. The checked-in `js/` must be regenerated to match.
- `templates/targets/javascript_wam/runtime.js.mustache` — the interpreter +
  profiler hooks (use `UW_PROFILE=1`/`json` before/after per change; paste tables).
- `docs/WAM_BACKEND_CONVENTIONS.md` §7/§8 — the new conventions; your lowered
  functions must honour both (Execute-of-builtin return path; no Allocate-framing
  assumptions).

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Lowered-emitter extensions + runtime changes; regenerated
  `examples/cli_args/wamjs/js/` (mixed mode) + updated `build.pl`/`build.sh`.
- Ground-fact term memoization with its trail-safety justification.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: performance section — before/after
  differential timings, per-predicate profile tables, which predicates lower vs
  interpret in the argparser build, the memoization rule.
- Extend `tests/test_wam_javascript_lowered.pl` (or builtins) with probes for each
  newly-lowered shape (node-vs-SWI) and a memoized-fact correctness probe
  (call twice, mutate nothing, bindings independent across calls).
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, `wam_target.pl`, `wam_text_parser.pl`, the conformance
  harness, or ANY frozen file under `examples/cli_args/` outside `wamjs/`
  (`cli_args.pl`, `oracle/`, the harness, `patternjs/`, `cljs/` are all frozen).

### Constraints — CRITICAL
- Wrong-but-fast is a failure: any lowered path that cannot reproduce interpreter
  semantics exactly must keep falling back. The differential IS the semantics gate.
- Profiling stays zero-cost when off (D38's bar); your changes must not regress the
  profiler's numbers' meaning (lowered predicates already report calls-only).

### Acceptance (must pass before handoff)
1. Corpus 17/17; differential 5067 lines, 0/0 — WITH the before/after wall times.
2. All suites green; `CONFORMANCE_TARGETS=javascript` green.
3. `UW_PROFILE=1` before/after tables for the 200-line sample, showing where the
   time went and where it goes now; state the achieved speedup factor.
4. The fallback list: which argparser predicates still interpret, and why.

### Handoff format
Return: changed files, the speedup factor + timing evidence, per-change profile
deltas, which predicates now lower vs interpret, the memoization rule + safety
argument, and any residual (with the next perf lever you'd pull).

## ↑↑↑ Copy to here
