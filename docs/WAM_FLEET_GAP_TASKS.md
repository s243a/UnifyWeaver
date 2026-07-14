# Hybrid WAM fleet — gap task breakdown

Atomic, hand-off-ready task cards for the fleet-wide gaps identified in
[`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md)
(§"Future-work scope") and the per-target
[`WAM_*_STATUS.md`](WAM_HYBRID_TARGETS_COMPARISON.md) docs. Each card is
self-contained so a single coding agent can pick it up in isolation.

## How to use this doc

- Each card names the **files to touch**, a **reference implementation
  to copy from** (with predicate/line anchors), terse **steps**, and an
  **acceptance** command. Grab one card, one branch, one PR.
- Line numbers are anchors captured during recon; treat them as "look
  near here", not exact — verify against the current tree.
- Inline **(verify: …)** notes flag a detail the recon could not fully
  pin down; resolve it as the first step of the task.
- **Size:** S ≈ <½ day (localized edit), M ≈ 1–2 days (one subsystem),
  L ≈ multi-day (new runtime surface + tests).
- Cards were produced by source-reading subagents; **re-confirm before
  implementing** — nothing here has been compiled or run.

## Index

| ID | Lever | Target | Size | Depends on |
|---|---|---|---|---|
| CONF-FSHARP | Conformance adapter | F# | M | — |
| CONF-LLVM | Conformance adapter | LLVM | L | — |
| CONF-R | Conformance adapter | R | M | — |
| CONF-CLOJURE | Conformance adapter | Clojure | L | — |
| CONF-LUA | Conformance adapter | Lua | M | — |
| CONF-KOTLIN ✅ | Conformance adapter | Kotlin | M | done — opt-in (`cursor/conf-kotlin-f421`); append green, 5 xfails |
| KT-LIST-BACKTRACK ✅ | Conformance gap fix | Kotlin | M | done — X heap-identity vars (`cursor/kt-list-backtrack-f421`) |
| KT-ARITH-SLASH-FUNCTOR ✅ | Conformance gap fix | Kotlin | S | done — `///2` last-slash parse (`cursor/kt-arith-slash-functor-f421`) |
| KT-Y-ENV-RECURSION ✅ | Conformance gap fix | Kotlin | M | done — Y heap-identity vars (`cursor/kt-y-env-recursion-f421`) |
| PARSE-C | Runtime-parser entry | C | S | — |
| PARSE-GO | Runtime-parser entry | Go | S | — |
| PARSE-SCALA | Runtime-parser entry | Scala | S | — |
| PARSE-CLOJURE | Runtime-parser entry | Clojure | S | — |
| PARSE-LUA | Runtime-parser entry | Lua | S | — |
| LMDB-GO | LMDB policy tiers | Go | M | — |
| LMDB-SCALA | LMDB policy tiers | Scala | M | — |
| LMDB-C-0 | LMDB lookup source (prereq) | C | M | — |
| LMDB-C | LMDB policy tiers | C | L | LMDB-C-0 |
| LMDB-R-0 | LMDB lookup source (prereq) | R | M | — |
| LMDB-R | LMDB policy tiers | R | L | LMDB-R-0 |
| ISO-C | ISO three-form (new) | C | L | — |
| ISO-GO | ISO three-form (new) | Go | L | — |
| ISO-SCALA | ISO three-form (new) | Scala | L | — |
| ISO-R | ISO three-form (new) | R | L | — |
| ISO-PYTHON | ISO three-form (finish) | Python | S | — |
| ISO-FSHARP | ISO three-form (finish) | F# | S | — |
| KERN-FSHARP | Finish F# kernel templates | F# | L | — |
| EMIT-ILASM | Lowered emitter | ILAsm | L | — |
| EMIT-JVM | Lowered emitter | JVM | L | — |
| EMIT-KOTLIN ✅ | Lowered emitter | Kotlin | M | done — flat facts/unify (`cursor/emit-kotlin-lowered-f421`) |
| EMIT-KOTLIN-2 ✅ | Lowered emitter (structures) | Kotlin | M | done — write-mode structures (`cursor/emit-kotlin-structures-f421`) |
| EMIT-KOTLIN-3 ✅ | Multi-clause deterministic | Kotlin | M | done — T5/T4 no call (`cursor/emit-kotlin-multi-clause-f421`) |
| EMIT-KOTLIN-4 ✅ | Last-call `execute` | Kotlin | M | done — tail execute (`cursor/emit-kotlin-execute-f421`) |
| EMIT-KOTLIN-5 | Mid-body `call` | Kotlin | M | fib/ack / non-tail continuation |
| BENCH-KOTLIN ✅ | Lowered vs interpreter timing | Kotlin | S | done — mostly no win (`cursor/bench-kotlin-f421`) |
| BENCH-LLVM | Effective-distance bench row | LLVM | L | — |
| BENCH-CPP | Effective-distance bench row | C++ | L | — |
| BENCH-C | Effective-distance bench row | C | M | — |
| BENCH-GO | Effective-distance bench row | Go | M | — |
| BENCH-R | Effective-distance bench row | R | L | — |

Suggested ordering: **start with `EMIT-KOTLIN`** — lowest-risk (least-mature
target, plumbing already present) and fully spec'd below as the first
hand-off. Then the **S** parser cards are quick wins; **conformance
adapters** for F#/LLVM/R give the biggest correctness-visibility return;
LMDB and ISO cards for C/R carry prerequisites; the F# kernel-template
card unblocks a Primary-tier target's headline gap.

---

## Lever: Conformance adapters

Register unregistered hybrid WAM targets in
`tests/test_wam_cross_target_conformance.pl`. All stay **opt-in** (no
`ct_default_target`) because each builds a per-program project with an
external toolchain.

### CONF-FSHARP: Register F# in the cross-target conformance harness
- **Lever:** Conformance adapters  **Target:** F#  **Size:** M  **Depends on:** —
- **Goal:** Add an F# adapter (`conformance_target(fsharp)` + `ct_toolchain`/`ct_build`/`ct_run`/`ct_teardown`) so the shared WAM spec runs against the F# backend.
- **Files to touch:** `tests/test_wam_cross_target_conformance.pl`
- **Reference to copy from:** `tests/test_wam_cross_target_conformance.pl` — the **Python adapter** block (lines ~705-739, interpreted/scripted style) for overall shape; project writer is `write_wam_fsharp_project/3` in `src/unifyweaver/targets/wam_fsharp_target.pl:4926`.
- **Steps:**
  1. Add `:- use_module('../src/unifyweaver/targets/wam_fsharp_target', [write_wam_fsharp_project/3]).` near the other target imports (lines ~50-71).
  2. Add `conformance_target(fsharp).` to the registry (lines ~77-85). Leave it opt-in (do NOT add `ct_default_target`).
  3. Add `ct_toolchain(fsharp, [dotnet]).` (alongside lines ~283-291). (verify: exact executable — `dotnet` drives `dotnet run`/`dotnet build` for an F# project.)
  4. `ct_build(fsharp, Preds, Queries, fsharp_ctx(Dir, Map))`: mirror Python's body — `ct_tmp_dir('tmp_ct_fsharp', Dir)`, `synth_wrappers`, `strip_pred`, `qualify_user`, then `write_wam_fsharp_project(AllPreds, [module_name(wam_ct)], Dir)`. Since F# is compiled, add a `dotnet build` gate via `run_proc` like the Go/Rust adapters (throw on nonzero). (verify: whether the generated project self-builds via a `.fsproj` and how the 0-arity wrapper is queried at runtime — inspect what `write_wam_fsharp_project` emits as an entry point / CLI shim; may need a driver overwrite like Go/Rust.)
  5. `ct_run(fsharp, fsharp_ctx(Dir, Map), K, A, Bool)`: look up wrapper name in `Map`, run `dotnet run -- <key>` via `run_proc_out`, map stdout to bool with `bool_of_string` (or absence-of-`false.` like Python — match whatever the generated runner prints).
  6. `ct_teardown(fsharp, fsharp_ctx(Dir, Map)) :- cleanup_dir(Dir), abolish_wrappers(Map).`
  7. Add `test(fsharp, [condition(ct_available(fsharp))]) :- run_target_conformance(fsharp).` inside the `begin_tests/end_tests` block (lines ~323-334).
- **Acceptance:** `CONFORMANCE_TARGETS=fsharp swipl -g run_tests tests/test_wam_cross_target_conformance.pl` passes (skips cleanly if `dotnet` absent).

### CONF-LLVM: Register LLVM in the cross-target conformance harness
- **Lever:** Conformance adapters  **Target:** LLVM  **Size:** L  **Depends on:** —
- **Goal:** Add an LLVM adapter that emits IR, compiles it to a native binary, and runs the shared spec.
- **Files to touch:** `tests/test_wam_cross_target_conformance.pl`
- **Reference to copy from:** `tests/test_wam_cross_target_conformance.pl` — the **C adapter** block (lines ~895-923, gcc + exit-code boolean) is the closest fit; project writer is `write_wam_llvm_project/3` in `src/unifyweaver/targets/wam_llvm_target.pl:1181`.
- **Caveat:** `write_wam_llvm_project/3` signature is `(+Predicates, +Options, +OutputFile)` — it writes a **single output file (LLVM IR), NOT a project directory** like every other target. The adapter must create the tmp dir itself, write the `.ll` into it, then compile with `llc`+`clang` (or `clang` directly on the `.ll`). (verify: whether the emitted IR contains a `main`/entry that takes a `pred/arity` argv and returns 0/1 by exit code, or whether a hand-written C driver + generated object must be linked like the C adapter's `driver.c`.)
- **Steps:**
  1. Add `:- use_module('../src/unifyweaver/targets/wam_llvm_target', [write_wam_llvm_project/3]).` to the imports.
  2. Add `conformance_target(llvm).` (opt-in only).
  3. Add `ct_toolchain(llvm, [clang, llc]).` (verify: whether both `clang` and `llc` are needed or `clang` alone suffices to compile `.ll` → binary).
  4. `ct_build(llvm, Preds, Queries, llvm_ctx(Dir, Map))`: `ct_tmp_dir('tmp_ct_llvm', Dir)`, `synth_wrappers`, prep preds, `directory_file_path(Dir, 'wam.ll', LlPath)`, `write_wam_llvm_project(AllPreds, [no_kernels(true)], LlPath)`, then compile to `runner` with `clang` via `run_proc` (throw on nonzero). If IR has no CLI entry, generate + link a driver like the C adapter (`c_runner_driver` pattern).
  5. `ct_run(llvm, llvm_ctx(Dir, Map), K, A, Bool)`: exec `runner <key>` via `shell/2`, map exit status 0→true, 1→false (mirror C/C++ adapters).
  6. `ct_teardown(llvm, llvm_ctx(Dir, Map)) :- cleanup_dir(Dir), abolish_wrappers(Map).`
  7. Add `test(llvm, [condition(ct_available(llvm))]) :- run_target_conformance(llvm).` in the tests block.
- **Acceptance:** `CONFORMANCE_TARGETS=llvm swipl -g run_tests tests/test_wam_cross_target_conformance.pl` passes (skips if `clang`/`llc` absent).

### CONF-R: Register R in the cross-target conformance harness
- **Lever:** Conformance adapters  **Target:** R  **Size:** M  **Depends on:** —
- **Goal:** Add an R adapter so the shared WAM spec runs against the R backend (interpreted, no build step).
- **Files to touch:** `tests/test_wam_cross_target_conformance.pl`
- **Reference to copy from:** `tests/test_wam_cross_target_conformance.pl` — the **Python adapter** (lines ~705-739; R is interpreted, same no-build shape); project writer is `write_wam_r_project/3` in `src/unifyweaver/targets/wam_r_target.pl:1903`.
- **Steps:**
  1. Add `:- use_module('../src/unifyweaver/targets/wam_r_target', [write_wam_r_project/3]).`
  2. Add `conformance_target(r).` (opt-in only).
  3. Add `ct_toolchain(r, ['Rscript']).` (verify: exact executable name/case — `Rscript`).
  4. `ct_build(r, Preds, Queries, r_ctx(Dir, Map))`: exact Python shape — `ct_tmp_dir('tmp_ct_r', Dir)`, `synth_wrappers`, `strip_pred`/`qualify_user`, `write_wam_r_project(AllPreds, [module_name(wam_ct)], Dir)`. No compile gate. (verify: name of the generated entry script, e.g. `main.R`, and how it takes a `pred/arity` arg — inspect `write_wam_r_project` output.)
  5. `ct_run(r, r_ctx(Dir, Map), K, A, Bool)`: `run_proc_out('Rscript', ['main.R', KeyStr], Dir, _, OutStr)`, map to bool (absence-of-`false.` like Python, or `bool_of_string` — match the generated runner's output).
  6. `ct_teardown(r, r_ctx(Dir, Map)) :- cleanup_dir(Dir), abolish_wrappers(Map).`
  7. Add `test(r, [condition(ct_available(r))]) :- run_target_conformance(r).`
- **Acceptance:** `CONFORMANCE_TARGETS=r swipl -g run_tests tests/test_wam_cross_target_conformance.pl` passes (skips if `Rscript` absent).

### CONF-CLOJURE: Register Clojure in the cross-target conformance harness
- **Lever:** Conformance adapters  **Target:** Clojure  **Size:** L  **Depends on:** —
- **Goal:** Add a Clojure adapter so the shared WAM spec runs against the Clojure (JVM) backend.
- **Files to touch:** `tests/test_wam_cross_target_conformance.pl`
- **Reference to copy from:** `tests/test_wam_cross_target_conformance.pl` — the **Python adapter** for the generate-then-run shape (JVM startup makes it script-like, no separate native binary); project writer is `write_wam_clojure_project/3` in `src/unifyweaver/targets/wam_clojure_target.pl:101`.
- **Caveat:** Clojure runs on the JVM via `clojure`/`clj`; there is no native-binary step. Per-run JVM startup is slow (like Haskell) — keep it opt-in and expect slowness. (verify: whether the "+JNI" hint implies a native-interop path; the standard `write_wam_clojure_project/3` emits a JVM Clojure project — confirm no JNI toolchain is actually required, and confirm the generated project's run command and how it accepts a `pred/arity` argv.)
- **Steps:**
  1. Add `:- use_module('../src/unifyweaver/targets/wam_clojure_target', [write_wam_clojure_project/3]).`
  2. Add `conformance_target(clojure).` (opt-in only).
  3. Add `ct_toolchain(clojure, [clojure]).` (verify: `clojure` vs `clj` — `clj` is the interactive wrapper; `clojure` is the scripting entry).
  4. `ct_build(clojure, Preds, Queries, clojure_ctx(Dir, Map))`: `ct_tmp_dir('tmp_ct_clojure', Dir)`, `synth_wrappers`, prep preds, `write_wam_clojure_project(AllPreds, [module_name(wam_ct)], Dir)`. Add a warm-up/AOT gate via `run_proc` only if the project needs compilation. (verify: deps.edn vs project.clj layout emitted, and the run invocation.)
  5. `ct_run(clojure, clojure_ctx(Dir, Map), K, A, Bool)`: run the project (e.g. `clojure -M -m <ns> <key>`) via `run_proc_out`, map stdout to bool with `bool_of_string`. (verify: exact `-M`/`-X` invocation and main namespace from the generated project.)
  6. `ct_teardown(clojure, clojure_ctx(Dir, Map)) :- cleanup_dir(Dir), abolish_wrappers(Map).`
  7. Add `test(clojure, [condition(ct_available(clojure))]) :- run_target_conformance(clojure).`
- **Acceptance:** `CONFORMANCE_TARGETS=clojure swipl -g run_tests tests/test_wam_cross_target_conformance.pl` passes (skips if `clojure` absent).

### CONF-LUA: Register Lua in the cross-target conformance harness
- **Lever:** Conformance adapters  **Target:** Lua  **Size:** M  **Depends on:** —
- **Goal:** Add a Lua adapter so the shared WAM spec runs against the Lua backend (interpreted, no build step).
- **Files to touch:** `tests/test_wam_cross_target_conformance.pl`
- **Reference to copy from:** `tests/test_wam_cross_target_conformance.pl` — the **Python adapter** (lines ~705-739); project writer is `write_wam_lua_project/3` in `src/unifyweaver/targets/wam_lua_target.pl:742`.
- **Steps:**
  1. Add `:- use_module('../src/unifyweaver/targets/wam_lua_target', [write_wam_lua_project/3]).`
  2. Add `conformance_target(lua).` (opt-in only).
  3. Add `ct_toolchain(lua, [lua]).` (verify: interpreter name — `lua` vs `lua5.4`/`luajit`).
  4. `ct_build(lua, Preds, Queries, lua_ctx(Dir, Map))`: exact Python shape — `ct_tmp_dir('tmp_ct_lua', Dir)`, `synth_wrappers`, prep preds, `write_wam_lua_project(AllPreds, [module_name(wam_ct)], Dir)`. No compile gate. (verify: generated entry filename, e.g. `main.lua`, and its `pred/arity` argv handling.)
  5. `ct_run(lua, lua_ctx(Dir, Map), K, A, Bool)`: `run_proc_out(lua, ['main.lua', KeyStr], Dir, _, OutStr)`, map to bool (match generated runner's output convention).
  6. `ct_teardown(lua, lua_ctx(Dir, Map)) :- cleanup_dir(Dir), abolish_wrappers(Map).`
  7. Add `test(lua, [condition(ct_available(lua))]) :- run_target_conformance(lua).`
- **Acceptance:** `CONFORMANCE_TARGETS=lua swipl -g run_tests tests/test_wam_cross_target_conformance.pl` passes (skips if `lua` absent).

### CONF-KOTLIN: Register Kotlin in the cross-target conformance harness
- **Lever:** Conformance adapters  **Target:** Kotlin  **Size:** M  **Depends on:** EMIT-KOTLIN-2
- **Status:** ✅ **Landed** on `cursor/conf-kotlin-f421` (2026-07-12). Opt-in `conformance_target(kotlin)` + `kotlin_functions` (interpreter vs `emit_mode(functions)`). Added `WamRuntime.tryRun` + `conformance_main(true)` so Main prints `true`/`false` without changing the human-facing register dump. Gradle `compileKotlin` build gate. **Measured:** `append/3` green; `ct_xfail` for member/reverse (list CDR `Var(Xn)` clobber under backtrack), builtins (`///2` functor split in `evalArith`), fib/ack (scoped Y-regs unbound after recursive call). Follow-ups: KT-LIST-BACKTRACK, KT-ARITH-SLASH-FUNCTOR, KT-Y-ENV-RECURSION.
- **Goal:** Add a Kotlin adapter so the shared WAM classic-program spec runs against the Kotlin hybrid backend (interpreter and functions modes).
- **Files to touch:** `tests/test_wam_cross_target_conformance.pl`, `templates/targets/kotlin_wam/WamRuntime.kt.mustache`, `templates/targets/kotlin_wam/Main.kt.mustache`, `src/unifyweaver/targets/wam_kotlin_target.pl`.
- **Acceptance:** `CONFORMANCE_TARGETS=kotlin,kotlin_functions LANG=C.UTF-8 swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` runs both adapters (skips if `gradle` absent); xfails documented; append passes.

### KT-LIST-BACKTRACK: Fix list placeholder clobber under backtracking (Kotlin)
- **Lever:** Conformance gap fix  **Target:** Kotlin  **Size:** M  **Depends on:** CONF-KOTLIN
- **Status:** ✅ **Landed** on `cursor/kt-list-backtrack-f421` (2026-07-12; stacks on KT-ARITH-SLASH-FUNCTOR). `newVariable` for X-registers now allocates a heap identity `H<n>` (Struct/list cells hold `Var(H<n>)`); rebinding `Xn` via `unify_variable` no longer mutates already-built terms. Y-registers still use scoped names (KT-Y-ENV-RECURSION). Removed member/reverse `ct_xfail`s.
- **Goal:** Retire `ct_xfail(kotlin, member)` / `reverse`. Heap-built lists store CDR as `Var(Xn)`; recursive clauses reuse `Xn` via `unify_variable` and overwrite the binding that shared Structs still reference.
- **Hint:** deep-copy structs at choice points, or heap refs instead of register-named vars for structure args (same class Haskell fixed with cons-cell finalize).

### KT-ARITH-SLASH-FUNCTOR: Parse `//` functor without split-on-`/` (Kotlin)
- **Lever:** Conformance gap fix  **Target:** Kotlin  **Size:** S  **Depends on:** CONF-KOTLIN
- **Status:** ✅ **Landed** on `cursor/kt-arith-slash-functor-f421` (2026-07-12). `evalArith` now uses `functorName` (strip trailing `/<digits>` via `substringBeforeLast`) so `"///2"` → `"//"`. Removed `ct_xfail` for builtins; append+builtins green under kotlin and kotlin_functions.
- **Goal:** Retire builtins xfail for `cbi_arith`. `evalArith` does `functor.split("/")` so `///2` yields empty name. Strip only a trailing `/<digits>` (WAT/Haskell `bareArithOp` / last-component fix).

### KT-Y-ENV-RECURSION: Y-register bind-through across recursive call (Kotlin)
- **Lever:** Conformance gap fix  **Target:** Kotlin  **Size:** M  **Depends on:** CONF-KOTLIN
- **Status:** ✅ **Landed** on `cursor/kt-y-env-recursion-f421` (2026-07-12; stacks on KT-LIST-BACKTRACK). Extends heap-identity `newVariable` to Y-registers (drops scoped `Y@E` Var names that recursive `allocate`/`call` could miss on deref). Removed fib/ack `ct_xfail`s; **all classic programs green**, no remaining Kotlin xfails.
- **Goal:** Retire fib/ack xfails. After recursive `call`/`execute`, `is/2` sees unbound scoped temps (`Y5@E9`) inside `+/2` trees — permanent-variable / environment bank gap.

---

## Lever: Runtime-parser capability entries

Add entries to `src/unifyweaver/targets/wam_runtime_parser_capability.pl`
so `wam_target_runtime_parser(<t>, …)` resolves a real mode. All five
targets get a `compiled(prolog_term_parser)` opt-in mode with `none`
default (no confirmed native Prolog-term parser); each needs the
`normalize_runtime_parser_target/2` alias pair added before the
catch-all clause.

### PARSE-C: Add C runtime-parser capability entry
- **Lever:** Runtime-parser capability entries  **Target:** C  **Size:** S  **Depends on:** —
- **Goal:** Register `wam_c`'s runtime-parser capability so `wam_target_runtime_parser(c, ...)` resolves a real mode instead of falling through to `none`/`domain_error`.
- **Files to touch:** `src/unifyweaver/targets/wam_runtime_parser_capability.pl`
- **Reference to copy from:** same file — the `wam_fsharp` entries (`target_runtime_parser_default(wam_fsharp, none).` at line ~138 and `target_runtime_parser_mode_(wam_fsharp, compiled(prolog_term_parser)).` at line ~146), plus the `normalize_runtime_parser_target(fsharp, wam_fsharp)` alias pair at lines ~154-155.
- **Steps:**
  1. Add alias pair: `normalize_runtime_parser_target(c, wam_c) :- !.` and `normalize_runtime_parser_target(wam_c, wam_c) :- !.` before the catch-all clause (line ~160).
  2. Add capability fact `target_runtime_parser_mode_(wam_c, compiled(prolog_term_parser)).` in the `target_runtime_parser_mode_/2` block (lines ~141-148). (verify: whether the C target has a native runtime parser — C++ has a hand-written one (`native(parse_term)`), but the C target is separate; if C also ships a native canonical-form parser, add `native(parse_term)` mode + default too. Default to `compiled`-only unless a native parser is confirmed in `wam_c_target.pl`.)
  3. Add default `target_runtime_parser_default(wam_c, none).` (keep opt-in, matching F#/Haskell/Rust) — or `native(parse_term)` if step 2 confirms a shipping native parser.
- **Acceptance:** `swipl -g "use_module('src/unifyweaver/targets/wam_runtime_parser_capability'), (wam_target_runtime_parser(c, [runtime_parser(compiled)], M), M==compiled(prolog_term_parser) -> writeln(ok) ; (writeln(fail),halt(1))), halt" -t 'halt(1)'` prints `ok`; and existing `swipl -g run_tests tests/test_wam_runtime_parser*.pl` (verify: exact test filename) stays green.

### PARSE-GO: Add Go runtime-parser capability entry
- **Lever:** Runtime-parser capability entries  **Target:** Go  **Size:** S  **Depends on:** —
- **Goal:** Register `wam_go`'s runtime-parser capability so `wam_target_runtime_parser(go, ...)` resolves a mode instead of `none`/`domain_error`.
- **Files to touch:** `src/unifyweaver/targets/wam_runtime_parser_capability.pl`
- **Reference to copy from:** same file — `wam_rust` entries (`target_runtime_parser_default(wam_rust, none).` line ~139, `target_runtime_parser_mode_(wam_rust, compiled(prolog_term_parser)).` line ~148) + the `normalize_runtime_parser_target(rust, wam_rust)` alias pair (lines ~158-159).
- **Steps:**
  1. Add alias pair `normalize_runtime_parser_target(go, wam_go) :- !.` / `normalize_runtime_parser_target(wam_go, wam_go) :- !.` before the catch-all.
  2. Add `target_runtime_parser_mode_(wam_go, compiled(prolog_term_parser)).`
  3. Add `target_runtime_parser_default(wam_go, none).` (Go has no native runtime term parser — compiled portable parser only, opt-in; mirror Rust). (verify: no native parser in `wam_go_target.pl`.)
- **Acceptance:** `swipl -g "use_module('src/unifyweaver/targets/wam_runtime_parser_capability'), (wam_target_runtime_parser(go, [runtime_parser(compiled)], M), M==compiled(prolog_term_parser) -> writeln(ok) ; halt(1)), halt" -t 'halt(1)'` prints `ok`; parser-capability unit tests stay green.

### PARSE-SCALA: Add Scala runtime-parser capability entry
- **Lever:** Runtime-parser capability entries  **Target:** Scala  **Size:** S  **Depends on:** —
- **Goal:** Register `wam_scala`'s runtime-parser capability so `wam_target_runtime_parser(scala, ...)` resolves a mode.
- **Files to touch:** `src/unifyweaver/targets/wam_runtime_parser_capability.pl`
- **Reference to copy from:** same file — `wam_haskell` entries (default `none` line ~138, mode `compiled(prolog_term_parser)` line ~147) + a `normalize_runtime_parser_target(haskell, wam_haskell)` alias pair (lines ~156-157).
- **Steps:**
  1. Add alias pair `normalize_runtime_parser_target(scala, wam_scala) :- !.` / `normalize_runtime_parser_target(wam_scala, wam_scala) :- !.` before the catch-all.
  2. Add `target_runtime_parser_mode_(wam_scala, compiled(prolog_term_parser)).`
  3. Add `target_runtime_parser_default(wam_scala, none).` (compiled-portable-parser only, opt-in; no native Scala term parser — Scala's `compile_wam_predicate_to_scala/4` is even a stub, so only the compiled WAM path applies). (verify: no native parser in `wam_scala_target.pl`.)
- **Acceptance:** `swipl -g "use_module('src/unifyweaver/targets/wam_runtime_parser_capability'), (wam_target_runtime_parser(scala, [runtime_parser(compiled)], M), M==compiled(prolog_term_parser) -> writeln(ok) ; halt(1)), halt" -t 'halt(1)'` prints `ok`; parser-capability unit tests stay green.

### PARSE-CLOJURE: Add Clojure runtime-parser capability entry
- **Lever:** Runtime-parser capability entries  **Target:** Clojure  **Size:** S  **Depends on:** —
- **Goal:** Register `wam_clojure`'s runtime-parser capability so `wam_target_runtime_parser(clojure, ...)` resolves a mode.
- **Files to touch:** `src/unifyweaver/targets/wam_runtime_parser_capability.pl`
- **Reference to copy from:** same file — `wam_haskell` entries (default `none` line ~138, mode `compiled(prolog_term_parser)` line ~147) + the haskell alias pair (lines ~156-157).
- **Steps:**
  1. Add alias pair `normalize_runtime_parser_target(clojure, wam_clojure) :- !.` / `normalize_runtime_parser_target(wam_clojure, wam_clojure) :- !.` before the catch-all.
  2. Add `target_runtime_parser_mode_(wam_clojure, compiled(prolog_term_parser)).`
  3. Add `target_runtime_parser_default(wam_clojure, none).` (compiled-only, opt-in). (verify: no native runtime term parser in `wam_clojure_target.pl`; Clojure's host `read-string` is NOT a Prolog-term parser, so do not advertise `native`.)
- **Acceptance:** `swipl -g "use_module('src/unifyweaver/targets/wam_runtime_parser_capability'), (wam_target_runtime_parser(clojure, [runtime_parser(compiled)], M), M==compiled(prolog_term_parser) -> writeln(ok) ; halt(1)), halt" -t 'halt(1)'` prints `ok`; parser-capability unit tests stay green.

### PARSE-LUA: Add Lua runtime-parser capability entry
- **Lever:** Runtime-parser capability entries  **Target:** Lua  **Size:** S  **Depends on:** —
- **Goal:** Register `wam_lua`'s runtime-parser capability so `wam_target_runtime_parser(lua, ...)` resolves a mode.
- **Files to touch:** `src/unifyweaver/targets/wam_runtime_parser_capability.pl`
- **Reference to copy from:** same file — `wam_haskell`/`wam_rust` entries (default `none`, mode `compiled(prolog_term_parser)`) + their alias pairs (lines ~156-159).
- **Steps:**
  1. Add alias pair `normalize_runtime_parser_target(lua, wam_lua) :- !.` / `normalize_runtime_parser_target(wam_lua, wam_lua) :- !.` before the catch-all.
  2. Add `target_runtime_parser_mode_(wam_lua, compiled(prolog_term_parser)).`
  3. Add `target_runtime_parser_default(wam_lua, none).` (compiled-only, opt-in; no native Lua Prolog-term parser — Lua's `compile_wam_predicate_to_lua/4` is a stub and the target has no shipping native parser). (verify: no native parser in `wam_lua_target.pl`.)
- **Acceptance:** `swipl -g "use_module('src/unifyweaver/targets/wam_runtime_parser_capability'), (wam_target_runtime_parser(lua, [runtime_parser(compiled)], M), M==compiled(prolog_term_parser) -> writeln(ok) ; halt(1)), halt" -t 'halt(1)'` prints `ok`; parser-capability unit tests stay green.

---

## Lever: Two-level lazy/cached LMDB materialisation policies

Bring the F#/Haskell eager/lazy/cached (L1/L2) LMDB policy surface to
targets with flat/load-everything LMDB. **Go** and **Scala** already
have an on-demand lookup abstraction → single card each. **C** and **R**
are load-everything only → each gets a prerequisite lookup-source
sub-task (`LMDB-*-0`) the tier card depends on. Reference lever:
`wam_fsharp_target.pl` `resolve_auto_lmdb_materialisation_fs/2` (~4833),
`resolve_auto_lmdb_cache_tier_fs/2` (~4873), options
`lmdb_materialisation`/`lmdb_l2_capacity` (~5218) +
`templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`; Haskell
analogue `resolve_auto_lmdb_cache_mode/2` (`wam_haskell_target.pl:4589`).

**Abstraction-maturity order (easiest → hardest):** Go (`AtomFact2Source`
interface already exists in `templates/targets/go_wam/state.go.mustache:38`
— just add cached/eager impls) > Scala (`LmdbFactSource` class exists but
handler dispatch is ad-hoc per backend, no shared trait) > C (unified
`WamFactSource` for *loading* only; LMDB currently wired for reverse-CSR
offsets, not general facts) > R (no on-demand path at all — greenfield).
**Two cross-cutting refactors** worth doing once, not per-target: (a) the
inline LMDB runtime blocks in C/Go/R are not templated — consider
extracting each into `templates/targets/<t>_wam/lmdb_fact_source.*` first
(mirroring how F#/Haskell read a template), then add tiers; (b) F#'s
`compute_lmdb_materialisation_fs/2` and Haskell's `_hs` variant are
near-duplicates — rather than adding a 5th/6th copy, consider extracting a
shared `resolve_auto_lmdb_materialisation/3` (target atom + options) in a
core module. **Naming caution (C):** the reverse-CSR `lmdb_offset` index
(`wam_reverse_csr_*`) is a *different* feature from forward-fact
materialisation — the LMDB-C cards target the forward `wam_fact_source_*`
path, not the reverse index.

### LMDB-GO: Add eager/lazy/cached materialisation tiers to Go LMDB fact source
- **Lever:** LMDB policy tiers  **Target:** Go  **Size:** M  **Depends on:** —
- **Goal:** Give the Go `lmdbAtomFact2Source` an `lmdb_materialisation(eager|lazy|cached|auto)` knob mirroring F#, adding an eager in-memory map and a cached L1/L2 layer over the existing on-demand lookup.
- **Files to touch:** `src/unifyweaver/targets/wam_go_target.pl` (inline runtime around lines 2841–2945: `registerLmdbAtomFact2/2`, `lmdbAtomFact2Source`, `newLmdbAtomFact2Source`, `Scan`, `LookupArg1`, `run`; and foreign-setup emit `go_foreign_setup_line/2` line 982); optionally `templates/targets/go_wam/runtime.go.mustache`; add test `tests/test_go_lmdb_materialisation.pl`.
- **Reference to copy from:** `src/unifyweaver/targets/wam_fsharp_target.pl` — `resolve_auto_lmdb_materialisation_fs/2` (4833), `compute_lmdb_materialisation_fs/2` (4841), `apply_edge_store_fs/3` (4825); tier semantics in `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache` (eager loaders vs cursor-lazy vs cached L2). Also Haskell `resolve_auto_lmdb_cache_mode/2` (`wam_haskell_target.pl:4589`).
- **Steps:**
  1. Add option plumbing: parse `lmdb_materialisation(Mode)` (default `cached`) + `lmdb_l2_capacity(N)` (default 4096) where the Go LMDB register line is built (`go_foreign_setup_line`, ~982) and thread Mode into `registerLmdbAtomFact2`.
  2. Port `resolve_auto_lmdb_materialisation_fs/2` + `compute_lmdb_materialisation_fs/2` as `resolve_auto_lmdb_materialisation_go/2` (auto → eager/lazy/cached using fact-count/demand heuristics from F#).
  3. Extend the emitted `lmdbAtomFact2Source` struct: eager = call `Scan()` once at init into a `map[string][]AtomPair` and serve `LookupArg1` from it; lazy = current shell-out per lookup (no cache); cached = wrap `LookupArg1` with an L1 map + bounded L2 (LRU sized by `lmdb_l2_capacity`).
  4. Dispatch on Mode in `newLmdbAtomFact2Source` so the three code paths are selected at construction.
  5. (verify: whether arity>2 needs the same map-value split the F#/Scala sources do — Go source is currently arity-2 `AtomPair` only.)
- **Acceptance:** `swipl -q -g run_tests -t halt -l tests/test_go_lmdb_materialisation.pl` passes, asserting generated Go contains distinct eager (`Scan()` at init), lazy (per-call `run`), and cached (L2/LRU) code for the three modes; plus existing `swipl -q -l tests/test_go_wam_builtins.pl -g run_tests -t halt` still green. (verify: repo's canonical test invocation via `run_all_tests.pl`.)

### LMDB-SCALA: Add eager/lazy/cached tiers to Scala LmdbFactSource
- **Lever:** LMDB policy tiers  **Target:** Scala  **Size:** M  **Depends on:** —
- **Goal:** Extend the existing Scala `LmdbFactSource` (already on-demand `lookupByArg1`/`streamAll`) with eager (preload into Map) and cached (L1/L2 over `lookupByArg1`) modes selected by `lmdb_materialisation` + `lmdb_l2_capacity`, matching F#.
- **Files to touch:** `src/unifyweaver/targets/wam_scala_target.pl` — `fact_source_spec_to_handler_code(Arity, lmdb(SpecOpts), Code)` (line 1079, ForeignHandler emit) and its option parsing; `templates/targets/scala_wam/runtime.scala.mustache` — `LmdbFactSource` class (259), `streamAll` (327), `lookupByArg1` (339); extend `tests/test_wam_scala_lmdb_runtime_smoke.pl`.
- **Reference to copy from:** `src/unifyweaver/targets/wam_fsharp_target.pl` — `resolve_auto_lmdb_materialisation_fs/2` (4833), `resolve_auto_lmdb_cache_tier_fs/2` (4873), option defaults at 5218–5219 (`lmdb_materialisation` cached, `lmdb_l2_capacity` 4096); tier code in `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`.
- **Steps:**
  1. In the `lmdb(SpecOpts)` handler codegen, read `lmdb_materialisation(Mode)` (default `cached`) and `lmdb_l2_capacity(N)` from `SpecOpts`/Options; port `resolve_auto_lmdb_materialisation_fs` as `_scala`.
  2. In `runtime.scala.mustache`, add to `LmdbFactSource`: an eager path that runs `streamAll()` once at init into an immutable `Map[WamTerm, Seq[...]]` and serves `lookupByArg1` from it; a cached path adding a `LinkedHashMap`/bounded L1+L2 cache (capacity param) around the current cursor lookup; leave the current path as `lazy`.
  3. Add a `mode`/`l2Capacity` constructor param to `LmdbFactSource` and thread it from the handler's `new LmdbFactSource(...)` string (line 1098–1100).
  4. Keep the ground-arg1 vs `streamAll` dispatch in `apply` unchanged; only the backing store per mode changes.
- **Acceptance:** `swipl -q -g run_tests -t halt -l tests/test_wam_scala_lmdb_runtime_smoke.pl` passes with new cases asserting the generated handler/runtime contains eager-preload, lazy-cursor, and cached-L1/L2 variants for `lmdb_materialisation(eager|lazy|cached)`; existing `tests/test_wam_scala_runtime_smoke.pl` stays green.

### LMDB-C-0 (prerequisite): Cursor-based on-demand C fact lookup source
- **Lever:** LMDB policy tiers  **Target:** C  **Size:** M  **Depends on:** —
- **Goal:** Build the missing on-demand fact-lookup abstraction in the C runtime (a `WamFactSource` variant that queries LMDB by key via a cursor instead of loading everything), so lazy/cached tiers have something to sit on.
- **Files to touch:** `src/unifyweaver/targets/wam_c_target.pl` (fact-source struct near `wam_fact_source_load_lmdb` line 4448; reuse cursor mechanics from `wam_reverse_csr_find_lmdb_offset` line 5163); `tests/test_wam_c_target.pl`.
- **Reference to copy from:** C's own `wam_reverse_csr_find_lmdb_offset` (`wam_c_target.pl:5163`, `#ifdef WAM_C_ENABLE_LMDB`) for cursor/get shape; Scala `LmdbFactSource.lookupByArg1` (`runtime.scala.mustache:339`) and Go `lmdbAtomFact2Source.LookupArg1` (`wam_go_target.pl:2932`) for the interface contract (lookup-by-arg1 + stream-all).
- **Steps:**
  1. Add a `wam_fact_source_lookup_lmdb(state, source, key, ...)` C function opening a read txn + cursor and returning matching tuples for a ground arg1 (arity ≥ 2, tab-joined value split like Scala/Go).
  2. Add a `stream_all` cursor path equivalent to eager but streaming (no full table build).
  3. Wire it as a selectable fact-source backend alongside the existing load-everything one.
- **Acceptance:** `swipl -q -g run_tests -t halt -l tests/test_wam_c_target.pl` passes a new case asserting generated C contains the cursor lookup fn and it is dispatchable; compiles under `-DWAM_C_ENABLE_LMDB`.

### LMDB-C: Add eager/lazy/cached tiers to C LMDB fact source
- **Lever:** LMDB policy tiers  **Target:** C  **Size:** L  **Depends on:** LMDB-C-0
- **Goal:** Once a cursor-based on-demand C fact lookup source exists, add `lmdb_materialisation(eager|lazy|cached|auto)` + `lmdb_l2_capacity` so C matches F# eager/lazy/cached (L1/L2) fact materialisation.
- **Files to touch:** `src/unifyweaver/targets/wam_c_target.pl` — fact-source path `wam_fact_source_load_lmdb` (line 4448, currently load-everything under `#ifdef WAM_C_ENABLE_LMDB`) and the fact-source struct/dispatch; option parsing where the C LMDB fact source is set up; `templates/targets/c_wam/` if runtime is templated (verify — C runtime appears emitted inline as format strings); test `tests/test_wam_c_target.pl`.
- **Reference to copy from:** `src/unifyweaver/targets/wam_fsharp_target.pl` — `resolve_auto_lmdb_materialisation_fs/2` (4833), `resolve_auto_lmdb_cache_tier_fs/2` (4873); F# cursor+L2 pattern in `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`. Note: C already has a cursor lookup for the *reverse-CSR index* (`wam_reverse_csr_find_lmdb_offset`, line 5163) as a shape example — but that is the index, not the fact source.
- **Steps:**
  1. Parse `lmdb_materialisation(Mode)` (default `cached`) + `lmdb_l2_capacity(N)` in the C LMDB fact-source setup; port the F# auto-resolvers as `_c`.
  2. eager = keep `wam_fact_source_load_lmdb` (load all into the in-memory table); lazy = use the LMDB-C-0 cursor lookup source with no cache; cached = cursor lookup + a bounded L1/L2 cache struct (capacity = `lmdb_l2_capacity`).
  3. Dispatch on Mode at fact-source construction; gate all under `#ifdef WAM_C_ENABLE_LMDB`.
- **Acceptance:** `swipl -q -g run_tests -t halt -l tests/test_wam_c_target.pl` passes new assertions that generated C emits eager (`wam_fact_source_load_lmdb`), lazy (cursor find, no cache), and cached (cursor + L2 capacity) code per mode; C still compiles with `-DWAM_C_ENABLE_LMDB` (verify: repo's C-build smoke, e.g. `test_wam_c_target.pl` build step).

### LMDB-R-0 (prerequisite): On-demand LMDB lookup source in the R runtime
- **Lever:** LMDB policy tiers  **Target:** R  **Size:** M  **Depends on:** —
- **Goal:** R currently only has `read_facts_lmdb` (load-everything, `wam_r_target.pl:1458`); add a per-key LMDB query function to `runtime.R.mustache` so lazy/cached tiers are possible.
- **Files to touch:** `templates/targets/r_wam/runtime.R.mustache` (add `WamRuntime$lookup_facts_lmdb`/`stream_facts_lmdb`); `src/unifyweaver/targets/wam_r_target.pl` (`fact_source_loader_call` line 1458, `emit_external_fact_source` 1414); `tests/test_wam_r_generator.pl`.
- **Reference to copy from:** Scala `LmdbFactSource.lookupByArg1`/`streamAll` (`runtime.scala.mustache:339,327`) and Go `lmdbAtomFact2Source.LookupArg1` (`wam_go_target.pl:2932`) for the lookup-by-arg1 + stream-all contract; existing R `read_facts_lmdb` for the LMDB-open/intern-table conventions.
- **Steps:**
  1. Add an R function opening the LMDB env and returning tuples for a given ground arg1 key (arity ≥ 2, split the tab-joined value into registers 2..N as F#/Scala do), plus a streaming full-scan variant.
  2. Expose both via `WamRuntime$` alongside `read_facts_lmdb` and make the fact-source loader able to select the on-demand path.
- **Acceptance:** `swipl -q -g run_tests -t halt -l tests/test_wam_r_generator.pl` passes a case asserting the generated R runtime defines the per-key lookup + stream functions and they are selectable as a fact-source backend.

### LMDB-R: Add eager/lazy/cached tiers to R LMDB fact source
- **Lever:** LMDB policy tiers  **Target:** R  **Size:** L  **Depends on:** LMDB-R-0
- **Goal:** After R gains an on-demand LMDB lookup source, add `lmdb_materialisation(eager|lazy|cached|auto)` + `lmdb_l2_capacity` so R matches the F# tier surface (today R is load-everything only).
- **Files to touch:** `src/unifyweaver/targets/wam_r_target.pl` — `fact_source_loader_call(lmdb(Path), _Arity, ...)` (line 1458, explicitly "Step-1 backend: load-everything") and `emit_external_fact_source/…` (1414), `r_fact_source_spec/4` (1397); `templates/targets/r_wam/runtime.R.mustache` (contains the `read_facts_lmdb` runtime); tests `tests/test_r_target.pl` / `tests/test_wam_r_generator.pl`.
- **Reference to copy from:** `src/unifyweaver/targets/wam_fsharp_target.pl` — `resolve_auto_lmdb_materialisation_fs/2` (4833) + `resolve_auto_lmdb_cache_tier_fs/2` (4873); tier semantics in `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`.
- **Steps:**
  1. Parse `lmdb_materialisation(Mode)` (default `cached`) + `lmdb_l2_capacity(N)` in `fact_source_loader_call`/`emit_external_fact_source`; port the F# auto-resolvers as `_r`.
  2. eager = keep `WamRuntime$read_facts_lmdb` (load all); lazy = call the LMDB-R-0 per-key lookup (no cache); cached = wrap lookup with an environment-backed L1 list + bounded L2 (capacity = `lmdb_l2_capacity`).
  3. Emit the correct R loader/handler per Mode from `runtime.R.mustache`.
- **Acceptance:** `swipl -q -g run_tests -t halt -l tests/test_wam_r_generator.pl` (and `tests/test_r_target.pl`) pass new assertions that generated R selects `read_facts_lmdb` for eager, per-key lookup for lazy, and cached L1/L2 for cached; existing R tests stay green.

---

## Lever: ISO three-form error contract

Adopt the shared ISO error three-form contract (config loader + rewrite
+ audit + catch/throw + error constructors + `is_iso`/`is_lax` + ISO/lax
arithmetic compares + `succ_iso`/`succ_lax` + lax IEEE-754 divide).
Spec: `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` §"What Counts
As Adoption". References: `wam_cpp_target.pl` (runtime shapes:
`throw_iso_error`, `wam_cpp_iso_audit/3`) and `wam_haskell_target.pl` /
`wam_fsharp_target.pl` (Prolog-side wiring via the shared `iso_errors`
core module). C/Go/Scala/R are **new** adopters (L); Python/F# are
**finish-the-delta** (S).

**Build on `src/unifyweaver/core/iso_errors.pl`** — the shared config /
rewrite / audit-walk helpers Python/Elixir/F# `use_module`. Do **not** copy
C++'s ~340-line inline implementation (C++ is intentionally unmigrated);
copy C++ only for exact error-term shapes and the classify-order
(unbound → zero-divide → non-evaluable). New adopters keep only the
target-specific parts: key-table assertions, an `iso_errors_rewrite_text/4`
variant, and a `wam_<t>_iso_audit/3` wrapper. **Test convention:** pair a
Prolog-only `tests/test_wam_<t>_iso_unit.pl` (config/rewrite/audit, no
toolchain) with a generate+build+run `test_wam_<t>_iso_smoke.pl` (template:
`tests/core/test_wam_fsharp_iso_smoke.pl` — hand-written driver asserts
PASS/FAIL per predicate, exits nonzero on any fail). The shared
`tests/test_iso_errors.pl` already covers the core module — don't re-test
plumbing there.

### ISO-C: New ISO three-form adoption for WAM C target
- **Lever:** ISO three-form  **Target:** C  **Size:** L  **Depends on:** —
- **Goal:** Bring the WAM C target from non-adopter (only Prolog-level `catch/throw` meta-calls in the generator) to full ISO three-form adoption mirroring F#/Haskell.
- **Files to touch:** `src/unifyweaver/targets/wam_c_target.pl`, `src/unifyweaver/targets/wam_c_lowered_emitter.pl`, new `tests/test_wam_c_iso_smoke.pl`
- **Reference to copy from:** `src/unifyweaver/targets/wam_haskell_target.pl` (most recent full adopter, mirrors F#) and `wam_fsharp_target.pl` for the Prolog-side wiring; `wam_cpp_target.pl` for the runtime shapes (`throw_iso_error`, `wam_cpp_iso_audit/3`, `is_iso`/`is_lax`, `succ_iso`/`succ_lax`, `evaluation_error`/`zero_divisor`); spec = `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` §"What Counts As Adoption" (7-point checklist).
- **Steps:**
  1. `use_module('../core/iso_errors', [...])` importing the 6 shared preds (`iso_errors_resolve_options/2`, `_load_config/2`, `_mode_for/3`, `_warn_multi_module/2`, `_rewrite/4`, `_rewrite_text/4`), as F#/Python do.
  2. Assert C key tables: `iso_errors:iso_errors_default_to_iso/2` and `_default_to_lax/2` facts for `is/2`, the 6 arithmetic compares (`</2 >/2 =</2 >=/2 =:=/2 =\=/2`), and `succ/2`.
  3. Emit C runtime: `catch/3`+`throw/1` substrate, `error(ErrorType,Context)` constructors + `throw_iso_error` helper, `is_iso`/`is_lax`, ISO/lax arithmetic compares, `succ_iso`/`succ_lax`, lax IEEE-754 divide (float→inf/nan, integer div-by-zero→`evaluation_error(zero_divisor)` in ISO).
  4. Wire per-predicate default rewrite into the text path (`builtin_call`,`put_structure`,`call`,`execute`) via `iso_errors_rewrite_text/4`.
  5. Add `wam_c_iso_audit/3` + `wam_c_iso_audit_report/1` (mirror `wam_python_iso_audit/3` at wam_python_target.pl:2046).
  6. Add inline-option/config plumbing (`iso_errors_config/1`, bare-PI multi-module warning) via `iso_errors_resolve_options/2`.
- **Acceptance:** `swipl -g run_tests -t halt tests/test_wam_c_iso_smoke.pl` passes (mirror `tests/test_wam_haskell_iso_smoke.pl` structure: key-tables registered, ISO-default throws, lax-default fails/lax, explicit override bypasses rewrite, audit report); `swipl -g run_tests -t halt tests/test_iso_errors.pl` still green.

### ISO-GO: New ISO three-form adoption for WAM Go target
- **Lever:** ISO three-form  **Target:** Go  **Size:** L  **Depends on:** —
- **Goal:** Bring the WAM Go target from non-adopter (only generator-side `catch/3` meta-calls) to full ISO three-form adoption mirroring F#/Haskell.
- **Files to touch:** `src/unifyweaver/targets/wam_go_target.pl`, `src/unifyweaver/targets/wam_go_lowered_emitter.pl`, new `tests/test_wam_go_iso_smoke.pl`
- **Reference to copy from:** `wam_haskell_target.pl` + `wam_fsharp_target.pl` (Prolog wiring, shared `iso_errors` module import); `wam_cpp_target.pl` for runtime pred shapes; spec `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` §"What Counts As Adoption".
- **Steps:**
  1. Import shared `iso_errors` module (6 preds) as F#/Python.
  2. Assert Go key tables (`iso_errors_default_to_iso/lax` for `is/2`, 6 compares, `succ/2`).
  3. Emit Go runtime: `catch/3`+`throw/1` substrate (Go `panic`/`recover` or error-value convention — verify: which the Go runtime already uses), `error(...)` constructors + `throw_iso_error`, `is_iso`/`is_lax`, ISO/lax compares, `succ_iso`/`succ_lax`, lax IEEE-754 divide (float inf/nan; int div-by-zero→`evaluation_error(zero_divisor)`).
  4. Wire text-path rewrite (`builtin_call`,`put_structure`,`call`,`execute`) via `iso_errors_rewrite_text/4`.
  5. Add `wam_go_iso_audit/3` + report (mirror `wam_python_iso_audit/3`).
  6. Add config/inline-override plumbing + bare-PI multi-module warning.
- **Acceptance:** `swipl -g run_tests -t halt tests/test_wam_go_iso_smoke.pl` passes (mirror `tests/test_wam_haskell_iso_smoke.pl`); `tests/test_iso_errors.pl` still green.

### ISO-SCALA: New ISO three-form adoption for WAM Scala target
- **Lever:** ISO three-form  **Target:** Scala  **Size:** L  **Depends on:** —
- **Goal:** Bring the WAM Scala target from non-adopter (only generator-side `catch/3`) to full ISO three-form adoption mirroring F#/Haskell.
- **Files to touch:** `src/unifyweaver/targets/wam_scala_target.pl`, `src/unifyweaver/targets/wam_scala_lowered_emitter.pl`, new `tests/test_wam_scala_iso_smoke.pl`
- **Reference to copy from:** `wam_haskell_target.pl` + `wam_fsharp_target.pl` (closest JVM-family analog for exception substrate); `wam_cpp_target.pl` for runtime pred shapes; spec `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`.
- **Steps:**
  1. Import shared `iso_errors` module (6 preds).
  2. Assert Scala key tables (`is/2`, 6 compares, `succ/2`).
  3. Emit Scala runtime: `catch/3`+`throw/1` (JVM exceptions), `error(...)` constructors + `throw_iso_error`, `is_iso`/`is_lax`, ISO/lax compares, `succ_iso`/`succ_lax`, lax IEEE-754 divide (float inf/nan; int div-by-zero→`evaluation_error(zero_divisor)` — verify JVM `ArithmeticException` mapping).
  4. Wire text-path rewrite via `iso_errors_rewrite_text/4`.
  5. Add `wam_scala_iso_audit/3` + report.
  6. Config/inline-override plumbing + bare-PI warning.
- **Acceptance:** `swipl -g run_tests -t halt tests/test_wam_scala_iso_smoke.pl` passes (mirror `tests/test_wam_haskell_iso_smoke.pl`); `tests/test_iso_errors.pl` still green.

### ISO-R: New ISO three-form adoption for WAM R target
- **Lever:** ISO three-form  **Target:** R  **Size:** L  **Depends on:** —
- **Goal:** Bring the WAM R target from non-adopter (only `tryCatch`/generator `catch/3`) to full ISO three-form adoption mirroring F#/Haskell.
- **Files to touch:** `src/unifyweaver/targets/wam_r_target.pl`, `src/unifyweaver/targets/wam_r_lowered_emitter.pl`, new `tests/test_wam_r_iso_smoke.pl`
- **Reference to copy from:** `wam_haskell_target.pl` + `wam_fsharp_target.pl`; `wam_cpp_target.pl` for runtime shapes; spec `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`. Note: R already keeps a native inline parser hot-path (see spec §"Relationship To Runtime Parser Transpilation") — ISO work is independent of that.
- **Steps:**
  1. Import shared `iso_errors` module (6 preds).
  2. Assert R key tables (`is/2`, 6 compares, `succ/2`).
  3. Emit R runtime: `catch/3`+`throw/1` on R `tryCatch`/`stop`/`condition` substrate (verify: extend existing `tryCatch` usage at wam_r_target.pl:65), `error(...)` constructors + `throw_iso_error`, `is_iso`/`is_lax`, ISO/lax compares, `succ_iso`/`succ_lax`, lax IEEE-754 divide (R `/` already yields Inf/NaN; add ISO int div-by-zero→`evaluation_error(zero_divisor)`).
  4. Wire text-path rewrite via `iso_errors_rewrite_text/4`.
  5. Add `wam_r_iso_audit/3` + report.
  6. Config/inline-override plumbing + bare-PI warning.
- **Acceptance:** `swipl -g run_tests -t halt tests/test_wam_r_iso_smoke.pl` passes (mirror `tests/test_wam_haskell_iso_smoke.pl`); `tests/test_iso_errors.pl` still green.

### ISO-PYTHON: Finish Python three-form delta (remaining concrete builtins)
- **Lever:** ISO three-form  **Target:** Python  **Size:** S  **Depends on:** —
- **Goal:** Close the Python delta — it already ships config/rewrite/`wam_python_iso_audit/3`/catch-throw/constructors/`is_iso`/`is_lax`/6 arithmetic compares/`succ` family/lax IEEE-754 divide; the only gap per spec is that remaining concrete builtins don't yet have three-form keys.
- **Files to touch:** `src/unifyweaver/targets/wam_python_target.pl` (key-table assertions near lines 1939–1953 and runtime emit), new `tests/test_wam_python_iso_smoke.pl` (verify: no Python iso smoke test currently exists; only `tests/test_wam_fsharp_iso_unit.pl`/`_smoke.pl` and haskell)
- **Reference to copy from:** `wam_fsharp_target.pl` for the fuller builtin surface; spec `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` §"Current Implementation Status" note ("not fully compatible until remaining concrete builtins also adopt three-form keys").
- **Steps (missing pieces ONLY):**
  1. Enumerate audited builtins still lacking `_iso`/`_lax` keys beyond `is/2`+6 compares+`succ/2` (verify against the C++ table in `wam_cpp_target.pl`); add `iso_errors:iso_errors_default_to_iso/2` + `_default_to_lax/2` assertions for each.
  2. Emit the corresponding `_iso`/`_lax` runtime variants + register in the Call/Execute meta-builtin routing.
  3. Add `tests/test_wam_python_iso_smoke.pl` covering key-tables, ISO-throw/lax-fail/override-bypass, audit report.
- **Acceptance:** `swipl -g run_tests -t halt tests/test_wam_python_iso_smoke.pl` passes (mirror `tests/test_wam_haskell_iso_smoke.pl`); `tests/test_iso_errors.pl` still green.

### ISO-FSHARP: Finish F# three-form delta (lax integer divide + remaining builtins)
- **Lever:** ISO three-form  **Target:** F#  **Size:** S  **Depends on:** —
- **Goal:** Close the F# delta — full substrate/config/rewrite/`wam_fsharp_iso_audit/3`/`is_iso`/`is_lax`/6 compares/`succ` family already ship; remaining gaps are the lax IEEE-754 **integer** divide (currently fails silently) and remaining concrete builtins lacking three-form keys.
- **Files to touch:** `src/unifyweaver/targets/wam_fsharp_target.pl`, `src/unifyweaver/targets/wam_fsharp_lowered_emitter.pl`, extend `tests/core/test_wam_fsharp_iso_smoke.pl` and/or `tests/test_wam_fsharp_iso_unit.pl`
- **Reference to copy from:** `wam_cpp_target.pl` lax-divide + `zero_divisor` handling (spec explicitly flags F# float div returns nan/inf via CLR but integer div-by-zero fails silently — §"Current Implementation Status" last row); spec `docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`.
- **Steps (missing pieces ONLY):**
  1. Fix integer div/mod by zero: ISO path throws `evaluation_error(zero_divisor)`; lax path uses defined lax behavior instead of silent CLR `DivideByZeroException`/silent fail (verify current emit in `wam_fsharp_lowered_emitter.pl`).
  2. Add three-form keys for any audited builtins still missing `_iso`/`_lax` beyond the current set; register in Call/Execute routing.
  3. Extend the existing F# smoke/unit tests with an integer-div-by-zero ISO-throw + lax case and any newly-keyed builtins.
- **Acceptance:** `swipl -g run_tests -t halt tests/core/test_wam_fsharp_iso_smoke.pl` and `swipl -g run_tests -t halt tests/test_wam_fsharp_iso_unit.pl` pass with the new integer-div-by-zero assertions; `tests/test_iso_errors.pl` still green.

---

## Lever: Finish F# foreign-kernel templates

### KERN-FSHARP: Port the 6 missing F# foreign-kernel templates
- **Lever:** Finish F# foreign-kernel templates  **Target:** F#  **Size:** L  **Depends on:** —
- **Goal:** Add the 6 missing `.fs.mustache` kernel templates so F#'s detector-fired kinds (transitive_closure2, transitive_distance3, transitive_parent_distance4, transitive_step_parent_distance5, weighted_shortest_path3, astar_shortest_path4) emit real F# instead of the "no F# template available" stub.
- **Consolidation note:** Emitted as ONE card, not six. The 6 items are mechanically identical — each is a syntax port of the already-complete Haskell template of the same base name into F#, wired through the same code path (`render_kernel_function_fs/2` swaps `.hs.mustache`→`.fs.mustache` and reads `templates/targets/fsharp_wam/`). A Grok agent should do them in one branch, one kind at a time, in the order below.
- **Files to touch:**
  - `templates/targets/fsharp_wam/kernel_transitive_closure.fs.mustache` (new)
  - `templates/targets/fsharp_wam/kernel_transitive_distance.fs.mustache` (new)
  - `templates/targets/fsharp_wam/kernel_transitive_parent_distance.fs.mustache` (new)
  - `templates/targets/fsharp_wam/kernel_transitive_step_parent_distance.fs.mustache` (new)
  - `templates/targets/fsharp_wam/kernel_weighted_shortest_path.fs.mustache` (new)
  - `templates/targets/fsharp_wam/kernel_astar_shortest_path.fs.mustache` (new)
  - No `.pl` change expected: `wam_fsharp_target.pl` `render_kernel_function_fs/2` (lines 3554–3585) already derives each F# filename from `kernel_template_file/2` by suffix swap; dropping the files in place activates them. (verify: confirm no per-kind allow-list gate elsewhere restricts F# to the 2 shipped kinds.)
- **Reference to copy from:**
  - Structure/algorithm per kind: `templates/targets/haskell_wam/kernel_<base>.hs.mustache` (all exist and are complete) — pair each new file with the same base name.
  - F# syntax + mustache-var conventions: the two working F# templates `templates/targets/fsharp_wam/kernel_category_ancestor.fs.mustache` and `kernel_bidirectional_ancestor.fs.mustache`.
  - Var binding contract: `wam_fsharp_target.pl` `config_ops_to_template_vars_fs/2` (just below line 3585) — mustache vars come from the kernel's `ConfigOps` (e.g. `edge_pred`, `max_depth`); mirror the var names the matching `.hs.mustache` consumes.
- **Steps:**
  1. For each base name, open the Haskell template and the two working F# templates side by side; identify the mustache vars used.
  2. Translate the Haskell kernel body to idiomatic F# (recursive/`Seq`/`Map` traversal calling `ILookupSource.Lookup`), keeping the same mustache placeholders and function-name shape as the F# category/bidirectional templates.
  3. Ensure the emitted F# function name and signature match what `program.fs.mustache` / `execute_foreign.fs.mustache` dispatch to (check `generate_execute_foreign_fs/2` + `emit_execute_foreign_entry_fs/1`, lines ~3586–3600; expect `kernel_register_layout/2` + `kernel_native_call/2` already defined per kind).
  4. Do the simplest kind first (transitive_closure), validate end-to-end, then the weighted/astar kinds.
- **Acceptance:** For each kind, generating a project with that kernel produces a template (no `// Kernel …: no F# template available` marker in `WamRuntime.fs`) and `dotnet build` succeeds. Extend the emission+build assertion pattern of `tests/core/test_wam_fsharp_bidirectional_e2e.pl` (Step 5, lines ~125–168) to each new kind; `swipl -q -g run_tests -t halt tests/test_wam_fsharp_target.pl` plus `tests/run_wam_fsharp_tests.pl` pass.

---

## Lever: Lowered emitters for early scaffolds

Add a lowered emitter (deterministic clause-1 fast path + interpreter
fallback) to the three Tier-D targets. Reference small emitters:
`wam_lua_lowered_emitter.pl` (590 lines), `wam_wat_lowered_emitter.pl`
(518 lines); call-site partition pattern in `wam_lua_target.pl:526-527`.

### EMIT-ILASM: Add ILAsm lowered emitter (clause-1 fast path + WAM fallback)
- **Lever:** Lowered emitters for early scaffolds  **Target:** ILAsm  **Size:** L  **Depends on:** —
- **Goal:** Give the ILAsm target a lowered emitter that emits a deterministic clause-1 fast path in CIL and falls back to the existing switch-dispatch WAM interpreter for non-lowerable predicates.
- **Files to touch:**
  - `src/unifyweaver/targets/wam_ilasm_lowered_emitter.pl` (new)
  - `src/unifyweaver/targets/wam_ilasm_target.pl` (wire lowered path into `write_wam_ilasm_project/3`, line 49; partition before the CIL switch-dispatch emit at lines 134–147)
  - `tests/core/test_wam_ilasm_lowered_smoke.pl` (new)
- **Reference to copy from:** `wam_lua_lowered_emitter.pl` (590 lines) — mirror its module shape: exports `wam_lua_lowerable/3`, `lower_predicate_to_lua/4`, `lua_lowered_func_name/2`; `build_emission_plan/2` → `classify_clause_shape/2` (T4/T5/multi_clause_1 plans, lines 33–60); and the call-site partition in `wam_lua_target.pl` lines 526–527 (`catch(wam_lua_lowerable(...))` then `lower_predicate_to_lua(...)`, else interpreter). For CIL idioms reuse `wam_ilasm_target.pl`'s existing instruction encoders and the `switch (...)` jump-table emit (line 147).
- **Steps:**
  1. Create module `wam_ilasm_lowered_emitter` exporting `wam_ilasm_lowerable/3`, `lower_predicate_to_ilasm/4`; reuse `wam_ite_structurer`, `wam_clause_chain`, `wam_text_parser` as Lua does.
  2. Port `build_emission_plan/2`/`classify_clause_shape/2` to detect the deterministic clause-1 / clause-chain shapes over WAM text.
  3. Emit CIL methods for the fast path using the target's existing encoders; name via an `ilasm_lowered_func_name/2`.
  4. In `write_wam_ilasm_project/3`, partition predicates: lowerable → native CIL method, rest → existing switch-dispatch interpreter (unchanged fallback).
- **Acceptance:** `swipl -q -g run_tests -t halt tests/core/test_wam_ilasm_lowered_smoke.pl` passes (lowered method present in emitted `.il`, interpreted predicate still routes through switch dispatch), and the existing `tests/…test_wam_ilasm*` suite still passes. (verify: exact existing ILAsm test entrypoint name.)

### EMIT-JVM: Add JVM lowered emitter (clause-1 fast path + tableswitch fallback)
- **Lever:** Lowered emitters for early scaffolds  **Target:** JVM  **Size:** L  **Depends on:** —
- **Goal:** Add a lowered emitter that emits a deterministic clause-1 fast path as JVM bytecode (Jamaica/Krakatau) and falls back to the existing `tableswitch` WAM interpreter.
- **Files to touch:**
  - `src/unifyweaver/targets/wam_jvm_lowered_emitter.pl` (new)
  - `src/unifyweaver/targets/wam_jvm_target.pl` (wire into `write_wam_jvm_project/4`, line 20; both `jamaica`/`krakatau` var styles at lines 53–54)
  - `tests/core/test_wam_jvm_lowered_smoke.pl` (new)
- **Reference to copy from:** `wam_wat_lowered_emitter.pl` (518 lines, the smaller reference) for the lower-then-fallback module shape; Lua's call-site partition (`wam_lua_target.pl` 526–527) for the wiring pattern. Bytecode emission must go through the existing `../core/jvm_bytecode` helpers (`use_module` at `wam_jvm_target.pl` line 38) and honor `jvm_wam_var_style/2` (symbolic for jamaica, numeric for krakatau, lines 53–54) plus `jvm_wam_comment/3`.
- **Steps:**
  1. Create module `wam_jvm_lowered_emitter` exporting `wam_jvm_lowerable/3`, `lower_predicate_to_jvm/4` (thread `OutputFmt` so both jamaica and krakatau assembly are emittable).
  2. Port the clause-shape classifier from the wat/lua emitter over WAM text.
  3. Emit fast-path methods via `jvm_bytecode` encoders, parameterized on var style.
  4. Partition in `write_wam_jvm_project/4`: lowerable → native method, rest → existing tableswitch interpreter (unchanged).
- **Acceptance:** `swipl -q -g run_tests -t halt tests/core/test_wam_jvm_lowered_smoke.pl` passes for BOTH `jamaica` and `krakatau` formats (lowered method emitted; interpreted predicate still uses `tableswitch`), existing JVM tests still green.

### EMIT-KOTLIN: Add Kotlin WAM-lowered emitter (deterministic clause-1 native dispatch)
- **Lever:** Lowered emitters for early scaffolds  **Target:** Kotlin  **Size:** M  **Depends on:** —
- **Status:** ✅ **Landed (narrowed)** on `cursor/emit-kotlin-lowered-f421` (2026-07-12). The `registerNative` runtime seam + native-first dispatch with interpreter fallback shipped and work. **Correctness caveat found in review:** write-mode structure/list construction lowered to unbound vars (silent wrong answer, returned `true` so fallback never fired). Fix narrows the lowerable set to **flat facts + register unification only** (`get/put_constant|integer|nil`, `get_variable`, `get_value`, `put_variable`, `put_value`); any predicate that constructs or unifies a structure/list declines and stays on the correct bytecode interpreter. See follow-up **EMIT-KOTLIN-2**. Historical context below retained.

**Context — read this first (verified against source 2026-07-11).** The Kotlin target is NOT missing its partition. `wam_kotlin_target.pl` already ships:
- `wam_kotlin_resolve_emit_mode/2` (`interpreter` | `functions` | `mixed(List)`) — lines 57–69; default `interpreter`.
- `wam_kotlin_partition_predicates/5` → `native(PI,Code)` / `wam(PI,WamText)` / `failed(PI)` buckets — lines 71–99.
- A WAM-fallback path that already works end-to-end: `compile_wam_predicate_to_kotlin/4` registers an `Instruction` list on `WamProgram`, and `WamRuntime.run` interprets it (the passing gradle test `generated_project_compiles_and_runs_fact_variable_and_terms` proves this).

**The actual gap.** The *native* bucket is a dead end. `should_attempt_native/2` (lines 101–103) routes lowerable predicates to the **non-WAM** `kotlin_target:compile_predicate_to_kotlin/3` (line 85), and `compile_native_parts/2` (lines 312–323) then emits that code **only as an audit block comment** — `/* Native Kotlin lowering selected … Direct native dispatch wiring is a follow-up … */`. A predicate placed in the Native bucket is therefore neither natively dispatched nor WAM-registered: **it is not runnable.** (The existing `functions_mode_partitions_native_when_available` test only checks the *partition*, not that the project runs the native predicate.) This task closes that: emit a real Kotlin function from WAM and dispatch to it.

- **Goal:** Add `wam_kotlin_lowered_emitter.pl` that lowers a **deterministic single-clause** predicate's WAM text into a native Kotlin function operating on the existing `WamState`, and wire it into `WamProgram`/`WamRuntime` so that predicate runs natively (bypassing the bytecode loop), with the WAM interpreter as the safety fallback for everything else.

- **Which hybrid targets to use as reference (ranked):**
  1. **Lua — PRIMARY.** `src/unifyweaver/targets/wam_lua_lowered_emitter.pl` (~590 lines) is the smallest dedicated *WAM-text → native-function* emitter and Lua's runtime model is structurally identical to Kotlin's (a `WamState` with registers + `run_predicate`, vs Kotlin's `WamState` + `WamRuntime.run`). Copy: its module shape and exports (`wam_lua_lowerable/3`, `lower_predicate_to_lua/4`, `lua_lowered_func_name/2`); `build_emission_plan/2` → `classify_clause_shape/2` (start with just the `multi_clause_1`/single-clause arm); and **crucially the wrapper wiring** — `wam_lua_target.pl:526–530` (partition calls `wam_lua_lowerable` then `lower_predicate_to_lua`) and `emit_lua_lowered_wrapper/4` (`wam_lua_target.pl:692`), which emits the public entry that builds a state, loads args into registers, and calls the lowered native fn. Kotlin needs the exact same two-step.
  2. **WAT — SECONDARY.** `src/unifyweaver/targets/wam_wat_lowered_emitter.pl` (~518 lines) for the **fallback-safety replay idiom** its header describes: "try the lowered clause-1 path first, and if it fails the generated public entry reinitialises state and falls back to the complete bytecode interpreter." Adopt this so a mis-lowered predicate stays correct — lowered failure must re-init and hand off to `WamRuntime.run`, not silently fail.
  3. Rust / Haskell / F# lowered emitters — reference **only** for richer clause shapes (T4–T6 / ITE) in a *later* pass; they are large and their native paths are more entangled. Do not model the first cut on them.

- **Shared machinery to reuse (already used by Lua):** `wam_ite_structurer` (`structure_ite/2`), `wam_clause_chain` (`clause_chain/2`), `wam_text_parser` (`wam_tokenize_line/2`, `wam_classify_constant_token/2` — already imported by `wam_kotlin_target.pl`). Do not re-parse WAM by hand.

- **Runtime seam (all in `templates/targets/kotlin_wam/`):**
  - `WamRuntime.kt.mustache` — `sealed class Value { Atom, IntVal, FloatVal, Var, Ref, Struct, ListVal }` (line 11), `data class Instruction` (21), `class WamState` with `registers`, `readRegister`/`writeRegister`/`bind`/`deref`/`resolve`/environment frames (lines 60+), `class WamProgram` (`register(key, listOf(Instruction))`, `predicateNames()`), and `class WamRuntime(program).run(predicate, state)`.
  - `Main.kt.mustache` — `buildProgram()` emits `{{native_predicates}}` / `{{wam_predicates}}` then `{{registrar_calls}}`. The generated `main` calls `WamRuntime(program).run(predicate, stateFromCliArgs(...))`.
  - **Recommended wiring:** add a native-dispatch seam to the runtime template — e.g. `WamProgram.registerNative(key, (WamState) -> Boolean)` plus a check at the top of `WamRuntime.run` that invokes the native fn (with WAT-style fall-through to the interpreter on `false`). Then `compile_native_parts` emits (a) the lowered `fun` and (b) a `program.registerNative("p/2", ::p_2_native)` line into the registrar list — replacing the audit comment. (verify: confirm `WamProgram`/`WamRuntime` exact API names in the .kt template before editing; keep the change additive so the interpreter path is untouched.)

- **Files to touch:**
  - `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl` (new — mirror `wam_lua_lowered_emitter.pl`)
  - `src/unifyweaver/targets/wam_kotlin_target.pl` — replace the audit-comment body of `compile_native_parts/2` (lines 312–323) with real lowering + a `registerNative` registrar entry; call `wam_kotlin_lowerable/3` + `lower_predicate_to_kotlin/4` in `partition_predicates_/8` (lines 84–88) instead of delegating to `kotlin_target:compile_predicate_to_kotlin/3` (or keep that as a secondary path but stop discarding it to a comment)
  - `templates/targets/kotlin_wam/WamRuntime.kt.mustache` + `Main.kt.mustache` — add the `registerNative`/native-dispatch seam
  - `tests/test_wam_kotlin_target.pl` — extend (do NOT create a new core file; Kotlin's tests live here with a ready-made `gradle_available/0` guard + `run_gradle/5` helper at lines 24–47)

- **Scope of the first cut (keep it small):** support only the **deterministic single-clause** shape (facts and single-clause rules — e.g. the existing `kt_guard/2`, `kt_fact/2`, `kt_same/2` test predicates). Anything else must decline in `wam_kotlin_lowerable/3` and stay on the WAM-register path. No T4/T5/ITE, no multi-clause, no aggregates in this task.

- **Steps:**
  1. Create `wam_kotlin_lowered_emitter` exporting `wam_kotlin_lowerable/3`, `lower_predicate_to_kotlin/4`, `kotlin_lowered_func_name/2`; reuse `wam_text_parser`/`wam_clause_chain`/`wam_ite_structurer`. Port Lua's `build_emission_plan`/`classify_clause_shape` but keep only the single-clause arm.
  2. Emit a native `fun <name>(state: WamState): Boolean` implementing the clause head unifications + deterministic body against the `WamState` API (`readRegister`/`writeRegister`/`bind`/`deref`/`resolve`, `Value.*`). Return `false` on any unification failure so the caller can fall back.
  3. Add the `registerNative` seam to `WamRuntime.kt.mustache` + `Main.kt.mustache`; check native dispatch first in `run`, WAT-style: on `false`, re-init state and run the interpreter.
  4. In `wam_kotlin_target.pl`, wire lowering into the Native bucket and emit both the `fun` and its `registerNative(...)` registrar line; leave `compile_wam_parts`/`registrar_names` (the WAM fallback, lines 325–354) untouched.
  5. Extend `tests/test_wam_kotlin_target.pl`: a unit test asserting a `functions`-mode project for a single-clause predicate emits a real `fun …(state: WamState)` + `registerNative` (NOT `/* Native Kotlin lowering selected`), and — under `[condition(gradle_available)]` — a gradle `run` of that predicate producing the same result as the interpreter path (compare against an `emit_mode(interpreter)` run of the same predicate).
- **Acceptance:** `swipl -q -g run_tests -t halt tests/test_wam_kotlin_target.pl` passes including the new native-dispatch cases; the existing interpreter gradle test still passes; a `functions`-mode single-clause predicate is now actually runnable (native), and a non-lowerable predicate still round-trips through the WAM interpreter.
- **Out of scope / follow-ups (leave as new cards):** T4/T5/ITE/multi-clause Kotlin lowering; deciding whether to also surface `kotlin_target.pl`'s non-WAM native output through this seam.

### EMIT-KOTLIN-2: Lower write-mode structure/list construction (Kotlin)
- **Lever:** Lowered emitters for early scaffolds  **Target:** Kotlin  **Size:** M  **Depends on:** EMIT-KOTLIN
- **Status:** ✅ **Landed** on `cursor/emit-kotlin-structures-f421` (2026-07-12). Root cause was not a write-mode register-ordering bug in the helpers: the emitter wrapped `get_variable`/`put_variable` in a bare Kotlin `{ ... }` block, which is a **discarded lambda** (never invoked). Head vars stayed unbound; `kotlinLoUnifyValue` then fabricated fresh `Var(Xn)` via `?: newVariable`, producing silent wrong answers (`[X1,X2]` instead of `[alpha,beta]`). Fix: emit `run { ... }` so the block executes; re-enable `parts_supported` for structure/list/`set_*`/`unify_*`. Runtime helpers were already correct. Tests assert structure/list/nested builders LOWER and match `emit_mode(interpreter)` via gradle.
- **Goal:** Extend the Kotlin lowered emitter to correctly build structures/lists in the head (write mode), then re-enable those ops in `parts_supported` so predicates like `p(X, wrap(X))` / `p(X,Y,[X,Y])` lower instead of declining to the interpreter.
- **Why deferred:** EMIT-KOTLIN's first cut lowered `put_structure`/`put_list`/`set_*`/`unify_*` incorrectly — write-mode arg pushes read the register *before* the head variable was bound, so the built term contained unbound vars (e.g. `kt_make_list(X,Y,[X,Y])` with `alpha beta` produced `[X1,X2]` not `[alpha,beta]`), and the lowered fn returned `true` so the interpreter fallback never fired. The fix narrowed the lowerable set; this card does it properly.
- **Files to touch:** `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl` (re-add the `parts_supported/1` facts for `get/put_structure`, `get/put_list`, `set_*`, `unify_*`; fix `emit_line_parts/2` for the write-mode ops), `templates/targets/kotlin_wam/WamRuntime.kt.mustache` (the `kotlinLoUnify*` helpers + `pushWriteArg`/`beginStructure*` already exist — audit their read-vs-write-mode contract), `tests/test_wam_kotlin_target.pl`.
- **Reference to copy from:** the WAM-register interpreter already in `WamRuntime.kt.mustache` (`beginStructure`/`beginStructurePut`/`pushWriteArg`/`nextReadArg`, and the main `run` loop's handling of `get_structure`/`unify_*`) — the lowered emit must reproduce exactly that read/write-mode sequencing. Lua's `wam_lua_lowered_emitter.pl` structure handling is the cross-target analog.
- **Steps:**
  1. Reproduce the bug: generate `kt_make_list/3` under `emit_mode(functions)`, gradle-run `kt_make_list/3 alpha beta`, confirm `A3` has unbound vars.
  2. Fix the emitted write-mode sequence so head-variable bindings are visible before the arg push (mirror the interpreter's ordering; likely the `set_value`/`unify_value` register read must deref through the just-bound register, and `get_variable` must bind before any subsequent `put_structure`).
  3. Re-add the declined `parts_supported/1` facts; keep the read-mode path (which already worked) intact.
  4. Add gradle e2e tests asserting lowered output == interpreter output for a structure builder AND a list builder (the regression tests added in EMIT-KOTLIN already assert the *decline* path; convert/extend them to assert the *lowered* path once fixed).
- **Acceptance:** `swipl -q -g run_tests -t halt tests/test_wam_kotlin_target.pl` passes with structure/list predicates lowering (not declining) and producing bindings identical to `emit_mode(interpreter)`.

### EMIT-KOTLIN-3: Deterministic multi-clause clause selection (Kotlin)
- **Lever:** Lowered emitters for early scaffolds  **Target:** Kotlin  **Size:** M  **Depends on:** EMIT-KOTLIN-2
- **Status:** ✅ **Landed** on `cursor/emit-kotlin-multi-clause-f421` (2026-07-12). T5 first-arg `clause_chain` + T4 `multi_clause_n` (snapshot/restore between clauses). Predicates with `call`/`execute` still declined at landing; **EMIT-KOTLIN-4** later added last-call `execute`.
- **Goal:** Extend `wam_kotlin_lowered_emitter.pl` so **deterministic multi-clause** predicates whose bodies contain only unification/builtins (NO `call`/`execute`) lower to native clause selection instead of declining. Mirror Lua's T5 `clause_chain` (first-arg constant dispatch) and T4 `multi_clause_n` (all clauses inline with trail/register restore between attempts).
- **Boundary (this card):**
  - **Lowers:** multi-clause facts (`p(a). p(b). p(c).`), first-arg-indexed constant chains (T5), and other multi-clause deterministic bodies without inter-predicate calls (T4).
  - **Still declines (at EMIT-KOTLIN-3 landing):** any clause with `call`/`execute` — superseded in part by **EMIT-KOTLIN-4** (`execute`) / **EMIT-KOTLIN-5** (`call`). ITE/soft-cut, cut, aggregates also out of scope.
- **Native mechanics:** per clause: `snapshotForNative`, try head + body, on failure `restoreFromSnapshot` and try next; return true on first success, false if all fail (T5 unbound A1 returns false so `tryRun` falls back to the interpreter).
- **Files to touch:** `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl`, `tests/test_wam_kotlin_target.pl`, `docs/WAM_KOTLIN_STATUS.md`, this card.
- **Reference:** `wam_lua_lowered_emitter.pl` `build_emission_plan`/`classify_clause_shape`; shared `wam_clause_chain`.
- **Acceptance:** multi-clause facts + T4/T5 preds LOWER under `emit_mode(functions)` and match `emit_mode(interpreter)` true/false (incl. non-matching); call/execute preds still decline; unit suite + `CONFORMANCE_TARGETS=kotlin,kotlin_functions` stay green.

### EMIT-KOTLIN-4: Lower last-call `execute` (tail recursion / inter-predicate)
- **Lever:** Lowered emitters for early scaffolds  **Target:** Kotlin  **Size:** M  **Depends on:** EMIT-KOTLIN-3
- **Status:** ✅ **Landed** on `cursor/emit-kotlin-execute-f421` (2026-07-13). Dispatch seam `(state, dispatch)` + `return dispatch("P/N", state)`. member/append/acc-reverse LOWER; fib/ack with mid-body `call` still decline.
- **Goal:** Allow a clause's final `execute P/N` in lowered bodies so tail-recursive predicates (`member/2`, `append/3`, accumulator `reverse`) LOWER under `emit_mode(functions)`. Mid-body `call` stays declined → **EMIT-KOTLIN-5**.
- **Dispatch seam:** native fn signature is `(WamState, dispatch:(String,WamState)->Boolean) -> Boolean`; `tryRun` passes `this::tryRun` so `return dispatch("P/N", state)` is native-first then bytecode.
- **Boundary:**
  - **Lowers:** last-call `execute` only (member, append, `crev_acc` / `clist_reverse`).
  - **Still declines:** mid-body `call` (fib, ack clause with call), `builtin_call` (except as already unsupported), cut/ITE/aggregates.
- **Depth caveat:** native `execute` recursion uses the JVM call stack. Measured ~1000 peano-depth OK, ~2000 → `StackOverflowError` on default stack. Conformance list lengths (3) are safe. Prefer decline over wrong answers if a workload overflows.
- **Acceptance:** member/append/(acc)reverse LOWER and match interpreter via gradle differential; fib still declines; unit + `kotlin`/`kotlin_functions` conformance green with no xfails; emitted project registers `lowered_cmem_2` / etc. under kotlin_functions.

### EMIT-KOTLIN-5: Lower mid-body `call P/N` (non-tail / continuation)
- **Lever:** Lowered emitters for early scaffolds  **Target:** Kotlin  **Size:** M  **Depends on:** EMIT-KOTLIN-4
- **Goal:** Emit non-tail `call` with saved continuation / environment so fib/ack and non-tail reverse lower.
- **Out of scope here:** cut, ITE/soft-cut, aggregates, multi-solution enumeration.
- **Note from BENCH-KOTLIN:** today's `tryRun`+snapshot dispatch makes even tail `execute` slower than the interpreter for append; EMIT-KOTLIN-5 should cheapen dispatch (or accept correctness-only value) before expecting a win.
- **Acceptance:** fib (or equivalent) LOWERS under `emit_mode(functions)` and matches interpreter; conformance remains green.

### BENCH-KOTLIN: Measure lowered vs interpreter (in-process)
- **Lever:** Effective-distance / perf evidence  **Target:** Kotlin  **Size:** S  **Depends on:** EMIT-KOTLIN-4
- **Status:** ✅ **Landed** on `cursor/bench-kotlin-f421` (2026-07-14). See [`docs/WAM_KOTLIN_BENCH.md`](WAM_KOTLIN_BENCH.md).
- **Goal:** Answer whether `emit_mode(functions)` pays off vs the bytecode interpreter — timing **execution**, not JVM/`gradle` startup.
- **Design:** `benchmark_main(Pred, Iterations)` project option (warmup + median of timed `tryRun` batches). Harness `examples/benchmark/run_wam_kotlin_lowered_vs_interpreter.pl` builds each program in both modes, asserts `registerNative` under functions, reports speedup.
- **Finding (honest):** lowering does **not** broadly win. Facts/T5/list-builder/append regress (~0.5–0.8×); T4 and member are modest wins (~1.07–1.19×). Likely `tryRun` snapshot cost on every native/recursive entry.
- **Acceptance:** harness runnable; results table documented; unit + kotlin/kotlin_functions conformance stay green.

---

## Lever: Effective-distance benchmark rows

Add each target to the scale-300 effective-distance matrix in
`docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` and the
`examples/benchmark/run_wam_cross_target_benchmark.sh` harness. Workload
`examples/benchmark/effective_distance.pl`; facts
`data/benchmark/300/facts.pl`. Generator template:
`examples/benchmark/generate_wam_c_effective_distance_benchmark.pl`
(compiled targets) or `…generate_wam_python_effective_distance_benchmark.pl`
(interpreted targets).

### BENCH-LLVM: Add LLVM row to effective-distance scale-300 matrix
- **Lever:** Effective-distance benchmark rows  **Target:** LLVM  **Size:** L  **Depends on:** —
- **Goal:** Produce a scale-300 effective-distance number for the LLVM WAM target and add its row to the benchmark matrix.
- **Files to touch:**
  - `examples/benchmark/generate_wam_llvm_effective_distance_benchmark.pl` (new)
  - `examples/benchmark/run_wam_cross_target_benchmark.sh` (append an LLVM per-target block, following the Rust block at lines 46–76)
  - `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` (new row in the "Executive Summary — Effective Distance, Scale 300" table, lines 11–25; add a "LLVM WAM" subsection under Target-by-Target)
  - Optionally extend `tests/core/test_wam_llvm_realdata_benchmark.pl` (exists, but dev-scale + single-kernel only) toward scale-300 + full `effective_distance/3`.
- **Reference to copy from:** `examples/benchmark/generate_wam_c_effective_distance_benchmark.pl` (contract: `main/0`, `generate/3-4`, loads `category_parent/2`, runs `category_ancestor/4` kernel, `max_depth(10)`, reference DFS for kernels-off) and the Rust generator `generate_wam_effective_distance_benchmark.pl`. Emit via `wam_llvm_target.pl` `write_wam_llvm_project/3` (line 1181).
- **Steps:**
  1. Clone the C/Rust generator; swap in `wam_llvm_target:write_wam_llvm_project/3`; drive `llc`/`clang` build like `test_wam_llvm_realdata_benchmark.pl` does.
  2. Wire the outer `effective_distance/3` enumeration (setof/findall over article×root) — the piece the current LLVM realdata test leaves out.
  3. Add the bash block to `run_wam_cross_target_benchmark.sh` (mkdir, `swipl … generate`, build, run at scale 300).
  4. Record query_ms/total_ms and add the doc row.
- **Acceptance:** `./examples/benchmark/run_wam_cross_target_benchmark.sh 300` builds and runs the LLVM benchmark end-to-end and prints query_ms/total_ms; the doc table gains a populated LLVM row (no `--` placeholders). (verify: `llc`/`clang` toolchain availability in the target env.)

### BENCH-CPP: Add C++ row to effective-distance scale-300 matrix
- **Lever:** Effective-distance benchmark rows  **Target:** C++  **Size:** L  **Depends on:** —
- **Goal:** Create a scale-300 effective-distance benchmark for the C++ WAM target and add its matrix row.
- **Files to touch:**
  - `examples/benchmark/generate_wam_cpp_effective_distance_benchmark.pl` (new — no C++ generator exists today)
  - `examples/benchmark/run_wam_cross_target_benchmark.sh` (append C++ block)
  - `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` (new table row + subsection)
- **Reference to copy from:** `generate_wam_c_effective_distance_benchmark.pl` (closest sibling — same C-family build/run shape) and `generate_wam_effective_distance_benchmark.pl`. Emit via `wam_cpp_target.pl` `write_wam_cpp_project/3` (exported line 28).
- **Steps:**
  1. Clone the C generator; swap to `write_wam_cpp_project/3`; build with a C++ compiler (`g++`/`clang++`) instead of `cc`.
  2. Ensure the emitted runner loads `category_parent/2` and runs the `category_ancestor/4` kernel over the full effective-distance enumeration.
  3. Add the bash block; capture query_ms/total_ms; add the doc row + subsection.
- **Acceptance:** `./examples/benchmark/run_wam_cross_target_benchmark.sh 300` builds and runs the C++ benchmark and prints timings; doc table gains a populated C++ row.

### BENCH-C: Add C row to effective-distance scale-300 matrix
- **Lever:** Effective-distance benchmark rows  **Target:** C  **Size:** M  **Depends on:** —
- **Goal:** Wire the already-existing C effective-distance generator into the scale-300 harness and add its matrix row (generator exists; harness wiring + doc row do not).
- **Files to touch:**
  - `examples/benchmark/run_wam_cross_target_benchmark.sh` (append C block; the generator is done)
  - `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` (new table row + subsection — C is currently absent from the matrix entirely)
  - (No new generator: `generate_wam_c_effective_distance_benchmark.pl` already exists and is the most feature-complete of the family.)
- **Reference to copy from:** the existing generator's usage header (`… <facts.pl> <output-dir> [kernels_on|kernels_off] [facts_tsv|facts_lmdb] [layout_profile]`); the Rust per-target block in `run_wam_cross_target_benchmark.sh` lines 46–76 for the bash-block pattern.
- **Steps:**
  1. Add a C bash block: `swipl … generate_wam_c_effective_distance_benchmark.pl -- data/benchmark/300/facts.pl <dir> kernels_on facts_tsv`, then `cc` build, then run.
  2. Capture query_ms/total_ms (pick a representative layout profile, e.g. `parent_only`).
  3. Add the populated doc row + a "C WAM" subsection.
- **Acceptance:** `./examples/benchmark/run_wam_cross_target_benchmark.sh 300` runs the C benchmark and prints timings; doc table gains a populated C row.

### BENCH-GO: Complete Go row in effective-distance scale-300 matrix
- **Lever:** Effective-distance benchmark rows  **Target:** Go  **Size:** M  **Depends on:** —
- **Goal:** Finish the Go effective-distance benchmark (generator exists; doc row is present but shows `--` / "driver in progress") so the Go row carries real scale-300 numbers.
- **Files to touch:**
  - `examples/benchmark/generate_wam_go_effective_distance_benchmark.pl` (exists — complete the "fact data not yet wired into runtime" gap noted in the doc's Go subsection)
  - `examples/benchmark/run_wam_cross_target_benchmark.sh` (Go block lines 136–157 — currently has the "benchmark driver not yet wired up for fact loading" comment; finish it)
  - `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` (replace `--`/`in progress` in the Go row, lines ~25 and the `### Go WAM (In Progress)` subsection lines ~285–290)
- **Reference to copy from:** `generate_wam_c_effective_distance_benchmark.pl` for the fact-loading + full-enumeration wiring that Go is missing; existing Go generator's own `generate/4` and its module-loading note (uses `:- module(...)`, needs `user:load_files`). Emit via `wam_go_target.pl` `write_wam_go_project/3`.
- **Steps:**
  1. In the Go generator/runtime, wire `category_parent/2` fact loading into the emitted Go program (the currently-missing piece).
  2. Run the full `effective_distance/3` query, `go build ./...`, execute at scale 300.
  3. Complete the bash block; capture timings; replace the `--` doc row and rewrite the "In Progress" subsection with results.
- **Acceptance:** `./examples/benchmark/run_wam_cross_target_benchmark.sh 300` builds and runs the Go benchmark and prints query_ms/total_ms; the Go doc row no longer shows `--` and the "In Progress" note is removed.

### BENCH-R: Add R row to effective-distance scale-300 matrix
- **Lever:** Effective-distance benchmark rows  **Target:** R  **Size:** L  **Depends on:** —
- **Goal:** Create a scale-300 effective-distance benchmark for the R WAM target and add its matrix row.
- **Files to touch:**
  - `examples/benchmark/generate_wam_r_effective_distance_benchmark.pl` (new — no R generator exists)
  - `examples/benchmark/run_wam_cross_target_benchmark.sh` (append R block)
  - `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` (new table row + subsection)
- **Reference to copy from:** `generate_wam_python_effective_distance_benchmark.pl` (closest analog — interpreted/scripted target, no compile step, run via interpreter) and the Rust generator for the pipeline docstring. Emit via `wam_r_target.pl` `write_wam_r_project/3` (line 1903, exported line 36).
- **Steps:**
  1. Clone the Python generator; swap to `wam_r_target:write_wam_r_project/3`; run the emitted program with `Rscript` instead of `python`.
  2. Ensure `category_parent/2` loading + `category_ancestor/4` kernel over the full effective-distance enumeration.
  3. Add the bash block (guarded on `Rscript` availability, mirroring the Python block); capture timings; add the doc row + "R WAM" subsection.
- **Acceptance:** `./examples/benchmark/run_wam_cross_target_benchmark.sh 300` runs the R benchmark and prints query_ms/total_ms; doc table gains a populated R row. (verify: `Rscript` present in the target env.)

---

## Document status

Task breakdown generated 2026-07-11 from the fleet gap analysis, via
source-reading subagents. Anchors (line numbers, predicate names) were
captured during recon and should be re-confirmed at implementation
time. As tasks land, check them off here and update the corresponding
`WAM_*_STATUS.md` "Path forward".
