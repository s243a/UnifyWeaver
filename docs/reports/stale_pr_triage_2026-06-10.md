# Stale PR Triage Report — 2026-06-10

Scope: all open PRs older than one week (created on or before 2026-06-03).
Five PRs qualified: #98, #2002, #2019, #2375, #2703. The four newer PRs
(#2822, #2929, #2953, #2954) were left untouched.

A structural finding that shaped every decision: the repository's `main`
history was restructured in mid-May 2026. Branches created before the
restructure (#98, #2002, #2019) share **no merge base** with current
`main` and cannot be rebased or merged mechanically. Branches created
after (#2375, #2703) merge normally.

## Closed (obsolete)

### #98 — feat: unified generator mode with common_generator abstraction (2025-11-28)
- **Decision: closed.** Fully superseded by `main`.
- `src/unifyweaver/targets/common_generator.pl` exists on `main` with the
  same module interface the PR introduced (`build_variable_map/2`,
  `translate_expr_common/4`, `translate_builtin_common/4`,
  `prepare_negation_data/4`).
- `src/unifyweaver/targets/csharp_target.pl` on `main` is ~5,200 lines
  larger than the PR's version; the `csharp_query_target.pl`
  backward-compat shim also exists on `main`.
- Every fix in the PR description (accessor format, N-way join VarMap
  threading, negation translation) is present on `main` in evolved form.

### #2002 — docs(wam-r): handoff refresh + LMDB plan (2026-05-10)
- **Decision: closed.** Doc-only; superseded by reality.
- The PR *planned* a two-step WAM-R LMDB backend. Step 1
  (load-everything via `source(Pred/Arity, lmdb(Path))`) has since been
  **implemented** on `main`, and `docs/handoff/wam_r_session_handoff.md`
  already lists step 2 (probe-on-demand dispatch) as the top follow-up.
  Re-applying the PR would overwrite current docs with stale planning text.

### #2019 — feat(wam-r): mode-analysis phase 4 (2026-05-11)
- **Decision: closed.** Phase 4 already delivered on `main`, in a newer
  design.
- `wam_r_lowered_emitter.pl` on `main` implements `multi_clause_n`;
  `docs/design/WAM_R_MODE_ANALYSIS_PLAN.md` marks phase 4 (multi_clause_n,
  `get_value` head match, bench harness) as delivered.
- The PR's `state$tail_call` trampoline was replaced on `main` by
  iter-style retry choice points, so the PR's runtime changes would
  conflict with rather than extend the current code.

## Updated (still relevant)

### #2703 — fix(wam-scala): LMDB cursor-scan reflection (2026-06-03)
- **Decision: updated.** The fix is still needed: `main` still has the
  broken `getMethod("`val`")` reflective lookups (backticks embedded in
  the name string) at `templates/targets/scala_wam/runtime.scala.mustache`
  lines 400 and 418.
- Merged `origin/main` into the branch (was 597 commits behind); resolved
  the single conflict (`CHANGELOG.md`, both sides added entries — kept both).
- **Tested post-merge** with Scala 2.13.16 / OpenJDK 21 / lmdbjava 0.9.0:
  all 3 gated LMDB tests pass, including the new `lmdb_backed_kernel`
  capstone test (kernel reading edges from LMDB via the `streamAll`
  cursor-scan path the bug broke).
- Pushed merge commit to `claude/wam-scala-lmdb-kernel`.

### #2375 — test(csharp-query): ClosurePairPlanStrategy coverage (2026-05-21)
- **Decision: updated.** Merged `origin/main` (was 1,430 commits behind;
  clean merge, no conflicts). The 4 new strategy tests
  (`MemoizedBySource`, `MemoizedByTarget`, `Backward`,
  `MixedDirectionWithPairProbeCache`) survived the merge intact.
- Re-ran the 4 new tests post-merge (.NET SDK 8.0.127): all 4 **PASS**
  with runtime execution and `STRATEGY_USED:...=true` assertions.
- The full 247-test suite was not re-run end-to-end in this session
  (each test performs a `dotnet build`; the suite exceeds the session
  command time limit). A full-suite run is recommended before merge.
- Pushed merge commit to `claude/csharp-strategy-coverage`.

## Relation to future todos

1. **Probe-on-demand LMDB dispatch for WAM-R** — the top item in
   `docs/handoff/wam_r_session_handoff.md`. Closing #2002 does not lose
   this: the handoff doc on `main` already carries the up-to-date version
   of the plan, with the Scala implementation (the subject of #2703) as
   the reference shape (`lookupByArg1` for ground arg1, `streamAll`
   otherwise). Note that #2703's reflection fix is a prerequisite for
   copying that reference shape — the `streamAll` path was silently broken.
2. **LMDB-backed graph kernels as the large-graph path** — #2703 is the
   capstone validation that `kernel_dispatch` composes with LMDB fact
   sources. Merging it unblocks the documented >100k-fact workloads and
   the roadmap items "LMDB fact-source" for other targets
   (`docs/WAM_TARGET_ROADMAP.md` lists LLVM as lacking LMDB integration).
3. **T4 multi_clause_n sweep** — #2019's closure is safe because phase 4
   landed independently; the live frontier is the per-target T4 sweep
   (Scala #2941, Rust #2942, Go #2952, C++ #2953, Haskell #2954 done;
   fsharp, clojure, llvm, lua remaining per #2954's notes). WAM-R phase 5
   candidates remain listed in `WAM_R_MODE_ANALYSIS_PLAN.md`.
4. **C# query strategy coverage** — merging #2375 closes the test-coverage
   gap left by #2371 (every non-Auto `ClosurePairPlanStrategy` variant
   asserted at runtime). Its harness addition
   (`harness_source_with_strategy_flag_options/5`) is reusable for future
   strategy tests, e.g. when the auto-selector is taught to pick
   `Backward` / `MixedDirectionWithPairProbeCache` naturally — currently
   only reachable via configured override.
5. **Generator-mode unification (#98)** — no follow-up needed; the
   abstraction lives on `main` as `common_generator.pl` and is shared by
   the current target implementations.

## Housekeeping observations

- The May-2026 history restructure silently orphaned every pre-restructure
  branch. Any other old branches on the remote that predate it will have
  the same no-merge-base problem; closing their PRs promptly (as done
  here) avoids confusing future triage.
- Testing toolchain used (installed in the session container): SWI-Prolog
  9.0.4, .NET SDK 8.0.127, Scala 2.13.16, OpenJDK 21, lmdbjava 0.9.0 +
  jnr/asm transitive JARs fetched from Maven Central.
