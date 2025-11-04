# ⚙️ C# Query Target Test Plan (v0.1)

**Date:** February 2026  
**Release:** v0.1 – C# query runtime mutual recursion support  
**Platforms:** Linux, WSL, macOS, Windows (PowerShell)  
**Estimated Time:** 15–30 minutes (core checks) + 15 minutes (optional manual runtime validation)

---

## 1. Objectives
- Verify that the C# query runtime compiles plans for facts, joins, arithmetic, comparisons, linear recursion, and mutual recursion.
- Confirm parity with Bash semantics for distinct handling and SCC evaluation.
- Exercise both automated (SWI-Prolog driven) and manual (`dotnet`) execution paths.
- Provide guidance for platform-specific considerations (WSL vs. PowerShell) even though the target is largely platform agnostic.

---

## 2. Prerequisites

| Tool | Minimum Version | Notes |
|------|-----------------|-------|
| `swipl` | 9.2.9 | Required to run the Prolog regression suite. |
| `.NET SDK` | 9.0.2+ | Needed for manual build & execution of generated plans. |
| `bash` / `pwsh` | latest | Only used to launch the test commands; no target-specific dependencies. |

### Environment Notes
- The automated Prolog tests behave identically across WSL and native PowerShell because they execute inside SWI-Prolog.
- Manual `.NET` steps work from any shell as long as the SDK is on `PATH`.
- Windows users should ensure `dotnet` is available in both PowerShell and WSL environments if running tests from both locations.

---

## 3. Core Test Checklist (Priority 1)

Run **all** steps in order. Use `SKIP_CSHARP_EXECUTION=1` to avoid the known `dotnet run` pipe deadlock (see `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md`).

### 3.1 C# Query Target Regression
```bash
SKIP_CSHARP_EXECUTION=1 \
swipl -q \
     -f tests/core/test_csharp_query_target.pl \
     -g test_csharp_query_target:test_csharp_query_target \
     -t halt
```

**Validates:**
- Facts (`test_fact/2`)
- Joins (`test_link/2`)
- Selection/constraints (`test_filtered/1`, `test_positive/1`)
- Arithmetic (`test_increment/2`, `test_factorial/2`)
- Linear recursion (`test_reachable/2`)
- Mutual recursion (`test_even/1`, `test_odd/1`)

Expected output shows “dotnet execution skipped” for each block followed by `=== C# query target tests complete ===`.

### 3.2 full control-plane suite (optional but recommended)
```bash
SKIP_CSHARP_EXECUTION=1 \
swipl -q -f run_all_tests.pl -g main -t halt
```
Confirms that integrating the C# target does not regress other subsystems (firewall, data sources, bash targets).

---

## 4. Manual Runtime Validation (Priority 2)

Run these only if you have `.NET` available and want to verify end-to-end execution. They exercise the **build-first** workaround documented in `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md`.

### 4.1 Generate artefacts and run compiled binary
```bash
SKIP_CSHARP_EXECUTION=1 \
swipl -q \
     -f tests/core/test_csharp_query_target.pl \
     -g "test_csharp_query_target:configure_csharp_query_options, \
         test_csharp_query_target:setup_test_data, \
         test_csharp_query_target:build_manual_plan(test_even/1, Dir), \
         halt."
```
> Produces a project under `output/csharp/<uuid>/`.

From that directory:
```bash
dotnet build --no-restore
dotnet bin/Debug/net9.0/<generated>.dll   # if DLL produced
# or
./bin/Debug/net9.0/<generated>            # if self-contained binary produced
```
Expected output: `0`, `2`, `4` (mutual recursion parity).

*(If you prefer, run `dotnet run --no-restore` after reading the hang workaround.)*

### 4.2 Bash parity spot-check (optional)
Run the equivalent Bash test to compare results:
```bash
swipl -q \
     -g "use_module('tests/core/test_recursive_csharp_target'), \
         test_recursive_csharp_target:test_cf_fact, \
         halt."
```
Ensure C# and Bash outputs match for the same predicates (focus on mutual recursion and arithmetic cases).

---

## 5. Platform Considerations

| Platform | Notes |
|----------|-------|
| WSL / native Linux | All commands above run as-is. Ensure `dotnet` is installed in the Linux environment. |
| Windows PowerShell | Use `SKIP_CSHARP_EXECUTION=1` to avoid pipe deadlocks. Manual build steps work identically; ensure `dotnet` is on the PowerShell `PATH`. |
| CI / Automation | Set `SKIP_CSHARP_EXECUTION=1` in the job environment. Persist artefacts (`output/csharp/...`) as build artifacts if you need to inspect the generated C# plans. |

---

## 6. Troubleshooting
- **Missing `dotnet`:** install the SDK (`winget install Microsoft.DotNet.SDK.9` on Windows, `sudo apt install dotnet-sdk-9.0` on Ubuntu/WSL).
- **Pipe deadlock during tests:** verify `SKIP_CSHARP_EXECUTION` is set; if you need runtime validation, use the manual build-first flow.
- **Hash collision or duplicate rows:** inspect generated C# module to confirm `DistinctNode` is present; ensure test data isn’t altered.
- **Mutual recursion fails:** check that the predicate group is allowed by the firewall and no stale artefacts remain in `tmp/` or `output/csharp/`.

---

## 7. Summary Checklist
- [ ] `tests/core/test_csharp_query_target.pl` passes with skip flag.
- [ ] Optional: `run_all_tests.pl` passes with skip flag.
- [ ] Optional: manual build-first project executes and outputs `alice,charlie` (join) and `0/2/4` (mutual recursion).
- [ ] No unexpected files remain in `tmp/` unless intentionally kept (`--csharp-query-keep`).
- [ ] Results recorded in release notes or PR checklist.

---

## 8. References
- `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md` – rationale for build-first workflow.
- `docs/targets/csharp-query-runtime.md` – architectural details of the query runtime.
- `docs/targets/comparison.md` – backend capabilities comparison.
- `tests/core/test_csharp_query_target.pl` – source for helper predicates referenced above.

This plan will evolve alongside the C# targets. Update it whenever the runtime gains new features (ordered dedup, memoisation, distributed execution, etc.).
