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
SKIP_CSHARP_EXECUTION=1 swipl -q \
     -f init.pl \
     tests/core/test_csharp_query_target.pl \
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

#### 3.1b Running the Generated C# Code

  **Note:** The test above with `SKIP_CSHARP_EXECUTION=1` only validates that C# code generation succeeds. To verify the
  generated code actually compiles and produces correct results, follow these steps:

  1. **Navigate to the generated C# project directory:**

```bash


# Assuming you are in the root folder of your test environment
# We navigate to the most newly created output folder.
$ cd $(ls -td tmp/csharp_query_* /tmp/csharp_query_* 2>/dev/null | head -1)
```
  2. Run the generated C# project:
 ```bash
$  dotnet run

alice,charlie
```
**Note:** `dotnet run` automatically builds the project. To do this in two steps see section 4.


  3. This confirms that the grandparent/2 predicate correctly finds Alice as Charlie's grandparent through the parent/2
  facts.

  Troubleshooting:
  - If dotnet build fails, check that .NET SDK 6.0+ is installed: dotnet --version
  - If you get "No such file or directory" errors, verify the test created a `csharp_query_*` directory during executio
  - If the output is empty or incorrect, the query runtime may have issues with the fixpoint evaluation

  This text includes everything needed: context that the SKIP flag only validates generation, the actual build/run commands
 
### 3.2 full control-plane suite (optional but recommended)
```bash
SKIP_CSHARP_EXECUTION=1 \
swipl -q -f init.pl -g "test_all" -t halt
```
Confirms that integrating the C# target does not regress other subsystems (firewall, data sources, bash targets).

---

## 4. Manual Runtime Validation (Priority 2)

Run these only if you have `.NET` available and want to verify end-to-end execution. They exercise the **build-first** workaround documented in `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md`.

### 4.1 Generate artefacts and run compiled binary
```bash
swipl -q -f init.pl generate_csharp_manual.pl
```
> Produces a project under /tmp/test_even_manual

Navigate to this directory:

```bash
cd /tmp/test_even_manual
```

then build the project

```bash
dotnet build --no-restore
```


#### Running the Compiled Code

**Option 1: Run with dotnet command (recommended)**
```bash
dotnet bin/Debug/net9.0/test_even_manual.dll
```

**Expected output:**
```
0
2
4
```

**Option 2: Build self-contained executable**

If you want a native binary that can be executed directly (e.g., `./test_even_manual`):
```bash
dotnet publish -c Release --self-contained -r linux-x64
./bin/Release/net9.0/linux-x64/publish/test_even_manual
```

This bundles the .NET runtime with your application, resulting in a larger file but no runtime dependency.

**Option 3: Configure binfmt_misc handler (advanced)**

On Linux, you can configure the kernel to automatically invoke dotnet for .dll files:
```bash
# Register .NET DLL handler (requires root)
echo ':dotnet:M::MZ::/usr/bin/dotnet:' | sudo tee /proc/sys/fs/binfmt_misc/register

# Then you can run DLLs directly
chmod +x bin/Debug/net9.0/test_even_manual.dll
./bin/Debug/net9.0/test_even_manual.dll
```

**Note:** This configuration is not standard and may not persist across reboots.

#### About Mono

**Note:** Some older Linux systems allowed double-clicking `.exe` files built with **Mono** (an older .NET Framework implementation). However:
- Modern .NET (net9.0) is a different runtime from Mono
- Mono cannot run modern .NET Core/5+/9 assemblies
- The `binfmt_misc` handlers configured for Mono do not work with modern .NET DLLs

For maximum compatibility across systems, **use Option 1** (`dotnet <path-to-dll>`).

---

**Validation:** All three options should produce the same output: `0`, `2`, `4` (the even numbers from the mutual recursion test).
### 4.2 Bash parity spot-check (optional)
Run the equivalent Bash test to compare results:
```bash
swipl -q -f init.pl \
       -g "use_module('tests/core/test_recursive_csharp_target'), \
           test_recursive_csharp_target:test_recursive_csharp_target, \
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
- [ ] Optional: full control-plane suite (`init.pl` with `test_all`) passes with skip flag.
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
