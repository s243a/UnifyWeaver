# C# Query Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing C# query playbook:
[playbooks/csharp_query_playbook.md](../../../../playbooks/csharp_query_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/csharp_query_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to compile recursive Prolog predicates (like Fibonacci) to C# using the `csharp_query` target and execute the resulting program. The aim is to ensure:
- Recursive predicates compile correctly to C#
- The query runtime handles fixpoint evaluation
- Execution produces expected output

## Inputs & Artifacts
- Playbook file: `playbooks/csharp_query_playbook.md`
- Example records: `playbooks/examples_library/csharp_examples.md`
- Generated Prolog script: `tmp/fib_csharp.pl`
- Generated C# project: `tmp/csharp_fib_project/`
- Temporary directory for artifacts: `tmp/`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Perl installed for record extraction.
3. .NET SDK 6.0+ installed (`dotnet` available).
4. Run all commands from the repository root.

## Execution Steps

### For Linux/macOS (Bash)

1. **Navigate to Project Root**
   ```bash
   cd /path/to/UnifyWeaver
   ```

2. **Extract the Record**
   ```bash
   perl scripts/utils/extract_records.pl \
     -f content \
     -q "unifyweaver.execution.csharp_fibonacci" \
     playbooks/examples_library/csharp_examples.md \
     > tmp/run_csharp_fibonacci.sh
   ```

3. **Run the Bash Script**
   ```bash
   chmod +x tmp/run_csharp_fibonacci.sh
   bash tmp/run_csharp_fibonacci.sh
   ```

### For Windows (PowerShell)

1. **Navigate to Project Root**
   ```powershell
   cd C:\path\to\UnifyWeaver
   ```

2. **Extract the Record**
   ```powershell
   perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.csharp_fibonacci_ps" playbooks/examples_library/csharp_examples.md | Out-File -FilePath tmp/run_csharp_fibonacci.ps1
   ```

3. **Run the PowerShell Script**
   ```powershell
   ./tmp/run_csharp_fibonacci.ps1
   ```

## Verification

**Expected output:**
```
Compiling Prolog to C#...
Executing C# program...
8: 21
Success: C# program compiled and executed successfully.
```

**Success criteria:**
- Script exits with status code 0
- Output contains `8: 21` (8th Fibonacci number is 21)
- No compilation errors from `dotnet build`
- Generated C# project in `tmp/csharp_fib_project/`

## Key Features Tested

1. **Recursive predicate compilation** (Fibonacci)
2. **C# query runtime** with fixpoint evaluation
3. **Cross-platform execution** (Bash and PowerShell)
4. **Auto-detection of .NET SDK version**

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "No compatible .NET SDK found" | .NET SDK not installed | Install .NET SDK 6.0+ |
| "Could not find swipl.exe" (PowerShell) | SWI-Prolog not in PATH | Add swipl to PATH or modify script |
| Wrong Fibonacci result | Base case or recursion issue | Verify predicate definition |
| Compilation errors | Missing module | Check `csharp_query_target` module |
| Timeout during execution | Infinite recursion | Check termination conditions |

## Related Material

- Agent-facing playbook: [playbooks/csharp_query_playbook.md](../../../../playbooks/csharp_query_playbook.md)
- Example records: `playbooks/examples_library/csharp_examples.md`
- C# query target module: `src/unifyweaver/targets/csharp_query_target.pl`
- C# test plan: `docs/development/testing/v0_1_csharp_test_plan.md`
- Extraction skill: `skills/skill_extract_records.md`
