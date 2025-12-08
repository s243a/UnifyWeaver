# C# Codegen Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing C# codegen playbook:
[playbooks/csharp_codegen_playbook.md](../../../../playbooks/csharp_codegen_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/csharp_codegen_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to compile non-recursive Prolog predicates (like `grandparent/2`) to C# source code and execute the resulting program. The aim is to ensure:
- Prolog facts and rules compile correctly to C#
- The generated C# code builds without errors
- Execution produces expected output

## Inputs & Artifacts
- Playbook file: `playbooks/csharp_codegen_playbook.md`
- Example records: `playbooks/examples_library/csharp_nonrecursive_examples.md`
- Generated Prolog script: `tmp/grandparent_csharp.pl`
- Generated C# project: `tmp/csharp_grandparent_project/`
- Temporary directory for artifacts: `tmp/`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Perl installed for record extraction.
3. .NET SDK 6.0, 7.0, or 8.0 installed (`dotnet` available).
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
     -q "unifyweaver.execution.csharp_grandparent_bash" \
     --file-filter=all \
     playbooks/examples_library/csharp_nonrecursive_examples.md \
     > tmp/run_csharp_grandparent.sh
   ```

3. **Run the Bash Script**
   ```bash
   chmod +x tmp/run_csharp_grandparent.sh
   bash tmp/run_csharp_grandparent.sh
   ```

### For Windows (PowerShell)

1. **Navigate to Project Root**
   ```powershell
   cd C:\path\to\UnifyWeaver
   ```

2. **Extract the Record**
   ```powershell
   perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.csharp_grandparent_ps" --file-filter=all playbooks/examples_library/csharp_nonrecursive_examples.md | Out-File -FilePath tmp/run_csharp_grandparent.ps1
   ```

3. **Run the PowerShell Script**
   ```powershell
   ./tmp/run_csharp_grandparent.ps1
   ```

## Verification

**Expected output:**
```
Compiling Prolog to C#...
Executing C# program...
anne:charles
anne:diana
Success: C# program compiled and executed successfully.
```

**Success criteria:**
- Script exits with status code 0
- Output contains grandparent relationships: `anne:charles` and `anne:diana`
- No compilation errors from `dotnet build`
- Generated C# project in `tmp/csharp_grandparent_project/`

## Key Features Tested

1. **Auto-detection of .NET SDK version** (8.0, 7.0, 6.0)
2. **Cross-platform execution** using `dotnet run`
3. **Here-document handling** for `.csproj` file creation
4. **Non-recursive predicate compilation** (`grandparent/2`)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "No compatible .NET SDK found" | .NET SDK not installed | Install .NET SDK 6.0+ |
| "Could not find swipl.exe" (PowerShell) | SWI-Prolog not in PATH | Add swipl to PATH or modify script |
| "arguments not sufficiently instantiated" | Module loading issue | Verify `csharp_stream_target` module loads |
| Compilation errors | Missing module or facts | Check Prolog script syntax |
| Output missing relationships | Fact definitions incorrect | Verify parent/2 facts in generated script |

## Related Material

- Agent-facing playbook: [playbooks/csharp_codegen_playbook.md](../../../../playbooks/csharp_codegen_playbook.md)
- Example records: `playbooks/examples_library/csharp_nonrecursive_examples.md`
- C# stream target module: `src/unifyweaver/targets/csharp_stream_target.pl`
- Extraction skill: `skills/skill_extract_records.md`
