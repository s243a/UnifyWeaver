# PowerShell Inline .NET Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing PowerShell inline .NET playbook:
[playbooks/powershell_inline_dotnet_playbook.md](../../../../playbooks/powershell_inline_dotnet_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/powershell_inline_dotnet_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to embed inline .NET code within PowerShell scripts for rapid prototyping and workflow integration. The aim is to ensure:
- C# code compiles correctly via `Add-Type`
- DLL caching provides significant performance improvement
- PowerShell integration works seamlessly
- Cross-platform execution functions properly

## Inputs & Artifacts
- Playbook file: `playbooks/powershell_inline_dotnet_playbook.md`
- Example records: `playbooks/examples_library/powershell_dotnet_examples.md`
- Generated PowerShell scripts: `tmp/*.ps1`
- DLL cache: `$env:TEMP/unifyweaver_dotnet_cache/`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Perl installed for record extraction.
3. PowerShell 7+ for cross-platform support.
4. .NET SDK 6.0+ installed.
5. Run all commands from the repository root.

## Execution Steps

### For Linux/macOS (Bash launching PowerShell)

1. **Navigate to Project Root**
   ```bash
   cd /path/to/UnifyWeaver
   ```

2. **Extract the Record**
   ```bash
   perl scripts/utils/extract_records.pl \
     -f content \
     -q "unifyweaver.execution.dotnet_string_reverser_bash" \
     playbooks/examples_library/powershell_dotnet_examples.md \
     > tmp/run_string_reverser.sh
   ```

3. **Run the Script**
   ```bash
   chmod +x tmp/run_string_reverser.sh
   bash tmp/run_string_reverser.sh
   ```

### For Windows (PowerShell)

1. **Navigate to Project Root**
   ```powershell
   cd C:\path\to\UnifyWeaver
   ```

2. **Extract and Run**
   ```powershell
   perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.dotnet_string_reverser_ps" playbooks/examples_library/powershell_dotnet_examples.md | Out-File -FilePath tmp/run_string_reverser.ps1
   ./tmp/run_string_reverser.ps1
   ```

## Verification

**Expected output:**
```
# First run (~386ms - compiles C#)
dlroW olleH

# Cached run (~2.8ms - uses cached DLL)
dlroW olleH
```

**Success criteria:**
- Script exits with status code 0
- String reversed correctly
- First run compiles successfully
- Cached runs are significantly faster (138x)
- No `Add-Type` errors

## Key Features Tested

1. **Inline C# compilation** via `Add-Type`
2. **DLL caching** for 138x performance improvement
3. **PowerShell integration** with .NET classes
4. **Cross-platform execution** (Windows, Linux, macOS)
5. **NuGet package references** (System.Text.Json, etc.)

## Performance Characteristics

| Mode | First Run | Cached Run | Speedup |
|------|-----------|------------|---------|
| No cache | ~386ms | ~386ms | 1x |
| With `pre_compile(true)` | ~386ms | ~2.8ms | **138x** |

## Comparison with C# Codegen

| Factor | Inline .NET | C# Codegen |
|--------|-------------|------------|
| Deployment | PowerShell script | Standalone .exe |
| First Run | 386ms | 500ms |
| Cached Run | 2.8ms | 50ms |
| Dependencies | PowerShell required | .NET runtime only |
| Use Case | Scripting, automation | Production apps |
| Platform | Windows-first | Cross-platform |

## Available Example Records

### Bash Execution
- `unifyweaver.execution.dotnet_string_reverser_bash`
- `unifyweaver.execution.dotnet_json_validator_bash`
- `unifyweaver.execution.dotnet_csv_transformer_bash`

### PowerShell Execution
- `unifyweaver.execution.dotnet_string_reverser_ps`
- `unifyweaver.execution.dotnet_json_validator_ps`
- `unifyweaver.execution.dotnet_csv_transformer_ps`

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "Add-Type: Cannot find type" | Namespace mismatch | Ensure namespace matches handler class |
| "Assembly already loaded" | Cached assembly conflict | New PowerShell session or `Remove-Module` |
| Performance slower than expected | Caching disabled | Enable `pre_compile(true)` |
| "pwsh not found" | PowerShell 7 not installed | Install PowerShell 7+ |
| Compilation errors | Missing NuGet reference | Add to `references([...])` |

## Key Configuration Options

```prolog
:- dynamic_source(
    my_handler/2,
    [source_type(dotnet), target(powershell)],
    [csharp_inline('...'),
     pre_compile(true),           % Enable DLL caching
     references(['System.Text.Json'])  % NuGet packages
    ]
).
```

| Option | Description | Default |
|--------|-------------|---------|
| `source_type(dotnet)` | Use inline .NET | Required |
| `target(powershell)` | Generate PowerShell | Required |
| `csharp_inline/1` | C# code to compile | Required |
| `pre_compile(true)` | Cache compiled DLL | `false` |
| `references/1` | NuGet package references | `[]` |

## Related Material

- Agent-facing playbook: [playbooks/powershell_inline_dotnet_playbook.md](../../../../playbooks/powershell_inline_dotnet_playbook.md)
- Example records: `playbooks/examples_library/powershell_dotnet_examples.md`
- JSON to LiteDB playbook: [playbooks/json_litedb_playbook.md](../../../../playbooks/json_litedb_playbook.md)
- C# codegen playbook: [playbooks/csharp_codegen_playbook.md](../../../../playbooks/csharp_codegen_playbook.md)
- .NET source module: `src/unifyweaver/sources/dotnet_source.pl`
- Examples: `examples/powershell_dotnet_example.pl`
