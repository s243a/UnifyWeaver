# JSON to LiteDB Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing JSON to LiteDB playbook:
[playbooks/json_litedb_playbook.md](../../../../playbooks/json_litedb_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/json_litedb_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to stream JSON data into a LiteDB NoSQL database using inline .NET code within PowerShell scripts. The aim is to ensure:
- JSON dynamic source parses data correctly
- Typed schemas generate proper C# record types
- LiteDB insertion and querying work correctly
- PowerShell + inline .NET integration functions properly

## Inputs & Artifacts
- Playbook file: `playbooks/json_litedb_playbook.md`
- Example records: `playbooks/examples_library/json_litedb_examples.md`
- Test data: `test_data/test_products.json`
- LiteDB library: `lib/LiteDB.dll`
- Generated database: `products.db`
- Temporary directory for artifacts: `tmp/`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Perl installed for record extraction.
3. .NET SDK 6.0+ installed (`dotnet` available).
4. PowerShell 7+ for cross-platform support.
5. LiteDB library installed (run setup script).
6. Run all commands from the repository root.

### Installing LiteDB
```bash
# Automated (for LLMs)
bash scripts/setup/setup_litedb.sh -y

# Interactive
bash scripts/setup/setup_litedb.sh
```

## Execution Steps

### For Linux/macOS (Bash)

1. **Navigate to Project Root**
   ```bash
   cd /path/to/UnifyWeaver
   ```

2. **Install LiteDB (if needed)**
   ```bash
   bash scripts/setup/setup_litedb.sh -y
   ```

3. **Extract the Record**
   ```bash
   perl scripts/utils/extract_records.pl \
     -f content \
     -q "unifyweaver.execution.json_to_litedb_bash" \
     playbooks/examples_library/json_litedb_examples.md \
     > tmp/run_json_litedb.sh
   ```

4. **Run the Bash Script**
   ```bash
   chmod +x tmp/run_json_litedb.sh
   bash tmp/run_json_litedb.sh
   ```

### For Windows (PowerShell)

1. **Navigate to Project Root**
   ```powershell
   cd C:\path\to\UnifyWeaver
   ```

2. **Install LiteDB (if needed)**
   ```powershell
   .\scripts\setup\setup_litedb.ps1 -Yes
   ```

3. **Extract and Run**
   ```powershell
   perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.json_to_litedb_ps" playbooks/examples_library/json_litedb_examples.md | Out-File -FilePath tmp/run_json_litedb.ps1
   ./tmp/run_json_litedb.ps1
   ```

## Verification

**Expected output:**
```
Loading JSON data into LiteDB...
Inserted: Widget Pro
Inserted: Gadget X
Inserted: Tool Master
Inserted: Device Alpha
✅ 4 products loaded

Querying products by category 'Electronics'...
Widget Pro:$29.99
Gadget X:$49.99

Success: JSON data streamed into LiteDB
```

**Success criteria:**
- Script exits with status code 0
- 4 products inserted into LiteDB
- Category query returns 2 electronics products
- Database file `products.db` created
- No LiteDB or .NET errors

## Key Features Tested

1. **JSON dynamic source** with typed schema
2. **Inline .NET code** compilation via `Add-Type`
3. **LiteDB NoSQL** insertion and querying
4. **PowerShell + .NET** integration
5. **DLL caching** for performance (138x speedup)

## Performance Characteristics

| Mode | First Run | Cached Run |
|------|-----------|------------|
| No cache | ~500ms | ~500ms |
| With `pre_compile(true)` | ~500ms | ~5ms |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "LiteDB.dll not found" | Library not installed | Run `scripts/setup/setup_litedb.sh` |
| "Database is locked" | Previous connection open | Close PowerShell, GC collect |
| "Add-Type: Cannot find type" | Namespace mismatch | Check class name matches handler |
| JSON parse errors | Invalid JSON format | Validate `test_products.json` |
| Performance slow | Caching disabled | Enable `pre_compile(true)` |

## Related Material

- Agent-facing playbook: [playbooks/json_litedb_playbook.md](../../../../playbooks/json_litedb_playbook.md)
- Example records: `playbooks/examples_library/json_litedb_examples.md`
- LiteDB setup script: `scripts/setup/setup_litedb.sh`
- PowerShell inline .NET playbook: [playbooks/powershell_inline_dotnet_playbook.md](../../../../playbooks/powershell_inline_dotnet_playbook.md)
- JSON source module: `src/unifyweaver/sources/json_source.pl`
- Codex's guide: `docs/guides/json_to_litedb_streaming.md`
