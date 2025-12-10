# Playbook Test Results

**Test Date**: 2025-12-10
**Test Method**: Automated execution with difficulty rating by AI agents

## Summary

| Playbook | Haiku Rating | Sonnet Rating | Status |
|----------|-------------|---------------|--------|
| sqlite_source | 2-5/10 | - | PASS |
| sql_window | 3-4/10 | - | PASS |
| bash_parallel | 3-6/10 | - | PASS |
| csharp_generator | 3/10 | - | PASS (incomplete) |
| powershell_inline_dotnet | 4/10 | - | PASS |
| awk_advanced | 4-5/10 | - | PASS |
| json_litedb | 7/10 | 4/10 | PASS |
| csharp_query | 8/10 | - | BLOCKED |

## Difficulty Scale

- **1-3**: Very clear, deterministic steps - copy-paste execution
- **4-5**: Some interpretation needed - requires basic context
- **6-7**: Requires context understanding - needs architectural knowledge
- **8-10**: Complex reasoning required - may have blockers or missing functionality

## Detailed Results

### sqlite_source_playbook.md

**Ratings**: 2/10, 5/10 (two runs)
**Status**: PASS

Key findings:
- Linear, sequential steps with exact commands
- Copy-paste executable
- Clear expected outputs
- Basic bash/Prolog knowledge helpful

### sql_window_playbook.md

**Ratings**: 3/10, 4/10 (two runs)
**Status**: PASS

Key findings:
- Explicit numbered steps
- Window functions demo worked correctly
- GROUP BY and recursive CTE examples succeeded
- Very deterministic workflow

### bash_parallel_playbook.md

**Ratings**: 3/10, 6/10 (two runs)
**Status**: PASS

Key findings:
- Varied ratings due to extraction tool quirks
- All partitioning strategies executed correctly
- Worker pool pattern demonstrated successfully
- `extract_records.pl` format understanding needed

### csharp_generator_playbook.md

**Rating**: 3/10
**Status**: PASS (incomplete playbook)

Key findings:
- Playbook lacks explicit execution steps
- Shows code patterns without commands
- Underlying `compile_predicate_to_csharp/3` works
- Recommendation: Add extraction records and step-by-step instructions

### powershell_inline_dotnet_playbook.md

**Rating**: 4/10
**Status**: PASS

Key findings:
- Clear 3-step workflow
- Module discovery required (library paths)
- Configuration options like `pre_compile(true)` need some understanding
- Good documentation structure

### awk_advanced_playbook.md

**Ratings**: 4/10, 5/10 (two runs)
**Status**: PASS

Key findings:
- Aggregation, tail recursion, and constraint compilation all worked
- Extraction tool query format (`:::` delimiters) requires learning
- Clear expected outputs documented
- Optional sections marked appropriately

### json_litedb_playbook.md

**Ratings**: 7/10 (Haiku), 4/10 (Sonnet)
**Status**: PASS

Key findings:
- Complex multi-system integration (Prolog → C# → PowerShell → LiteDB)
- LiteDB installation has interactive prompts
- Sonnet handled interactive setup more easily
- Model capability affects perceived difficulty

### csharp_query_playbook.md

**Rating**: 8/10
**Status**: BLOCKED

Key findings:
- Reports missing `build_unifyweaver_project/0` predicate
- Underlying functionality (`compile_predicate_to_csharp/3`) works
- Playbook may reference aspirational/unimplemented API
- Manual testing showed C# code generation succeeds

## Issues Identified

### 1. Extract Tool Query Matching

The `extract_records.pl` tool uses regex matching, which can cause:
- Partial matches returning multiple records
- `csharp_sum_pair` matches both bash and PowerShell examples
- Fix: Use exact match with `^query$` anchors

### 2. Missing Predicates

- `build_unifyweaver_project/0` referenced in csharp_query_playbook doesn't exist
- The underlying compilation machinery works via `compile_predicate_to_csharp/3`

### 3. Interactive Setup

- Some playbooks require handling interactive prompts (e.g., LiteDB setup)
- More capable models handle this better

## Recommendations

1. **Add exact match examples** to playbooks that use `extract_records.pl`
2. **Update csharp_query_playbook** to use `compile_predicate_to_csharp/3` directly
3. **Add execution records** to csharp_generator_playbook
4. **Document interactive setup handling** for playbooks with dependencies

## Test Environment

- Platform: Linux (PRoot-Distro)
- SWI-Prolog: Available
- .NET SDK: 9.0 (net9.0)
- Claude Models: Haiku 4.5, Sonnet 4
- Gemini Models: 2.5 Pro (quota limited), 2.5 Flash

## Notes

- Gemini tests mostly failed due to API quota exhaustion
- Some tests ran multiple times for consistency verification
- Model capability significantly affects perceived playbook difficulty
