# Playbook Testing Procedure

This document provides a complete procedure for testing all UnifyWeaver playbooks with LLM agents.

## Quick Reference: Which Models to Use

**Recommended: Test with 4 models for comprehensive coverage:**
1. **Claude Haiku 4.5** - Fast baseline, good tool use
2. **Claude Sonnet 4** - Higher capability, validates complex playbooks
3. **Gemini 2.5 Pro** - Cross-vendor validation
4. **Gemini 2.5 Flash** - Fast, catches tool use edge cases

| Category | Playbooks | Test With | Notes |
|----------|-----------|-----------|-------|
| **Standard** | csv, xml, tree_recursion, mutual_recursion, csharp_codegen, prolog_generation, parallel_execution | All 4 models | Both vendors work well |
| **Complex** | json_litedb | Haiku + Sonnet preferred | Gemini may timeout; Sonnet: 4/10, Haiku: 7/10 |
| **Fixed** | csharp_query | All 4 models | Now uses `csharp_sum_pair` records |
| **Haiku Only** | powershell_inline_dotnet | Haiku 4.5 only | Gemini rate limited; Haiku passes (4/10) |
| **Issues** | large_xml_streaming | Gemini only (so far) | Has path errors, Python syntax issues |
| **Design Only** | csharp_generator | Do not test | Lacks executable steps; needs extraction record |
| **Complex** | csharp_xml_fragments | Needs work | Too complex for current playbook structure |

## Environment Setup

### Standard Linux/macOS

```bash
cd /path/to/UnifyWeaver
```

### Termux with proot-distro (Debian)

For .NET playbooks in Termux, you must set memory limits:

```bash
# Required for .NET in proot-distro Debian
export DOTNET_GCHeapHardLimit=1C0000000

# Then run your tests
```

**Affected playbooks requiring this setting:**
- `csharp_codegen_playbook.md`
- `csharp_query_playbook.md` (blocked anyway)
- `csharp_xml_fragments_playbook.md`
- `csharp_generator_playbook.md`
- `json_litedb_playbook.md`
- `powershell_inline_dotnet_playbook.md`

## Test Commands

### Testing with Claude Haiku 4.5

```bash
claude -p "Pretend you have fresh context and run the playbook at playbooks/<PLAYBOOK_NAME>.md

After completing, rate the difficulty 1-10 where:
- 1-3: Very clear, deterministic steps
- 4-5: Some interpretation needed
- 6-7: Requires context understanding
- 8-10: Complex reasoning required

Explain your rating." \
  --model claude-haiku-4-5-20251001 \
  --allowedTools "Bash(perl:*),Bash(chmod:*),Bash(bash:*),Bash(swipl:*),Bash(python3:*),Bash(mkdir:*),Bash(cat:*),Bash(ls:*),Bash(dotnet:*),Bash(pwsh:*),Read,Glob,Grep"
```

### Testing with Claude Sonnet 4

```bash
claude -p "Pretend you have fresh context and run the playbook at playbooks/<PLAYBOOK_NAME>.md

After completing, rate the difficulty 1-10 where:
- 1-3: Very clear, deterministic steps
- 4-5: Some interpretation needed
- 6-7: Requires context understanding
- 8-10: Complex reasoning required

Explain your rating." \
  --model claude-sonnet-4-20250514 \
  --allowedTools "Bash(perl:*),Bash(chmod:*),Bash(bash:*),Bash(swipl:*),Bash(python3:*),Bash(mkdir:*),Bash(cat:*),Bash(ls:*),Bash(dotnet:*),Bash(pwsh:*),Read,Glob,Grep"
```

### Testing with Gemini 2.5 Pro

```bash
gemini --model gemini-2.5-pro \
  --prompt "Pretend you have fresh context and run the playbook at playbooks/<PLAYBOOK_NAME>.md

After completing, rate the difficulty 1-10 where:
- 1-3: Very clear, deterministic steps
- 4-5: Some interpretation needed
- 6-7: Requires context understanding
- 8-10: Complex reasoning required

Explain your rating." \
  --yolo
```

### Testing with Gemini 2.5 Flash

```bash
gemini --model gemini-2.5-flash \
  --prompt "Pretend you have fresh context and run the playbook at playbooks/<PLAYBOOK_NAME>.md

After completing, rate the difficulty 1-10 where:
- 1-3: Very clear, deterministic steps
- 4-5: Some interpretation needed
- 6-7: Requires context understanding
- 8-10: Complex reasoning required

Explain your rating." \
  --yolo
```

## Complete Playbook Test List

### Tier 1: Test with Both Haiku 4.5 and Gemini 2.5 Pro

These playbooks are well-tested and work with both models:

| Playbook | Avg Difficulty | Status |
|----------|----------------|--------|
| `csv_data_source_playbook.md` | 1.5/10 | Both pass |
| `xml_data_source_playbook.md` | 1.5/10 | Both pass |
| `tree_recursion_playbook.md` | 2/10 | Both pass |
| `mutual_recursion_playbook.md` | 1.5/10 | Both pass |
| `csharp_codegen_playbook.md` | 1.5/10 | Both pass |
| `prolog_generation_playbook.md` | 2/10 | Both pass |
| `parallel_execution_playbook.md` | 3/10 | Both pass |

**Test command (run both in parallel):**
```bash
# Haiku
claude -p "..." --model claude-haiku-4-5-20251001 ... 2>&1 &

# Gemini
gemini --model gemini-2.5-pro ... --yolo 2>&1 &

wait
```

### Tier 2: Test with Haiku 4.5 Only

These playbooks have issues with Gemini (usually timeouts):

| Playbook | Difficulty | Issue with Gemini |
|----------|------------|-------------------|
| `json_litedb_playbook.md` | 7/10 | Times out; complex multi-system |

**Test command:**
```bash
claude -p "Pretend you have fresh context and run the playbook at playbooks/json_litedb_playbook.md..." \
  --model claude-haiku-4-5-20251001 \
  --allowedTools "Bash(perl:*),Bash(chmod:*),Bash(bash:*),Bash(swipl:*),Bash(python3:*),Bash(mkdir:*),Bash(cat:*),Bash(ls:*),Bash(dotnet:*),Bash(pwsh:*),Read,Glob,Grep"
```

### Tier 3: Recently Fixed (Needs Retesting)

These playbooks had blocking issues that have been resolved:

| Playbook | Fix Applied | Status |
|----------|-------------|--------|
| `csharp_query_playbook.md` | Now uses `csharp_sum_pair` records | Ready for testing |

### Tier 4: Needs Evaluation (Untested)

These playbooks need initial testing to determine which models work:

| Playbook | Notes |
|----------|-------|
| `large_xml_streaming_playbook.md` | Multi-stage pipeline, likely complex |
| `csharp_xml_fragments_playbook.md` | XML streaming with C# |
| `powershell_inline_dotnet_playbook.md` | PowerShell + .NET |
| `csharp_generator_playbook.md` | May have similar issues to csharp_query |

### Tier 5: Future - Requires Smarter Models

Reserved for playbooks that may need Sonnet 4 or Opus 4.5:

| Playbook | Why Smarter Model Needed |
|----------|--------------------------|
| *(none currently)* | |

## Running Full Test Suite

### Parallel Batch Test Script

```bash
#!/bin/bash
# test_all_playbooks.sh

cd /root/UnifyWeaver

# For Termux proot-distro
export DOTNET_GCHeapHardLimit=1C0000000

# Tier 1: Both models
TIER1_PLAYBOOKS=(
  "csv_data_source_playbook"
  "xml_data_source_playbook"
  "tree_recursion_playbook"
  "mutual_recursion_playbook"
  "csharp_codegen_playbook"
  "prolog_generation_playbook"
  "parallel_execution_playbook"
)

for pb in "${TIER1_PLAYBOOKS[@]}"; do
  echo "Testing $pb with Haiku 4.5..."
  claude -p "Pretend you have fresh context and run the playbook at playbooks/${pb}.md" \
    --model claude-haiku-4-5-20251001 \
    --allowedTools "Bash(*),Read,Glob,Grep" 2>&1 > "results/${pb}_haiku.log" &

  echo "Testing $pb with Gemini 2.5 Pro..."
  gemini --model gemini-2.5-pro \
    --prompt "Pretend you have fresh context and run the playbook at playbooks/${pb}.md" \
    --yolo 2>&1 > "results/${pb}_gemini.log" &
done

# Tier 2: Haiku only
echo "Testing json_litedb_playbook with Haiku 4.5 only..."
claude -p "Pretend you have fresh context and run the playbook at playbooks/json_litedb_playbook.md" \
  --model claude-haiku-4-5-20251001 \
  --allowedTools "Bash(*),Read,Glob,Grep" 2>&1 > "results/json_litedb_haiku.log" &

wait
echo "All tests complete. Check results/ directory."
```

## Recording Results

After running tests, update the test matrix at:
- `website/docs/development/testing/playbooks/PLAYBOOK_TEST_MATRIX.md`

### Result Format

```markdown
| Playbook | Haiku 4.5 | Gemini 2.5 Pro | Notes |
|----------|-----------|----------------|-------|
| `playbook_name` | Pass/Fail (difficulty/10) | Pass/Fail (difficulty/10) | Any issues |
```

## Troubleshooting

### .NET Memory Issues in Termux

```
Error: OutOfMemoryException
```
**Solution:** Set `DOTNET_GCHeapHardLimit=1C0000000`

### Gemini Timeout

If Gemini times out on a playbook:
1. Note it as "Timeout" in the matrix
2. Move playbook to "Haiku Only" tier
3. Consider if playbook needs simplification

### Missing Predicate Errors

If a playbook fails due to missing predicates:
1. Mark as "BLOCKED" in the matrix
2. Note the specific missing predicate
3. File an issue or implement the predicate

## See Also

- [PLAYBOOK_TEST_MATRIX.md](../website/docs/development/testing/playbooks/PLAYBOOK_TEST_MATRIX.md) - Full test results
- Individual playbook files in `playbooks/` directory
