# Playbook LLM Test Matrix

This document tracks which LLMs can successfully execute each playbook, serving as both:
1. **Automated test coverage** - Verify playbooks work correctly
2. **Documentation quality metric** - If simpler models succeed, our docs are highly deterministic
3. **Capability benchmarking** - Understand what level of intelligence each task requires

## Test Models (Ordered by Capability)

| Tier | Model | Notes |
|------|-------|-------|
| **Tier 1 (Basic)** | Haiku 3.5 | Fast, cheap, good baseline |
| **Tier 2 (Standard)** | Haiku 4.5 | Better reasoning, still fast |
| **Tier 3 (Advanced)** | Gemini 2.0 Flash | Good balance of speed/capability |
| **Tier 4 (Strong)** | Gemini 2.5 Pro | Strong reasoning |
| **Tier 5 (Frontier)** | Claude Sonnet 4 | High capability |
| **Tier 6 (Expert)** | Claude Opus 4.5 | Highest capability |

## Test Matrix

### Legend
- ✅ **Pass** - Completed successfully on first attempt
- ⚠️ **Pass with retry** - Required 1-2 retries
- ❌ **Fail** - Could not complete
- 🔄 **Partial** - Completed some steps
- ➖ **Not tested** - Awaiting test
- 📝 **Score: X/10** - Difficulty rating from advanced model

### Data Source Playbooks

| Playbook | Haiku 3.5 | Haiku 4.5 | Gemini 2.0 Flash | Gemini 2.5 Pro | Notes |
|----------|-----------|-----------|------------------|----------------|-------|
| `csv_data_source_playbook` | ➖ | ✅ Pass (2/10) | ➖ | ✅ Pass (1/10) | Both pass after bug fix |
| `xml_data_source_playbook` | ➖ | ➖ | ➖ | ➖ | Python XML parsing |
| `json_litedb_playbook` | ➖ | ➖ | ➖ | ➖ | .NET + LiteDB |
| `large_xml_streaming_playbook` | ➖ | ➖ | ➖ | ➖ | Multi-stage pipeline |

### C# Compilation Playbooks

| Playbook | Haiku 3.5 | Haiku 4.5 | Gemini 2.0 Flash | Gemini 2.5 Pro | Notes |
|----------|-----------|-----------|------------------|----------------|-------|
| `csharp_codegen_playbook` | ➖ | ➖ | ➖ | ➖ | Non-recursive |
| `csharp_query_playbook` | ➖ | ➖ | ➖ | ➖ | Recursive (Fibonacci) |
| `csharp_xml_fragments_playbook` | ➖ | ➖ | ➖ | ➖ | XML streaming |

### Recursion Playbooks

| Playbook | Haiku 3.5 | Haiku 4.5 | Gemini 2.0 Flash | Gemini 2.5 Pro | Notes |
|----------|-----------|-----------|------------------|----------------|-------|
| `tree_recursion_playbook` | ➖ | ➖ | ➖ | ➖ | Tree traversal |
| `mutual_recursion_playbook` | ➖ | ➖ | ➖ | ➖ | Even/odd recursion |

### Execution Playbooks

| Playbook | Haiku 3.5 | Haiku 4.5 | Gemini 2.0 Flash | Gemini 2.5 Pro | Notes |
|----------|-----------|-----------|------------------|----------------|-------|
| `parallel_execution_playbook` | ➖ | ➖ | ➖ | ➖ | Parallel processing |
| `prolog_generation_playbook` | ➖ | 🔄 Partial (6/10) | ➖ | 🔄 Partial (7/10) | Bug: no auto-execute in generated script |
| `powershell_inline_dotnet_playbook` | ➖ | ➖ | ➖ | ➖ | Inline .NET |

## Difficulty Ratings (From Advanced Models)

When a Tier 5+ model runs a playbook, it should provide a difficulty rating:

| Playbook | Difficulty (1-10) | Reasoning |
|----------|-------------------|-----------|
| `csv_data_source_playbook` | 1-2/10 | Gemini 2.5 Pro: 1/10, Haiku 4.5: 2/10 - purely mechanical steps |
| `prolog_generation_playbook` | 6-7/10 | Haiku 4.5: 6/10, Gemini 2.5 Pro: 7/10 - required debugging missing output |
| `xml_data_source_playbook` | ➖ | |
| `csharp_codegen_playbook` | ➖ | |
| ... | | |

### Rating Criteria
- **1-3**: Very straightforward, clear step-by-step instructions
- **4-5**: Some interpretation needed, but well-documented
- **6-7**: Requires understanding context, may need troubleshooting
- **8-9**: Complex, requires significant reasoning
- **10**: Requires expert knowledge or creative problem-solving

## Test Protocol

### Running a Playbook Test

1. **Fresh context**: Start with no prior conversation history
2. **Standard prompt**:
   ```
   Pretend you have fresh context and run the playbook at playbooks/<playbook_name>.md
   ```
3. **Record results**:
   - Did it complete successfully?
   - How many retries needed?
   - What errors occurred?
   - Time to completion (if available)

### For Advanced Models (Tier 5+)

After completing a playbook, ask:
```
Rate this playbook's difficulty on a scale of 1-10, where:
- 1-3: Very clear, deterministic steps
- 4-5: Some interpretation needed
- 6-7: Requires context understanding
- 8-10: Complex reasoning required

Explain your rating and suggest improvements to make it easier to follow.
```

## Test Results Log

### 2025-12-08 - csv_data_source_playbook - Gemini 2.5 Pro

**Result**: ⚠️ Pass with fix

**Execution**:
- Model followed playbook steps correctly
- Extracted and ran the bash script
- Output was missing "Alice" record
- Model investigated and found bug in generated `tmp/users.sh`
- Bug: In lookup mode, `NR > 2` should be `NR > 1` (skips first data row)
- Model fixed the bug and verified correct output

**Output after fix**:
```
1:Alice:30
2:Bob:25
3:Charlie:35
```

**Bug Found**: Real bug in `csv_source.pl` - lookup mode uses `NR > 2` instead of `NR > 1`
- Streaming mode (no key): `NR > 1` ✅ correct
- Lookup mode (with key): `NR > 2` ❌ wrong - skips first data row

**Difficulty Rating**: 7/10
- Easy: Clear goal, good project structure
- Difficult: Required debugging when output was wrong, no explicit guidance on troubleshooting

**Observations**:
- Gemini couldn't read files in `tmp/` due to gitignore patterns
- Worked around by using `cat` via shell command
- This test discovered a real bug that should be fixed

**Action Items**:
- [x] Fix `NR > 2` bug in `src/unifyweaver/sources/csv_source.pl` (fixed by Claude Opus 4.5)
- [x] Re-test after fix to verify clean pass

---

### 2025-12-08 - csv_data_source_playbook - Bug Fix Verification (Claude Opus 4.5)

**Result**: ✅ Pass (after fix)

**Root Cause Analysis**:
- Bug was in `sources.pl` + `csv_source.pl` interaction
- `sources.pl:108-111` already adds `skip_lines(1)` when `has_header(true)` is set
- `csv_source.pl:168-171` was ALSO adding 1 when `HeaderMode = auto`
- Result: Double-counting → `TotalSkip = 2` instead of `1`

**Fix Applied**:
- Modified `csv_source.pl:generate_csv_bash/11` to NOT add 1 for headers
- Changed `TotalSkip is SkipLines + 1` to just `TotalSkip = SkipLines`
- Comment added explaining that `sources.pl` already handles header skip

**Verification Output**:
```
=== All users ===
1:Alice:30
2:Bob:25
3:Charlie:35

=== Lookup user 1 ===
1:Alice:30

Success: CSV source compiled and executed
```

**Both streaming and lookup modes now correctly use `NR > 1`**

---

### 2025-12-08 - csv_data_source_playbook - Haiku 4.5

**Result**: ✅ Pass (first attempt)

**Execution**:
- Model followed all playbook steps correctly
- Extracted and ran the bash script successfully
- All expected output matched perfectly

**Output**:
```
Creating Prolog script...
Compiling CSV source to bash...
Registered source type: csv -> csv_source
Compiling dynamic source: users/3 using csv
  Compiling CSV source: users/3
Compiled CSV source to tmp/users.sh
To use: source tmp/users.sh && users

Testing generated bash script...
Loading users function...

Calling users() to get all records:
1:Alice:30
2:Bob:25
3:Charlie:35

Success: CSV source compiled and executed
```

**Difficulty Rating**: 2/10

**Reasoning from Haiku 4.5**:
> This playbook scores very low in difficulty (2/10) for the following reasons:
> - **Deterministic and sequential**: The playbook provides 4 clear, numbered steps to follow in exact order
> - **Unambiguous instructions**: Each step has a specific bash command that requires no interpretation
> - **Explicit output expectations**: The expected output section is detailed and specific, making validation trivial
> - **Error prevention**: The "Common Mistakes to Avoid" section clearly warns against a wrong approach
> - **No decision-making required**: There's no branching logic or conditional steps—just a straight path forward

**Key Insight**: Haiku 4.5 (Tier 2 model) passing on first attempt with a 2/10 difficulty rating indicates **highly deterministic documentation**. This validates the playbook's quality for automated LLM testing.

---

### 2025-12-08 - csv_data_source_playbook - Gemini 2.5 Pro (Re-test after fix)

**Result**: ✅ Pass (first attempt)

**Execution**:
- Model followed all playbook steps correctly
- Extracted and ran the bash script successfully
- All expected output matched perfectly

**Output confirmed**:
```
1:Alice:30
2:Bob:25
3:Charlie:35
Success: CSV source compiled and executed
```

**Difficulty Rating**: 1/10

**Reasoning from Gemini 2.5 Pro**:
> The playbook's instructions were exceptionally clear and deterministic. It provided the exact shell commands to execute in a precise sequence. There was no need for interpretation, context-specific knowledge, or complex reasoning. The steps were purely mechanical, leading directly to the expected outcome.

**Comparison with pre-fix run**: Previous rating was 7/10 because model had to debug the missing "Alice" bug. After fix, rating dropped to 1/10 - demonstrating that playbook difficulty is heavily influenced by whether the underlying code works correctly.

---

### 2025-12-08 - prolog_generation_playbook - Haiku 4.5

**Result**: 🔄 Partial

**Execution**:
- Model followed all playbook steps correctly
- Extracted and ran the bash script successfully
- Script ran but `5:120` output was missing
- Model diagnosed the issue: generated `factorial.sh` has no auto-execute block

**Output**:
```
Generating Prolog code for factorial...
✓ Generated Prolog code: tmp/factorial.pl
...
Generated scripts: [tmp/factorial.sh]

Testing generated factorial script...
Running factorial(5):
                      <-- Missing 5:120 here
Success: Factorial compiled and executed correctly
```

**Bug Found**: Generated `tmp/factorial.sh` defines `factorial` function but doesn't invoke it when run directly. The script lacks the auto-execute pattern:
```bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    factorial_stream "$@"
fi
```

**Difficulty Rating**: 6/10

**Reasoning from Haiku 4.5**:
> The playbook has explicit step-by-step instructions that are easy to follow. However, executing it successfully requires understanding the context and purpose of each step, not just mechanically following commands. The non-deterministic aspects (missing output) required understanding what went wrong.

---

### 2025-12-08 - prolog_generation_playbook - Gemini 2.5 Pro

**Result**: 🔄 Partial (manually fixed to complete)

**Execution**:
- Model followed all playbook steps correctly
- Identified same bug as Haiku 4.5
- Attempted to patch `factorial.sh` but script was regenerated
- Manually re-ran compilation steps and patched script to demonstrate fix

**Bug Found**: Same as Haiku 4.5 - missing auto-execute block in generated script.

**Difficulty Rating**: 7/10

**Reasoning from Gemini 2.5 Pro**:
> The playbook's instructions were clear, but a successful execution required significant debugging and context-awareness beyond what was written. The underlying `compiler_driver` tool produced a non-functional script, and I had to diagnose this by inspecting multiple generated files, understand the shell execution flow, and devise a manual, multi-step workaround.

**Action Items**:
- [ ] Fix recursive compiler to add auto-execute block to generated bash scripts
- [ ] Re-test after fix to verify clean pass

---

### Template for Recording Results

```markdown
### [Date] - [Playbook] - [Model]

**Result**: ✅ Pass / ⚠️ Retry / ❌ Fail / 🔄 Partial

**Attempt 1**:
- Started: [timestamp]
- Completed: [timestamp]
- Errors: [none / description]

**Notes**:
[Any observations about model behavior, documentation issues, etc.]

**Difficulty Rating** (if Tier 5+): X/10
**Suggested Improvements**: [from model feedback]
```

---

## Analysis Goals

### Documentation Quality
- If **Tier 1-2 models pass**: Documentation is highly deterministic ✅
- If **only Tier 3+ passes**: May need to simplify instructions
- If **only Tier 5+ passes**: Task genuinely requires intelligence

### Test Coverage Priorities

1. **High Priority** (test with multiple models):
   - `csv_data_source_playbook` - Core functionality
   - `csharp_codegen_playbook` - C# compilation
   - `tree_recursion_playbook` - Recursion handling

2. **Medium Priority**:
   - `json_litedb_playbook` - .NET integration
   - `parallel_execution_playbook` - Advanced features

3. **Lower Priority** (test with capable models first):
   - `csharp_xml_fragments_playbook` - Specialized
   - `large_xml_streaming_playbook` - Complex pipeline

## Next Steps

1. [x] Run `csv_data_source_playbook` with Haiku 4.5 - ✅ Pass (2/10)
2. [x] Run `csv_data_source_playbook` with Gemini 2.5 Pro - ⚠️ Pass+fix (found bug)
3. [x] Fix bug and verify - ✅ Fixed by Claude Opus 4.5
4. [ ] Test `csharp_codegen_playbook` with multiple models
5. [ ] Expand to other playbooks

## Related Documentation

- [Playbook Philosophy](../../../playbooks/../docs/development/playbooks/philosophy.md)
- [Playbook Specification](../../../playbooks/../docs/development/playbooks/specification.md)
- [Best Practices](../../../playbooks/../docs/development/playbooks/best_practices.md)
