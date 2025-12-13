# Playbook Test Results

**Test Date**: 2025-12-10 to 2025-12-12
**Test Method**: Automated execution with difficulty rating by AI agents (Haiku 4.5, Gemini 2.5 Pro)

## Summary

| Playbook | Haiku Rating | Gemini Rating | Status |
|----------|-------------|---------------|--------|
| sqlite_source | 2-5/10 | - | PASS |
| sql_window | 3-4/10 | - | PASS |
| bash_parallel | 3-6/10 | - | PASS |
| csharp_generator | 3/10 | 4/10 | PASS |
| powershell_inline_dotnet | 4/10 | 8/10 | PASS / RATE LIMITED |
| awk_advanced | 4-5/10 | - | PASS |
| json_litedb | 7/10 | - | PASS |
| csharp_query | 8/10 | - | PASS (verified) |
| **csv_data_source** | **2/10** | **1/10** | **PASS** |
| **xml_data_source** | **2/10** | **1/10** | **PASS** |
| **csharp_codegen** | **2/10** | **1/10** | **PASS** |
| **tree_recursion** | **2/10** | **1/10** | **PASS** |
| **mutual_recursion** | **2/10** | **1/10** | **PASS** |
| **parallel_execution** | **4/10** | **1/10** | **PASS** |
| **prolog_generation** | **3/10** | **1/10** | **PASS** |
| **large_xml_streaming** | **2/10** | **-** | **PASS** |
| **powershell_binding** | **3/10** | **-** | **PASS** |
| **template_system** | **3/10** | **-** | **PASS** |
| **http_source** | **3/10** | **-** | **PASS** |
| **csharp_xml_fragments** | **2/10** | **-** | **BLOCKED** |
| **cross_target_glue** | **6/10** | **-** | **PARTIAL** |

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
- Complex multi-system integration (Prolog -> C# -> PowerShell -> LiteDB)
- LiteDB installation has interactive prompts
- Sonnet handled interactive setup more easily
- Model capability affects perceived difficulty

### csharp_query_playbook.md

**Rating**: 8/10
**Status**: PASS (verified manually)

Key findings:
- Haiku initially reported missing `build_unifyweaver_project/0` but this was an error
- Playbook correctly uses `compile_predicate_to_csharp/3` (verified in source)
- Manual verification confirmed the bash example works correctly
- High difficulty rating due to multi-step C# compilation workflow
- Haiku consistently misreports this predicate across multiple runs (hallucination)

## 2025-12-12 Gemini Batch Testing

**Model**: Gemini 2.5 Pro
**Playbooks Tested**: 7
**Method**: Manual testing via Gemini CLI

### Results Summary

- **7/7 Passed**: All playbooks now pass (bugs fixed in commit b04fe8d)
- **All rated 1/10 difficulty**: Extremely clear, deterministic instructions
- **Average time**: ~1 minute per playbook
- **Bug fixes verified**: Both Gemini 2.5 Pro and Haiku 4.5 retests confirm fixes work

### Key Findings

1. **Documentation Quality Validated**
   - All 7 playbooks rated 1/10 by Gemini 2.5 Pro
   - Confirms playbooks are highly deterministic
   - Instructions described as "precise" with "expected output"

2. **Bugs Discovered and Fixed** (commit b04fe8d):
   - **tree_recursion_playbook**: Missing auto-execute block → Fixed, now outputs `30`
   - **mutual_recursion_playbook**: Wrong group passed to compiler → Fixed, both functions work
   - Both bugs were in code generation, not playbook documentation
   - Fixes verified with Gemini 2.5 Pro and Haiku 4.5 retests

3. **Cross-Model Validation**
   - Gemini consistently rates playbooks lower difficulty than Haiku
   - Confirms that documentation works across vendors
   - Average difficulty drop: Haiku 2-4/10 → Gemini 1/10

### Detailed Gemini Results

**csv_data_source_playbook**: ✅ Pass (1/10)
- Output: `1:Alice:30`, `2:Bob:25`, `3:Charlie:35`
- Time: ~1 minute

**xml_data_source_playbook**: ✅ Pass (1/10)
- Output: `Total price: 1300`
- Time: ~1 minute

**csharp_codegen_playbook**: ✅ Pass (1/10)
- Output: `anne:charles`, `anne:diana`
- Time: ~2 minutes

**tree_recursion_playbook**: ✅ Pass (1/10)
- Compilation: ✓ Succeeded
- Execution: Outputs `30` correctly ✅ (Fixed in b04fe8d)
- Time: ~1 minute

**mutual_recursion_playbook**: ✅ Pass (1/10)
- Compilation: ✓ Succeeded
- Execution: Both `is_even(4)` and `is_odd(3)` work ✅ (Fixed in b04fe8d)
- Time: ~1 minute

**parallel_execution_playbook**: ✅ Pass (1/10)
- Output: `SUCCESS: Final sum is 500500`
- Time: ~1 minute

**prolog_generation_playbook**: ✅ Pass (1/10)
- Output: `5:120`
- Time: ~1 minute

## 2025-12-12 Additional Playbook Testing

**Model**: Haiku 4.5
**Playbooks Tested**: 6 new playbooks
**Method**: Automated testing with Haiku agents

### Results Summary

- **4/6 Passed**: large_xml_streaming, powershell_binding, template_system, http_source
- **1/6 Blocked**: csharp_xml_fragments (incomplete placeholder)
- **1/6 Partial**: cross_target_glue (bugs in 2 of 4 phases)
- **Average difficulty**: 2.8/10 (excluding blocked playbook)

### Detailed Results

**large_xml_streaming_playbook**: ✅ Pass (2/10)
- All 4 examples executed successfully
- Extracted 2 trees, 0 pearls from sample.rdf
- Memory-efficient streaming pipeline validated
- Fixed path references from non-existent context file to tests/test_data/sample.rdf (commit 2abbc49)

**powershell_binding_playbook**: ✅ Pass (3/10)
- All 5 examples executed successfully
- Generated PowerShell bindings for Math, Cmdlets, File Operations, Pipeline Operations
- Test suite verified 68 PowerShell bindings
- Clear copy-paste instructions with exact expected outputs

**template_system_playbook**: ✅ Pass (3/10)
- 5 of 6 examples passed
- Basic template rendering, function generation, caching, file loading all worked
- Test suite: 7/7 internal tests passed
- 1 example failed due to missing optional constraint_analyzer module (expected)

**http_source_playbook**: ✅ Pass (3/10)
- All 5 examples executed successfully
- Generated HTTP GET/POST scripts with caching
- Verified wget-based and curl-based implementations
- Cache management functions validated

**csharp_xml_fragments_playbook**: ❌ Blocked (2/10)
- Incomplete placeholder playbook
- Contains only Prolog configuration declarations (no executable steps)
- Missing compile_dynamic_source/3 predicate
- Documented as "Needs work" in testing procedure
- Rated 2/10 because the incompleteness is immediately obvious (not ambiguous)

**cross_target_glue_playbook**: ⚠️ Partial (6/10)
- Phase 1 (Shell/AWK): Failed - unclosed braces in generated AWK code
- Phase 2 (Go): ✅ Success - JSON processing pipeline worked correctly
- Phase 3 (.NET): Failed - references non-existent predicate generate_csharp_pipeline_class/3
- Phase 4 (Rust): ✅ Success - TSV aggregation pipeline worked correctly
- Extraction tool format mismatch (uses ::: notation instead of standard metadata)
- Rated 6/10 due to architectural knowledge required and untested examples

## Issues Identified

### 1. Extract Tool Query Matching

The `extract_records.pl` tool uses regex matching, which can cause:
- Partial matches returning multiple records
- `csharp_sum_pair` matches both bash and PowerShell examples
- Fix: Use exact match with `^query$` anchors

### 2. AI Agent Misreporting

- Haiku consistently reports `build_unifyweaver_project/0` as missing from csharp_query_playbook
- Manual verification showed this was incorrect - the playbook uses `compile_predicate_to_csharp/3`
- Agent hallucination or misinterpretation of playbook instructions
- This occurred across multiple independent test runs

### 3. Interactive Setup

- Some playbooks require handling interactive prompts (e.g., LiteDB setup)
- More capable models handle this better

### 4. Code Generation Bugs (Found 2025-12-12)

**tree_recursion_playbook**:
- Playbook documentation: ✅ Clear (1/10 difficulty)
- Compilation: ✅ Succeeds
- Execution: ❌ Produces no output for test case `[10,[5,[],[]],[15,[],[]]]`
- Expected output: `...:30`
- Issue: Bug in bash code generator for tree recursion

**mutual_recursion_playbook**:
- Playbook documentation: ✅ Clear (1/10 difficulty)
- Compilation: ✅ Succeeds
- Execution: ❌ Fails with "Unknown function: 4" and "Unknown function: 3"
- Issue: Bug in bash code generator for mutual recursion

## Recommendations

1. **Add exact match examples** to playbooks that use `extract_records.pl`
2. **Add execution records** to csharp_generator_playbook (currently incomplete)
3. **Document interactive setup handling** for playbooks with dependencies
4. **Consider agent validation** - manual verification may be needed for high-difficulty ratings
5. ~~**Fix tree_recursion bash generator**~~ → **COMPLETE** (commit b04fe8d)
6. ~~**Fix mutual_recursion bash generator**~~ → **COMPLETE** (commit b04fe8d)
7. ~~**Fix large_xml_streaming_playbook path references**~~ → **COMPLETE** (commit 2abbc49)
8. **Complete csharp_xml_fragments_playbook** - Add executable examples and compile_dynamic_source/3 calls
9. **Fix cross_target_glue_playbook Phase 1** - AWK code generation produces unclosed braces
10. **Fix cross_target_glue_playbook Phase 3** - Update to use correct .NET pipeline predicate name
11. **Standardize extraction tool format** - cross_target_glue uses ::: notation instead of standard metadata

## Test Environment

- Platform: Linux (PRoot-Distro)
- SWI-Prolog: Available
- .NET SDK: 9.0 (net9.0)
- Claude Models: Haiku 4.5, Sonnet 4
- Gemini Models: 2.5 Pro (quota limited), 2.5 Flash

## Notes

- **2025-12-12 Update - Morning**: Gemini 2.5 Pro successfully tested 7 playbooks (all pass after bug fixes)
- **2025-12-12 Update - Evening**: Haiku 4.5 tested 6 additional playbooks (4 pass, 1 blocked, 1 partial)
- Initial Gemini tests (2025-12-10) failed due to API quota exhaustion - resolved by manual testing
- Some tests ran multiple times for consistency verification
- Model capability significantly affects perceived playbook difficulty
- Gemini consistently rates playbooks as easier (1/10) compared to Haiku (2-4/10)
- Cross-vendor testing validates documentation quality
- **Total playbooks tested**: 19 (15 pass, 1 blocked, 1 partial, 2 rate-limited)
- **Bugs found and fixed**: 3 (tree_recursion, mutual_recursion, large_xml_streaming)
- **New issues identified**: 3 (csharp_xml_fragments incomplete, cross_target_glue phases 1&3 broken, extraction tool format mismatch)
