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
| `csv_data_source_playbook` | ➖ | ➖ | ➖ | ➖ | Basic extraction |
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
| `prolog_generation_playbook` | ➖ | ➖ | ➖ | ➖ | Prolog dialects |
| `powershell_inline_dotnet_playbook` | ➖ | ➖ | ➖ | ➖ | Inline .NET |

## Difficulty Ratings (From Advanced Models)

When a Tier 5+ model runs a playbook, it should provide a difficulty rating:

| Playbook | Difficulty (1-10) | Reasoning |
|----------|-------------------|-----------|
| `csv_data_source_playbook` | ➖ | |
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

1. [ ] Run `csv_data_source_playbook` with Haiku 4.5
2. [ ] Run `csv_data_source_playbook` with Gemini 2.5 Pro (get difficulty rating)
3. [ ] Document results and iterate on playbook if needed
4. [ ] Expand to other playbooks

## Related Documentation

- [Playbook Philosophy](../../../playbooks/../docs/development/playbooks/philosophy.md)
- [Playbook Specification](../../../playbooks/../docs/development/playbooks/specification.md)
- [Best Practices](../../../playbooks/../docs/development/playbooks/best_practices.md)
