# Playbook Test Prompts

Standard prompts for testing playbooks with different LLMs.

## Basic Execution Prompt

Use this for all models:

```
Pretend you have fresh context and run the playbook at playbooks/<PLAYBOOK_NAME>.md

Important:
- Follow the steps exactly as written
- Report any errors you encounter
- Confirm the expected output matches what you see
```

## Playbook-Specific Prompts

### csv_data_source_playbook
```
Pretend you have fresh context and run the playbook at playbooks/csv_data_source_playbook.md

Expected final output should include:
- "1 Alice 30" or "1:Alice:30"
- "2 Bob 25" or "2:Bob:25"
- "3 Charlie 35" or "3:Charlie:35"
- "Success: CSV source compiled and executed"
```

### xml_data_source_playbook
```
Pretend you have fresh context and run the playbook at playbooks/xml_data_source_playbook.md

Expected final output should include:
- "Total price: 1300"
```

### csharp_codegen_playbook
```
Pretend you have fresh context and run the playbook at playbooks/csharp_codegen_playbook.md

Expected final output should include:
- "anne:charles"
- "anne:diana"
- "Success: C# program compiled and executed successfully"
```

### csharp_query_playbook
```
Pretend you have fresh context and run the playbook at playbooks/csharp_query_playbook.md

Expected final output should include:
- "8: 21" (8th Fibonacci number)
- "Success: C# program compiled and executed successfully"
```

### tree_recursion_playbook
```
Pretend you have fresh context and run the playbook at playbooks/tree_recursion_playbook.md
```

### mutual_recursion_playbook
```
Pretend you have fresh context and run the playbook at playbooks/mutual_recursion_playbook.md
```

### parallel_execution_playbook
```
Pretend you have fresh context and run the playbook at playbooks/parallel_execution_playbook.md
```

### prolog_generation_playbook
```
Pretend you have fresh context and run the playbook at playbooks/prolog_generation_playbook.md
```

### json_litedb_playbook
```
Pretend you have fresh context and run the playbook at playbooks/json_litedb_playbook.md

Expected final output should include:
- "4 products loaded"
- "Widget Pro" and "Gadget X" in query results
- "Success: JSON data streamed into LiteDB"
```

### large_xml_streaming_playbook
```
Pretend you have fresh context and run the playbook at playbooks/large_xml_streaming_playbook.md

Expected output should include Prolog facts like:
- tree(2492215, 'Hacktivism', ...)
- pearl(root, ...)
```

### powershell_inline_dotnet_playbook
```
Pretend you have fresh context and run the playbook at playbooks/powershell_inline_dotnet_playbook.md

Expected output: reversed string (e.g., "Hello World" → "dlroW olleH")
```

### csharp_xml_fragments_playbook
```
Pretend you have fresh context and run the playbook at playbooks/csharp_xml_fragments_playbook.md
```

## Difficulty Rating Prompt (For Tier 5+ Models)

After the model completes a playbook, use this follow-up:

```
Now that you've completed the playbook, please rate its difficulty:

Scale:
- 1-3: Very clear, deterministic steps - a simple model could follow this
- 4-5: Some interpretation needed, but documentation is good
- 6-7: Requires understanding context, may need troubleshooting knowledge
- 8-9: Complex, requires significant reasoning or domain expertise
- 10: Requires expert knowledge or creative problem-solving

Please provide:
1. Your rating (1-10)
2. What made it easy or difficult
3. Specific suggestions to improve the playbook for less capable models
4. Any ambiguous instructions you encountered
```

## Recording Results Template

After each test, record:

```markdown
## Test: [playbook_name] with [model_name]
**Date**: YYYY-MM-DD
**Result**: ✅ Pass / ⚠️ Pass with retry / ❌ Fail / 🔄 Partial

### Execution Log
- Step 1: [outcome]
- Step 2: [outcome]
- ...

### Errors Encountered
[List any errors or unexpected behavior]

### Output Verification
- Expected: [expected output]
- Actual: [actual output]
- Match: Yes/No

### Difficulty Rating (if applicable)
Rating: X/10
Reasoning: [model's explanation]

### Suggestions for Improvement
[From model feedback or observations]
```

## Batch Testing Script Concept

For automated testing (future implementation):

```bash
#!/bin/bash
# test_playbook.sh - Run a playbook with a specific model

PLAYBOOK=$1
MODEL=$2  # e.g., "haiku-4.5", "gemini-2.5-pro"

# Load the appropriate prompt
PROMPT=$(cat docs/development/testing/playbooks/test_prompts.md | \
         grep -A5 "### ${PLAYBOOK}" | tail -n +2)

# Run with the model (pseudocode - actual implementation depends on API)
# claude/gemini API call with $PROMPT

# Capture output and verify against expected results
# Log results to PLAYBOOK_TEST_MATRIX.md
```

## CI/CD Integration (Future)

```yaml
# .github/workflows/playbook-tests.yml
name: Playbook LLM Tests

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  test-playbooks:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        playbook:
          - csv_data_source_playbook
          - csharp_codegen_playbook
          - tree_recursion_playbook
        model:
          - haiku-4.5
          - gemini-2.5-pro
    steps:
      - uses: actions/checkout@v4
      - name: Run playbook test
        run: ./scripts/test_playbook.sh ${{ matrix.playbook }} ${{ matrix.model }}
      - name: Update test matrix
        run: ./scripts/update_test_matrix.sh
```
