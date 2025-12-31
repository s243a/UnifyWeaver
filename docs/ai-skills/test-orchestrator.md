# Test Orchestrator Skill

This document teaches AI assistants how to act as a test orchestrator for playbook LLM testing.

## Overview

As a test orchestrator, you coordinate running playbooks across multiple LLM models to:
1. Verify playbooks execute correctly
2. Measure documentation quality (simpler models succeeding = better docs)
3. Collect difficulty ratings from models
4. Record results in the test matrix

## Prerequisites

### Required Access
- Shell access to run commands
- Read/write access to `docs/development/testing/playbooks/`
- Ability to invoke external LLM CLIs (Gemini, Claude)

### Required Files
- Playbooks in `playbooks/` directory
- Test matrix at `docs/development/testing/playbooks/PLAYBOOK_TEST_MATRIX.md`
- Test prompts at `docs/development/testing/playbooks/test_prompts.md`

## Model Tiers

Models are ranked by capability for benchmarking:

| Tier | Model | Model ID | Notes |
|------|-------|----------|-------|
| Tier 1 | Haiku 3.5 | `claude-3-5-haiku-20241022` | Basic baseline |
| Tier 2 | Haiku 4.5 | `claude-haiku-4-5-20251001` | Better reasoning |
| Tier 3 | Gemini 2.0 Flash | `gemini-2.0-flash` | Fast, capable |
| Tier 4 | Gemini 2.5 Pro | `gemini-2.5-pro` | Strong reasoning |
| Tier 5 | Claude Sonnet 4 | `claude-sonnet-4-20250514` | High capability |
| Tier 6 | Claude Opus 4.5 | `claude-opus-4-5-20251101` | Highest capability |

## Running Tests

### Method 1: Gemini CLI (Headless)

Use for Gemini models:

```bash
cd /path/to/UnifyWeaver

gemini --model gemini-2.5-pro --prompt "YOUR_PROMPT_HERE" --yolo
```

**Flags:**
- `--model`: Specify model (gemini-2.5-pro, gemini-2.0-flash, etc.)
- `--yolo`: Auto-approve all tool calls (required for headless)
- `--prompt` or `-p`: The prompt to execute

**Example:**
```bash
gemini --model gemini-2.5-pro --prompt "Pretend you have fresh context and run the playbook at playbooks/csv_data_source_playbook.md

Expected final output should include:
- \"1:Alice:30\"
- \"2:Bob:25\"
- \"3:Charlie:35\"
- \"Success: CSV source compiled and executed\"

Follow the steps exactly. After completing, rate difficulty 1-10." --yolo
```

### Method 2: Claude CLI (Headless)

Use for Claude models:

```bash
cd /path/to/UnifyWeaver

claude -p "YOUR_PROMPT_HERE" --model MODEL_ID --allowedTools "TOOLS"
```

**Flags:**
- `-p`: Print mode (headless, runs once and exits)
- `--model`: Specify model ID
- `--allowedTools`: Comma-separated list of allowed tools

**Example:**
```bash
claude -p "Pretend you have fresh context and run the playbook at playbooks/csv_data_source_playbook.md

Expected final output should include:
- \"1:Alice:30\"
- \"2:Bob:25\"
- \"3:Charlie:35\"
- \"Success: CSV source compiled and executed\"

Follow the steps exactly. After completing, rate difficulty 1-10." \
  --model claude-haiku-4-5-20251001 \
  --allowedTools "Bash(swipl:*),Bash(bash:*),Read,Glob,Grep"
```

**Note:** `--dangerously-skip-permissions` cannot be used with root/sudo. Use `--allowedTools` instead.

### Method 3: Subagent (Claude Code Internal)

When running as Claude Code, you can launch subagents:

```
Use the Task tool with:
- subagent_type: "general-purpose"
- model: "haiku" (for Haiku 4.5)
- prompt: Your playbook execution prompt
```

**Example prompt for Task tool:**
```
Run the playbook at playbooks/csv_data_source_playbook.md

Steps:
1. Read the playbook
2. Execute each step exactly as written
3. Verify output matches expected results
4. Rate difficulty 1-10 with explanation

Expected output:
- 1:Alice:30
- 2:Bob:25
- 3:Charlie:35
- Success message
```

## Standard Test Prompt Template

Use this template for all playbook tests:

```
Pretend you have fresh context and run the playbook at playbooks/<PLAYBOOK_NAME>.md

Expected final output should include:
<LIST_EXPECTED_OUTPUTS>

Follow the steps exactly as written and report any errors you encounter.
Confirm the expected output matches what you see.

After completing, rate this playbook's difficulty on a scale of 1-10, where:
- 1-3: Very clear, deterministic steps
- 4-5: Some interpretation needed
- 6-7: Requires context understanding
- 8-10: Complex reasoning required

Explain your rating.
```

## Recording Results

### 1. Update Test Matrix Table

Edit `docs/development/testing/playbooks/PLAYBOOK_TEST_MATRIX.md`:

```markdown
| `playbook_name` | ➖ | ✅ Pass (2/10) | ➖ | ✅ Pass (1/10) | Notes here |
```

Result symbols:
- ✅ **Pass** - Completed successfully on first attempt
- ⚠️ **Pass with retry** - Required 1-2 retries
- ❌ **Fail** - Could not complete
- 🔄 **Partial** - Completed some steps
- ➖ **Not tested** - Awaiting test

### 2. Add Test Results Log Entry

Add detailed entry to the Test Results Log section:

```markdown
### YYYY-MM-DD - playbook_name - Model Name

**Result**: ✅ Pass (first attempt)

**Execution**:
- Model followed all playbook steps correctly
- [Any notable observations]

**Output**:
```
[Relevant output here]
```

**Difficulty Rating**: X/10

**Reasoning from Model**:
> [Quote the model's explanation]

---
```

### 3. Update Difficulty Ratings Table

If model provided a rating:

```markdown
| `playbook_name` | X/10 | Model reasoning summary |
```

## Workflow Example

Here's a complete orchestration workflow:

### 1. Select Playbook and Models

```
Playbook: csv_data_source_playbook
Models to test: Haiku 4.5, Gemini 2.5 Pro
```

### 2. Run Tests

```bash
# Test with Haiku 4.5
claude -p "..." --model claude-haiku-4-5-20251001 --allowedTools "..."

# Test with Gemini 2.5 Pro
gemini --model gemini-2.5-pro --prompt "..." --yolo
```

### 3. Collect Results

| Model | Result | Difficulty |
|-------|--------|------------|
| Haiku 4.5 | ✅ Pass | 2/10 |
| Gemini 2.5 Pro | ✅ Pass | 1/10 |

### 4. Update Documentation

- Update test matrix table
- Add log entries for each test
- Update difficulty ratings
- Commit changes

### 5. Analyze Results

- If Tier 1-2 passes: Documentation is highly deterministic
- If only Tier 3+ passes: May need to simplify instructions
- If only Tier 5+ passes: Task genuinely requires intelligence

## Troubleshooting

### Gemini can't read files in tmp/

Gemini may have gitignore restrictions. Workaround:
```bash
# Use cat via shell instead of file read
cat tmp/users.sh
```

### Claude permission errors

If `--dangerously-skip-permissions` fails (root user):
```bash
# Use explicit tool allowlist instead
--allowedTools "Bash(swipl:*),Bash(bash:*),Read,Glob,Grep"
```

### Model finds a bug

This is valuable! Record:
1. What bug was found
2. Where in the code
3. How to reproduce
4. Create a fix task

Then re-test after fixing for fair comparison.

### Test output doesn't match expected

1. Check if playbook instructions are ambiguous
2. Check if underlying code has bugs
3. Check if expected output in playbook is wrong
4. Update playbook or code as needed

## Best Practices

1. **Always use fresh context** - Start each test with "Pretend you have fresh context"

2. **Test lower tiers first** - If Haiku passes, higher tiers likely will too

3. **Get difficulty ratings** - These help calibrate documentation quality

4. **Re-test after fixes** - Always re-run tests after fixing bugs for fair comparison

5. **Document everything** - Future orchestrators need the history

6. **Sync to website** - Copy updated docs to `website/docs/` directory

## Related Files

- `playbooks/` - All playbook files
- `docs/development/testing/playbooks/PLAYBOOK_TEST_MATRIX.md` - Test results
- `docs/development/testing/playbooks/test_prompts.md` - Playbook-specific prompts
- `docs/development/playbooks/` - Playbook writing guides
