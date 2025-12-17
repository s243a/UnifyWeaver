# Playbook: AWK Foreign Function Binding

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to call AWK scripts as foreign functions, binding existing AWK tools to Prolog predicates.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "awk_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use awk source"


## Workflow Overview
Use UnifyWeaver to bind AWK as a foreign function:
1. Declare a data source using the `awk_source` plugin with `awk_command` or `awk_file`
2. UnifyWeaver compiles this to bash that executes AWK and captures output
3. Call the generated function to retrieve AWK-processed data as Prolog facts

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `awk_command_basic`, `awk_file_basic`, `awk_separator`, `awk_aggregate` in `playbooks/examples_library/awk_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/awk_source.pl` implements the plugin
3. **Extraction Tool** – `scripts/extract_records.pl` for extracting examples

## Execution Guidance

**IMPORTANT**: The records in [1] contain **BASH SCRIPTS**, not Prolog code. Extract and run with `bash`, not `swipl`.

### Example 1: Basic AWK Command

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script**
```bash
perl scripts/extract_records.pl playbooks/examples_library/awk_source_examples.md \
    awk_command_basic > tmp/awk_basic.sh
```

**Step 3: Make executable and run**
```bash
chmod +x tmp/awk_basic.sh
bash tmp/awk_basic.sh
```

**Expected Output:**
```
Compiling AWK source: count_lines/1
Generated: tmp/count_lines.sh
Testing count_lines/1:
5
Success: AWK command binding works
```

### Example 2: AWK File Binding

Extract and run `awk_file_basic` query to bind an AWK script file.

### Example 3: Custom Field Separator

Extract and run `awk_separator` query for CSV parsing with custom delimiter.

### Example 4: AWK Aggregation

Extract and run `awk_aggregate` query for sum/grouping operations.

## Expected Outcome
- AWK commands/files compile to bash functions
- Generated functions execute AWK and return results
- Exit code 0 with "Success" message
- Data available as Prolog facts for further processing

## Citations
[1] playbooks/examples_library/awk_source_examples.md
[2] src/unifyweaver/sources/awk_source.pl
[3] scripts/extract_records.pl
