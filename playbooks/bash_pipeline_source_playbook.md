# Playbook: Bash Pipeline Composer

## Audience
This playbook is a high-level guide for coding agents. It demonstrates composing multi-stage bash pipelines as UnifyWeaver data sources.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "bash_pipeline_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use bash pipeline source"


## Workflow Overview
Use UnifyWeaver to compose bash pipelines:
1. Declare a data source using the `bash_pipeline` plugin with pipeline stages
2. Each stage specifies tool (grep, awk, sort, etc.), script, and arguments
3. UnifyWeaver chains stages into a unified pipeline

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `pipeline_basic`, `pipeline_complex`, `pipeline_aggregate` in `playbooks/examples_library/bash_pipeline_examples.md`
2. **Source Module** – `src/unifyweaver/sources/bash_pipeline_source.pl`
3. **Extraction Tool** – `scripts/extract_records.pl`

## Execution Guidance

**IMPORTANT**: Records contain **BASH SCRIPTS**. Extract and run with `bash`, not `swipl`.

### Example 1: Grep + AWK Pipeline

**Step 1: Navigate and extract**
```bash
cd /path/to/UnifyWeaver
perl scripts/extract_records.pl playbooks/examples_library/bash_pipeline_examples.md \
    pipeline_basic > tmp/pipeline_basic.sh
chmod +x tmp/pipeline_basic.sh
bash tmp/pipeline_basic.sh
```

**Expected Output:**
```
Compiling bash pipeline: find_errors/1
Generated: tmp/find_errors.sh
Testing find_errors/1:
ERROR:Connection failed
ERROR:Timeout occurred
Success: Grep + AWK pipeline works
```

### Example 2: Multi-Stage Aggregation

Extract and run `pipeline_complex` query for AWK aggregation with sort.

### Example 3: Sort + Uniq

Extract and run `pipeline_aggregate` query for deduplication pipeline.

## Expected Outcome
- Pipeline stages execute in sequence
- Data flows correctly through all stages
- Final output matches expected results
- Exit code 0 with "Success" message

## Citations
[1] playbooks/examples_library/bash_pipeline_examples.md
[2] src/unifyweaver/sources/bash_pipeline_source.pl
[3] scripts/extract_records.pl
