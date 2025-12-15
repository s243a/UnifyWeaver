# Playbook: JSON Data Source (jq)

## Audience
This playbook is a high-level guide for coding agents. It demonstrates JSON processing using jq through UnifyWeaver's json_source plugin.

## Workflow Overview
Use UnifyWeaver to process JSON with jq:
1. Declare a data source using the `json_source` plugin with jq filter
2. UnifyWeaver generates bash that pipes JSON through jq
3. Execute to retrieve filtered/transformed JSON data

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `json_basic`, `json_filter`, `json_array` in `playbooks/examples_library/json_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/json_source.pl`
3. **Extraction Tool** – `scripts/extract_records.pl`

## Execution Guidance

**IMPORTANT**: Records contain **BASH SCRIPTS**. Extract and run with `bash`, not `swipl`.

### Example 1: Basic jq Filtering

**Step 1: Navigate and extract**
```bash
cd /path/to/UnifyWeaver
perl scripts/extract_records.pl playbooks/examples_library/json_source_examples.md \
    json_basic > tmp/json_basic.sh
chmod +x tmp/json_basic.sh
bash tmp/json_basic.sh
```

**Expected Output:**
```
Compiling JSON source: get_names/1
Generated: tmp/get_names.sh
Testing get_names/1:
Alice
Bob
Charlie
Success: JSON basic filtering works
```

### Example 2: Conditional Filtering

Extract and run `json_filter` query for jq select() operations.

### Example 3: Array to TSV

Extract and run `json_array` query for array transformation with jq.

## Expected Outcome
- JSON files processed successfully via jq
- jq filters applied correctly
- Data transformed to desired format
- Exit code 0 with "Success" message

## Citations
[1] playbooks/examples_library/json_source_examples.md
[2] src/unifyweaver/sources/json_source.pl
[3] scripts/extract_records.pl
