# Playbook: YAML Data Source

## Audience
This playbook is a high-level guide for coding agents. It demonstrates processing YAML files as data sources using UnifyWeaver's yaml_source plugin via Python PyYAML.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "yaml_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use yaml source"


## Workflow Overview
Use UnifyWeaver to process YAML data:
1. Declare a data source using the `yaml_source` plugin with YAML file and optional filter
2. UnifyWeaver generates Python code using PyYAML to parse and filter data
3. Execute the generated bash function to retrieve structured YAML data

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `yaml_basic`, `yaml_filter`, `yaml_array` in `playbooks/examples_library/yaml_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/yaml_source.pl`
3. **Extraction Tool** – `scripts/extract_records.pl`

## Execution Guidance

**IMPORTANT**: Records contain **BASH SCRIPTS**. Extract and run with `bash`, not `swipl`.

### Example 1: Basic YAML Reading

**Step 1: Navigate and extract**
```bash
cd /path/to/UnifyWeaver
perl scripts/extract_records.pl playbooks/examples_library/yaml_source_examples.md \
    yaml_basic > tmp/yaml_basic.sh
chmod +x tmp/yaml_basic.sh
bash tmp/yaml_basic.sh
```

**Expected Output:**
```
Compiling YAML source: read_config/1
Generated: tmp/read_config.sh
Testing read_config/1:
appname:MyApp
version:1.0.0
port:8080
Success: YAML basic reading works
```

### Example 2: Filtering with Python Expressions

Extract and run `yaml_filter` query for list comprehension filtering.

### Example 3: Array Processing

Extract and run `yaml_array` query for array element extraction.

## Expected Outcome
- YAML files parsed successfully via Python PyYAML
- Python filter expressions applied correctly
- Data returned as Prolog-compatible facts
- Exit code 0 with "Success" message

## Citations
[1] playbooks/examples_library/yaml_source_examples.md
[2] src/unifyweaver/sources/yaml_source.pl
[3] scripts/extract_records.pl
