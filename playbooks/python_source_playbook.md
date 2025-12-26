# Playbook: Python Foreign Function Binding

## Audience
This playbook is a high-level guide for coding agents. It demonstrates using Python code as data sources through UnifyWeaver's foreign function binding system.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "python_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use python source"


## Workflow Overview
Use UnifyWeaver to bind Python as a foreign function:
1. Declare a data source using the `python_source` plugin with inline code, file, or SQLite query
2. UnifyWeaver generates bash that executes Python and captures output
3. Call the generated function to retrieve Python-processed data

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `python_inline_basic`, `python_file_basic`, `python_sqlite` in `playbooks/examples_library/python_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/python_source.pl`
3. **Extraction Tool** – `scripts/extract_records.pl`

## Execution Guidance

**IMPORTANT**: Records contain **BASH SCRIPTS**. Extract and run with `bash`, not `swipl`.

### Example 1: Inline Python Code

**Step 1: Navigate and extract**
```bash
cd /path/to/UnifyWeaver
perl scripts/extract_records.pl playbooks/examples_library/python_source_examples.md \
    python_inline_basic > tmp/python_inline.sh
chmod +x tmp/python_inline.sh
bash tmp/python_inline.sh
```

**Expected Output:**
```
Compiling Python source: fibonacci/1
Generated: tmp/fibonacci.sh
Testing fibonacci/1:
1
1
2
3
5
8
13
21
34
55
Success: Python inline code works
```

### Example 2: Python File

Extract and run `python_file_basic` query to execute external Python scripts.

### Example 3: Python with SQLite

Extract and run `python_sqlite` query for SQLite database queries via Python.

## Expected Outcome
- Python code executes successfully (inline or file-based)
- SQLite integration works for database queries
- Results returned as Prolog facts
- Exit code 0 with "Success" message

## Citations
[1] playbooks/examples_library/python_source_examples.md
[2] src/unifyweaver/sources/python_source.pl
[3] scripts/extract_records.pl
