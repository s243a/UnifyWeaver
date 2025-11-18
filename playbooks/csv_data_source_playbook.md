# Playbook: CSV Data Source

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents do not handwrite scripts here—they orchestrate UnifyWeaver to generate and run the CSV processing pipeline by referencing example records and skills.

## Workflow Overview
Use UnifyWeaver to synthesize the entire CSV processing flow:
1. Define a data source that reads CSV data using the csv_source plugin.
2. The CSV source will parse the file with headers and make data available as Prolog predicates.
3. UnifyWeaver will compile the CSV source to bash and execute it.

## Agent Inputs
Reference the following artifacts instead of embedding raw commands:
1. **Executable Record** – `unifyweaver.execution.csv_data_source` in `playbooks/examples_library/csv_examples.md`.
2. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md` explains how to set up the Prolog environment and run scripts from the project root.
3. **Parser Catalog** – `docs/playbooks/parsing/README.md` lists the available extractors (Perl, Python, `parsc`) and usage order.
4. **Extraction Skill** – `skills/skill_extract_records.md` documents CLI flags and environment notes.
5. **Reviewer Reference** – `docs/development/testing/playbooks/csv_data_source_playbook__reference.md` for validation details.

## Execution Guidance

**IMPORTANT**: The record in [1] contains a **BASH SCRIPT**, not Prolog code. You must extract it and run it with `bash`, not with `swipl`.

### Step-by-Step Instructions

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.csv_data_source" \
  playbooks/examples_library/csv_examples.md \
  > tmp/run_csv_example.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_csv_example.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_csv_example.sh
```

**Expected Output**:
```
Creating Prolog script...
Compiling CSV source to bash...
...
Calling users() to get all records:
1 Alice 30
2 Bob 25
3 Charlie 35
Success: CSV source compiled and executed
```

### What the Script Does

The bash script you extracted will:
1. Create a Prolog script in `tmp/csv_example.pl`
2. Run SWI-Prolog to compile the CSV source to bash
3. Execute the generated bash function
4. Output the CSV data records

### Common Mistakes to Avoid

❌ **DO NOT** try to consult the extracted file as Prolog:
```bash
# WRONG - This will fail!
swipl -g "consult('tmp/run_csv_example.sh'), ..."
```

✅ **DO** run it as a bash script:
```bash
# CORRECT
bash tmp/run_csv_example.sh
```

## Expected Outcome
- Successful runs print the CSV records and "Success: CSV source compiled and executed"
- Exit code 0
- Errors typically stem from misconfiguration or missing test data; reconsult [2]/[3]

## Citations
[1] playbooks/examples_library/csv_examples.md (`unifyweaver.execution.csv_data_source`)
[2] skills/skill_unifyweaver_environment.md
[3] docs/playbooks/parsing/README.md
[4] skills/skill_extract_records.md
[5] docs/development/testing/playbooks/csv_data_source_playbook__reference.md
