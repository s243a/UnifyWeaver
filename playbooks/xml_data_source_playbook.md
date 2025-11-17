# Playbook: XML Data Source

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents do not handwrite scripts here—they orchestrate UnifyWeaver to generate and run the XML processing pipeline by referencing example records and skills.

## Workflow Overview
Use UnifyWeaver to synthesize the entire XML processing flow:
1. Define a data source that uses a Python script to parse XML data.
2. The Python script will read an embedded XML string, sum the prices of products, and print the total.
3. UnifyWeaver will execute the Python script and capture the output.

## Agent Inputs
Reference the following artifacts instead of embedding raw commands:
1. **Executable Record** – `unifyweaver.execution.xml_data_source` in `playbooks/examples_library/xml_examples.md`.
2. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md` explains how to set up the Prolog environment and run scripts from the project root.
3. **Parser Catalog** – `docs/playbooks/parsing/README.md` lists the available extractors (Perl, Python, `parsc`) and usage order.
4. **Extraction Skill** – `skills/skill_extract_records.md` documents CLI flags and environment notes.
5. **Reviewer Reference** – `docs/development/testing/playbooks/xml_data_source_playbook__reference.md` for validation details.

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
  -q "unifyweaver.execution.xml_data_source" \
  playbooks/examples_library/xml_examples.md \
  > tmp/run_xml_example.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_xml_example.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_xml_example.sh
```

**Expected Output**:
```
Creating Prolog script...
Compiling Python source to bash...
...
Total price: 1300
```

### What the Script Does

The bash script you extracted will:
1. Create a Prolog script in `tmp/xml_example.pl`
2. Run SWI-Prolog to compile the Python source to bash
3. Execute the generated bash script
4. Output the result: "Total price: 1300"

### Common Mistakes to Avoid

❌ **DO NOT** try to consult the extracted file as Prolog:
```bash
# WRONG - This will fail!
swipl -g "consult('tmp/run_xml_example.sh'), ..."
```

✅ **DO** run it as a bash script:
```bash
# CORRECT
bash tmp/run_xml_example.sh
```

## Expected Outcome
- Successful runs print `Total price: 1300` and exit 0.
- Errors typically stem from misconfiguration or unsupported parser choice; reconsult [2]/[3].

## Citations
[1] playbooks/examples_library/xml_examples.md (`unifyweaver.execution.xml_data_source`)
[2] skills/skill_unifyweaver_environment.md
[3] docs/playbooks/parsing/README.md
[4] skills/skill_extract_records.md
[5] docs/development/testing/playbooks/xml_data_source_playbook__reference.md
