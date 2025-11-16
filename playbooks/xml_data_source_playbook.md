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
2. **Parser Catalog** – `docs/playbooks/parsing/README.md` lists the available extractors (Perl, Python, `parsc`) and usage order.
3. **Extraction Skill** – `skills/skill_extract_records.md` documents CLI flags and environment notes.
4. **Reviewer Reference** – `docs/development/testing/playbooks/xml_data_source_playbook__reference.md` for validation details.

## Execution Guidance
1. Choose a parser per [2] (preferred order: Perl script `scripts/utils/extract_records.pl`, Python implementation, then `parsc`). Extract record [1] into a temporary bash file.
2. Execute the extracted script. It will generate the necessary Prolog and Python code to run the XML data source.
3. Confirm the final log matches the expectation in [4]. For failures, inspect the generated script.

## Expected Outcome
- Successful runs print `Total price: 1300` and exit 0.
- Errors typically stem from misconfiguration or unsupported parser choice; reconsult [2]/[3].

## Citations
[1] playbooks/examples_library/xml_examples.md (`unifyweaver.execution.xml_data_source`)  
[2] docs/playbooks/parsing/README.md  
[3] skills/skill_extract_records.md  
[4] docs/development/testing/playbooks/xml_data_source_playbook__reference.md
