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
1. **Setup environment**: Read [2] (`skills/skill_unifyweaver_environment.md`) to understand environment requirements. Ensure you are in the project root directory.
2. **Extract record**: Choose a parser per [3] (preferred order: Perl script `scripts/utils/extract_records.pl`, Python implementation, then `parsc`). Extract record [1] into a temporary bash file.
3. **Execute script**: Run the extracted script. It will generate the necessary Prolog and Python code to run the XML data source.
4. **Verify output**: Confirm the final log matches the expectation in [5]. For failures, inspect the generated script and check environment setup per [2].

## Expected Outcome
- Successful runs print `Total price: 1300` and exit 0.
- Errors typically stem from misconfiguration or unsupported parser choice; reconsult [2]/[3].

## Citations
[1] playbooks/examples_library/xml_examples.md (`unifyweaver.execution.xml_data_source`)
[2] skills/skill_unifyweaver_environment.md
[3] docs/playbooks/parsing/README.md
[4] skills/skill_extract_records.md
[5] docs/development/testing/playbooks/xml_data_source_playbook__reference.md
