# XML Data Source Playbook — Reviewer Reference

## Overview
This document is a reviewer’s guide and checklist for validating the agent-facing XML data source playbook:
[playbooks/xml_data_source_playbook.md](../../../../playbooks/xml_data_source_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/xml_data_source_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to process XML data using a Python data source. The aim is to ensure that the Python script is correctly generated and executed, and that the output is captured as expected.

## Inputs & Artifacts
- Playbook file: `playbooks/xml_data_source_playbook.md`
- Orchestration script: `${TMP_FOLDER:-tmp}/xml_pipeline.pl`
- Temporary directory for artifacts: `${TMP_FOLDER:-tmp}`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Python 3 installed (`python3` available).
3. Run all commands from the repository root.
4. TMP guidance described in `docs/development/ai-skills/workflow_environment.md`.

## Execution Steps

1. **Extract the Record**
   - An agent will use an extractor to get the `unifyweaver.execution.xml_data_source` record from `playbooks/examples_library/xml_examples.md`.
   - The agent will then save this to a temporary file, e.g., `${TMP_FOLDER:-tmp}/xml_pipeline.pl`.

2. **Run the Pipeline**
   ```bash
   swipl -g "consult('${TMP_FOLDER:-tmp}/xml_pipeline.pl'), data_source_execute(xml_data_source, Result), halt"
   ```

## Verification

- Expected output:  
  ```
  Total price: 1300
  ```
- Script exits with status code 0.
- Temporary artifacts are created and used in `${TMP_FOLDER:-tmp}`.

## Troubleshooting

| Symptom                                   | Likely Cause              | Fix                                                                  |
| ------------------------------------------ | ------------------------- | --------------------------------------------------------------------- |
| Python script errors                       | Issues in the Python code | Check the Python script embedded in the Prolog record for correctness. |
| SWI-Prolog errors loading modules          | Bad paths or module names | Verify `use_module` paths match repo structure and file names.        |
| Artifacts missing or wrong TMP folder      | Environment variables     | Confirm `${TMP_FOLDER}` or use default `tmp`.                        |

## Related Material

- Agent-facing playbook: [playbooks/xml_data_source_playbook.md](../../../../playbooks/xml_data_source_playbook.md)
- Workflow environment guidance: `docs/development/ai-skills/workflow_environment.md`
- Example record: `playbooks/examples_library/xml_examples.md`
