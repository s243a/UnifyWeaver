# CSV Data Source Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing CSV data source playbook:
[playbooks/csv_data_source_playbook.md](../../../../playbooks/csv_data_source_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/csv_data_source_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to process CSV data using the csv_source plugin. The aim is to ensure that the CSV source is correctly compiled to bash and that the data is accessible as expected.

## Inputs & Artifacts
- Playbook file: `playbooks/csv_data_source_playbook.md`
- Example record: `playbooks/examples_library/csv_examples.md`
- Test data: `test_data/test_users.csv`
- Generated script: `tmp/users.sh`
- Temporary directory for artifacts: `tmp/`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Perl installed for record extraction.
3. Test data file exists: `test_data/test_users.csv`
4. Run all commands from the repository root.

## Execution Steps

1. **Navigate to Project Root**
   ```bash
   cd /path/to/UnifyWeaver
   ```

2. **Extract the Record**
   ```bash
   perl scripts/utils/extract_records.pl \
     -f content \
     -q "unifyweaver.execution.csv_data_source" \
     playbooks/examples_library/csv_examples.md \
     > tmp/run_csv_example.sh
   ```

3. **Run the Bash Script**
   ```bash
   chmod +x tmp/run_csv_example.sh
   bash tmp/run_csv_example.sh
   ```

## Verification

**Expected output:**
```
Creating Prolog script...
Compiling CSV source to bash...
Registered source type: csv -> csv_source
Registered dynamic source: users/3 using csv
Defined source: users/3 using csv
Compiling dynamic source: users/3 using csv
  Compiling CSV source: users/3

Compiled CSV source to tmp/users.sh

To use: source tmp/users.sh && users

Loading users function...

Calling users() to get all records:
1:Alice:30
2:Bob:25
3:Charlie:35

Success: CSV source compiled and executed
```

**Success criteria:**
- Script exits with status code 0
- All three users (Alice, Bob, Charlie) are displayed
- CSV data is correctly parsed with headers
- Generated bash function `users()` is callable
- Temporary artifacts created in `tmp/`

## Troubleshooting

| Symptom                                   | Likely Cause              | Fix                                                                  |
| ------------------------------------------ | ------------------------- | --------------------------------------------------------------------- |
| "file not found: test_data/test_users.csv" | Missing test data         | Ensure test_data/test_users.csv exists in project root               |
| "Unknown procedure: source/3"              | Module not loaded         | Check that sources module is being loaded correctly                   |
| SWI-Prolog errors loading modules          | Bad paths or module names | Verify `use_module` paths match repo structure and file names        |
| Artifacts missing in tmp/                  | Directory doesn't exist   | Create tmp/ directory in project root                                 |
| CSV parsing errors                         | Header/format issues      | Check that test_users.csv has proper CSV format with headers          |

## Related Material

- Agent-facing playbook: [playbooks/csv_data_source_playbook.md](../../../../playbooks/csv_data_source_playbook.md)
- Environment setup skill: `skills/skill_unifyweaver_environment.md`
- Example record: `playbooks/examples_library/csv_examples.md`
- Test data: `test_data/test_users.csv`
- CSV source module: `src/unifyweaver/sources/csv_source.pl`
