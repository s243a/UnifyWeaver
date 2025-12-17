# Playbook: Parallel Execution Pipeline

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents do not handwrite scripts here—they orchestrate UnifyWeaver to generate and run the MapReduce pipeline by referencing example records and skills.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "parallel_execution" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use parallel execution"


## Workflow Overview
Use UnifyWeaver to synthesize the entire MapReduce flow:
1. Partition the logical input stream (1..1000) into fixed-size chunks; each chunk becomes one worker invocation.
2. Declare the worker (“map”) component once; UnifyWeaver emits the bash script that sums integers from STDIN.
3. In Prolog, register the `fixed_size` partitioner and the `bash_fork` backend, then call `backend_execute/4` so UnifyWeaver forks bash jobs for every partition.
4. Reduce by iterating over the `result(PartitionID, Output)` terms and summing their payloads. Compare the aggregate with the arithmetic-series expectation (see [4]).

## Agent Inputs
Reference the following artifacts instead of embedding raw commands:
1. **Executable Record** – `unifyweaver.execution.parallel_sum_pipeline` in `playbooks/examples_library/parallel_examples.md`.
2. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md` explains how to set up the Prolog environment and run scripts from the project root.
3. **Parser Catalog** – `docs/playbooks/parsing/README.md` lists the available extractors (Perl, Python, `parsc`) and usage order.
4. **Extraction Skill** – `skills/skill_extract_records.md` documents CLI flags and environment notes.
5. **Reviewer Reference** – `docs/development/testing/playbooks/parallel_execution_playbook__reference.md` for validation details.

## Execution Guidance
1. Choose a parser per [3] (preferred order: Perl script `scripts/utils/extract_records.pl`, Python implementation, then `parsc`). Extract record [1] into a temporary bash file.
2. Ensure `TMP_FOLDER` (and optional `WORKER_SCRIPT`) resolve to a writable directory. Execute the extracted script; it materializes the worker, generates the Prolog orchestrator, and drives `backend_execute/4` to launch the parallel run.
3. Confirm the final log matches the expectation in [5]. For failures, inspect the generated script or the `bash_fork` logs; parser-specific troubleshooting lives in [3] and [4].

## Expected Outcome
- Successful runs print `SUCCESS: ...` with the arithmetic sum and exit 0 (see [5] for exact wording).
- Errors typically stem from TMP misconfiguration or unsupported parser choice; reconsult [2]/[3]/[4].

## Citations
[1] playbooks/examples_library/parallel_examples.md (`unifyweaver.execution.parallel_sum_pipeline`)
[2] skills/skill_unifyweaver_environment.md
[3] docs/playbooks/parsing/README.md
[4] skills/skill_extract_records.md
[5] docs/development/testing/playbooks/parallel_execution_playbook__reference.md
