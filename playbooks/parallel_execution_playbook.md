# Playbook: Parallel Execution Pipeline

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents do not handwrite scripts here—they orchestrate UnifyWeaver to generate and run the MapReduce pipeline by referencing example records and skills.

## Workflow Overview
Use UnifyWeaver to synthesize the entire MapReduce flow:
1. Partition the logical input stream (1..1000) into fixed-size chunks; each chunk becomes one worker invocation.
2. Declare the worker (“map”) component once; UnifyWeaver emits the bash script that sums integers from STDIN.
3. In Prolog, register the `fixed_size` partitioner and the `bash_fork` backend, then call `backend_execute/4` so UnifyWeaver forks bash jobs for every partition.
4. Reduce by iterating over the `result(PartitionID, Output)` terms and summing their payloads. Compare the aggregate with the arithmetic-series expectation (see [4]).

## Agent Inputs
Reference the following artifacts instead of embedding raw commands:
1. **Executable Record** – `unifyweaver.execution.parallel_sum_pipeline` in `playbooks/examples_library/parallel_examples.md`.
2. **Parser Catalog** – `docs/playbooks/parsing/README.md` lists the available extractors (Perl, Python, `parsc`) and usage order.
3. **Extraction Skill** – `skills/skill_extract_records.md` documents CLI flags and environment notes.
4. **Reviewer Reference** – `docs/development/testing/playbooks/parallel_execution_playbook__reference.md` for validation details.

## Execution Guidance
1. Choose a parser per [2] (preferred order: Perl script `scripts/utils/extract_records.pl`, Python implementation, then `parsc`). Extract record [1] into a temporary bash file.
2. Ensure `TMP_FOLDER` (and optional `WORKER_SCRIPT`) resolve to a writable directory. Execute the extracted script; it materializes the worker, generates the Prolog orchestrator, and drives `backend_execute/4` to launch the parallel run.
3. Confirm the final log matches the expectation in [4]. For failures, inspect the generated script or the `bash_fork` logs; parser-specific troubleshooting lives in [2] and [3].

## Expected Outcome
- Successful runs print `SUCCESS: ...` with the arithmetic sum and exit 0 (see [4] for exact wording).
- Errors typically stem from TMP misconfiguration or unsupported parser choice; reconsult [2]/[3].

## Citations
[1] playbooks/examples_library/parallel_examples.md (`unifyweaver.execution.parallel_sum_pipeline`)  
[2] docs/playbooks/parsing/README.md  
[3] skills/skill_extract_records.md  
[4] docs/development/testing/playbooks/parallel_execution_playbook__reference.md
