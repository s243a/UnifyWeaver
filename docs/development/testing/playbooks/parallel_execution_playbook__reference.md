# Parallel Execution Playbook — Reviewer Reference

## Overview
This document is a reviewer’s guide and checklist for validating the agent-facing parallel execution playbook:
[examples/parallel_execution_playbook.md](../../../../examples/parallel_execution_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the examples folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at examples/parallel_execution_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to perform MapReduce-style parallel data processing. The aim is to ensure partitioning, worker execution, aggregation, and verification all function correctly when orchestrated by an agent using the playbook.

## Inputs & Artifacts
- Playbook file: `examples/parallel_execution_playbook.md`
- Worker script: `${TMP_FOLDER:-tmp}/sum_worker.sh`
- Orchestration script: `${TMP_FOLDER:-tmp}/parallel_pipeline.pl`
- Temporary directory for artifacts: `${TMP_FOLDER:-tmp}`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Bash shell and basic core utilities available.
3. Run all commands from the repository root.
4. TMP guidance described in `docs/development/ai-skills/workflow_environment.md`.

## Execution Steps

1. **Create Worker Script**
   - Make a Bash script that sums stdin numbers and outputs the sum.
   - Save to `${TMP_FOLDER:-tmp}/sum_worker.sh` and `chmod +x`.

2. **Create Prolog Orchestration Script**
   - Sets up and registers required modules and backends.
   - Generates test data, partitions it, runs workers in parallel, aggregates results, and verifies correctness.

3. **Run Pipeline**
   ```bash
   swipl -g "consult('${TMP_FOLDER:-tmp}/parallel_pipeline.pl'), run_pipeline, halt"
   ```

## Verification

- Expected output:  
  ```
  SUCCESS: Final sum is 500500
  ```
- Script exits with status code 0.
- Temporary artifacts are created and used in `${TMP_FOLDER:-tmp}`.
- All components (partitioner, parallel backend, aggregator) load and execute without errors.

## Troubleshooting

| Symptom                                   | Likely Cause              | Fix                                                                  |
| ------------------------------------------ | ------------------------- | --------------------------------------------------------------------- |
| Bash worker not executable                 | Missing `chmod +x`        | Run `chmod +x ${TMP_FOLDER}/sum_worker.sh` after script creation.     |
| Incorrect sum or failure message           | Logic or partition config | Check partition size, aggregator predicate, and worker script logic.  |
| SWI-Prolog errors loading modules          | Bad paths or module names | Verify `use_module` paths match repo structure and file names.        |
| Artifacts missing or wrong TMP folder      | Environment variables     | Confirm `${TMP_FOLDER}` or use default `tmp`.                        |

## Related Material

- Agent-facing playbook: [examples/parallel_execution_playbook.md](../../../../examples/parallel_execution_playbook.md)
- Workflow environment guidance: `docs/development/ai-skills/workflow_environment.md`
- Demo script: `examples/demo_partition_parallel.pl`
