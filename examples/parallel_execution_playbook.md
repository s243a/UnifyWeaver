# Playbook: Parallel Execution Pipeline

## 1. Introduction & Intent
This playbook demonstrates UnifyWeaver's MapReduce-style parallel execution capabilities. The goal is to partition a dataset, process the partitions in parallel using multiple bash processes, and aggregate the results.

This playbook serves as a high-level guide for an agent. The concrete implementation details are encapsulated in a structured, executable example in the example library.

## 2. Workflow Overview
The overall workflow follows a MapReduce pattern:
1.  A "worker" script is defined to perform the "map" step on a single partition of data.
2.  A Prolog script orchestrates the pipeline, first partitioning a dataset into chunks.
3.  The orchestrator then uses a parallel backend (e.g., `bash_fork`) to execute the worker script across all partitions simultaneously.
4.  Finally, the results from all workers are collected and aggregated (the "reduce" step) to produce a final result.

## 3. Inputs & Outputs
### Inputs
- An agent command to extract and execute a script from the example library.

### Outputs
- **Final Result:** The string `SUCCESS: Final sum is 500500` printed to standard output.
- **Exit Code:** The script should exit with status code `0`.

## 4. Execution Directions
To run this parallel pipeline, extract the content of the `unifyweaver.execution.parallel_sum_pipeline` record from the example library and execute it as a bash script.

### Example Agent Command
```bash
# 1. Extract the execution script from the library
scripts/utils/extract_records.pl --format content --query "unifyweaver.execution.parallel_sum_pipeline" "education/UnifyWeaver_Education/book-workflow/examples_library/parallel_examples.md" > tmp/run_parallel_test.sh

# 2. Execute the extracted script
bash tmp/run_parallel_test.sh
```

## 5. References
- **Implementation Example:** `unifyweaver.execution.parallel_sum_pipeline` in `education/UnifyWeaver_Education/book-workflow/examples_library/parallel_examples.md`
- **Original Demo:** `examples/demo_partition_parallel.pl`
- **Extraction Skill:** `skills/skill_extract_records.md`

## 6. Troubleshooting/Validation
- **Expected Output:** The command should execute without errors and print the line `SUCCESS: Final sum is 500500`.
- **Common Errors:** If the script fails, inspect the contents of `tmp/run_parallel_test.sh` and the output of the `swipl` command within it to diagnose the issue. Ensure all paths are correct and necessary modules are available.