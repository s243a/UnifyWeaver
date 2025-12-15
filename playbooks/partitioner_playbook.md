# Playbook: Data Partitioning Strategies

## Audience
This playbook is a high-level guide for coding agents. It demonstrates UnifyWeaver's partitioning system for splitting data streams for parallel processing.

## Workflow Overview
Use UnifyWeaver's partitioner module:
1. Initialize a partitioner with strategy (fixed_size, hash_based, or key_based) and configuration
2. Partition data stream using the configured strategy
3. Process partitions independently (often in parallel)

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `partition_fixed`, `partition_hash`, `partition_key` in `playbooks/examples_library/partitioner_examples.md`
2. **Core Module** – `src/unifyweaver/core/partitioner.pl`
3. **Strategy Modules** – `src/unifyweaver/core/partitioners/{fixed_size,hash_based,key_based}.pl`
4. **Extraction Tool** – `scripts/extract_records.pl`

## Execution Guidance

**IMPORTANT**: Records contain **BASH SCRIPTS**. Extract and run with `bash`, not `swipl`.

### Example 1: Fixed-Size Partitioning

**Step 1: Navigate and extract**
```bash
cd /path/to/UnifyWeaver
perl scripts/extract_records.pl playbooks/examples_library/partitioner_examples.md \
    partition_fixed > tmp/partition_fixed.sh
chmod +x tmp/partition_fixed.sh
bash tmp/partition_fixed.sh
```

**Expected Output:**
```
Testing fixed_size partitioner (chunk_size=3):
Partition 0: [1,2,3]
Partition 1: [4,5,6]
Partition 2: [7,8,9]
Partition 3: [10]
Success: Fixed-size partitioning works
```

### Example 2: Hash-Based Distribution

Extract and run `partition_hash` query for load-balanced distribution across workers.

### Example 3: Key-Based Grouping

Extract and run `partition_key` query for grouping by extracted keys.

## Expected Outcome
- Data partitioned according to strategy
- Partition counts and distributions correct
- Suitable for parallel processing workflows
- Exit code 0 with "Success" message

## Citations
[1] playbooks/examples_library/partitioner_examples.md
[2] src/unifyweaver/core/partitioner.pl
[3] src/unifyweaver/core/partitioners/*.pl
[4] scripts/extract_records.pl
