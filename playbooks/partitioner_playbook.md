# Playbook: Data Partitioning Strategies

## Audience
This playbook demonstrates UnifyWeaver's partitioning system for splitting data streams for parallel processing.

## Overview
The `partitioner` module provides pluggable strategies for partitioning data: fixed_size, hash_based, and key_based.

## When to Use

✅ **Use partitioner when:**
- Splitting data for parallel processing
- Need load balancing across workers
- Want consistent key-based distribution
- Processing large datasets in chunks

## Agent Inputs

1. **Executable Records** – `playbooks/examples_library/partitioner_examples.md`
2. **Core Module** – `src/unifyweaver/core/partitioner.pl`
3. **Strategies** – `src/unifyweaver/core/partitioners/{fixed_size,hash_based,key_based}.pl`

## Execution Guidance

### Example 1: Fixed-Size Partitioning

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
```

### Example 2: Hash-Based Partitioning

```bash
perl scripts/extract_records.pl playbooks/examples_library/partitioner_examples.md \
    partition_hash > tmp/partition_hash.sh
chmod +x tmp/partition_hash.sh
bash tmp/partition_hash.sh
```

**Expected Output:**
```
Testing hash_based partitioner (workers=3):
Partition 0: [alice,diana]
Partition 1: [bob,eve]
Partition 2: [charlie]
```

### Example 3: Key-Based Partitioning

```bash
perl scripts/extract_records.pl playbooks/examples_library/partitioner_examples.md \
    partition_key > tmp/partition_key.sh
chmod +x tmp/partition_key.sh
bash tmp/partition_key.sh
```

**Expected Output:**
```
Testing key_based partitioner:
Partition north: [record1,record3]
Partition south: [record2]
Partition east: [record4]
```

## Partitioning Strategies

### 1. Fixed-Size (Chunk-Based)
- Splits data into fixed-size chunks
- Config: `chunk_size(N)`
- Use: Simple batch processing

### 2. Hash-Based (Load Balancing)
- Distributes items via hash function
- Config: `num_partitions(N)` or `num_workers(N)`
- Use: Parallel worker pools

### 3. Key-Based (Grouping)
- Groups by extracted key field
- Config: `key_field(Field)` or `key_extractor(Pred)`
- Use: Group-by operations

## See Also

- `playbooks/bash_parallel_playbook.md` - Uses partitioning for parallel execution
- `playbooks/parallel_execution_playbook.md` - Parallel processing patterns

## Summary

**Key Concepts:**
- ✅ Pluggable partitioning strategies
- ✅ Fixed-size, hash-based, key-based
- ✅ Supports parallel processing
- ✅ Load balancing and grouping
