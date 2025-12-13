# Partitioner Examples

This file contains executable records for the partitioner playbook.

## Example 1: Fixed-Size Partitioning

Query: `partition_fixed`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Partitioner Demo: Fixed Size ==="

cat > tmp/partition_fixed.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/partitioner').
:- use_module('src/unifyweaver/core/partitioners/fixed_size').

main :-
    % Initialize partitioner with chunk_size=3
    partitioner_init(fixed_size, [chunk_size(3)], Handle),

    % Partition data [1..10]
    Data = [1,2,3,4,5,6,7,8,9,10],
    partitioner_partition(Handle, Data, Partitions),

    % Display results
    format("Testing fixed_size partitioner (chunk_size=3):~n"),
    forall(
        nth0(N, Partitions, Part),
        format("Partition ~w: ~w~n", [N, Part])
    ),

    partitioner_cleanup(Handle),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/partition_fixed.pl
echo "Success: Fixed-size partitioning works"
```

## Example 2: Hash-Based Partitioning

Query: `partition_hash`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Partitioner Demo: Hash-Based ==="

cat > tmp/partition_hash.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/partitioner').
:- use_module('src/unifyweaver/core/partitioners/hash_based').

main :-
    % Initialize with 3 workers
    partitioner_init(hash_based, [num_workers(3)], Handle),

    % Partition usernames
    Data = [alice, bob, charlie, diana, eve],
    partitioner_partition(Handle, Data, Partitions),

    % Display results
    format("Testing hash_based partitioner (workers=3):~n"),
    forall(
        nth0(N, Partitions, Part),
        format("Partition ~w: ~w~n", [N, Part])
    ),

    partitioner_cleanup(Handle),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/partition_hash.pl
echo "Success: Hash-based partitioning works"
```

## Example 3: Key-Based Partitioning

Query: `partition_key`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Partitioner Demo: Key-Based ==="

cat > tmp/partition_key.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/partitioner').
:- use_module('src/unifyweaver/core/partitioners/key_based').

main :-
    % Initialize with key field
    partitioner_init(key_based, [key_field(1)], Handle),

    % Partition records by region (first field)
    Data = [
        record(north, data1),
        record(south, data2),
        record(north, data3),
        record(east, data4)
    ],
    partitioner_partition(Handle, Data, Partitions),

    % Display results
    format("Testing key_based partitioner:~n"),
    forall(
        member(partition(Key, Items), Partitions),
        format("Partition ~w: ~w~n", [Key, Items])
    ),

    partitioner_cleanup(Handle),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/partition_key.pl
echo "Success: Key-based partitioning works"
```
