# PR Description: Phase 9f - Window Functions (Go/Bbolt)

**Title:** `feat(go): implement Phase 9f - Window Functions (row_number, rank)`

## Overview
This PR implements **Phase 9f** of the Go/Bbolt target, adding support for Window Functions within the aggregation framework. This enables advanced analytical queries that require sorting and ranking records *within* a group, rather than just summarizing them.

## Key Features

### 1. Window Operations
- **`row_number(OrderField, Result)`**: Assigns a unique, sequential integer (1, 2, 3...) to records within a group, ordered by `OrderField`.
- **`rank(OrderField, Result)`**: Supported (currently maps to sequential numbering).
- **`dense_rank(OrderField, Result)`**: Supported (currently maps to sequential numbering).

### 2. Semantic Shift
- **Standard Aggregation (`sum`, `count`)**: Collapses a group of $N$ records into **1** summary record.
- **Window Function (`row_number`)**: Preserves all $N$ records in the group, sorting them and enriching them with a rank field. Output is $N$ records per group.

### 3. Implementation
- **Accumulation**: Collects full records into a slice `[]Record` in memory during the grouping phase.
- **Sorting**: Uses Go's `sort.Slice` to sort the collected records based on the specified ordering field.
- **Output**: Iterates through the sorted slice, assigns ranks, and emits each record individually.
- **Modes**: Supported in both **Bbolt** (database) and **JSONL** (stream) modes.

## Verification
- Verified with `tests/test_go_window.pl`.
- Confirmed correct sorting and rank assignment in generated Go code.
