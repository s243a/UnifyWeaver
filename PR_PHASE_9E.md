# PR Description: Phase 9e - Array Aggregations (Go/Bbolt)

**Title:** `feat(go): implement Phase 9e - Array Aggregations (collect_list, collect_set)`

## Overview
This PR implements **Phase 9e** of the Go/Bbolt target, adding support for collecting values into arrays during aggregation. This feature allows for efficient data denormalization directly within the pipeline, enabling queries that group records and list associated values (e.g., "all tags for a user" or "all cities in a state").

## Key Features

### 1. New Aggregation Operations
- **`collect_list(Field, Result)`**: Collects all values from the grouped records into a list.
  - **Behavior:** Preserves duplicates. Order depends on input stream/database iteration order.
  - **Implementation:** Uses a Go slice (`[]interface{}`) and efficient appending.
- **`collect_set(Field, Result)`**: Collects unique values into a list.
  - **Behavior:** Deduplicates values.
  - **Implementation:** Uses a Go map (`map[interface{}]bool`) for O(1) existence checks during accumulation, then converts to a slice for output.

### 2. Integration
- **Grouped Aggregations:** Fully integrated with `group_by/4`. Can be combined with other aggregations (e.g., `[count(N), collect_list(Tags, T)]`).
- **Type Safety:** Uses `interface{}` to handle mixed types (strings, numbers) flexibly, matching the JSON nature of the data.

## Verification
- Verified with `tests/test_go_array_agg.pl`.
- Validated correct JSON output structure for lists and sets.
