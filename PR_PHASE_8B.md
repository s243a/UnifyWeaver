# PR Description: Phase 8b - Enhanced Filtering & Key Optimization (Go/Bbolt)

**Title:** `feat(go): implement Phase 8b - Enhanced Filtering & Key Optimization`

## Overview
This PR implements **Phase 8b** of the Go/Bbolt target development, focusing on advanced filtering capabilities and database access optimizations. It enables UnifyWeaver to efficiently query Bbolt databases using Secondary Indexes and Primary Keys, significantly improving read performance for filtered queries. It also adds standard Prolog string and list operations to the Go target.

## Key Features

### 1. Secondary Indexes (`:- index/2`)
- **Declaration:** Users can declare indexes using `:- index(Predicate/Arity, Field).`.
- **Write Path:** Automatically creates index buckets (`index_Pred_Field`) and maintains `Value:PrimaryKey` entries when records are written.
- **Read Path:** Uses `cursor.Seek()` on index buckets to perform efficient range scans when a filter matches an indexed field.

### 2. Key Optimization Strategies
- **Direct Lookup:** Uses `bucket.Get()` when the filter exactly matches the Primary Key.
- **Prefix Scan:** Uses `cursor.Seek()` when the filter matches a prefix of a Composite Primary Key.
- **Index Scan:** Uses `cursor.Seek()` on Secondary Indexes (as described above).
- **Fallback:** Gracefully falls back to full bucket scan if no optimization is applicable.

### 3. String & List Operations
- **Case-Insensitive Equality:** `Field =@= "Value"` compiles to `strings.EqualFold()`.
- **Substring Match:** `contains(Field, "Sub")` compiles to `strings.Contains()`.
- **List Membership:** `member(Field, ["A", "B"])` compiles to efficient Go loops/slices.

## Implementation Details
- **New Module:** `src/unifyweaver/core/index_analyzer.pl` handles index declarations.
- **Modified:** `src/unifyweaver/targets/go_target.pl`:
    - Updated `compile_database_read_mode` to analyze optimization opportunities.
    - Added `generate_index_scan_code`, `generate_direct_lookup_code` (refactored), etc.
    - Added support for `=@=`, `contains/2`, `member/2` in `constraint_to_go_check`.

## Verification
- Verified with `tests/test_go_index.pl` (Write Path).
- Verified with `tests/test_go_index_read.pl` (Read Path/Optimization).
- Verified with `tests/test_go_phase_8b.pl` (String/List Ops).
- Regression tested with `tests/test_go_parallel.pl`.
