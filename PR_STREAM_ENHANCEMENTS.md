# PR Description: Stream Processing Enhancements (Go)

**Title:** `feat(go): implement stream enhancements (error aggregation, progress reporting)`

## Overview
This PR implements observability and robustness enhancements for the Go target's JSONL stream processing mode. It addresses the need for better error handling in ETL pipelines and visibility into long-running processes.

## Key Features

### 1. Error Aggregation (`error_file`)
- **Option:** `error_file('errors.jsonl')`
- **Function:** Captures validation and parsing errors into a dedicated side-channel file instead of crashing or logging to stderr.
- **Format:** JSONL records with timestamp, error message, and the raw input line.
- **Concurrency:** Thread-safe implementation for both sequential and parallel modes.

### 2. Progress Reporting (`progress`)
- **Option:** `progress(interval(N))` or `progress(true)` (default 1000).
- **Function:** logs processed record counts to stderr periodically.
- **Implementation:** Uses `sync/atomic` for accurate counting in parallel execution.

### 3. Refactoring
- **Unified Logic:** Refactored `compile_json_input_mode` to use the robust `compile_json_to_go_typed_noschema` generator even for untyped inputs. This ensures consistent error handling architecture across all JSONL input scenarios.

## Verification
- Verified with `tests/test_go_stream_enhancements.pl`.
- Checked both sequential and parallel execution paths.
- Confirmed imports (`time`, `sync/atomic`) are only added when needed.
