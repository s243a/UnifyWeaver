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

### 3. Error Thresholds (`error_threshold`)
- **Option:** `error_threshold(count(N))`
- **Function:** Provides fail-fast behavior by terminating the process if the error count exceeds N.
- **Benefits:** Prevents long-running jobs from consuming resources when the input data is significantly malformed or incompatible with the schema.

### 4. Metrics Export (`metrics_file`)
- **Option:** `metrics_file('metrics.json')`
- **Function:** Produces a structured JSON summary of the run, including start/end times, record counts, error counts, and overall throughput (rec/sec).
- **Benefits:** Enables automated performance monitoring and integration with observability dashboards.

### 5. Refactoring
- **Unified Logic:** Refactored `compile_json_input_mode` to use the robust `compile_json_to_go_typed_noschema` generator even for untyped inputs. This ensures consistent error handling and observability architecture across all JSONL input scenarios.

## Verification
- Verified with `tests/test_go_stream_enhancements.pl`.
- Checked sequential and parallel execution paths for all new options.
- Confirmed correct concurrency handling (mutexes for files, atomic for counters).
- Validated that optional imports (`time`, `sync/atomic`) are only added when needed.
