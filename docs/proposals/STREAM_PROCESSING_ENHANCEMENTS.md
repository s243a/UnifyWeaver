# Stream Processing Enhancements: Error Aggregation & Progress Reporting

## Overview
Enhance the Go target's stream processing capabilities (JSONL input) with robust observability and error management features. This addresses the "Stream Processing Enhancements" item in `FUTURE_WORK.md`.

## Goals
1.  **Error Aggregation**: capturing validation failures and malformed records into a dedicated error file instead of just logging to stderr or dropping them.
2.  **Progress Reporting**: providing real-time feedback on processing throughput and counts for long-running jobs.

## Feature 1: Error Aggregation (`error_file`)

**Problem**: Currently, when a record fails schema validation or JSON parsing, it is often skipped or logged to stderr intermixed with other logs. For ETL pipelines, it is crucial to capture these "bad records" for later analysis and remediation.

**Proposed Syntax**:
```prolog
compile_predicate_to_go(process_user/1, [
    json_input(true),
    schema(user_schema),
    error_file('errors.jsonl')  % New option
], Code).
```

**Implementation Logic**:
- Open the error file at the start of the program.
- Define a standard error record structure:
  ```json
  {
    "timestamp": "2025-12-23T10:00:00Z",
    "error": "Schema validation failed: missing field 'email'",
    "raw_record": "{\"name\": \"Bob\"}",
    "source_line": 150
  }
  ```
- Wrap validation/parsing logic in a block that catches errors and writes to this file.
- Thread-safe writing if parallel processing is enabled.

## Feature 2: Progress Reporting (`progress`)

**Problem**: For large datasets (millions of records), the user has no visibility into progress until the job finishes.

**Proposed Syntax**:
```prolog
compile_predicate_to_go(process_user/1, [
    json_input(true),
    progress(interval(1000))  % Report every 1000 records
], Code).
```

**Implementation Logic**:
- Maintain atomic counters for `processed_count`, `success_count`, `error_count`.
- Start a background goroutine (ticker) OR check in the main loop (modulo N).
- Output format (stderr):
  `[2025-12-23 10:00:05] Processed: 10000 | Success: 9950 | Errors: 50 | Rate: 2000 rec/s`

## Go Implementation Details

### Imports
- `time` (for timestamps and tickers)
- `sync/atomic` (for thread-safe counters)
- `os` (file I/O)

### Parallel Mode Integration
Both features must work with the existing worker pool implementation.
- **Errors**: Workers send error objects to a dedicated error channel. A separate "error writer" goroutine consumes this channel and writes to the file (avoids mutex contention on file I/O).
- **Progress**: Workers atomically increment shared counters. A separate "monitor" goroutine prints stats.

## Verification Plan
1.  **Test Case**: `tests/test_go_stream_enhancements.pl`
    - Input: A mix of valid and invalid JSON lines.
    - Compilation: Enable `error_file` and `progress`.
    - Execution: Run the binary.
    - Checks:
        - Verify `stdout` contains only valid transformed records.
        - Verify `errors.jsonl` exists and contains the expected invalid records with error messages.
        - Verify `stderr` contains progress logs (captured via redirection).

## Timeline
- Design & Plan: 30m
- Implementation: 2h
- Testing: 1h
