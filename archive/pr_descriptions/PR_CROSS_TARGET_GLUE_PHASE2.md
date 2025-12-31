# PR: Cross-Target Glue Phase 2 - Shell Integration

## Title
feat: Add cross-target glue Phase 2 - shell integration

## Summary

Phase 2 of the cross-target glue system implements complete shell script generation for AWK, Python, and Bash targets. This enables seamless pipeline construction across these languages using Unix pipes as the transport layer.

## Changes

### New Module: `src/unifyweaver/glue/shell_glue.pl`

Complete script generation with format-aware I/O glue:

**AWK Script Generation (`generate_awk_script/4`):**
- TSV/CSV/JSON input parsing with automatic field assignment
- Header skip support (`NR == 1 { next }`)
- Format-aware output generation

**Python Script Generation (`generate_python_script/4`):**
- stdin/stdout processing with format conversion
- Dictionary-based record handling
- JSON import/export support

**Bash Script Generation (`generate_bash_script/4`):**
- IFS-based field parsing for TSV/CSV
- jq integration for JSON processing
- Strict mode with `set -euo pipefail`

**Pipeline Orchestration (`generate_pipeline/3`):**
- Multi-step pipeline script generation
- Input/output file handling
- Proper pipe chaining with error handling

### Integration Tests: `tests/integration/glue/test_shell_glue.pl`

Comprehensive tests covering:
- AWK script generation (TSV, header skip, JSON output)
- Python script generation (TSV, JSON, header handling)
- Bash script generation (TSV, header skip)
- Pipeline generation (2-step, 3-step, with I/O files)
- Format option handling (input/output format detection)

### Example: `examples/cross-target-glue/`

Log analysis pipeline demonstrating cross-target data flow:

```
AWK (parse) → Python (analyze) → AWK (summarize)
```

**Files:**
- `log_pipeline.pl` - Pipeline definition and script generator
- `README.md` - Usage documentation

**Pipeline Stages:**
1. **AWK Parse**: Extract fields from Apache log format, filter errors (4xx/5xx)
2. **Python Analyze**: Categorize errors, assign severity, extract endpoint type
3. **AWK Summarize**: Aggregate counts, report unique IPs

## Test Results

All integration tests pass:
```
=== Shell Glue Integration Tests ===

Test: AWK script generation
  ✓ AWK sets TSV field separator
  ✓ AWK assigns field 1
  ✓ AWK assigns field 3
  ✓ AWK skips header
  ✓ AWK outputs JSON

Test: Python script generation
  ✓ Python imports sys
  ✓ Python splits on tab
  ✓ Python has process function
  ✓ Python imports json
  ✓ Python parses JSON
  ✓ Python outputs JSON
  ✓ Python skips header

Test: Bash script generation
  ✓ Bash has strict mode
  ✓ Bash uses tab separator
  ✓ Bash reads fields
  ✓ Bash skips header

Test: Pipeline generation
  ✓ Pipeline is bash script
  ✓ Pipeline includes AWK
  ✓ Pipeline includes Python
  ✓ Pipeline uses pipes
  ✓ Pipeline reads input file
  ✓ Pipeline writes output file
  ✓ Pipeline includes Bash
  ✓ Pipeline step 2
  ✓ Pipeline step 3

Test: Format options
  ✓ input_format extracts json
  ✓ format option used for input
  ✓ default input format is tsv
  ✓ output_format extracts json
  ✓ default output format is tsv

All tests passed!
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    shell_glue.pl                     │
├─────────────────────────────────────────────────────┤
│  Format Detection                                    │
│  ├── input_format/2                                 │
│  └── output_format/2                                │
├─────────────────────────────────────────────────────┤
│  Script Generation                                   │
│  ├── generate_awk_script/4                          │
│  │   ├── awk_begin_block/2                          │
│  │   ├── awk_field_assignments/3                    │
│  │   └── awk_output_code/3                          │
│  ├── generate_python_script/4                       │
│  │   ├── python_imports/3                           │
│  │   ├── python_reader/4                            │
│  │   └── python_writer/3                            │
│  └── generate_bash_script/4                         │
│      ├── bash_reader/4                              │
│      └── bash_writer/3                              │
├─────────────────────────────────────────────────────┤
│  Pipeline Orchestration                              │
│  ├── generate_pipeline/3                            │
│  ├── steps_to_pipeline/2                            │
│  └── step_to_command/2                              │
└─────────────────────────────────────────────────────┘
```

## Relationship to Phase 1

Builds on Phase 1 infrastructure:
- Uses `target_registry.pl` for target metadata
- Uses `target_mapping.pl` for predicate-to-target declarations
- Extends `pipe_glue.pl` concepts with full script generation

## Next Steps (Phase 3+)

- Go/Rust compiled target glue
- Process management (fork/exec orchestration)
- Socket-based communication for long-running targets
- .NET interop via IronPython
