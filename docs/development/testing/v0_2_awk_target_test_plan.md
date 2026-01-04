# AWK Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: AWK code generation target testing

## Overview

This test plan covers the AWK target for UnifyWeaver, which generates portable AWK scripts for text processing, data transformation, and ETL pipelines.

## Prerequisites

### System Requirements

- GNU AWK (gawk) 5.0+ or POSIX-compatible awk
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned
- Bash shell (for integration tests)

### Verification

```bash
# Verify AWK installation
awk --version  # GNU AWK
# or
awk -W version  # Some systems

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify AWK code generation without executing the generated scripts.

#### 1.1 Basic Generator Tests

```bash
# Run AWK generator tests
swipl -g "use_module('tests/core/test_awk_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `field_extraction` | Extract CSV fields | Generates `$1`, `$2` references |
| `pattern_matching` | Regex patterns | Generates `/pattern/` blocks |
| `field_separator` | Custom FS | Generates `BEGIN { FS="..." }` |
| `aggregation` | Sum/count/avg | Generates accumulator variables |
| `output_formatting` | Print statements | Generates proper `printf`/`print` |

#### 1.2 Pipeline Mode Tests

```bash
swipl -g "use_module('tests/core/test_awk_pipeline_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `single_stage` | One AWK stage | Single script generation |
| `multi_stage` | Chained AWK stages | Multiple scripts with pipes |
| `mixed_pipeline` | AWK + other targets | Proper glue code |

### 2. Integration Tests (Compilation + Execution)

These tests compile Prolog to AWK, execute the script, and verify output.

#### 2.1 CSV Processing Tests

```bash
./tests/integration/test_awk_csv_processing.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| CSV field extraction | Select specific columns | Correct column values |
| CSV filtering | Row-level predicates | Filtered subset |
| CSV aggregation | Group-by with sum | Aggregated results |
| CSV join | Two-file join | Merged records |

#### 2.2 Enhanced Chaining Tests

```bash
./tests/integration/test_awk_enhanced_chaining.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| AWK → AWK chain | Multi-stage pipeline | Correct data flow |
| AWK → Bash chain | Script handoff | Environment preservation |
| Parallel AWK | GNU parallel integration | Parallel execution |

### 3. Compatibility Tests

#### 3.1 AWK Dialect Tests

Test across different AWK implementations:

```bash
# Test with GNU AWK
UNIFYWEAVER_AWK=gawk ./tests/integration/test_awk_compatibility.sh

# Test with mawk (fast AWK)
UNIFYWEAVER_AWK=mawk ./tests/integration/test_awk_compatibility.sh

# Test with original awk (if available)
UNIFYWEAVER_AWK=nawk ./tests/integration/test_awk_compatibility.sh
```

**Compatibility Matrix**:
| Feature | gawk | mawk | nawk | POSIX |
|---------|------|------|------|-------|
| Regex | Full | Basic | Basic | Basic |
| Arrays | Yes | Yes | Yes | Yes |
| printf | Yes | Yes | Yes | Yes |
| getline | Yes | Yes | Yes | Yes |
| User functions | Yes | Yes | Yes | Yes |

### 4. Performance Tests

#### 4.1 Large File Processing

```bash
# Generate test data (1M lines)
seq 1 1000000 | awk '{print $1","$1*2","$1*3}' > /tmp/large_test.csv

# Run performance test
time swipl -g "test_awk_large_file" -t halt tests/perf/test_awk_performance.pl
```

**Benchmarks**:
| Test | File Size | Expected Time |
|------|-----------|---------------|
| Field extraction | 1M lines | < 5s |
| Pattern filter | 1M lines | < 10s |
| Aggregation | 1M lines | < 15s |

## Test Commands Reference

### Quick Smoke Test

```bash
# Generate and run a simple AWK script
swipl -g "
    use_module('src/unifyweaver/targets/awk_target'),
    compile_to_awk(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
# Run all AWK tests
./tests/run_awk_tests.sh

# Or individually:
swipl -g "use_module('tests/core/test_awk_generator'), run_tests" -t halt
swipl -g "use_module('tests/core/test_awk_pipeline_generator'), run_tests" -t halt
./tests/integration/test_awk_enhanced_chaining.sh
```

## Known Issues

1. **mawk limitations**: Some regex features not supported
2. **Large field counts**: Performance degrades with >100 fields
3. **Binary data**: Not suitable for binary file processing

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UNIFYWEAVER_AWK` | AWK interpreter path | `gawk` |
| `AWK_FIELD_SEPARATOR` | Default field separator | `,` |
| `KEEP_AWK_SCRIPTS` | Preserve generated scripts | `0` |

## Related Documentation

- [AWK Target Implementation](../../architecture/targets/awk_target.md)
- [Pipeline Generator Mode](../../architecture/pipeline_generator_mode.md)
- [CSV Data Source Playbook](playbooks/csv_data_source_playbook__reference.md)
