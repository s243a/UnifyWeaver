# Quick Testing Guide

This guide provides commands for rapid regression testing after making changes or pulling updates.

## When to Use

- After pulling remote changes (especially before pushing)
- Before creating a release
- After making significant code changes
- To verify cross-platform compatibility

## Core Test Suite

Run all core tests (stream compiler, recursive compiler, advanced recursion, constraints):

```bash
swipl -l init.pl -g "test_all, halt" -t halt
```

## Integration Test

Full integration test with all data sources (CSV, JSON, Python, SQLite) and ETL pipeline:

```bash
swipl -l init.pl -g main -t halt examples/integration_test.pl
```

### With test data preservation:

```bash
KEEP_TEST_DATA=true swipl -l init.pl -g main -t halt examples/integration_test.pl
```

### Different emoji levels:

```bash
# Full emoji support (requires Unicode terminal)
UNIFYWEAVER_EMOJI_LEVEL=full swipl -l init.pl -g main -t halt examples/integration_test.pl

# BMP emoji only (default)
UNIFYWEAVER_EMOJI_LEVEL=bmp swipl -l init.pl -g main -t halt examples/integration_test.pl

# ASCII fallbacks only
UNIFYWEAVER_EMOJI_LEVEL=ascii swipl -l init.pl -g main -t halt examples/integration_test.pl
```

## Firewall Tests

Test the firewall security system:

```bash
swipl -g "use_module('tests/core/test_firewall_enhanced'), test_firewall_enhanced" -t halt
```

## Data Source Tests

### CSV Source

```bash
swipl -g "use_module('tests/core/test_csv_source'), test_csv_source" -t halt
```

### Python Source

```bash
swipl -g "use_module('tests/core/test_python_source'), test_python_source" -t halt
```

## C# Query Tests

Fast verification (code generation only, skip execution):

```bash
SKIP_CSHARP_EXECUTION=1 swipl -q -f init.pl \
  -s tests/core/test_csharp_query_target.pl \
  -g 'test_csharp_query_target:test_csharp_query_target' \
  -t halt -- --csharp-query-keep
```

Full test with execution:

```bash
swipl -q -f init.pl \
  -s tests/core/test_csharp_query_target.pl \
  -g 'test_csharp_query_target:test_csharp_query_target' \
  -t halt
```

## Platform-Specific Tests

### Platform Detection

```bash
swipl -g "use_module('src/unifyweaver/core/platform_detection')" \
     -g "platform_detection:test_platform_detection" -t halt
```

### Bash Executor

```bash
swipl -g "use_module('src/unifyweaver/core/bash_executor')" \
     -g "bash_executor:test_bash_executor" -t halt
```

## Quick Smoke Test

Run just the essentials to verify basic functionality:

```bash
# 1. Core tests
swipl -l init.pl -g "test_all, halt" -t halt

# 2. Integration test
swipl -l init.pl -g main -t halt examples/integration_test.pl

# 3. Firewall test
swipl -g "use_module('tests/core/test_firewall_enhanced'), test_firewall_enhanced" -t halt
```

If all three pass, the system is likely in good shape.

## Test Output Verification

### Expected Results

All tests should output:
- ✅ Success indicators for passing tests
- Clear error messages for failures
- Test summaries showing pass/fail counts

### Common Issues

**Integration test hangs:**
- Check that `examples/integration_test.pl` uses `process_create` with saved scripts
- Verify temp file paths are not causing issues
- See `POST_RELEASE_TODO_v0_1.md` Priority 5, item 2 for details

**Module import warnings:**
- Some warnings about missing exports are known issues
- See `POST_RELEASE_TODO_v0_1.md` Priority 6 for tracking

**C# tests fail:**
- Ensure .NET SDK is installed: `dotnet --version`
- Use `SKIP_CSHARP_EXECUTION=1` to test code generation only

## Continuous Integration

For CI/CD pipelines, run this complete test sequence:

```bash
#!/bin/bash
set -e  # Exit on first failure

echo "Running core tests..."
swipl -l init.pl -g "test_all, halt" -t halt

echo "Running integration test..."
swipl -l init.pl -g main -t halt examples/integration_test.pl

echo "Running firewall tests..."
swipl -g "use_module('tests/core/test_firewall_enhanced'), test_firewall_enhanced" -t halt

echo "Running data source tests..."
swipl -g "use_module('tests/core/test_csv_source'), test_csv_source" -t halt
swipl -g "use_module('tests/core/test_python_source'), test_python_source" -t halt

echo "Running C# tests (code generation only)..."
SKIP_CSHARP_EXECUTION=1 swipl -q -f init.pl \
  -s tests/core/test_csharp_query_target.pl \
  -g 'test_csharp_query_target:test_csharp_query_target' \
  -t halt -- --csharp-query-keep

echo "✅ All tests passed!"
```

## See Also

- [PowerShell Test Plan](v0_0_2_powershell_test_plan.md) - Comprehensive PowerShell/WSL testing
- [Linux Test Plan](v0_0_2_linux_test_plan.md) - Comprehensive Linux testing
- [CSV Data Source Playbook](playbooks/csv_data_source_playbook.md) - Detailed CSV testing scenarios
