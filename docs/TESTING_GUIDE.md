# UnifyWeaver Testing Guide

**Last Updated:** 2025-10-26
**Status:** Active

---

## Overview

UnifyWeaver has a multi-tiered testing strategy:
1. **Comprehensive Automated Test Suite** - Run all tests quickly
2. **Feature-Specific Tests** - Deep dive into specific features
3. **Integration Tests** - End-to-end workflows
4. **Demo/Example Files** - Educational and manual testing

---

## Quick Start

### Run All Tests
```bash
# Comprehensive test suite (recommended)
swipl examples/run_all_tests.pl

# With verbose output
VERBOSE=1 swipl examples/run_all_tests.pl
```

### Run Specific Test Categories
```bash
# Core compiler tests
swipl examples/tests/test_compilers.pl

# Data source tests
swipl examples/tests/test_data_sources.pl

# PowerShell tests
swipl examples/tests/test_powershell.pl

# Firewall tests
swipl examples/test_firewall_powershell.pl
```

---

## Test Categories

### 1. Automated Test Suite

**Purpose:** Fast, comprehensive validation of all major features

**Location:** `examples/run_all_tests.pl` (to be created)

**What it tests:**
- Stream compiler
- Recursive compiler
- All data sources (CSV, JSON, HTTP, AWK, Python)
- PowerShell compilation (pure and BaaS)
- Firewall policies
- Template system
- Error handling

**Run time:** < 30 seconds

**Usage:**
```bash
swipl examples/run_all_tests.pl
```

---

### 2. Feature-Specific Tests

#### A. PowerShell Tests

**Files:**
- `examples/test_pure_powershell.pl` - Pure PowerShell mode tests
- `examples/test_firewall_powershell.pl` - Firewall integration tests
- `examples/generate_test_scripts.pl` - Generate actual PowerShell scripts

**What they test:**
- Pure PowerShell template generation
- BaaS mode fallback
- Firewall-driven mode selection
- Real PowerShell script execution

**Usage:**
```bash
# Test pure PowerShell compilation
swipl examples/test_pure_powershell.pl

# Test firewall integration
swipl examples/test_firewall_powershell.pl

# Generate and manually test PowerShell scripts
swipl examples/generate_test_scripts.pl
pwsh test_output/csv_pure.ps1
pwsh test_output/json_pure.ps1
```

#### B. Data Source Tests

**Files:**
- `examples/data_sources_demo.pl` - Comprehensive data source demos
- Individual pipeline demos (awk, json, http, python)

**What they test:**
- CSV source with headers
- JSON source with jq filters
- HTTP source with APIs
- AWK pipeline integration
- Python pipeline integration

**Usage:**
```bash
# Run all data source demos
swipl examples/data_sources_demo.pl

# Individual source tests
swipl examples/json_pipeline_demo.pl
swipl examples/http_pipeline_demo.pl
```

#### C. Compiler Tests

**Files:**
- `examples/integration_test.pl` - Stream and recursive compiler tests

**What they test:**
- Fact compilation
- Simple recursion
- Complex recursion (fibonacci, binomial)
- Constraint handling
- Pipeline integration

**Usage:**
```bash
swipl examples/integration_test.pl
```

#### D. Firewall Tests

**Files:**
- `examples/test_firewall_powershell.pl` - Firewall policy tests

**What they test:**
- Permissive mode
- Pure PowerShell policy
- Strict security policy
- Service denial
- Mode derivation

**Usage:**
```bash
swipl examples/test_firewall_powershell.pl
```

---

### 3. Integration Tests

**Purpose:** Test end-to-end workflows

**File:** `examples/integration_test.pl`

**What it tests:**
- Complete compilation workflows
- Multiple data sources in one script
- Error handling
- Output validation

**Usage:**
```bash
KEEP_TEST_DATA=true swipl examples/integration_test.pl
```

---

### 4. Demo/Example Files

**Purpose:** Educational, manual testing, and documentation

**Files:**
- `examples/load_demo.pl` - Quick feature showcase
- `examples/pipeline_demo.pl` - Pipeline examples
- `examples/constraints_demo.pl` - Constraint examples
- `examples/powershell_compilation_demo.pl` - PowerShell examples

**Usage:**
```bash
# Interactive demo
swipl examples/load_demo.pl

# Run specific demo
swipl -g test_csv_source -t halt examples/data_sources_demo.pl
```

---

## Test Organization Structure

### Current Structure
```
examples/
├── run_all_tests.pl              # Comprehensive automated suite (TO CREATE)
├── integration_test.pl           # Integration tests
├── test_firewall_powershell.pl   # Firewall tests
├── test_pure_powershell.pl       # Pure PowerShell tests
├── generate_test_scripts.pl      # Script generation
├── data_sources_demo.pl          # Data source demos
├── *_pipeline_demo.pl            # Pipeline-specific demos
└── *_demo.pl                     # Feature demos
```

### Recommended Structure (Future)
```
examples/
├── run_all_tests.pl              # Main test runner
├── tests/                        # Organized test suites
│   ├── test_compilers.pl         # Compiler tests
│   ├── test_data_sources.pl      # Data source tests
│   ├── test_powershell.pl        # PowerShell tests
│   ├── test_firewall.pl          # Firewall tests
│   └── test_templates.pl         # Template tests
├── demos/                        # Educational demos
│   ├── data_sources_demo.pl
│   ├── pipeline_demo.pl
│   └── constraints_demo.pl
└── integration/                  # Integration tests
    └── integration_test.pl
```

---

## Creating the Comprehensive Test Suite

### Design Goals

1. **Fast**: Run all tests in < 30 seconds
2. **Comprehensive**: Cover all major features
3. **Clear Output**: Pass/fail for each test
4. **Detailed on Failure**: Show what went wrong
5. **CI-Ready**: Exit code 0 on success, 1 on failure

### Test Categories to Include

```prolog
% examples/run_all_tests.pl structure
main :-
    run_test_suite([
        % Core compilers
        test_stream_compiler,
        test_recursive_compiler,

        % Data sources
        test_csv_source,
        test_json_source,
        test_http_source,
        test_awk_source,
        test_python_source,

        % PowerShell
        test_powershell_pure_mode,
        test_powershell_baas_mode,
        test_powershell_auto_mode,

        % Firewall
        test_firewall_permissive,
        test_firewall_pure_powershell,
        test_firewall_strict,

        % Templates
        test_template_rendering,
        test_template_fallback,

        % Error handling
        test_error_cases
    ]).
```

---

## Best Practices

### When to Use Each Test Type

1. **Automated Suite** (`run_all_tests.pl`):
   - Before committing changes
   - During CI/CD
   - Quick validation

2. **Feature-Specific Tests**:
   - When developing a new feature
   - When debugging a specific issue
   - When adding new functionality

3. **Integration Tests**:
   - Before releases
   - When multiple components interact
   - When testing complete workflows

4. **Demos**:
   - Learning the system
   - Documentation examples
   - Manual exploration

### Test Development Workflow

1. Write feature-specific test first
2. Implement feature
3. Run feature test until passing
4. Add test to comprehensive suite
5. Run comprehensive suite
6. Commit

---

## Test Data

### Test Files Location
```
test_data/
├── test_users.csv          # CSV test data
├── test_products.json      # JSON test data
└── test_api_response.json  # HTTP mock responses
```

### Test Output Location
```
test_output/               # Generated during tests (gitignored)
├── *.ps1                  # Generated PowerShell scripts
├── *.sh                   # Generated bash scripts
└── *.log                  # Test logs
```

---

## Continuous Integration

### GitHub Actions (Future)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install SWI-Prolog
        run: sudo apt-get install swi-prolog
      - name: Run Tests
        run: swipl examples/run_all_tests.pl
```

---

## Next Steps

1. ✅ Document test structure (this file)
2. ⏳ Create `examples/run_all_tests.pl`
3. ⏳ Organize tests into `examples/tests/` directory
4. ⏳ Add CI/CD integration
5. ⏳ Create test coverage reporting

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
**License:** MIT OR Apache-2.0
