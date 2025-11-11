# 🐧 **Linux/WSL Test Plan for UnifyWeaver v0.0.2**

**Date:** October 20, 2025 (Updated)
**Release:** v0.0.2 Data Source Plugin System + Native Execution Support
**Platform:** Linux, WSL, Docker, macOS (native bash execution environments)
**Estimated Testing Time:** 30-60 minutes (Priorities 1-2), +30 min (Priority 3), +60 min (Priority 4)

**Last Updated:** Added native execution support, platform detection, Windows emoji fix, and comprehensive integration tests.

**Note:** For Windows PowerShell testing, see `docs/development/testing/v0_0_2_powershell_test_plan.md`

---

## **Setup and Prerequisites**

Before running any tests, ensure your environment is properly configured:

### **1. System Requirements**

**Required tools:**
- `swipl` (SWI-Prolog 9.2.9 or later)
- `bash` (native execution environment)
- `jq` (for JSON data sources)
- `sqlite3` (for SQLite data sources)
- `python3` (for Python data sources)
- `awk` (for CSV data sources)
- `curl` (for HTTP data sources)

**Install missing tools on Ubuntu/Debian/WSL:**
```bash
sudo apt update
sudo apt install jq sqlite3 python3 gawk curl
```

**Verify installation:**
```bash
which jq sqlite3 python3 awk curl
```

### **2. Test Environment Setup**

#### **2.1. Create Test Environment**

Generate a fresh test environment for isolated testing:

```bash
cd scripts/testing
./init_testing.sh test_env7
cd test_env7
```

This creates a complete copy of UnifyWeaver with:
- ✅ All source modules
- ✅ Documentation
- ✅ Example files
- ✅ Test scripts
- ✅ Output directories

> **Note:** Test environments in `scripts/testing/test_env*` are not tracked in git (see `.gitignore`)

#### **2.2. SWI-Prolog PATH Configuration**

After creating the test environment, you need to ensure `swipl` is available on your PATH. There are **two methods** to do this:

##### **Method 1: Comprehensive Setup (Recommended)**

Use `find_swi-prolog.sh` for full configuration with persistence:

```bash
source ./scripts/testing/find_swi-prolog.sh
```

This provides:
- ✅ SWI-Prolog auto-detection (native Linux or Windows via WSL)
- ✅ Interactive prompts for configuration choices
- ✅ Config persistence to `.unifyweaver.conf`
- ✅ Wrapper creation for cross-platform scenarios
- ✅ UTF-8 locale configuration

First-time setup will prompt you to:
- Choose between native Linux swipl vs Windows swipl.exe (on WSL)
- Create PATH wrappers if needed
- Save preferences for future sessions

##### **Method 2: Quick Setup**

Use `init_swipl_env.sh` for simple PATH setup without persistence:

```bash
source ./scripts/init_swipl_env.sh
```

This provides:
- ✅ SWI-Prolog auto-detection and PATH setup
- ✅ UTF-8 locale configuration
- ✅ Faster, lighter (no interactive prompts)
- ❌ No config persistence
- ❌ No wrapper creation

**Use Method 1 for:** Initial setup, persistent configuration
**Use Method 2 for:** Quick sessions, automation, CI/CD

> **Note:** See `scripts/testing/README_SWIPL_ENV.md` for detailed documentation on environment setup scripts.

#### **2.3. Verify SWI-Prolog**

Check that swipl is available:

```bash
which swipl
swipl --version
```

Expected output:
```
SWI-Prolog version 9.2.9 for x86_64-linux
```

### **3. Platform Detection**

This test plan is designed for native bash execution environments (Linux, WSL, Docker, macOS). UnifyWeaver will automatically detect your platform and use native bash execution.

To verify platform detection:
```bash
swipl -g "use_module('src/unifyweaver/core/platform_detection')" \
     -g "platform_detection:detect_platform(P), format('Platform: ~w~n', [P])" \
     -t halt
```

---

## **Priority 1: Critical Functionality Tests** ⚡

### **1. Data Source Plugin Loading Tests**
```bash
# Test 1: Verify plugins load without errors or warnings
swipl -g "use_module('src/unifyweaver/sources/csv_source')" -t halt
swipl -g "use_module('src/unifyweaver/sources/python_source')" -t halt
swipl -g "use_module('src/unifyweaver/sources/http_source')" -t halt
swipl -g "use_module('src/unifyweaver/sources/json_source')" -t halt

# Expected: All should load without error messages or warnings
# ✅ FIXED: No more singleton variable warnings
# ✅ FIXED: No more deprecation warnings
```

### **1a. Platform Detection Test** 🆕
```bash
# Test 1a: Verify platform detection works correctly
swipl -g "use_module('src/unifyweaver/core/platform_detection')" \
     -g "platform_detection:test_platform_detection" -t halt

# Expected output:
# === Platform Detection Test ===
# Platform: [docker|wsl|linux|macos|windows]
# Execution Mode: [direct_bash|powershell_wsl]
#
# Platform Checks:
#   Windows: [YES|NO]
#   WSL: [YES|NO]
#   Docker: [YES|NO]
#   Native Linux: [YES|NO]
#   macOS: [YES|NO]
#
# Execution Capability:
#   Can execute bash directly: [YES|NO]
```

### **1b. Native Bash Execution Test** 🆕
```bash
# Test 1b: Verify native bash execution (Linux/WSL/Docker/macOS only)
swipl -g "use_module('src/unifyweaver/core/bash_executor')" \
     -g "bash_executor:test_bash_executor" -t halt

# Expected: All 5 tests pass
# - Simple echo command
# - Command with stdin input
# - Bash pipeline
# - Multiple commands
# - File execution

# Note: Skips automatically on Windows (requires PowerShell compatibility layer)
```

### **2. Unit Test Execution**
```bash
# Test 2: Run individual plugin tests (if test files exist)
swipl -g "use_module('tests/core/test_csv_source'), test_csv_source" -t halt
swipl -g "use_module('tests/core/test_python_source'), test_python_source" -t halt
swipl -g "use_module('tests/core/test_firewall_enhanced'), test_firewall_enhanced" -t halt

# Expected: All tests pass with ✅ messages
# Note: Some test files may not exist yet - that's okay
```
### **3. Test Emoji Compatibility**

```SHELL
# Test 3: Verify emoji rendering works on your terminal
swipl -l init.pl -g quick_test -t halt examples/quick_emoji_test.pl
```

**Expected output:**
```
=== Quick Emoji Test ===
Current emoji level: full

Direct format/2 tests:
  ✅ Checkmark (BMP)
  🚀 Rocket (non-BMP)

safe_format/2 tests (should match above):
  ✅ Checkmark (BMP)
  🚀 Rocket (non-BMP)
```

**Note:** The integration test (section 3a) automatically detects emoji support and uses appropriate fallbacks. If your environment behaves differently than expected, you can override emoji detection by setting the `UNIFYWEAVER_EMOJI_LEVEL` environment variable:

```SHELL
# Force full emoji support
UNIFYWEAVER_EMOJI_LEVEL=full swipl -l init.pl -g main -t halt examples/integration_test.pl

# Use BMP-only (✅ ❌ ⚠ work, 🚀 📊 become [STEP] [DATA])
UNIFYWEAVER_EMOJI_LEVEL=bmp swipl -l init.pl -g main -t halt examples/integration_test.pl

# ASCII fallbacks only ([OK] [FAIL] [STEP] [DATA])
UNIFYWEAVER_EMOJI_LEVEL=ascii swipl -l init.pl -g main -t halt examples/integration_test.pl
```

For more information, see: `docs/development/UNICODE_SPECIFICATION.md`

### **3a. Integration Test Suite** 🆕
```bash
# Test 3: Comprehensive integration test
KEEP_TEST_DATA=true swipl -l init.pl -g main -t halt examples/integration_test.pl

# Expected output:
# 🧪 ========================================
#   UnifyWeaver v0.0.2 Integration Test
# ========================================
#
# 💾 Setting up test data...
#   ✅ Test data created
#
# 🔍 === Platform Detection Test ===
#   Platform: docker
#   Execution Mode: direct_bash
#   ✅ Native bash execution available
#
# 🎨 === Emoji Rendering Test ===
#   Emoji Level: full
#   BMP: ✅ ❌ ⚠ ℹ
#   Non-BMP: 🚀 📊 📈 🎉
#   ✅ Emoji rendering working
#
# 📊 === CSV Source Test ===
#   Loaded 2 products
#   ✅ CSV source working
#
# 📄 === JSON Source Test ===
#   Loaded 2 orders
#   ✅ JSON source working
#
# 🐍 === Python Source Test ===
#   Analysis Results:
#   Mouse	5	1
#   Laptop	2	1
#   Desk	1	1
#   ✅ Python processing working
#
# 💾 === SQLite Source Test ===
#   Top Products:
#   Mouse:5:1
#   Laptop:2:1
#   Desk:1:1
#   ✅ SQLite query working
#
# 🚀 === Complete ETL Pipeline Test ===
#   📡 Stage 1: Extract (JSON)
#     Extracted 2 records
#   📊 Stage 2: Transform & Load (Python)
#     Loaded 2 users into database
#   📈 Stage 3: Query (SQLite)
#     Results:
#     Bob:30:SF
#     Alice:25:NYC
#   ✅ ETL pipeline complete
#
# ========================================
# 🎉 All Integration Tests Passed!
# ========================================
#
# 🧹 Cleaning up test files...
#   ✅ Cleanup complete

# Note: Requires jq and sqlite3 installed
# Install with: sudo apt install jq sqlite3
```

---

### **3b. Test Generated Code** 🆕

```SHELL
# Test 3b: Verify generated bash scripts work correctly
# Note: init_testing.sh copies this to test_env root
bash ./test_generated_scripts.sh
```

**Expected output:**
```
=== Testing Generated UnifyWeaver Scripts ===

NOTE: This is an ad-hoc test script.
      Future versions will use test_runner_generator.pl

1. CSV Source (Products):
laptop,laptop,999.99,Electronics
mouse,mouse,29.99,Electronics

2. JSON Source (Orders):
{"orderId": 1, "product": "laptop", "quantity": 2}
{"orderId": 2, "product": "mouse", "quantity": 5}

3. Python ETL Pipeline (Orders → Analysis):
Mouse	5	1
Laptop	2	1
Desk	1	1

4. SQLite Query (Top Products):
Mouse:5:1
Laptop:2:1
Desk:1:1

✅ All scripts tested successfully!
```

**Note:** This test requires that Test 3a (Integration Test Suite) was run with `KEEP_TEST_DATA=true` to preserve the generated scripts in `test_output/`.
## **Priority 2: Real-World Functionality Tests** 🌍

### **4. Demo Execution Tests**

#### **Test 4a: Data Sources Demo (Syntax Example)**
```bash
# This demo shows data source syntax
# It demonstrates configuration but doesn't execute the sources

swipl -l init.pl -g main -t halt examples/data_sources_demo.pl

# Expected output:
# UnifyWeaver v0.0.2 Data Sources Demo
# ==========================================
# Created sample input data: input/sample_users.csv
# 📡 Starting ETL Pipeline...
# 🚀 Fetching posts from API...
# 📊 Parsing JSON data...
# 📈 Generating report...
# ✅ ETL Pipeline completed successfully!
# 👥 Validating user data...
# ✅ User validation completed!
#
# Demo completed!
#   Input data: input/sample_users.csv
#   (Output would be in output/ if sources were compiled and executed)

# ✅ FIXED: Emoji rendering works on Windows (uses Unicode escapes)
# ✅ FIXED: Directory organization improved (input/ for data, output/ for results)
# ✅ FIXED: Missing module imports added
```
This test will generate sample_users.csv in the input folder.
#### **Test 4b: JSON Source with jq** 🆕
```SHELL
# Test JSON data source with jq filtering
# Requires: jq installed

# Create test data
cat > /tmp/test_data.json << 'EOF'
{
  "users": [
    {"id": 1, "name": "Alice", "age": 25, "city": "NYC"},
    {"id": 2, "name": "Bob", "age": 30, "city": "SF"}
  ]
}
EOF

# Test in SWI-Prolog using consult(user):
cat << 'PROLOG' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/json_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(json, users, [
    json_file('/tmp/test_data.json'),
    jq_filter('.users[] | [.id, .name, .age, .city] | @tsv'),
    raw_output(true)
]).

test :-
    compile_dynamic_source(users/2, [], Code),
    write_and_execute_bash(Code, '', Output),
    format('Output:~n~w~n', [Output]).
PROLOG

# Expected output:
# 1	Alice	25	NYC
# 2	Bob	30	SF
```

#### **Test 4c: SQLite Source with Python** 🆕
```SHELL
# Test SQLite data source
# Requires: sqlite3 installed

# Create test database
sqlite3 /tmp/test.db << 'SQL'
DROP TABLE IF EXISTS employees;
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary INTEGER
);
INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000);
INSERT INTO employees VALUES (2, 'Bob', 'Sales', 75000);
INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 105000);
SQL

# Test in SWI-Prolog using consult(user):
cat << 'PROLOG' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/python_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(python, high_earners, [
    sqlite_query('SELECT name, department, salary FROM employees WHERE salary > 80000 ORDER BY salary DESC'),
    database('/tmp/test.db'),
    arity(3)
]).

test :-
    compile_dynamic_source(high_earners/3, [], Code),
    write_and_execute_bash(Code, '', Output),
    format('Output:~n~w~n', [Output]).
PROLOG

# Expected output:
# Charlie:Engineering:105000
# Alice:Engineering:95000
```

#### **Test 4d: Complete ETL Pipeline** 🆕
```SHELL
# Test multi-stage ETL pipeline: JSON → Python → SQLite → Query
# Requires: jq and sqlite3 installed

# Run the ETL test (creates temporary test data)
cat << 'PROLOG' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/json_source').
:- use_module('src/unifyweaver/sources/python_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Stage 1: Extract from JSON
:- source(json, raw_users, [
    json_file('/tmp/test_data.json'),
    jq_filter('.users[] | [.name, .age, .city] | @tsv'),
    raw_output(true),
    arity(3)
]).

% Stage 2: Load into SQLite
:- source(python, load_users, [
    python_inline('
import sys, sqlite3
conn = sqlite3.connect("/tmp/etl_test.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, age INTEGER, city TEXT)")
for line in sys.stdin:
    parts = line.strip().split("\\t")
    if len(parts) >= 3:
        cursor.execute("INSERT INTO users VALUES (?, ?, ?)", parts)
conn.commit()
print(f"Loaded {cursor.rowcount} users")
conn.close()
'),
    arity(1)
]).

% Stage 3: Query results
:- source(python, get_users, [
    sqlite_query('SELECT name, age, city FROM users ORDER BY age DESC'),
    database('/tmp/etl_test.db'),
    arity(3)
]).

test :-
    % Extract
    compile_dynamic_source(raw_users/3, [], ExtractCode),
    write_and_execute_bash(ExtractCode, '', ExtractOutput),
    % Load
    compile_dynamic_source(load_users/1, [], LoadCode),
    write_and_execute_bash(LoadCode, ExtractOutput, LoadOutput),
    format('Load: ~w~n', [LoadOutput]),
    % Query
    compile_dynamic_source(get_users/3, [], QueryCode),
    write_and_execute_bash(QueryCode, '', QueryOutput),
    format('Results:~n~w~n', [QueryOutput]).

:- test, halt.
PROLOG

# Expected:
# Load: Loaded 2 users
# Results:
# Bob:30:SF
# Alice:25:NYC
```

### **5. CSV Source Manual Test**
```SHELL
# Test 5: Create and test CSV processing
echo "name,age,city" > test_data.csv
echo "alice,25,nyc" >> test_data.csv
echo "bob,30,sf" >> test_data.csv

# In SWI-Prolog using consult(user):
cat << 'PROLOG' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').

:- source(csv, test_users, [csv_file('test_data.csv'), has_header(true)]).

test :-
    dynamic_source_compiler:dynamic_source_def(test_users/Arity, Type, Config),
    format('Registered: test_users/~w (type: ~w)~n', [Arity, Type]).
PROLOG

# Expected: Compiles without errors, source registered
```

### **6. Emoji Rendering Test** 🆕
```SHELL
# Test 6: Verify emoji rendering works across platforms
# Note: This test is redundant with Test 3 (already completed)
# You can skip this test if you already ran Test 3

swipl -l init.pl -g quick_test -t halt examples/quick_emoji_test.pl

# Expected: Displays emoji at appropriate level based on terminal
# - Windows: May use BMP or ASCII fallbacks
# - Linux/WSL/macOS: Full Unicode emoji

# ✅ FIXED: Windows emoji rendering issue resolved
# Non-BMP emoji (🚀📊📈) now display correctly on Windows
```

---

## **Priority 3: Security & Regression Tests** 🔒

### **7. Firewall Security Test**

#### **Test 7a: Basic Firewall Configuration**
```SHELL
# Test 7a: Verify firewall accepts configuration
cat << 'PROLOG' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/core/firewall').
:- use_module('src/unifyweaver/sources').

test :-
    assertz(firewall:firewall_default([denied([python3])])),
    source(python, blocked_test, [python_inline('print("test")')]),
    format('Firewall test: Source registered (guidance mode)~n', []).
PROLOG

# Expected output:
# Registered dynamic source: blocked_test/2 using python
# Defined source: blocked_test/2 using python
# Firewall test: Source registered (guidance mode)

# Note: denied([python3]) is guidance-based, not hard-blocking in v0.0.2
```

#### **Test 7b: Network Host Blocking (Interactive)**
```SHELL
# Test 7b: Verify firewall blocks unauthorized URLs
# This test requires interactive REPL

swipl -l init.pl
```

Inside the REPL:
```PROLOG
% Set up firewall to allow only specific hosts
?- assertz(firewall:firewall_default([network_hosts(['*.typicode.com', '*.github.com'])])).

% Load HTTP source module
?- use_module(library(unifyweaver/sources)).
?- use_module(library(unifyweaver/sources/http_source)).

% Test 1: Try blocked URL
?- source(http, blocked_api, [url('https://malicious.com')]).

% Expected: Firewall blocks it
% Firewall blocks network access to host: https://malicious.com
% Firewall validation failed for blocked_api/2. Compilation aborted.
% false.

% Test 2: Try allowed URL
?- source(http, allowed_api, [url('https://jsonplaceholder.typicode.com/posts')]).

% Expected: Allowed
% Registered dynamic source: allowed_api/2 using http
% Defined source: allowed_api/2 using http
% true.

?- halt.
```

**Expected behavior:**
- ✅ Blocked URL (`malicious.com`) is rejected with error message
- ✅ Allowed URL (`jsonplaceholder.typicode.com`) registers successfully
- ✅ Firewall validation happens before source registration
```

### **8. Backward Compatibility Test**
```bash
# Test 8: Ensure existing functionality still works
swipl << 'EOF'
[examples/fibonacci_fold].
[examples/binomial_fold].
halt.
EOF

# Expected output:
# true.
# true.
# (Returns true for each successful file load - no errors)

# ✅ VERIFIED: All existing examples load without errors
```

### **9. System Requirements Test**
```bash
# Test 9: Verify required tools are available
echo "=== System Requirements Check ==="
for tool in awk curl python3; do
    echo -n "$tool: "
    which $tool || echo "NOT FOUND"
done

# Optional tools (for full functionality)
for tool in jq sqlite3 wget; do
    echo -n "$tool (optional): "
    which $tool || echo "NOT FOUND"
done

# Expected: Core tools found
# jq: Required for JSON sources
# sqlite3: Required for SQLite queries
# wget: Alternative to curl for HTTP sources
```

---

## **Priority 4: Performance & Edge Cases** ⚡
*Optional but recommended for comprehensive validation*

### **10. Error Handling Tests**
- Test with non-existent CSV files
- Test with malformed JSON
- Test with unreachable URLs
- Test with invalid Python syntax
- Expected: Graceful error messages, no crashes

### **11. Large Data Test** (Optional)
- Create large CSV (1000+ rows) and test processing
- Test HTTP caching with repeated requests
- Expected: Reasonable performance, no memory issues

### **12. Edge Case Scenarios**
- CSV with embedded quotes and commas
- Python code with import restrictions
- HTTP requests with custom headers
- JSON with deeply nested structures
- Network timeout scenarios
- Cache expiration testing

### **13. Load Testing** (Optional)
- Multiple concurrent HTTP requests
- Large JSON file processing
- Extensive Python SQLite operations
- Memory usage monitoring
- Performance benchmarking

---

# 🚀 **Additional Pre-Release Recommendations**

## **Documentation Updates:**

### **1. Update VERSION File**
```bash
echo "0.0.2" > VERSION
```

### **2. Update CHANGELOG.md**
Add entry for v0.0.2 with feature summary:
- Native bash execution support (platform_detection.pl, bash_executor.pl)
- Windows emoji rendering fix
- PowerShell compatibility layer
- All data source types functional (CSV, JSON, HTTP, Python, SQLite)
- Zero warnings, zero deprecation issues

### **3. Update README.md**
Add section showcasing:
- New native execution capabilities
- Platform support matrix
- Data source examples

---

## **Release Checklist:**

### **Before Merge:**
- [x] All Priority 1 tests pass (critical functionality)
- [x] All Priority 2 tests pass (real-world scenarios)
- [x] Priority 3 tests pass (security & compatibility)
- [x] Demo runs successfully without errors
- [x] No firewall violations in normal usage
- [x] Backward compatibility verified with existing examples
- [x] Documentation updated (VERSION, CHANGELOG, README)
- [x] All singleton warnings fixed
- [x] All deprecation warnings fixed
- [x] Windows emoji rendering fixed
- [x] Native execution support added

### **After Merge:**
- [ ] Create GitHub release with tag v0.0.2
- [ ] Update project documentation
- [ ] Consider announcing new capabilities

### **Optional (Priority 4):**
- [ ] Edge case testing completed
- [ ] Performance testing satisfactory
- [ ] Load testing completed

---

## **New Features in v0.0.2:**

### **1. Native Bash Execution** 🆕
- Platform detection module (`platform_detection.pl`)
- Native bash executor (`bash_executor.pl`)
- Direct execution on Linux/WSL/Docker/macOS
- No compatibility layer needed on Linux-like platforms

### **2. Windows Emoji Fix** 🆕
- Fixed SWI-Prolog format/3 Unicode mangling on Windows
- Extract-and-pass workaround for non-BMP characters
- All source code uses Unicode escapes (`\U0001F680`)
- Comprehensive Unicode specification document

### **3. PowerShell Compatibility** 🆕
- `uw-*` command wrappers for Windows
- Backend switching (WSL/Cygwin)
- Seamless Windows integration

### **4. Integration Testing** 🆕
- Comprehensive integration test file
- Tests all data source types
- End-to-end ETL pipeline verification

---

## **Risk Assessment:**

### **Low Risk Items:** ✅
- New source plugins (self-contained architecture)
- Enhanced firewall (additive changes only)
- Test suite additions (no breaking changes)
- Platform detection (optional, non-breaking)
- Native execution (opt-in functionality)

### **Medium Risk Items:** ⚠️
- Emoji rendering changes (thoroughly tested)
- Template system integration under load
- Memory usage with large datasets

### **Monitor After Release:** 📊
- Error handling in production scenarios
- Performance with real-world data volumes
- Community feedback on new features
- Cross-platform emoji rendering edge cases

---

## **Test Execution Notes:**

**Priority 1 & 2 (Essential):** ~45 minutes
- These cover all critical functionality
- Must pass before any release consideration
- **New:** Includes platform detection and native execution tests

**Priority 3 (Important):** +30 minutes
- Covers security and regression testing
- Highly recommended before public release
- **New:** Includes emoji rendering verification

**Priority 4 (Comprehensive):** +60 minutes
- Edge cases and performance validation
- Can be deferred to post-release if time-constrained
- Valuable for production readiness assessment

---

## **Platform Testing Matrix:**

| Platform | Status | Native Execution | PowerShell Layer | Emoji Support |
|----------|--------|------------------|------------------|---------------|
| Linux | ✅ Tested | ✅ Yes | N/A | ✅ Full |
| WSL | ✅ Tested | ✅ Yes | ✅ Yes | ✅ Full |
| Docker | ✅ Tested | ✅ Yes | N/A | ✅ Full |
| macOS | ⚠️ Not Tested | ✅ Should Work | N/A | ✅ Should Work |
| Windows PowerShell | ⚠️ Partial | ❌ No | ✅ Yes | ✅ Yes (Fixed) |

**Note:** Windows PowerShell testing requires native Windows environment. All Linux-like platforms tested successfully.

---

**Final Note:** This updated test plan reflects the significant enhancements in v0.0.2, including native execution support, Windows emoji fixes, and comprehensive integration testing. The release is production-ready with all critical tests passing.
