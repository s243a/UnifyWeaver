# 🪟 **PowerShell Test Plan for UnifyWeaver v0.0.2 (Automated)**

**Date:** October 20, 2025 (Updated)
**Release:** v0.0.2 Data Source Plugin System + Native Execution Support
**Platform:** Windows PowerShell with WSL or Cygwin backend
**Status:** ✅ **Ready for Execution**

**Last Updated:** Added Windows emoji fix notes, platform detection information, and integration test guidance.

---

## **1. Overview**

This test plan provides a robust, automated method for testing UnifyWeaver on Windows. It leverages a **PowerShell Compatibility Layer** (`init_unify_compat.ps1`) that wraps native Linux commands, allowing them to be executed from a PowerShell terminal via either a WSL or Cygwin backend.

This approach allows us to follow the main WSL test plan almost identically, ensuring consistent testing across all environments.

---

## **2. Environment Setup**

### **2.1. Create Test Environment**

Before running any tests, create a fresh test environment:

1.a. **Navigate to the project directory in Windows Explorer**

1.b. **Open PowerShell at this location**
   - In Windows Explorer, click the file menu
   - Click the button that says "Open PowerShell window>"

1.c. **Navigate to the testing directory**
```POWERSHELL
cd scripts/testing
```

1.d. **Generate a new test environment** - it will be located at the path specified after the -p option
```POWERSHELL
./Init-TestEnvironment.ps1 -p test_env_ps1
```
* Directories in `scripts/testing/test_env*` are not tracked in git (see `.gitignore`)
* The `-p` option lets you specify the full path to the test environment
* The `-d` option lets you specify the directory containing the test environment (default name: `test_env`)

### **2.2. SWI-Prolog PATH Configuration**

After creating the test environment, you need to ensure `swipl` is available on your PATH. There are **three methods** to do this:

#### **Method 1: Use the Launcher Script (Recommended)**
The launcher script automatically sets up SWI-Prolog PATH and opens an enhanced terminal:

```POWERSHELL
cd test_env_ps1
.\start_test_env.bat
```

This opens a terminal with:
- ✅ SWI-Prolog on PATH (session only)
- ✅ UTF-8 encoding configured
- ✅ Compatibility layer ready to load

**Proceed to step 2.3** after the terminal opens.

#### **Method 2: Source the Compatibility Layer**
The compatibility layer automatically loads the SWI-Prolog environment:

```powershell
cd test_env_ps1
. .\scripts\init_unify_compat.ps1
```

This provides:
- ✅ SWI-Prolog on PATH (via `init_swipl_env.ps1`)
- ✅ Unix command wrappers (`uw-ls`, `uw-grep`, etc.)
- ✅ UTF-8 encoding

**Skip to step 3** - the compatibility layer is already loaded.

#### **Method 3: Manual Environment Setup**
Source the SWI-Prolog environment script directly:

```powershell
cd test_env_ps1
. .\scripts\init_swipl_env.ps1
```

This provides:
- ✅ SWI-Prolog on PATH (session only)
- ✅ UTF-8 encoding configured
- ❌ No Unix command wrappers (use Method 2 if you need them)

**Proceed to step 2.3** to load the compatibility layer if needed.

> **Note:** See `scripts/testing/README_SWIPL_ENV.md` for detailed documentation on the environment setup scripts.

### **2.3. Load Compatibility Layer (If Not Already Loaded)**

If you used Method 1 or Method 3, load the compatibility layer now:

```powershell
. .\scripts\init_unify_compat.ps1
```

You should see: `"UnifyWeaver compatibility layer loaded. Backend mode: cygwin"`

### **2.4. Verify Setup**

Run the compatibility layer smoke test:

```powershell
.\test_compat_layer.ps1
```

This verifies:
- Both backends (WSL and Cygwin) are accessible
- Basic pipeline support (`uw-ls | uw-grep`) works

### **2.5. Choose Your Backend (Optional)**

Set the environment variable to select your execution engine. **This only needs to be done once per session.** The default is `cygwin` because the PowerShell test plan is primarily intended for Windows testing, and arrow keys don't work in the WSL version of swipl (due to it not being compiled with Linux terminal libraries like readline).

**For WSL:**
```powershell
$env:UNIFYWEAVER_EXEC_MODE = 'wsl'
```

**For Cygwin (Default):**
```powershell
$env:UNIFYWEAVER_EXEC_MODE = 'cygwin'
```

---

## **3.1 Compatibility Layer Command Wrappers**

After sourcing `init_unify_compat.ps1`, the following helpers are available in your PowerShell session:

| Wrapper      | Backend Command          | Notes                                      |
| ------------ | ------------------------ | ------------------------------------------ |
| `uw-uname`   | `uname`                  | Backend detection                          |
| `uw-ls`      | `ls -1`                  | Deterministic listing (good for pipes)     |
| `uw-grep`    | `grep --color=never`     | Accepts pipeline input, exit 0/1 permitted |
| `uw-sqlite3` | `sqlite3`                | Runs SQLite queries through backend        |
| `uw-curl`    | `curl`                   | Full curl support                          |
| `uw-jq`      | `jq`                     | JSON parsing (requires jq installed)       |
| `uw-run`     | Arbitrary command string | Generic runner if a wrapper is missing     |

All `uw-*` filters consume PowerShell pipeline input and send it to the chosen backend (WSL or Cygwin). Use these wrappers instead of raw `bash` / `wsl` invocations in the tests below.

## **3. Test Execution**

This plan mirrors the structure of the main `v0_0_2_pre_release_test_plan.md`. The commands have been adapted for the PowerShell environment.

### **Priority 1 & 2: Core Functionality & Demos**

These tests correspond to **Tests 1-6** in the main plan.

#### **Test A: Prolog Module Loading**
*Verifies that the Prolog source files themselves are valid and load on Windows.*
```powershell
# This test does not require the compatibility layer.
swipl -g "use_module('src/unifyweaver/sources/csv_source')" -t halt
swipl -g "use_module('src/unifyweaver/sources/python_source')" -t halt
swipl -g "use_module('src/unifyweaver/sources/http_source')" -t halt
swipl -g "use_module('src/unifyweaver/sources/json_source')" -t halt

# Expected: All commands complete with `true` and no errors or warnings
# ✅ FIXED: No more singleton variable warnings
# ✅ FIXED: No more deprecation warnings
```

#### **Test B: Integration Test Execution**
*Runs the comprehensive integration test that verifies all data sources and creates a SQLite database.*

**Note:** This test now works on PowerShell with WSL/Cygwin! The platform detection has been fixed to properly execute bash scripts via WSL.

```powershell
# Run the integration test (requires init.pl for module paths)
$env:KEEP_TEST_DATA = 'true'
swipl -l init.pl -g main -t halt examples/integration_test.pl
```

**Expected Behavior:**
- Platform detection shows `windows` with `powershell_wsl` execution mode
- CSV, JSON, and Python sources compile successfully
- Bash scripts execute via WSL automatically
- Python source creates and populates `test_output/analysis.db`
- Generated bash scripts are created in `test_output/`
- Emoji display correctly in PowerShell
- Test completes with all checkmarks ✅

**Note:** The integration test will automatically use WSL to execute bash scripts. If you have Cygwin instead, set `$env:UNIFYWEAVER_EXEC_MODE = 'cygwin'` before running.

#### **Test C: Verify Integration Test Output**
*Uses the compatibility layer to inspect the database created by Test B.*

```powershell
# Make sure the PowerShell compatibility layer is sourced
. .\scripts\init_unify_compat.ps1

# List the generated files
uw-ls test_output

# Expected: Should list analysis.db and generated .sh scripts

# Query the database
"SELECT * FROM order_summary ORDER BY total_quantity DESC;" | uw-sqlite3 test_output/analysis.db

# Expected: Product order summaries (e.g., Laptop: 2, Mouse: 5, Desk: 1)
```

**Note:** This test verifies that the SQLite database was created correctly by the integration test in Test B.

#### **Test D: Compatibility Layer Flexibility**
*Verify the backend toggle using the provided wrappers.*

**Note:** This test can only be run from native Windows PowerShell, not from WSL or Docker. If you're running in a Linux environment (WSL, Docker, native Linux), this test does not apply - native bash execution is used instead.

```powershell
# Test 1: WSL backend
$env:UNIFYWEAVER_EXEC_MODE = 'wsl'
uw-uname -o
# Expected Output: GNU/Linux

# Test 2: Cygwin backend (if installed)
$env:UNIFYWEAVER_EXEC_MODE = 'cygwin'
uw-uname -o
# Expected Output: Cygwin

# Note: If running from WSL/Linux instead of Windows PowerShell:
# - This test is not applicable
# - Native bash execution is used instead (see platform_detection.pl)
# - You can verify native execution with:
#   swipl -g "use_module('src/unifyweaver/core/bash_executor')" \
#        -g "bash_executor:test_bash_executor" -t halt
```

### **Priority 3: Security & Regression**

These tests correspond to **Tests 7-9** in the main plan.

#### **Test E1: Firewall Security Test**
*This test is performed inside the Prolog REPL.*
```powershell
swipl -l init.pl

# --- Inside the SWI-Prolog REPL ---
?- use_module(library(unifyweaver/core/firewall)).
?- assertz(firewall:firewall_default([denied([python3])])).
?- use_module(library(unifyweaver/sources)).
?- source(python, blocked_test, [python_inline('print("test")')]).
?- halt.
```
**Expected Behavior:**
- The source registers successfully, as the firewall is currently for guidance, not hard-blocking.

**Test E2: Blocked and allowed URLs**

First start prolog w/ init.pl

```POWERSHELL
swipl -l init.pl
```

```PROLOG
1 ?- assertz(firewall:firewall_default([network_hosts(['*.typicode.com', '*.github.com'])])).
2 ?- use_module(library(unifyweaver/sources)).
3 ?- use_module(library(unifyweaver/sources/http_source)).
4 ?- use_module(library(unifyweaver/core/dynamic_source_compiler)).
5 ?- source(http, blocked_api, [url('https://malicious.com')]).
```

Should see:
```text
Firewall blocks network access to host: https://malicious.com
Firewall validation failed for blocked_api/2. Compilation aborted.
```
```PROLOG
8 ?- source(http, allowed_api, [url('https://jsonplaceholder.typicode.com/posts')]).
```
Should see:
```text
9 ?- compile_dynamic_source(allowed_api/2, [], Code).
Compiling dynamic source: allowed_api/2 using http
... # You'll actually see the compiled code here.
```
```prolog

```
#### **Test F: Backward Compatibility**
*Ensures old examples still load.*
```powershell
swipl -l init.pl

# --- Inside the SWI-Prolog REPL ---
?- [examples/fibonacci_fold].
?- [examples/binomial_fold].
?- halt.
```
**Expected Behavior:**
- Both files load with `true` and no errors.

#### **Test G: System Requirements**
*Uses the compatibility layer to check for tools in the selected backend.*
```powershell
# Check for required tools via the execution backend
uw-which awk
uw-which curl
uw-which jq
uw-which python3
uw-which sqlite3
```
**Expected Behavior:**
- The paths to the tools within the WSL or Cygwin environment should be displayed.
- **Required for full functionality:**
  - jq (for JSON sources)
  - sqlite3 (for SQLite queries)
- Install missing tools in WSL: `sudo apt install jq sqlite3`

---

## **4. New Features in v0.0.2** 🆕

### **Platform Detection & Native Execution**
UnifyWeaver now includes platform detection and native bash execution support. This means:

- **On Windows PowerShell:** Use the `uw-*` wrappers (this test plan)
- **On WSL/Linux/macOS:** Native bash execution is available
- **Platform auto-detection:** UnifyWeaver automatically detects the environment

**Testing Native Execution (Optional):**
If you want to test the native execution module from PowerShell (via WSL), you can run:
```powershell
swipl -g "use_module('src/unifyweaver/core/platform_detection')" -g "platform_detection:test_platform_detection" -t halt
swipl -g "use_module('src/unifyweaver/core/bash_executor')" -g "bash_executor:test_bash_executor" -t halt
```

### **Windows Emoji Fix**
Emoji now render correctly on Windows PowerShell. The fix involved:
- Using Unicode escapes (`\U0001F680`) instead of literal emoji in source code
- Modified `safe_format/3` to extract and pass emoji as arguments
- Comprehensive Unicode specification document created

**Expected Emoji Output:**
- BMP emoji (✅ ❌ ⚠ ℹ): Should display correctly
- Non-BMP emoji (🚀 📊 📈 💾): Should display correctly (fixed!)

### **Integration Testing**
A comprehensive integration test is now available:
```powershell
# Note: Requires jq and sqlite3 installed in WSL
swipl -g main -t halt examples/integration_test.pl
```
This test verifies:
- Platform detection
- All data source types (CSV, JSON, Python, SQLite)
- Complete ETL pipeline
- Emoji rendering
- Native bash execution

---

## **5. Summary**

This updated plan provides a comprehensive and automated way to validate UnifyWeaver on Windows. By abstracting the execution backend, it allows for consistent, powerful testing directly from PowerShell, fulfilling the same test objectives as the main test plan.

**Key Updates in This Revision:**
- Added notes about Windows emoji fix (now working!)
- Added platform detection and native execution information
- Added integration test guidance
- Added notes about Linux vs. Windows PowerShell environments
- Updated system requirements to include jq and sqlite3
