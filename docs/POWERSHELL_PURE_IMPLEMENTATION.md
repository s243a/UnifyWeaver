# Pure PowerShell Implementation

**Status:** Implemented (Phase 1)
**Version:** 1.0.0
**Date:** 2025-10-26
**Branch:** powershell-target-language

---

## Overview

This document describes the pure PowerShell implementation for UnifyWeaver's PowerShell target language feature. Pure PowerShell mode generates native PowerShell code with no bash dependency, providing better integration with Windows environments and satisfying firewall restrictions.

## What Was Implemented

### Implemented Features (Pure PowerShell)

The following data sources have **pure PowerShell templates** that use native PowerShell cmdlets:

#### 1. CSV Sources ✅
- **PowerShell Cmdlet:** `Import-Csv`
- **Why Pure PS is Better:** Native CSV parsing, automatic header detection, type inference
- **Templates:**
  - `csv_source_unary_powershell_pure` - Arity 1 predicates
  - `csv_source_binary_plus_powershell_pure` - Arity 2+ predicates

**Example:**
```powershell
function user {
    param([string]$Key)
    $data = Import-Csv -Path 'users.csv'
    # ... filtering logic ...
}
```

#### 2. JSON Sources ✅
- **PowerShell Cmdlet:** `ConvertFrom-Json`
- **Why Pure PS is Better:** Native JSON parsing, object pipeline support
- **Templates:**
  - `json_file_source_powershell_pure` - File-based JSON
  - `json_stdin_source_powershell_pure` - Piped JSON data

**Example:**
```powershell
function product {
    param([string]$Key)
    $data = Get-Content 'products.json' | ConvertFrom-Json
    # ... filtering logic ...
}
```

#### 3. HTTP Sources ✅
- **PowerShell Cmdlet:** `Invoke-RestMethod`
- **Why Pure PS is Better:** Native HTTP client, automatic JSON parsing, header support
- **Templates:**
  - `http_source_powershell_pure` - HTTP API calls

**Example:**
```powershell
function api_user {
    param([string]$Key)
    $response = Invoke-RestMethod -Uri 'https://api.example.com/users'
    # ... filtering logic ...
}
```

---

## What Was NOT Implemented (Future Work)

The following features still use **Bash-as-a-Service** (BaaS) mode and will be implemented in future phases:

### Deferred to Future (BaaS Only)

#### AWK Sources
- **Reason:** Complex AWK patterns difficult to replicate in PowerShell
- **Future:** Possible PowerShell equivalents using `-split`, regex, `Select-String`

####  Python Sources
- **Reason:** Cannot execute Python from pure PowerShell without subprocess
- **Future:** Remains BaaS-only

#### Simple Facts/Relations
- **Reason:** Not a priority (works fine with BaaS)
- **Future:** Phase 2b - native PowerShell arrays and hashtables

#### Joins/Recursion
- **Reason:** More complex, needs careful PowerShell design
- **Future:** Phase 2b - PowerShell object-based joins

---

## How It Works

### Automatic Mode Selection

The PowerShell compiler automatically selects between pure PowerShell and BaaS based on the data source type:

```prolog
% Automatically uses pure PowerShell for CSV
:- dynamic_source(user/2, external(csv, 'users.csv')).
?- compile_to_powershell(user/2, Code).
% Generates pure PowerShell using Import-Csv

% Automatically uses BaaS for AWK
:- dynamic_source(log/3, external(awk, "awk '{print $1,$2,$3}' app.log")).
?- compile_to_powershell(log/3, Code).
% Generates BaaS wrapper (uw-bash + AWK)
```

### Manual Mode Selection

Users can force a specific mode:

```prolog
% Force pure PowerShell (will fail if not supported)
?- compile_to_powershell(user/2, [powershell_mode(pure)], Code).

% Force BaaS (even for CSV/JSON/HTTP)
?- compile_to_powershell(user/2, [powershell_mode(baas)], Code).

% Auto-detect (default)
?- compile_to_powershell(user/2, [powershell_mode(auto)], Code).
```

### Implementation Architecture

```
compile_to_powershell(Predicate, Options, Code)
    |
    +--> Determine mode: pure, baas, or auto
    |
    +--> Check if pure mode supported
    |       |
    |       +--> supports_pure_powershell/2
    |              - Checks for source_type(csv|json|http)
    |
    +--> If pure supported:
    |       +--> compile_to_pure_powershell/3
    |              +--> Adds template_suffix('_powershell_pure')
    |              +--> Calls dynamic_source_compiler
    |                     +--> CSV/JSON/HTTP source plugins
    |                            +--> render_named_template with suffix
    |                                   +--> Pure PowerShell templates
    |
    +--> Else (BaaS fallback):
            +--> compile_to_baas_powershell/3
                   +--> Generate bash code
                   +--> Wrap in powershell_wrapper
                          +--> uw-bash compatibility layer
```

---

## Benefits of Pure PowerShell

### 1. No Bash Dependency
- Works in restricted Windows environments
- No WSL/Cygwin/Git Bash required
- Firewall-compliant

### 2. Native PowerShell Integration
- Uses built-in cmdlets (`Import-Csv`, `Invoke-RestMethod`)
- Better error handling
- PowerShell object pipeline support (future)

### 3. Performance
- No extra process invocation overhead
- Direct .NET API access
- Faster for small-to-medium datasets

### 4. Windows-First Design
- Leverages PowerShell strengths
- Better Windows path handling
- Integration with Windows APIs

---

## Mixed-Mode Pipelines

When a Prolog program uses both pure-supported and BaaS-only sources, the generated code will be **mixed**:

**Prolog:**
```prolog
:- dynamic_source(user/2, external(csv, 'users.csv')).        % Pure PS
:- dynamic_source(log/3, external(awk, "awk '...' app.log")). % BaaS

process_data(User, LogEntry) :-
    user(UserId, UserName),
    log(UserId, Timestamp, LogEntry).
```

**Generated PowerShell:**
```powershell
# user/2 - Pure PowerShell
function user {
    param([string]$Key)
    $data = Import-Csv -Path 'users.csv'
    # ... pure PowerShell ...
}

# log/3 - BaaS
$bashScript = @'
awk '{print $1 ":" $2 ":" $3}' app.log
'@
function log {
    uw-bash -c $bashScript
}

# process_data/2 - Mixes both
function process_data {
    $users = user
    $logs = log
    # ... join logic ...
}
```

**Result:** The system handles this automatically based on what each source type supports.

---

## Files Modified

### Source Plugins (Added Pure Templates)

1. **`src/unifyweaver/sources/csv_source.pl`**
   - Added `csv_source_unary_powershell_pure` template
   - Added `csv_source_binary_plus_powershell_pure` template
   - Updated `generate_csv_bash/10` to accept template_suffix option

2. **`src/unifyweaver/sources/json_source.pl`**
   - Added `json_file_source_powershell_pure` template
   - Added `json_stdin_source_powershell_pure` template

3. **`src/unifyweaver/sources/http_source.pl`**
   - Added `http_source_powershell_pure` template

### Compiler (Mode Selection Logic)

4. **`src/unifyweaver/core/powershell_compiler.pl`**
   - Added `powershell_mode(baas|pure|auto)` option
   - Added `supports_pure_powershell/2` - checks source type
   - Added `compile_to_pure_powershell/3` - generates pure code
   - Renamed existing logic to `compile_to_baas_powershell/3`
   - Mode selection happens in `compile_to_powershell/3`

---

## Usage Examples

### Example 1: CSV with Pure PowerShell

**Prolog:**
```prolog
:- dynamic_source(employee/3, external(csv, 'employees.csv')).

?- compile_to_powershell(employee/3, [output_file('employee.ps1')], _).
```

**Generated (employee.ps1):**
```powershell
function employee {
    param([string]$Key)

    $data = Import-Csv -Path 'employees.csv'

    if ($Key) {
        $keyColumn = $data[0].PSObject.Properties.Name[0]
        $matches = $data | Where-Object { $_.$keyColumn -eq $Key }
        foreach ($row in $matches) {
            $values = $row.PSObject.Properties | ForEach-Object { $_.Value }
            $values -join ":"
        }
    } else {
        foreach ($row in $data) {
            $values = $row.PSObject.Properties | ForEach-Object { $_.Value }
            $values -join ":"
        }
    }
}
```

**Run:**
```powershell
. .\employee.ps1
employee          # List all employees
employee "E001"   # Lookup employee E001
```

### Example 2: JSON API with Pure PowerShell

**Prolog:**
```prolog
:- dynamic_source(github_repo/2, external(http,
    'https://api.github.com/users/octocat/repos'
)).

?- compile_to_powershell(github_repo/2, [output_file('github.ps1')], _).
```

**Generated (github.ps1):**
```powershell
function github_repo {
    param([string]$Key)

    try {
        $response = Invoke-RestMethod -Uri 'https://api.github.com/users/octocat/repos'

        foreach ($item in $response) {
            $values = @($item.id, $item.name)
            $values -join ":"
        }
    }
    catch {
        Write-Error "HTTP request failed: $_"
    }
}
```

**Run:**
```powershell
. .\github.ps1
github_repo  # List all repos
```

---

## Testing

### Manual Testing

```prolog
% Load PowerShell compiler
?- [unifyweaver(core/powershell_compiler)].

% Test pure mode
?- compile_to_powershell(test_pred/2, [
    source_type(csv),
    csv_file('test.csv'),
    powershell_mode(pure)
], Code),
   write(Code).

% Test auto mode (should detect CSV and use pure)
?- compile_to_powershell(test_pred/2, [
    source_type(csv),
    csv_file('test.csv')
], Code),
   sub_string(Code, _, _, _, 'Import-Csv').  % Should succeed
```

### Integration Testing

Create test CSV/JSON files and verify generated PowerShell works:

```bash
# Generate test script
swipl -g "compile_to_powershell(user/2, [source_type(csv), csv_file('test.csv'), output_file('test.ps1')], _)" -t halt

# Run in PowerShell
pwsh -File test.ps1
```

---

## Future Work

### Phase 2b: Advanced Pure PowerShell (v0.0.6)

- [ ] Simple facts (arrays)
- [ ] Binary relations (hashtables, PSCustomObject arrays)
- [ ] Joins (nested loops with Where-Object)
- [ ] Recursion (native PowerShell recursion)
- [ ] SQL sources (.NET System.Data.SQLite)

### Phase 2c: PowerShell Object Pipeline (v0.0.7)

- [ ] Return PowerShell objects instead of colon-separated strings
- [ ] Support PowerShell pipeline chaining
- [ ] Type annotations for better IntelliSense

### Phase 3: Firewall Mode

- [ ] Firewall detection logic
- [ ] Enforce pure mode when firewall detected
- [ ] Clear error messages when pure mode unsupported

---

## Rationale

### Why Only CSV/JSON/HTTP?

These three source types have **superior** native PowerShell support compared to bash:

| Feature | Bash | PowerShell | Winner |
|---------|------|------------|--------|
| CSV parsing | AWK field splitting | `Import-Csv` (automatic headers, types) | **PowerShell** |
| JSON parsing | `jq` (external tool) | `ConvertFrom-Json` (built-in) | **PowerShell** |
| HTTP requests | `curl` (external tool) | `Invoke-RestMethod` (built-in, auto-parse) | **PowerShell** |
| AWK text processing | Native AWK | Complex regex + `-split` | **Bash** |
| Python integration | `python` (subprocess) | `python` (subprocess) | Tie (BaaS) |

**Decision:** Implement pure PowerShell where PowerShell is better or equal. Keep BaaS where bash/AWK is superior.

### Why Not Everything?

- **AWK:** PowerShell regex is verbose, AWK is concise and fast
- **Python:** No advantage to pure PowerShell (both need subprocess)
- **Simple facts:** BaaS works fine, not a priority
- **Joins:** Need more design work for efficient PowerShell implementation

### Automatic Mixed-Mode

By allowing both pure and BaaS in the same pipeline, users get:
- **Best of both worlds:** Pure PowerShell where beneficial, bash where needed
- **No breaking changes:** Existing BaaS code still works
- **Gradual migration:** Can move to pure incrementally

---

## Conclusion

**What we built:**
- Pure PowerShell templates for CSV, JSON, HTTP sources
- Automatic mode selection (pure vs BaaS)
- Mixed-mode pipeline support
- No breaking changes to existing BaaS functionality

**What users get:**
- Native PowerShell code for data sources (better integration)
- No bash dependency for CSV/JSON/HTTP processing
- Firewall-compliant scripts (when using pure mode)
- Automatic fallback to BaaS for unsupported features

**What's next:**
- Phase 2b: Extend pure mode to facts, relations, joins
- Phase 2c: PowerShell object pipeline
- Phase 3: Firewall mode enforcement

---

**Created:** 2025-10-26
**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
