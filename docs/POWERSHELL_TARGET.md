# PowerShell as a Target Language

**Status:** Implemented (Phases 1-11 Complete)
**Version:** 2.5.0
**Date:** 2025-12-10

---

## Overview

UnifyWeaver compiles Prolog predicates to PowerShell scripts with two compilation modes:

1. **Pure PowerShell** - Native PowerShell code with no bash dependency
2. **Bash-as-a-Service (BaaS)** - Bash scripts wrapped in PowerShell compatibility layer

### Key Features (v2.5.0)

| Feature | Description |
|---------|-------------|
| **Pure PowerShell** | Native PowerShell for CSV, JSON, HTTP, XML, .NET sources |
| **Binding System** | 68+ bindings for cmdlets, .NET methods, Windows automation |
| **Object Pipeline** | `ValueFromPipeline` parameters, `PSCustomObject` output |
| **Advanced Joins** | Hash-based and pipelined N-way joins with O(n+m) complexity |
| **Outer Joins** | LEFT, RIGHT, and FULL OUTER joins with automatic pattern detection |
| **Recursion** | Simple, transitive, mutual, tail recursion with memoization |
| **Firewall Security** | Per-predicate mode control (pure/baas/auto) |
| **C# Hosting** | In-process integration with .NET assemblies |

For detailed implementation status, see [POWERSHELL_PURE_IMPLEMENTATION.md](POWERSHELL_PURE_IMPLEMENTATION.md).

---

## Quick Start

### 1. Compile a Simple Predicate

```prolog
:- use_module(unifyweaver(core/powershell_compiler)).

?- compile_to_powershell(color/1, Code).
% Generates PowerShell script that wraps bash implementation
```

### 2. Save to File

```prolog
?- compile_to_powershell(
    grandparent/2,
    [output_file('grandparent.ps1')],
    _
).
% Creates grandparent.ps1
```

### 3. Run the Generated Script

```powershell
# First, load the compatibility layer
. .\scripts\init_unify_compat.ps1

# Then run the generated script
.\grandparent.ps1
```

---

## Outer Join Support

The PowerShell target supports LEFT, RIGHT, and FULL OUTER joins through automatic pattern detection.

### Syntax

```prolog
% LEFT JOIN - all left records, matched right or null
left_join(X, Z) :-
    left_table(X, Y),
    (right_table(Y, Z) ; Z = null).

% RIGHT JOIN - all right records, matched left or null
right_join(X, Z) :-
    (left_table(X, Y) ; X = null),
    right_table(Y, Z).

% FULL OUTER JOIN - all records from both sides
full_outer(X, Z) :-
    (left_table(X, Y) ; X = null),
    (right_table(Y, Z) ; Z = null).
```

### Implementation

The compiler generates PowerShell code using:
- Hashtable-based lookups for efficient joins
- `PSCustomObject` output with nullable properties
- Match tracking for unmatched records in FULL OUTER joins

Example generated output structure:
```powershell
[PSCustomObject]@{
    X = $left.X
    Z = if ($right) { $right.Z } else { $null }
}
```

---

## API Reference

### Main Predicates

#### `compile_to_powershell/2`

**Signature:**
```prolog
compile_to_powershell(+Predicate, -PowerShellCode)
```

**Description:** Compiles a Prolog predicate to PowerShell code with default options.

**Example:**
```prolog
?- compile_to_powershell(color/1, Code).
Code = "# Generated PowerShell Script\n...".
```

---

#### `compile_to_powershell/3`

**Signature:**
```prolog
compile_to_powershell(+Predicate, +Options, -PowerShellCode)
```

**Description:** Compiles a Prolog predicate to PowerShell code with custom options.

**Parameters:**
- `Predicate`: Pred/Arity indicator (e.g., `grandparent/2`)
- `Options`: List of compilation options (see below)
- `PowerShellCode`: Generated PowerShell script as string

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `powershell_mode(pure\|baas\|auto)` | Compilation mode | `auto` |
| `output_format(object\|text)` | Output type | `text` |
| `pipeline_input(true\|false)` | Enable ValueFromPipeline | `false` |
| `arg_names([...])` | Custom property names | `['X', 'Y', 'Z']` |
| `cmdlet_binding(true\|false)` | Generate [CmdletBinding()] | `false` |
| `output_file(Path)` | Write to file | — |
| `verbose_output(true\|false)` | Add Write-Verbose | `false` |
| `wrapper_style(inline\|tempfile)` | BaaS wrapper style | `inline` |
| `compat_check(true\|false)` | Add compat layer check | `true` |

**Examples:**
```prolog
% Pure PowerShell with object output
?- compile_to_powershell(user/2, [
    powershell_mode(pure),
    output_format(object)
], Code).

% Pipeline-enabled function with custom property names
?- compile_to_powershell(employee/2, [
    pipeline_input(true),
    output_format(object),
    arg_names(['EmployeeId', 'Name'])
], Code).

% Write to file with verbose output
?- compile_to_powershell(
    grandparent/2,
    [output_file('grandparent.ps1'), verbose_output(true)],
    _
).

% Force BaaS mode (uses bash subprocess)
?- compile_to_powershell(log_parser/3, [powershell_mode(baas)], Code).
```

---

### Wrapper Styles

#### Inline (Default)

**Pros:**
- Simple and clean
- No filesystem operations
- Bash code embedded in PowerShell as heredoc

**Cons:**
- Large bash scripts make PowerShell file large

**Example Output:**
```powershell
$bashScript = @'
#!/bin/bash
# bash implementation here
'@

uw-bash -c $bashScript
```

#### Tempfile

**Pros:**
- Better for very large bash scripts
- Easier debugging (temp file can be inspected)

**Cons:**
- Requires filesystem write permission
- Cleanup code needed

**Example Output:**
```powershell
$tempFile = [System.IO.Path]::GetTempFileName() + ".sh"
$bashScript = @'
#!/bin/bash
# bash implementation here
'@

try {
    Set-Content -Path $tempFile -Value $bashScript -Encoding UTF8
    uw-bash $tempFile
} finally {
    Remove-Item $tempFile -ErrorAction SilentlyContinue
}
```

---

## Generated Script Structure

### Typical Generated PowerShell Script

```powershell
# Generated PowerShell Script
# Script: grandparent.ps1
# Generated by UnifyWeaver PowerShell Compiler
# Generated: 2025-10-19 22:00:00
#
# This script wraps a bash implementation via the PowerShell compatibility layer.
# The bash code is executed using uw-bash from UnifyWeaver compatibility layer.

# Ensure compatibility layer is available
if (-not (Get-Command uw-bash -ErrorAction SilentlyContinue)) {
    Write-Error "UnifyWeaver PowerShell compatibility layer not loaded"
    Write-Host "Please run: . .\scripts\init_unify_compat.ps1" -ForegroundColor Yellow
    exit 1
}

# Embedded bash implementation
$bashScript = @'
#!/bin/bash
# Bash code generated by stream_compiler
awk 'BEGIN {
    # AWK implementation of grandparent/2
    ...
}' "$@"
'@

# Execute via compatibility layer
if ($Input) {
    $Input | uw-bash -c $bashScript
} else {
    uw-bash -c $bashScript
}
```

---

## Usage Examples

### Example 1: Compile Facts

**Prolog:**
```prolog
% Define facts
color(red).
color(green).
color(blue).

% Compile to PowerShell
?- compile_to_powershell(color/1, [output_file('colors.ps1')], _).
```

**Generated PowerShell:**
```powershell
# colors.ps1
# (wrapper code omitted for brevity)

$bashScript = @'
#!/bin/bash
awk 'BEGIN {
    print "red"
    print "green"
    print "blue"
}'
'@

uw-bash -c $bashScript
```

**Usage:**
```powershell
. .\scripts\init_unify_compat.ps1
.\colors.ps1
# Output:
# red
# green
# blue
```

---

### Example 2: Compile Join Query

**Prolog:**
```prolog
% Define relationships
parent(tom, bob).
parent(bob, ann).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% Compile to PowerShell
?- compile_to_powershell(grandparent/2, [output_file('grandparent.ps1')], _).
```

**Generated PowerShell:**
```powershell
# grandparent.ps1
# (includes join logic via AWK)

$bashScript = @'
#!/bin/bash
awk 'BEGIN {
    # parent/2 facts
    parent["tom","bob"] = 1
    parent["bob","ann"] = 1

    # Join logic
    for (key in parent) {
        split(key, xy, SUBSEP)
        x = xy[1]; y = xy[2]
        for (key2 in parent) {
            split(key2, yz, SUBSEP)
            if (yz[1] == y) {
                z = yz[2]
                print x "\t" z
            }
        }
    }
}'
'@

uw-bash -c $bashScript
```

**Usage:**
```powershell
.\grandparent.ps1
# Output:
# tom    ann
```

---

### Example 3: Pipeline Support

**Generated scripts support PowerShell pipelines:**

```powershell
# Read data from file and process
Get-Content users.csv | .\process_users.ps1

# Chain multiple scripts
.\get_users.ps1 | .\filter_active.ps1 | .\format_output.ps1
```

---

## Requirements

### Runtime Requirements

1. **PowerShell 5.1 or later**
   - Windows PowerShell 5.1 (built-in on Windows 10+)
   - PowerShell 7+ recommended for better cross-platform support

2. **UnifyWeaver PowerShell Compatibility Layer**
   - Located in `scripts/powershell-compat/`
   - Must be loaded before running generated scripts

3. **Bash (via compatibility layer)**
   - Git Bash, WSL, or Cygwin
   - Automatically invoked by compatibility layer

### Development Requirements

1. **SWI-Prolog 8.0 or later**
2. **UnifyWeaver core modules**
   - `stream_compiler` or `recursive_compiler`
   - `powershell_compiler` (this module)

---

## How to Use Generated Scripts

### Method 1: Manual Compatibility Layer Load

```powershell
# In PowerShell, one-time per session:
. .\scripts\init_unify_compat.ps1

# Now you can run generated scripts:
.\colors.ps1
.\grandparent.ps1
```

### Method 2: Auto-Load in Profile

Add to your PowerShell profile (`$PROFILE`):

```powershell
# Load UnifyWeaver compatibility layer
$UnifyWeaverPath = "C:\path\to\UnifyWeaver"
if (Test-Path "$UnifyWeaverPath\scripts\init_unify_compat.ps1") {
    . "$UnifyWeaverPath\scripts\init_unify_compat.ps1"
}
```

### Method 3: Disable Compat Check (Not Recommended)

If you're sure the compatibility layer is always available:

```prolog
?- compile_to_powershell(color/1, [compat_check(false)], Code).
```

This removes the compatibility check from generated scripts, but they'll fail if `uw-bash` is not available.

---

## Comparison: PowerShell vs Bash Output

### Same Predicate, Two Targets

**Prolog:**
```prolog
color(red).
color(green).
color(blue).
```

**Bash Output:**
```bash
#!/bin/bash
awk 'BEGIN {
    print "red"
    print "green"
    print "blue"
}'
```

**PowerShell Output:**
```powershell
# Generated PowerShell Script
# (wrapper code)

$bashScript = @'
#!/bin/bash
awk 'BEGIN {
    print "red"
    print "green"
    print "blue"
}'
'@

uw-bash -c $bashScript
```

**Key Difference:** PowerShell version wraps the same bash code in a PowerShell compatibility layer invocation.

**Result:** Identical output, different invocation mechanism.

---

## Performance Considerations

### Overhead

PowerShell scripts have a small overhead compared to direct bash execution:

```
Direct Bash:
  Prolog → Bash → AWK → Output

PowerShell (Phase 1):
  Prolog → PowerShell → uw-bash → Bash → AWK → Output
  (Extra layer: PowerShell compatibility wrapper)
```

**Impact:**
- Minimal for most use cases (< 100ms overhead)
- Negligible for data processing tasks
- More noticeable for many small script invocations

**Mitigation:**
- Use bash target for performance-critical code
- Phase 2 will eliminate this overhead with native PowerShell generation

---

## Limitations (Phase 1)

### Current Limitations

❌ **Not True Native PowerShell**
- Generated scripts call bash, not native PowerShell cmdlets
- Cannot directly integrate with PowerShell objects

❌ **No PowerShell Object Pipeline**
- Data passes as text, not as PowerShell objects
- Future enhancement for Phase 2

❌ **Requires Bash**
- Must have bash available via compatibility layer
- Not purely PowerShell-only

❌ **Performance Overhead**
- Extra layer of indirection (PowerShell → bash)

### What Works Well

✅ **Text Processing**
- AWK-based data processing
- File I/O
- String manipulation

✅ **PowerShell Integration**
- Can be called from PowerShell scripts
- Supports PowerShell pipelines (text-based)
- Works with PowerShell variables and control flow

✅ **Cross-Platform**
- Same code works on Windows, WSL, Linux (with PowerShell Core)

---

## Troubleshooting

### Error: "UnifyWeaver PowerShell compatibility layer not loaded"

**Solution:**
```powershell
. .\scripts\init_unify_compat.ps1
```

### Error: "uw-bash: command not found"

**Cause:** Compatibility layer not properly initialized

**Solution:**
1. Verify `scripts/powershell-compat/init_unify_compat.ps1` exists
2. Load it: `. .\scripts\init_unify_compat.ps1`
3. Check bash is available: `bash --version`

### Generated Script Produces No Output

**Possible Causes:**
1. Input data not provided correctly
2. Bash script has errors
3. AWK logic issue

**Debugging:**
```powershell
# Check if bash code works directly
$bashScript = @'
# ... paste bash code here ...
'@

# Run with uw-bash for debugging
uw-bash -c $bashScript
```

### Permission Denied When Running Script

**Solution:**
```powershell
# Set execution policy (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or bypass for single script
PowerShell -ExecutionPolicy Bypass -File .\script.ps1
```

---

## Future Enhancements (Phase 2)

### Native PowerShell Code Generation

Instead of wrapping bash, generate actual PowerShell code:

**Example (Future):**
```powershell
# Native PowerShell (Phase 2)
function Get-Color {
    return @('red', 'green', 'blue')
}

Get-Color
```

### Object Pipeline Support

Convert between Prolog terms and PowerShell objects:

```powershell
# Pipeline with objects (Phase 2)
$users | ConvertTo-Json | Invoke-PrologQuery | ConvertFrom-Json
```

### PowerShell-Specific Optimizations

- Use PowerShell cmdlets instead of external tools
- Leverage .NET framework
- Support PowerShell modules

---

## Testing

### Run Tests

```prolog
?- [unifyweaver(core/powershell_compiler)].
?- test_powershell_compiler.
```

### Expected Output

```
╔════════════════════════════════════════╗
║  PowerShell Compiler Tests            ║
╚════════════════════════════════════════╝

[Test 1] Inline wrapper generation
[✓] Inline wrapper contains expected components

[Test 2] Tempfile wrapper generation
[✓] Tempfile wrapper contains expected components

[Test 3] Compilation with various options
[✓] compat_check(false) omits compatibility check
[✓] Default includes compatibility check

╔════════════════════════════════════════╗
║  All PowerShell Compiler Tests Passed ║
╚════════════════════════════════════════╝
```

### Run Demos

```prolog
?- [examples/powershell_compilation_demo].
?- main.
```

This will demonstrate all compilation features and generate example PowerShell scripts.

---

## FAQ

### Q: Can I run PowerShell scripts without the compatibility layer?

**A:** No (in Phase 1). The generated scripts require the compatibility layer to invoke bash. Phase 2 will support native PowerShell generation.

### Q: Will PowerShell scripts work on Linux/Mac?

**A:** Yes, if you have PowerShell Core (pwsh) and bash installed. The compatibility layer works cross-platform.

### Q: Can I integrate generated scripts with existing PowerShell modules?

**A:** Yes! Generated scripts are normal PowerShell scripts and can be called from other PowerShell code:

```powershell
Import-Module MyModule
$data = Get-MyData
$data | .\process_prolog.ps1 | Export-Results
```

### Q: What's the performance difference vs bash?

**A:** Phase 1 has ~50-100ms overhead for the PowerShell wrapper. For data processing, this is usually negligible. Phase 2 native generation will eliminate this.

### Q: Can I edit the generated PowerShell scripts?

**A:** Yes, but regenerating will overwrite your changes. Best practice: keep Prolog as source of truth, or copy generated script to a new file for customization.

---

## Summary

### What You Get (Phase 1)

✅ Compile Prolog predicates to PowerShell scripts
✅ Reuse all existing bash compilation templates
✅ Integrate with PowerShell workflows
✅ Cross-platform support (Windows/Linux/Mac with PowerShell Core)
✅ Text-based pipeline support

### What's Coming (Phase 2)

⭐ Native PowerShell code generation
⭐ PowerShell object pipeline support
⭐ No bash dependency
⭐ Better performance
⭐ PowerShell-specific optimizations

---

## See Also

- **PowerShell Compatibility Layer:** `docs/POWERSHELL_COMPAT.md`
- **Stream Compiler:** `docs/STREAM_COMPILER.md`
- **Recursive Compiler:** `docs/RECURSIVE_COMPILER.md`
- **Examples:** `examples/powershell_compilation_demo.pl`

---

**Created:** 2025-10-19
**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
