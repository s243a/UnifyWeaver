<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# PowerShell Compatibility Layer

Cross-platform bash command execution from PowerShell using WSL or Cygwin backends.

## Quick Start

### From PowerShell

```powershell
# Source the compatibility layer
. .\scripts\powershell-compat\init_unify_compat.ps1

# Use uw-* prefixed wrappers
$env:UNIFYWEAVER_EXEC_MODE = 'wsl'      # or 'cygwin'
uw-ls | uw-grep '.pl$'
uw-uname -o
```

### From Bash/WSL

The recommended way to test the compatibility layer from Bash is to use the provided wrapper scripts:

```bash
# From the repository root or scripts/powershell-compat directory:

# Test with Cygwin backend (default)
./scripts/powershell-compat/test_from_bash.sh cygwin

# Test with WSL backend
./scripts/powershell-compat/test_from_bash.sh wsl
```

## Files

- **`init_unify_compat.ps1`** - Main compatibility layer (source this from PowerShell)
- **`test_compat_layer.ps1`** - Test suite (run from PowerShell directly)
- **`test_compat_layer_wsl.ps1`** - Wrapper to test with WSL backend
- **`test_compat_layer_cygwin.ps1`** - Wrapper to test with Cygwin backend
- **`test_from_bash.sh`** - Bash wrapper to invoke tests from WSL/Bash

## Why Wrapper Scripts?

When calling PowerShell from Bash, environment variable assignment in the `-Command` parameter causes escaping issues:

```bash
# ❌ This FAILS - Bash escaping breaks PowerShell parsing
powershell.exe -Command "$env:UNIFYWEAVER_EXEC_MODE='wsl'; .\test_compat_layer.ps1"
# Error: The term ':UNIFYWEAVER_EXEC_MODE=wsl' is not recognized...

# ✅ This WORKS - Using -File parameter with wrapper script
powershell.exe -File test_compat_layer_wsl.ps1
```

The `-File` parameter approach:
1. Avoids shell escaping issues
2. Provides cleaner cross-environment invocation
3. Makes it easy to test different backends

## Backend Selection

The `UNIFYWEAVER_EXEC_MODE` environment variable controls which backend to use:

- **`wsl`** - Windows Subsystem for Linux (requires WSL installed)
- **`cygwin`** - Cygwin bash (default, requires Cygwin installed)

If not set, Cygwin is used by default.

## Testing from Different Environments

### PowerShell → Cygwin

```powershell
$env:UNIFYWEAVER_EXEC_MODE = 'cygwin'
.\scripts\powershell-compat\test_compat_layer.ps1
```

### PowerShell → WSL

```powershell
$env:UNIFYWEAVER_EXEC_MODE = 'wsl'
.\scripts\powershell-compat\test_compat_layer.ps1
```

### Bash → Cygwin

```bash
./scripts/powershell-compat/test_from_bash.sh cygwin
```

### Bash → WSL

```bash
./scripts/powershell-compat/test_from_bash.sh wsl
```

## Direct PowerShell Invocation (Alternative)

If you prefer to use `-File` directly from Bash:

```bash
# With wrapper scripts (recommended)
powershell.exe -File ./scripts/powershell-compat/test_compat_layer_wsl.ps1
powershell.exe -File ./scripts/powershell-compat/test_compat_layer_cygwin.ps1

# Note: Paths must be Windows-style when called from Bash
```

## Implementation Details

The compatibility layer provides:

- **`Invoke-UnifyCommand`** - Core function to execute bash commands
- **`uw-*` prefixed wrappers** - Safe wrappers for common Unix commands
  - `uw-ls`, `uw-grep`, `uw-cat`, `uw-uname`, etc.
  - Avoids conflicts with PowerShell aliases and cmdlets
- **Path conversion** - Automatic Windows ↔ Unix path conversion
- **Working directory sync** - Keeps backend cwd in sync with PowerShell cwd
- **UTF-8 normalization** - Ensures consistent output encoding

## Troubleshooting

**Problem:** "PowerShell not found" when running from Bash

**Solution:** The `test_from_bash.sh` script searches for PowerShell in these locations:
1. `powershell.exe` in PATH
2. `pwsh.exe` in PATH (PowerShell Core)
3. `/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe`

If none are found, add PowerShell to your PATH or update the script.

**Problem:** "Cygwin bash not found" error

**Solution:** Install Cygwin or set `$env:UNIFYWEAVER_EXEC_MODE = 'wsl'` to use WSL instead.

**Problem:** WSL backend fails

**Solution:** Ensure WSL is installed and configured. Test with `wsl.exe bash -c 'echo test'`

## See Also

- `docs/FIREWALL_GUIDE.md` - Firewall configuration
- `examples/integration_test.pl` - Full integration tests
