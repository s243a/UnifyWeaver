# Claude Code on Windows - Setup and Environment Guide

## Overview

This guide documents how to properly set up and use Claude Code on Windows for UnifyWeaver development, particularly when working with cross-platform tools like PowerShell, Cygwin, and WSL.

**Target Audience:** Contributors using Claude Code on Windows
**Last Updated:** 2025-10-19

---

## Table of Contents

1. [Installation](#installation)
2. [Launch Environment Behavior](#launch-environment-behavior)
3. [Recommended Setup](#recommended-setup)
4. [PowerShell Access](#powershell-access)
5. [File Access Patterns](#file-access-patterns)
6. [Common Issues](#common-issues)
7. [Testing Emoji Support](#testing-emoji-support)

---

## Installation

### Prerequisites

- **Node.js and npm** installed on Windows
- **Git for Windows** (provides Git Bash/MSYS)
- **Cygwin** (recommended for full file access)
- **PowerShell 5 or 7** (usually pre-installed on Windows)
- **SWI-Prolog** (for running UnifyWeaver)

### Installing Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

This installs the Windows version of Claude Code, which uses MSYS/Git Bash for command execution.

---

## Launch Environment Behavior

### Critical Discovery: Launch Chain Matters

The way you launch Claude Code affects what file system access and executable paths are available.

#### Option 1: Direct Launch from PowerShell (❌ Limited Access)

```powershell
# In PowerShell
claude
```

**Result:**
- ❌ Claude runs in limited MSYS/Git Bash environment
- ❌ Can see Cygwin paths but can't access Windows executables in `Program Files`
- ❌ Limited file access to Windows system directories
- ⚠️ May see errors like "Permission denied" when accessing Windows folders

**Example of what fails:**
```bash
# In Claude Code launched from PowerShell
ls "/c/Program Files/swipl/bin"
# Error: Permission denied or Access denied
```

#### Option 2: Launch via Cygwin from PowerShell (✅ Full Access)

```powershell
# Step 1: Launch PowerShell (Windows Terminal recommended)
# Step 2: Launch Cygwin
bash

# Step 3: From Cygwin, launch Claude Code
claude
```

**Result:**
- ✅ Claude runs in MSYS but inherits Cygwin's environment
- ✅ Full access to Windows file system
- ✅ Can access executables using full paths
- ✅ All Windows directories accessible

**Example of what works:**
```bash
# In Claude Code launched from Cygwin
ls "/c/Program Files/swipl/bin"
# ✅ Success! Shows swipl.exe and other files

"/c/Program Files/PowerShell/7/pwsh.exe" -Version
# ✅ PowerShell 7.5.3
```

#### Option 3: Launch from WSL (✅ Full Access + Native PowerShell)

```powershell
# Step 1: Launch PowerShell (Windows Terminal recommended)
# Step 2: Launch WSL
wsl

# Step 3: From WSL, launch Claude Code
claude
```

**Result:**
- ✅ Claude runs in native Linux bash environment
- ✅ Full access to Windows file system via `/mnt/c/`
- ✅ PowerShell accessible (may require Windows compatibility setup)
- ✅ Best for Linux-native development workflows

**Requirements:**
- WSL installed on Windows (`wsl --install` in PowerShell as admin)
- Node.js/npm installed in WSL (not just Windows)
- May need to configure Windows executable interop

**Example:**
```bash
# In Claude Code launched from WSL
ls /mnt/c/Program\ Files/swipl/bin
# ✅ Works with WSL path syntax

# PowerShell access (if Windows interop enabled)
powershell.exe -Command "Get-Process"
# ✅ May work depending on WSL configuration
```

**Note:** PowerShell access from WSL may require enabling Windows executable interop. See WSL documentation for details.

### Environment Chain Visualization

**Limited Access Path:**
```
PowerShell (Windows)
    ↓
Claude Code (MSYS/Git Bash)
    ↓
    Limited file access ❌
```

**Full Access Path - Option A (Cygwin):**
```
PowerShell (Windows)
    ↓
Cygwin (bash with Windows integration)
    ↓
Claude Code (MSYS/Git Bash)
    ↓
    Full file access ✅
```

**Full Access Path - Option B (WSL):**
```
PowerShell (Windows)
    ↓
WSL (Linux subsystem)
    ↓
Claude Code (native Linux bash)
    ↓
    Full file access ✅
    Native Linux tools ✅
    Windows interop (if enabled) ✅
```

---

## Recommended Setup

### For Daily Development

#### Option A: Using Cygwin (Recommended for Windows-focused development)

1. **Launch Windows Terminal** (best emoji support)
2. **Start PowerShell** (usually the default)
3. **Launch Cygwin:**
   ```powershell
   bash
   ```
4. **Launch Claude Code:**
   ```bash
   claude
   ```

**Pros:**
- Full Windows file access
- Works with Windows npm-installed Claude Code
- Direct access to Windows executables

#### Option B: Using WSL (Recommended for Linux-focused development)

1. **Launch Windows Terminal** (best emoji support)
2. **Start PowerShell** (usually the default)
3. **Launch WSL:**
   ```powershell
   wsl
   ```
4. **Launch Claude Code:**
   ```bash
   claude
   ```

**Pros:**
- Native Linux environment
- Better for cross-platform development
- Full Linux toolchain available
- PowerShell accessible via Windows interop

**Cons:**
- Requires Claude Code installed in WSL (separate from Windows installation)
- May need WSL configuration for Windows executable access

### Quick Launch Scripts

#### Cygwin Launch Script (PowerShell)

Create `start-claude-cygwin.ps1`:
```powershell
# Start-ClaudeCygwin.ps1 - Launch Claude Code via Cygwin

Write-Host "Launching Claude Code with full Windows access..." -ForegroundColor Cyan
Write-Host "Using Cygwin environment..." -ForegroundColor Yellow

# Launch Cygwin with Claude Code auto-start
bash -c "echo 'Cygwin environment ready. Launching Claude Code...'; claude"
```

Usage:
```powershell
.\start-claude-cygwin.ps1
```

#### WSL Launch Script (PowerShell)

Create `start-claude-wsl.ps1`:
```powershell
# Start-ClaudeWSL.ps1 - Launch Claude Code via WSL

Write-Host "Launching Claude Code in WSL..." -ForegroundColor Cyan
Write-Host "Using native Linux environment..." -ForegroundColor Yellow

# Launch WSL with Claude Code auto-start
wsl bash -c "echo 'WSL environment ready. Launching Claude Code...'; claude"
```

Usage:
```powershell
.\start-claude-wsl.ps1
```

---

## PowerShell Access

### The Challenge

PowerShell executables are **not in PATH** by default when Claude Code runs in MSYS/Git Bash or WSL.

### Solution A: Use Full Paths (Cygwin/MSYS)

Claude Code can execute PowerShell commands using full executable paths:

#### PowerShell 5 (Windows built-in):
```bash
"/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" -Command "Get-Process"
```

#### PowerShell 7 (if installed):
```bash
"/c/Program Files/PowerShell/7/pwsh.exe" -Command "Get-ChildItem"
```

### Solution B: Windows Interop (WSL)

If you launched Claude Code from WSL, you can access PowerShell via Windows interop:

#### Check if Windows interop is enabled:
```bash
# Should show PowerShell help
powershell.exe -Help
```

#### Using PowerShell from WSL:
```bash
# PowerShell 5 (always available)
powershell.exe -Command "Get-Process"

# PowerShell 7 (if installed)
pwsh.exe -Command "Get-ChildItem"
```

#### Accessing Windows files from WSL:
```bash
# WSL uses /mnt/c/ instead of /c/
ls /mnt/c/Program\ Files/swipl/bin

# Or escape spaces differently
ls "/mnt/c/Program Files/swipl/bin"
```

#### If Windows interop is not enabled:

Edit `/etc/wsl.conf` in WSL:
```ini
[interop]
enabled = true
appendWindowsPath = false
```

Then restart WSL:
```powershell
# In Windows PowerShell
wsl --shutdown
wsl
```

### Running PowerShell Scripts

```bash
# Execute .ps1 script
"/c/Program Files/PowerShell/7/pwsh.exe" -File "./my-script.ps1"

# Execute with parameters
"/c/Program Files/PowerShell/7/pwsh.exe" -File "./script.ps1" -Param1 "value"

# Execute encoded command (for complex multi-line scripts)
"/c/Program Files/PowerShell/7/pwsh.exe" -EncodedCommand <base64-encoded-script>
```

### Example: Checking Windows Terminal Font

```bash
"/c/Program Files/PowerShell/7/pwsh.exe" -Command "\$settingsPath = \"\$env:LOCALAPPDATA\\Packages\\Microsoft.WindowsTerminal_8wekyb3d8bbwe\\LocalState\\settings.json\"; Get-Content \$settingsPath | Select-String -Pattern 'font' -Context 2,2"
```

**Important:**
- Escape dollar signs: `\$` instead of `$`
- Escape backslashes in paths: `\\` instead of `\`
- Use double quotes for outer command, single quotes inside PowerShell

---

## File Access Patterns

### Windows Paths in MSYS/Cygwin

| Windows Path | MSYS/Cygwin Path |
|--------------|------------------|
| `C:\Users\username` | `/c/Users/username` |
| `C:\Program Files` | `/c/Program Files` |
| `C:\Program Files (x86)` | `/c/Program Files (x86)` |
| `%LOCALAPPDATA%` | `/c/Users/username/AppData/Local` |
| `%APPDATA%` | `/c/Users/username/AppData/Roaming` |

### Common UnifyWeaver Paths

```bash
# Project root
cd /c/Users/johnc/Dropbox/projects/UnifyWeaver

# Test environment
cd /c/Users/johnc/Dropbox/projects/UnifyWeaver/scripts/testing/test_env_ps3

# SWI-Prolog installation
ls "/c/Program Files/swipl/bin"

# Windows Terminal settings
cat "/c/Users/johnc/AppData/Local/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json"
```

### Testing File Access

Quick test to verify full access:
```bash
# Should list swipl.exe and other files
ls "/c/Program Files/swipl/bin"

# Should show PowerShell version
"/c/Program Files/PowerShell/7/pwsh.exe" -Version

# Should display Windows Terminal settings
cat "/c/Users/$USER/AppData/Local/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json" | head -20
```

---

## Common Issues

### Issue 1: "Permission Denied" When Accessing Program Files

**Symptom:**
```bash
ls "/c/Program Files/swipl/bin"
# Error: Permission denied
```

**Cause:** Claude Code launched directly from PowerShell without Cygwin

**Solution:** Launch Claude Code from Cygwin (launched from PowerShell)

---

### Issue 2: PowerShell Not Found

**Symptom:**
```bash
pwsh.exe -Version
# Error: command not found
```

**Cause:** PowerShell not in PATH in MSYS environment

**Solution:** Use full path:
```bash
"/c/Program Files/PowerShell/7/pwsh.exe" -Version
```

---

### Issue 3: Emoji Showing as Mojibake in Claude Output

**Symptom:**
Claude Code shows `������` for emoji like 🚀 📊 📈

**Cause:** MSYS pipe mangles multi-byte UTF-8 emoji when piping output to Claude

**Impact:** None! This is cosmetic only. The actual terminal displays emoji correctly.

**Verification:** Run the command in native PowerShell/Windows Terminal to verify emoji render correctly for end users.

**Example:**
```bash
# Claude sees: ������ STEP | ������ DATA
# User sees: 🚀 STEP | 📊 DATA (correct!)
swipl -l init.pl -g "safe_format('🚀 STEP | 📊 DATA~n', []), halt"
```

**Solution:** Trust the platform detection and user verification. Don't rely on Claude's output view for emoji rendering tests.

---

### Issue 4: Cygwin Not Installed

**Solution:**

1. Download Cygwin installer: https://www.cygwin.com/
2. Install with default packages (bash, coreutils, etc.)
3. Add Cygwin to PATH or use full path:
   ```powershell
   C:\cygwin64\bin\bash.exe
   ```

**Alternative:** If you can't install Cygwin, some Windows commands may still work with limited access. Document which operations succeed vs. fail.

---

## Testing Emoji Support

UnifyWeaver includes a platform compatibility system that auto-detects terminal capabilities and provides appropriate emoji support.

### Quick Emoji Test

```bash
cd /c/Users/johnc/Dropbox/projects/UnifyWeaver/scripts/testing/test_env_ps3

# Run comprehensive emoji test
swipl -l init.pl -g "[examples/test_emoji_levels], main, halt"
```

### Expected Results by Terminal

| Terminal | Emoji Level | BMP Symbols | Non-BMP Emoji |
|----------|-------------|-------------|---------------|
| **Windows Terminal** | `full` | ✅ ❌ ⚠ ℹ ⚡ (color) | 🚀 📊 📈 💾 🎉 (color) |
| **ConEmu** | `bmp` | ✅ ❌ ⚠ ℹ ⚡ (mono) | `[STEP]` `[DATA]` (ASCII fallback) |
| **PowerShell console** | `bmp` | ✅ ❌ ⚠ ℹ ⚡ | `[STEP]` `[DATA]` (ASCII fallback) |
| **CMD.exe** | `bmp` | ✅ ❌ ⚠ ℹ ⚡ | `[STEP]` `[DATA]` (ASCII fallback) |

### Manual Verification Script

For accurate emoji verification, run this PowerShell script **outside of Claude Code**:

```powershell
cd C:\Users\johnc\Dropbox\projects\UnifyWeaver\scripts\testing\test_env_ps3
.\Fix-WindowsTerminalEmoji.ps1
```

This script:
- ✅ Tests emoji rendering directly in your terminal
- ✅ Checks Windows Terminal font configuration
- ✅ Verifies UTF-8 encoding
- ✅ Reports actual emoji support level

---

## Environment Detection

### How UnifyWeaver Detects Your Terminal

The `platform_compat.pl` module checks environment variables:

| Variable | Terminal Detected | Emoji Level |
|----------|-------------------|-------------|
| `WT_SESSION` | Windows Terminal | `full` |
| `WSL_DISTRO_NAME` | WSL | `full` |
| `ConEmuPID` | ConEmu | `bmp` |
| (none) | Unknown/PowerShell | `bmp` (safe default) |

### Verifying Detection

```bash
# Check what terminal was detected
swipl -l init.pl -g "use_module(unifyweaver(core/platform_compat)), detect_terminal(T), format('Terminal: ~w~n', [T]), halt"

# Check current emoji level
swipl -l init.pl -g "use_module(unifyweaver(core/platform_compat)), get_emoji_level(L), format('Emoji Level: ~w~n', [L]), halt"
```

---

## Summary: Best Practices for Contributors

### ✅ DO:

1. **Choose your launch environment:**
   - **Cygwin** (from PowerShell) for Windows-focused development with full access
   - **WSL** (from PowerShell) for Linux-focused development with Windows interop
   - **Avoid** launching Claude directly from PowerShell (limited access)
2. **Use Windows Terminal** for best emoji support
3. **Use full paths** for PowerShell executables:
   - Cygwin/MSYS: `/c/Program Files/PowerShell/7/pwsh.exe`
   - WSL: `pwsh.exe` (if Windows interop enabled) or `/mnt/c/Program Files/PowerShell/7/pwsh.exe`
4. **Verify emoji rendering** in native terminal, not just Claude's output view
5. **Use appropriate path syntax:**
   - Cygwin/MSYS: `/c/Users/...` instead of `C:\Users\...`
   - WSL: `/mnt/c/Users/...` instead of `C:\Users\...`

### ❌ DON'T:

1. **Don't launch Claude directly from PowerShell** if you need Program Files access
2. **Don't rely on PATH** for PowerShell executables (use full paths or Windows interop)
3. **Don't trust Claude's mojibake output** for emoji - verify in native terminal
4. **Don't use backslash paths** (`C:\...`) - use forward slash with appropriate prefix
5. **Don't mix WSL and Cygwin path syntax** - they use different mount points

### 🔧 Quick Reference Commands

```bash
# Check if you have full access
ls "/c/Program Files/swipl/bin"  # Should succeed

# Run PowerShell 7
"/c/Program Files/PowerShell/7/pwsh.exe" -Version

# Test emoji in SWI-Prolog
cd /c/Users/johnc/Dropbox/projects/UnifyWeaver/scripts/testing/test_env_ps3
swipl -l init.pl -g "[examples/test_emoji_levels], main, halt"

# Check terminal detection
swipl -l init.pl -g "use_module(unifyweaver(core/platform_compat)), detect_terminal(T), format('~w~n', [T]), halt"
```

---

## Related Documentation

- **Emoji Support:** `../../scripts/testing/test_env_ps3/docs/EMOJI_SUPPORT.md`
- **Terminal Launcher:** `../../scripts/testing/test_env_ps3/docs/TERMINAL_LAUNCHER.md`
- **Windows Terminal Tests:** `../../scripts/testing/test_env_ps3/docs/WINDOWS_TERMINAL_TESTS.md`
- **Platform Compatibility:** `../../src/unifyweaver/core/platform_compat.pl`

---

## Troubleshooting Checklist

Before filing an issue, verify:

- [ ] Launched Claude Code from Cygwin (not directly from PowerShell)
- [ ] Can access `/c/Program Files/swipl/bin` successfully
- [ ] PowerShell accessible via full path
- [ ] Using Windows Terminal (not ConEmu/CMD) for emoji tests
- [ ] Verified emoji rendering in native terminal, not just Claude output
- [ ] UTF-8 encoding active in Windows Terminal
- [ ] Font set to Cascadia Code (or other emoji-capable font)

---

**Contributors:** If you discover additional setup steps or workarounds, please update this document!

**Questions?** File an issue or discuss in the development channel.
