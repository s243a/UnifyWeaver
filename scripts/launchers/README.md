# UnifyWeaver Terminal Launchers

Terminal launcher scripts for starting UnifyWeaver test environments with optimal terminal support.

## Files

### Main Launchers

#### `Start-UnifyWeaverTerminal.ps1`
**Primary terminal launcher with auto-detection**

Automatically detects and prefers the best terminal in this order:
1. **Windows Terminal** (best emoji support - full color emoji)
2. **ConEmu** (BMP symbols only)
3. **Standard PowerShell** (fallback)

**Usage:**
```powershell
# Auto-detect best terminal
.\Start-UnifyWeaverTerminal.ps1

# Force specific terminal
.\Start-UnifyWeaverTerminal.ps1 -Terminal wt          # Windows Terminal
.\Start-UnifyWeaverTerminal.ps1 -Terminal conemu      # ConEmu
.\Start-UnifyWeaverTerminal.ps1 -Terminal powershell  # Standard PowerShell

# Show detailed output
.\Start-UnifyWeaverTerminal.ps1 -ShowDetails
```

**Features:**
- Detects Windows Terminal, ConEmu, and PowerShell
- Configures UTF-8 encoding automatically
- Sets up SWI-Prolog PATH (session-only)
- Displays emoji support level based on terminal

---

#### `start_powershell_env.bat`
**Batch file wrapper for easy launching**

Double-click to launch the unified terminal launcher with auto-detection.

**Usage:**
```batch
start_powershell_env.bat
```

Internally calls `Start-UnifyWeaverTerminal.ps1` with default settings.

---

#### `start_test_env.bat`
**Alternative batch launcher**

Similar to `start_powershell_env.bat` but with different naming for consistency with older test environments.

**Usage:**
```batch
start_test_env.bat
```

---

## Legacy Launchers

### `legacy/start_powershell_env.ps1`
**DEPRECATED - ConEmu-focused launcher**

This is the old launcher that only supported ConEmu or standard PowerShell. It has been replaced by `Start-UnifyWeaverTerminal.ps1` which adds Windows Terminal support and better auto-detection.

**Migration:**
- Replace `start_powershell_env.ps1` calls with `Start-UnifyWeaverTerminal.ps1`
- Or use the batch wrapper files which automatically use the new launcher

**Why deprecated:**
- No Windows Terminal support (misses out on color emoji)
- No terminal priority/preference system
- Hardcoded paths less flexible

---

## Terminal Comparison

| Terminal | Emoji Level | BMP Symbols | Non-BMP Emoji | Auto-Detected |
|----------|-------------|-------------|---------------|---------------|
| **Windows Terminal** | `full` | ✅ ❌ ⚠ ℹ ⚡ (color) | 🚀 📊 📈 💾 🎉 (color) | ✅ Yes (`WT_SESSION`) |
| **ConEmu** | `bmp` | ✅ ❌ ⚠ ℹ ⚡ (mono) | `[STEP]` `[DATA]` (ASCII fallback) | ✅ Yes (`ConEmuPID`) |
| **PowerShell Console** | `bmp` | ✅ ❌ ⚠ ℹ ⚡ | `[STEP]` `[DATA]` (ASCII fallback) | ✅ Yes (default) |

---

## For Developers

### How Launchers Work

1. **Detect terminal availability** (Windows Terminal, ConEmu, PowerShell)
2. **Build initialization script** with:
   - Working directory setup
   - SWI-Prolog added to PATH (session-only)
   - UTF-8 encoding configuration
   - Environment variables for locale
3. **Encode script** as Base64 to avoid escaping issues
4. **Launch terminal** with encoded command

### Platform Compatibility Integration

These launchers work with UnifyWeaver's platform compatibility system (`src/unifyweaver/core/platform_compat.pl`) which:

- Detects terminal type from environment variables
- Sets appropriate emoji level (`ascii`, `bmp`, or `full`)
- Provides `safe_format/2` and `safe_format/3` for emoji-aware output
- Gracefully degrades emoji to ASCII fallbacks when needed

See `docs/development/ai-tools/claude-code-windows-setup.md` for detailed documentation.

---

## Installation

These launchers are automatically copied by `scripts/testing/Init-TestEnvironment.ps1` when creating new test environments.

Manual installation:
```powershell
# Copy to test environment
Copy-Item scripts/launchers/*.ps1 <test-env-path>/
Copy-Item scripts/launchers/*.bat <test-env-path>/
```

---

## Related Documentation

- **Platform Compatibility:** `src/unifyweaver/core/platform_compat.pl`
- **Emoji Support Guide:** `scripts/testing/test_env_ps3/docs/EMOJI_SUPPORT.md`
- **Terminal Launcher Guide:** `scripts/testing/test_env_ps3/docs/TERMINAL_LAUNCHER.md`
- **Claude Code Setup:** `docs/development/ai-tools/claude-code-windows-setup.md`

---

## Troubleshooting

### Launcher doesn't find SWI-Prolog
Update the SWI-Prolog path in the launcher script or add swipl to your system PATH.

### Windows Terminal not auto-detected
Install Windows Terminal from the Microsoft Store or specify it manually:
```powershell
.\Start-UnifyWeaverTerminal.ps1 -Terminal wt
```

### Emoji not rendering correctly
Run the diagnostic tool:
```powershell
.\diagnostics\Fix-WindowsTerminalEmoji.ps1
```

---

**Last Updated:** 2025-10-19
**Maintainer:** UnifyWeaver Project
