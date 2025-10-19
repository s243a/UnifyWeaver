# UnifyWeaver Diagnostic Tools

Diagnostic and troubleshooting utilities for UnifyWeaver development on Windows.

## Files

### `Fix-WindowsTerminalEmoji.ps1`
**Windows Terminal emoji support diagnostic and configuration tool**

Checks and fixes emoji rendering issues in Windows Terminal by verifying font configuration and UTF-8 encoding settings.

#### Usage:

**Diagnostic mode (default):**
```powershell
.\Fix-WindowsTerminalEmoji.ps1
```

Runs four diagnostic tests:
1. **Emoji Rendering Test** - Visual check of BMP and non-BMP emoji
2. **Settings Location** - Finds Windows Terminal settings.json
3. **Font Configuration** - Checks current font and emoji support
4. **Encoding Verification** - Verifies UTF-8 is active

**Auto-fix mode:**
```powershell
.\Fix-WindowsTerminalEmoji.ps1 -AutoFix
```

Automatically:
- Backs up current settings
- Sets font to "Cascadia Code" (emoji-capable)
- Saves updated configuration
- Prompts to restart Windows Terminal

#### What It Checks:

✅ **Emoji Rendering**
- BMP symbols (✅ ❌ ⚠ ℹ ⚡) - should work in all terminals
- Non-BMP emoji (🚀 📊 📈 💾 🎉) - require emoji-capable font

✅ **Font Configuration**
- Current font face setting
- Whether font supports emoji fallback
- Recommends "Cascadia Code" if not set

✅ **UTF-8 Encoding**
- Console output encoding (should be UTF-8)
- Console input encoding (should be UTF-8)
- Code page verification (should be 65001)

✅ **Settings File**
- Locates Windows Terminal settings.json
- Checks for PowerShell profile configuration

#### Expected Output (Healthy System):

```
[Test 1] Emoji Rendering Test
BMP Symbols (should work): ✅ ❌ ⚠ ℹ ⚡
Non-BMP Emoji (need font fix): 🚀 📊 📈 💾 🎉

[✓] Found settings at: C:\Users\...\settings.json
[✓] Found PowerShell profile: Windows PowerShell
[INFO] Current font: Cascadia Code
[✓] Font supports emoji via fallback
[✓] UTF-8 encoding is active
```

#### When to Use:

- **Non-BMP emoji appear as boxes or garbled characters** (畜㡄䐳畜䕄〸)
- **Setting up new Windows Terminal installation**
- **After updating Windows Terminal**
- **Testing emoji support before committing code with emoji output**

#### Manual Fix (Alternative):

If you prefer to configure manually:

1. Open Windows Terminal (press `Ctrl+,` for settings)
2. Click your PowerShell profile
3. Under "Appearance" → "Font face"
4. Select **"Cascadia Code"**
5. Click "Save"
6. Restart Windows Terminal

---

## Platform Compatibility Integration

This diagnostic tool works with UnifyWeaver's platform compatibility system:

### Emoji Support Levels

| Level | Description | Terminals |
|-------|-------------|-----------|
| `ascii` | All emoji → ASCII fallbacks | CMD.exe, old terminals |
| `bmp` | BMP symbols (✅ ❌ ⚠) + ASCII for non-BMP | ConEmu, PowerShell console |
| `full` | All emoji including color (🚀 📊 📈) | Windows Terminal, WSL |

### Auto-Detection

UnifyWeaver's `platform_compat.pl` module automatically detects terminal capabilities:

```prolog
?- detect_terminal(T), get_emoji_level(L).
T = windows_terminal,
L = full.
```

Environment variables checked:
- `WT_SESSION` → Windows Terminal (full)
- `ConEmuPID` → ConEmu (bmp)
- `WSL_DISTRO_NAME` → WSL (full)
- (none) → Unknown (bmp, conservative)

---

## Related Documentation

- **Platform Compatibility Module:** `src/unifyweaver/core/platform_compat.pl`
- **Emoji Support Guide:** `scripts/testing/test_env_ps3/docs/EMOJI_SUPPORT.md`
- **Terminal Launcher Guide:** `scripts/testing/test_env_ps3/docs/TERMINAL_LAUNCHER.md`
- **Windows Terminal Tests:** `scripts/testing/test_env_ps3/docs/WINDOWS_TERMINAL_TESTS.md`
- **Claude Code Setup:** `docs/development/ai-tools/claude-code-windows-setup.md`

---

## Troubleshooting

### Issue: Still seeing garbled emoji after fix

**Solution:**
1. Restart Windows Terminal completely (close all tabs/windows)
2. Run diagnostic again to verify font was saved
3. Check if Windows Terminal was updated (may have reset settings)

### Issue: Script can't find settings.json

**Possible causes:**
- Windows Terminal not installed
- Non-standard installation location

**Solution:**
```powershell
# Find settings.json manually
Get-ChildItem "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal*\LocalState\settings.json"
```

### Issue: Cascadia Code not available

**Solution:**
Cascadia Code is included with Windows Terminal. If missing:
1. Update Windows Terminal from Microsoft Store
2. Or download Cascadia Code separately: https://github.com/microsoft/cascadia-code/releases

Alternative emoji-capable fonts:
- Consolas (partial emoji support)
- Segoe UI Emoji (emoji-only font)

### Issue: Emoji work in PowerShell but not in SWI-Prolog

**Cause:**
SWI-Prolog output is being piped through different encoding.

**Solution:**
1. Verify UTF-8 encoding in Prolog:
   ```prolog
   ?- stream_property(current_output, encoding(E)).
   E = utf8.
   ```

2. Check platform compatibility is loaded:
   ```prolog
   ?- use_module(unifyweaver(core/platform_compat)).
   ?- get_emoji_level(L).
   L = full.  % Should be 'full' in Windows Terminal
   ```

---

## For Developers

### Adding New Diagnostic Tools

When adding diagnostic scripts to this directory:

1. **Use clear naming:** `Fix-<Component><Issue>.ps1` or `Test-<Component>.ps1`
2. **Include help:** Add `-Help` parameter with usage examples
3. **Show status:** Use color-coded output (`Green` = success, `Yellow` = warning, `Red` = error)
4. **Provide auto-fix:** Include optional `-AutoFix` parameter when applicable
5. **Back up before changes:** Always backup configuration files
6. **Update this README:** Document what the tool does and when to use it

### Testing Diagnostic Tools

Test matrix for emoji diagnostic:

| Terminal | BMP Symbols | Non-BMP Emoji | Expected Level |
|----------|-------------|---------------|----------------|
| Windows Terminal (Cascadia Code) | ✅ | ✅ | full |
| Windows Terminal (Consolas) | ✅ | ⚠️ Partial | full |
| ConEmu | ✅ | ❌ | bmp |
| PowerShell Console | ✅ | ❌ | bmp |
| CMD.exe | ⚠️ Depends | ❌ | ascii/bmp |

---

**Last Updated:** 2025-10-19
**Maintainer:** UnifyWeaver Project
