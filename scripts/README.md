# UnifyWeaver Scripts

This directory contains launcher scripts and utilities for running UnifyWeaver on different platforms.

## Quick Start

**Windows users:** Double-click one of these:
- `start_unifyweaver_windows.bat` - Native Windows (recommended for most users)
- `start_unifyweaver_cygwin.bat` - Cygwin/MSYS2/WSL environments

**Linux/macOS users:** Run from terminal:
```bash
bash unifyweaver_console.sh
```

---

## Windows Launchers

### `start_unifyweaver_windows.bat` ⭐ NEW
**Pure Windows launcher - no Cygwin/WSL required**

Uses native Windows SWI-Prolog directly, making it the simplest option for Windows users.

**Features:**
- No dependencies on Cygwin, MSYS2, or WSL
- Automatic SWI-Prolog detection and PATH configuration
- **ConEmu terminal support** for enhanced console experience
- Falls back to standard Windows console if ConEmu not installed
- Helper commands pre-loaded (load_stream, test_advanced, etc.)

**Requirements:**
- SWI-Prolog for Windows: https://www.swi-prolog.org/download/stable
- Optional: ConEmu for better terminal: https://conemu.github.io/

**Usage:**
```cmd
REM Just double-click the .bat file
REM Or run from command prompt:
start_unifyweaver_windows.bat
```

**ConEmu Integration:**
The launcher automatically detects ConEmu in these locations:
- `%ProgramFiles%\ConEmu\ConEmu64.exe`
- `%ProgramFiles(x86)%\ConEmu\ConEmu.exe`
- `%LocalAppData%\ConEmu\ConEmu64.exe`

If found, launches in ConEmu with proper working directory and syntax highlighting.

---

### `start_unifyweaver_cygwin.bat`
**Unix-like environment launcher (Cygwin/MSYS2/WSL)**

For users who prefer Unix tools on Windows or need bash-based workflows.

**Automatically detects and uses (in priority order):**
1. Cygwin (with ConEmu support ⭐ NEW)
2. MSYS2 (with ConEmu support ⭐ NEW)
3. WSL (Windows Subsystem for Linux)

**Features:**
- Uses `setlocal EnableExtensions EnableDelayedExpansion` for proper variable capture
- Converts Windows paths to POSIX using `cygpath`
- Adds SWI-Prolog to PATH automatically
- **ConEmu terminal support** for Cygwin and MSYS2
- Launches bash console with UnifyWeaver environment

**Usage:**
```cmd
REM Just double-click the .bat file
REM Or run from command prompt:
start_unifyweaver_cygwin.bat
```

**Note:** ConEmu integration works with Cygwin and MSYS2. For WSL, use Windows Terminal for best experience.

## Console Launcher

### `unifyweaver_console.sh`
**Bash script that launches SWI-Prolog with UnifyWeaver environment**

Called by the `.bat` launcher but can also be run directly in Unix environments.

**Features:**
- Auto-configures library paths
- Defines helper predicates
- Works in Cygwin, MSYS2, WSL, and native Linux

**Usage:**
```bash
# From project root
bash scripts/unifyweaver_console.sh

# Or if UNIFYWEAVER_ROOT is set
export UNIFYWEAVER_ROOT=/path/to/project
bash scripts/unifyweaver_console.sh
```

**Available commands in console:**
- `load_recursive.` - Load recursive_compiler
- `load_stream.` - Load stream_compiler
- `load_template.` - Load template_system
- `load_all_core.` - Load all core modules
- `test_advanced.` - Run advanced recursion tests
- `help.` - Show help

## Testing Scripts

### `testing/init_testing.sh`
**Initialize a test environment (Bash)**

Creates a standalone testing directory with all UnifyWeaver modules. Works on Linux, WSL, Cygwin, and MSYS2.

**Usage:**
```bash
cd scripts/testing
./init_testing.sh

# Or specify custom location
UNIFYWEAVER_ROOT=/tmp/my_test ./init_testing.sh

# Force Windows SWI-Prolog for testing
./init_testing.sh --force-windows
```

### `testing/Init-TestEnvironment.ps1` ⭐ NEW
**Initialize a test environment (PowerShell)**

Pure Windows version of init_testing.sh for users who prefer PowerShell or don't have bash available.

**Requirements:**
- PowerShell 5.1 or later
- SWI-Prolog for Windows

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Usage:**
```powershell
cd scripts\testing
.\Init-TestEnvironment.ps1

# Specify custom location
.\Init-TestEnvironment.ps1 -TargetDir "C:\UnifyWeaver\test"

# Get help
.\Init-TestEnvironment.ps1 -Help
```

**Features:**
- Pure PowerShell - no bash required
- Same functionality as init_testing.sh
- Uses robocopy for fast file copying
- Creates Windows batch launcher
- Generates helpful README.txt

### `testing/find_swi-prolog.sh`
**Detect and configure SWI-Prolog**

Automatically sourced by `init_testing.sh`. Prioritizes native Linux SWI-Prolog over Windows version in WSL for better readline support (arrow keys).

---

## Terminal Customization

### ConEmu Terminal Emulator

ConEmu provides a vastly improved console experience on Windows with features like:
- Tabs and split panes
- Better copy/paste (with mouse support)
- 24-bit color support
- Customizable fonts and themes
- Unicode support
- Saved sessions

**Installation:**
Download from: https://conemu.github.io/

**Usage with UnifyWeaver:**
ConEmu support is automatically integrated into:
- `start_unifyweaver_windows.bat` (native Windows)
- `start_unifyweaver_cygwin.bat` (Cygwin/MSYS2)

Just install ConEmu and run the batch files - they'll detect and use it automatically!

**Manual ConEmu Setup:**
If you want to create a dedicated ConEmu task:

1. Open ConEmu Settings (Win+Alt+P)
2. Go to "Startup" → "Tasks"
3. Add new task with command:
```
cmd /k "%ProgramFiles%\swipl\bin\swipl.exe" -q -g "asserta(library_directory('C:/path/to/UnifyWeaver/src'))" -t prolog
```

**Legacy ConEmu Script:**
See `legacy/conemu_logiforge.bat` for the original LogiForge ConEmu integration pattern used as reference for the current implementation.

---

## Platform Recommendations

### Windows Users

**Recommended Setup:**
1. Install SWI-Prolog for Windows
2. Install ConEmu (optional but recommended)
3. Use `start_unifyweaver_windows.bat`

**Alternative (Unix-like):**
1. Install Cygwin or MSYS2
2. Install ConEmu (optional)
3. Use `start_unifyweaver_cygwin.bat`

### Linux/macOS Users

**Recommended Setup:**
1. Install SWI-Prolog from package manager
2. Run `bash unifyweaver_console.sh`

### WSL Users

**Recommended Setup:**
1. Install native Linux SWI-Prolog in WSL: `sudo apt install swi-prolog`
2. Use Windows Terminal for best experience
3. Run `bash unifyweaver_console.sh`

---

## Legacy Scripts (LogiForge)

The following scripts are from the LogiForge project and kept in `legacy/` for reference:
- `logiforge_console.sh` - Original console launcher
- `logiforge_console_old.sh` - Older version
- `logiforge_console_simp.sh` - Simplified version
- `logiforge_cygwin.sh` - Cygwin-specific version
- `start_logiforge_cygwin.bat` - Original Windows launcher
- `start_logiforge_cygwin_back.bat` - Backup version
- `conemu_logiforge.bat` - ConEmu specific launcher

**Key techniques preserved from legacy:**
- `EnableDelayedExpansion` for proper variable capture in batch files
- `!variable!` delayed expansion syntax
- ConEmu detection and fallback patterns
- Path conversion with cygpath

See `legacy/README.md` for detailed documentation of these techniques.
