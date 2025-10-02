# UnifyWeaver Scripts

## Windows Launchers

### `start_unifyweaver_cygwin.bat`
**Double-click to launch UnifyWeaver console from Windows**

Automatically detects and uses (in priority order):
1. Cygwin
2. MSYS2
3. WSL (Windows Subsystem for Linux)

**Features:**
- Uses `setlocal EnableExtensions EnableDelayedExpansion` for proper variable capture
- Converts Windows paths to POSIX using `cygpath`
- Adds SWI-Prolog to PATH automatically

**Usage:**
```cmd
REM Just double-click the .bat file
REM Or run from command prompt:
start_unifyweaver_cygwin.bat
```

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
**Initialize a test environment**

Creates a standalone testing directory with all UnifyWeaver modules.

**Usage:**
```bash
cd scripts/testing
./init_testing.sh

# Or specify custom location
UNIFYWEAVER_ROOT=/tmp/my_test ./init_testing.sh

# Force Windows SWI-Prolog for testing
./init_testing.sh --force-windows
```

### `testing/find_swi-prolog.sh`
**Detect and configure SWI-Prolog**

Automatically sourced by `init_testing.sh`. Prioritizes native Linux SWI-Prolog over Windows version in WSL for better readline support (arrow keys).

## Legacy Scripts (LogiForge)

The following scripts are from the LogiForge project and kept for reference:
- `logiforge_console.sh` - Original console launcher
- `logiforge_console_old.sh` - Older version
- `logiforge_console_simp.sh` - Simplified version
- `logiforge_cygwin.sh` - Cygwin-specific version
- `start_logiforge_cygwin.bat` - Original Windows launcher
- `start_logiforge_cygwin_back.bat` - Backup version
- `conemu_logiforge.bat` - ConEmu specific launcher

These can be safely deleted once the new UnifyWeaver scripts are confirmed working.
