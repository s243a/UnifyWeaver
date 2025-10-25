# SWI-Prolog Environment Setup Scripts

These scripts add SWI-Prolog to PATH so you can run `swipl` from any PowerShell or bash session, without needing to use the launcher scripts.

## Usage

### PowerShell

```powershell
# From project root or test environment
. .\scripts\testing\init_swipl_env.ps1

# Or from test environment created by Init-TestEnvironment.ps1
. .\scripts\init_swipl_env.ps1

# Now swipl is available
swipl --version
swipl -g "use_module('src/unifyweaver/sources/csv_source')" -t halt
```

### Bash/WSL

```bash
# From project root or test environment
source ./scripts/testing/init_swipl_env.sh

# Or from test environment created by init_testing.sh
source ./scripts/init_swipl_env.sh

# Now swipl is available
swipl --version
swipl -g "use_module('src/unifyweaver/sources/csv_source')" -t halt
```

## Features

Both scripts:
- ✅ Auto-detect SWI-Prolog installation in common locations
- ✅ Add SWI-Prolog to PATH (session only, non-permanent)
- ✅ Set up UTF-8 encoding
- ✅ Configure locale for SWI-Prolog (LANG, LC_ALL)
- ✅ Verify installation and show version
- ✅ Idempotent - safe to run multiple times

### PowerShell Script Features
- Searches: `C:\Program Files\swipl`, `C:\Program Files (x86)\swipl`, etc.
- Sets console encoding to UTF-8
- Quiet mode: `. .\scripts\init_swipl_env.ps1 -Quiet`

### Bash Script Features
- Platform detection (WSL, Cygwin, Linux, macOS)
- Platform-specific search paths
- Prevents execution (must be sourced)
- Quiet mode: `source ./scripts/init_swipl_env.sh --quiet`

## Integration with Compatibility Layer

The PowerShell compatibility layer (`init_unify_compat.ps1`) automatically sources `init_swipl_env.ps1` if found.

```powershell
# This automatically sets up SWI-Prolog and Unix compatibility
. .\scripts\init_unify_compat.ps1

# Now both swipl and uw-* commands are available
swipl --version
uw-ls | uw-grep '.pl$'
```

## Automatic Setup in Test Environments

When you create a test environment, these scripts are automatically copied:

### PowerShell Test Environments
```powershell
.\scripts\testing\Init-TestEnvironment.ps1 -p test_env_ps4
cd test_env_ps4

# Method 1: Use launcher (automatically sets up PATH)
.\start_test_env.bat

# Method 2: Manual PowerShell session
. .\scripts\init_swipl_env.ps1
swipl -l init.pl
```

### Bash/WSL Test Environments
```bash
./scripts/testing/init_testing.sh test_env7
cd scripts/testing/test_env7

# Method 1: Use find_swi-prolog.sh (more comprehensive setup)
source ./scripts/testing/find_swi-prolog.sh
swipl -l init.pl

# Method 2: Simple PATH setup only
source ./scripts/init_swipl_env.sh
swipl -l init.pl
```

## Difference from find_swi-prolog.sh

`find_swi-prolog.sh` is more comprehensive and includes:
- Config persistence to `.unifyweaver.conf`
- Interactive prompts for WSL native vs Windows swipl.exe
- Wrapper creation for cross-platform scenarios
- Ask-once configuration with persistent settings

`init_swipl_env.sh` is simpler:
- Just adds swipl to PATH
- No persistence or configuration
- No interactive prompts
- Faster and lighter for quick sessions

**Use `find_swi-prolog.sh` for:** Initial setup, persistent configuration
**Use `init_swipl_env.sh` for:** Quick sessions, automation, CI/CD

## Search Paths

### Windows/PowerShell
- `C:\Program Files\swipl\bin`
- `C:\Program Files (x86)\swipl\bin`
- `C:\swipl\bin`
- `%LocalAppData%\swipl\bin`

### WSL
- `/mnt/c/Program Files/swipl/bin` (Windows swipl.exe)
- `/usr/bin` (native Linux)
- `/usr/local/bin`
- `/opt/swipl/bin`
- `~/.local/bin`

### Linux
- `/usr/bin`
- `/usr/local/bin`
- `/opt/swipl/bin`
- `/snap/swi-prolog/current/bin`
- `~/.local/bin`

### macOS
- `/usr/local/bin` (Homebrew Intel)
- `/opt/homebrew/bin` (Homebrew Apple Silicon)
- `/opt/local/bin` (MacPorts)
- `/Applications/SWI-Prolog.app/Contents/MacOS`

## Installation Instructions

If SWI-Prolog is not found, install it:

### Windows
Download from: https://www.swi-prolog.org/download/stable

### Linux/WSL
```bash
sudo apt-add-repository ppa:swi-prolog/stable
sudo apt update && sudo apt install swi-prolog
```

### macOS
```bash
brew install swi-prolog
```

### Cygwin
Install from: https://www.swi-prolog.org/download/stable

## See Also

- `find_swi-prolog.sh` - Comprehensive setup with persistence
- `init_unify_compat.ps1` - PowerShell Unix compatibility layer
- `Start-UnifyWeaverTerminal.ps1` - Enhanced terminal launcher
- `docs/development/testing/v0_0_2_linux_test_plan.md` - Linux/WSL testing
- `docs/development/testing/v0_0_2_powershell_test_plan.md` - PowerShell testing
