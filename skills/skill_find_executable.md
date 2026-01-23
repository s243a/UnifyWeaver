# Skill: Find Executable

Locate executables that may not be in the system's PATH by searching common installation directories.

## When to Use

- Command fails with "command not found" but you know it's installed
- Need to find language runtimes (perl, python, swipl) on different OSes
- Setting up cross-platform scripts that need absolute paths

## Quick Start

```bash
# The skill searches OS-specific directories automatically
# Example: Find SWI-Prolog on the system

# On Linux, check these locations:
ls /usr/bin/swipl /usr/local/bin/swipl /snap/bin/swipl 2>/dev/null | head -1

# On macOS, also check:
ls /opt/homebrew/bin/swipl /Applications/SWI-Prolog.app/Contents/MacOS/swipl 2>/dev/null | head -1

# On Windows (PowerShell):
@("C:\Program Files\swipl\bin\swipl.exe", "C:\Program Files (x86)\swipl\bin\swipl.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
```

## Agent Inputs
- **Executable Name**: The name of the executable to find (e.g., `perl`, `swipl`).

## Execution Guidance
An agent should use this skill when a command fails with a "command not found" error. The agent can then use the returned path to execute the command.

### Example Usage
If `perl --version` fails, the agent can use this skill to find `perl`.

## Implementation
The skill is implemented by a script that searches for the executable in a predefined list of paths. The list of paths is specific to the operating system.

### Windows
- `C:\Strawberry\perl\bin`
- `C:\Program Files\swipl\bin`
- `C:\Program Files (x86)\swipl\bin`

### Linux
- `/usr/bin`
- `/usr/local/bin`
- `/snap/bin`

### macOS
- `/usr/bin`
- `/usr/local/bin`
- `/opt/homebrew/bin`
- `/Applications/SWI-Prolog.app/Contents/MacOS`

## Expected Outcome
- If the executable is found, the skill returns the absolute path to the executable.
- If the executable is not found, the skill returns an empty string.
- The agent can then use the returned path to execute the command, for example: `& "C:\Strawberry\perl\bin\perl.exe" my_script.pl`.
