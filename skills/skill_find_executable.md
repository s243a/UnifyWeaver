# Skill: Find Executable

## Audience
This skill is for coding agents that need to locate executables that may not be in the system's PATH.

## Workflow Overview
This skill attempts to find a specified executable by searching a list of common installation directories for different operating systems. It returns the full path to the executable if found.

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
