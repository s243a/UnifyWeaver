# PowerShell Target Implementation Decision

## Current Issue
The inline approach (embedding bash scripts as strings in PowerShell) is encountering persistent Cygwin exit 126 errors due to complex quoting/escaping requirements when passing multi-line bash scripts through PowerShell → Cygwin bash.

## Two Approaches Considered

### Approach 1: Inline (Current)
**Pros:**
- Less file clutter
- Self-contained PowerShell scripts

**Cons:**
- Complex quoting/escaping issues
- Difficult to debug when errors occur
- Not easily editable
- Fragile across different shells (Cygwin/WSL/Git Bash)

### Approach 2: Separate Bash Files (Recommended)
**Pros:**
- Clean separation of concerns
- Bash files can be executed directly OR via PowerShell wrapper
- Easier to debug and edit
- No quoting/escaping issues
- Matches existing `examples/bash_outputs/` pattern
- Users can choose their preferred invocation method

**Cons:**
- More files in output directory

## Decision
**Default: Approach 2 (Separate Files)**, with Approach 1 (Inline) preserved as an option for future use.

### Implementation Strategy

#### Default: Separate Files
1. Generate `.sh` bash file using existing bash compiler
2. Generate `.ps1` PowerShell wrapper that calls `uw-bash script.sh`
3. Users can run either:
   - `bash script.sh` (direct)
   - `pwsh script.ps1` (via wrapper)

#### Preserved: Inline Option
- Keep inline bash templates in `templates/powershell/inline/`
- Preserve quoting functions in compatibility layer
- Document inline approach for future use
- Can be enabled via compiler option if needed

### Template Organization
```
templates/powershell/
├── separate/           # Default: separate .sh files
│   └── wrapper.ps1     # Calls uw-bash script.sh
└── inline/             # Optional: embedded bash
    └── embedded.ps1    # Contains bash as heredoc string
```

### Benefits of Hybrid Approach
- **Separate files (default)**: Robust, debuggable, user-friendly
- **Inline option (preserved)**: Available for specific use cases
- **Maximum reuse**: Existing bash compiler unchanged
- **Future flexibility**: Easy to switch or support both
