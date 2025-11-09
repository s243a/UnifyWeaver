# Workflow Execution Environment

This document defines the execution context for UnifyWeaver workflows and playbooks.

## Project Structure

### Primary Repository
```
UnifyWeaver/
├── src/unifyweaver/          # Core source
│   ├── core/                 # Compiler, runtime
│   │   ├── compiler_driver.pl
│   │   ├── stream_compiler.pl
│   │   └── advanced/
│   │       ├── test_runner_generator.pl
│   │       └── test_runner_inference.pl
│   └── ...
├── education/                # Tutorial materials
│   └── output/               # Compiled outputs
│       └── advanced/         # Advanced recursion examples
├── examples/                 # Playbooks
├── docs/                     # Documentation
└── tests/                    # Test infrastructure
```

### Sandbox Repository (Development)
```
UnifyWeaver-sandbox/          # Experimental features
└── [same structure as primary]
```

### Education Repository
```
UnifyWeaver_Education-sandbox/
└── book-workflow/            # Workflow materials
    ├── examples_library/     # Example database
    │   ├── compilation_examples.md
    │   ├── testing_examples.md
    │   ├── recursion_examples.md
    │   └── log_examples.md
    ├── ch1_introduction/
    └── playbook_format.md
```

## Initialization Patterns

### When to Use init.pl

**Create init.pl when:**
- Working on a long-running development session
- Testing multiple related predicates
- Need persistent module configuration
- Using an interactive SWI-Prolog REPL

**Skip init.pl when:**
- Running single playbook execution
- Using one-off compilation commands
- Examples include full initialization inline
- Executing from scripts or automation

### init.pl Template

For interactive development sessions:

```prolog
% init.pl - Session initialization for UnifyWeaver development
:- initialization(setup_environment).

setup_environment :-
    % Set project root
    (   getenv('UNIFYWEAVER_HOME', Home) ->
        true
    ;   working_directory(Home, Home)  % Use current directory
    ),

    % Configure file search paths
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    asserta(file_search_path(education, 'education')),

    % Load common modules
    use_module(library(filesex)),
    use_module(library(readutil)),

    format('Environment initialized: ~w~n', [Home]).
```

### Inline Initialization (Preferred for Playbooks)

For reproducibility, playbooks should use inline initialization:

```bash
swipl -q -g "
    % Set up environment
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),

    % Execute task
    [factorial.pl],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [], Scripts),
    halt"
```

**Why inline?**
- Self-contained (no external dependencies)
- Works across platforms
- Reproducible from any starting point
- Clear what environment is being used

## Platform-Specific Considerations

### WSL (Windows Subsystem for Linux)
```bash
# Project root typically at:
/mnt/c/Users/username/path/to/UnifyWeaver

# Example:
cd /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/context/sandbox/UnifyWeaver-sandbox
```

### Termux (Android)
```bash
# Project root typically at:
/data/data/com.termux/files/home/UnifyWeaver

# Memory constraints - use streaming compilation:
swipl --stack-limit=512m -g "compile(...)"
```

### Native Linux
```bash
# Standard locations:
~/projects/UnifyWeaver
/opt/UnifyWeaver

# Full stack available
```

## Environment Variables

### UNIFYWEAVER_HOME
**Purpose:** Root directory of UnifyWeaver installation
**Default:** Current working directory
**Override:** Set before running playbooks

```bash
export UNIFYWEAVER_HOME=/path/to/UnifyWeaver
cd $UNIFYWEAVER_HOME
```

### UNIFYWEAVER_EXEC_MODE
**Purpose:** Execution environment (for compatibility layer)
**Values:** `wsl`, `termux`, `native`
**Used by:** PowerShell compatibility layer

## Working Directory Conventions

### For Compilation Tasks
```bash
# Run from project root
cd $UNIFYWEAVER_HOME
swipl -g "compile(...)"
```

**Why project root?**
- Consistent file_search_path behavior
- Generated files go to expected locations
- Documentation references work correctly

### For Testing
```bash
# Run from output directory
cd education/output/advanced
./test_runner.sh
```

Or with absolute paths:
```bash
$UNIFYWEAVER_HOME/education/output/advanced/test_runner.sh
```

### For Development
```bash
# Can run from anywhere if using absolute paths
# Inline initialization with explicit paths preferred
```

## Example Usage in Playbooks

### Minimal Reference
For simple, well-documented examples:

```markdown
**Environment:** Inline initialization (see [workflow_environment.md])

**Execute:**
```bash
cd $UNIFYWEAVER_HOME
swipl -q -g "asserta(file_search_path(unifyweaver, 'src/unifyweaver')), ..."
```
```

### Full Context
For complex scenarios:

```markdown
**Environment Setup:**

Before executing this playbook:
1. Navigate to project root: `cd $UNIFYWEAVER_HOME`
2. Verify you're in correct location: `ls src/unifyweaver/core`

**Platform Notes:**
- **WSL**: Use `/mnt/c/...` paths
- **Termux**: May need `--stack-limit=512m`
- **Linux**: Standard paths work

**Execution:**
[commands with inline initialization]
```

## Output Location Conventions

### Compiled Scripts
- **Default**: `education/output/advanced/`
- **Custom**: Pass `output_directory(Path)` option to `compile/3`

### Test Runners
- **Same as scripts**: `education/output/advanced/test_runner.sh`
- **Generated by**: `test_runner_inference.pl`

### Temporary Files
- **Prolog source**: `/tmp/` for examples
- **Working data**: `education/tmp/` for persistent intermediate files

## Common Patterns

### Pattern 1: Quick Test
```bash
# Generate, compile, test in one go
cd /tmp
cat > factorial.pl <<'EOF'
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
EOF

cd $UNIFYWEAVER_HOME
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/factorial.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [], Scripts),
    writeln(Scripts),
    halt"
```

### Pattern 2: Generate Test Runner
```bash
cd $UNIFYWEAVER_HOME
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/test_runner.sh'),
    halt"
```

### Pattern 3: Execute and Verify
```bash
chmod +x education/output/advanced/test_runner.sh
./education/output/advanced/test_runner.sh
```

## Troubleshooting

### "Cannot find module unifyweaver(core/compiler_driver)"
**Cause:** file_search_path not set correctly
**Fix:** Ensure `asserta(file_search_path(unifyweaver, 'src/unifyweaver'))` runs first

### "Permission denied" on generated scripts
**Cause:** Scripts not executable
**Fix:** `chmod +x path/to/script.sh`

### Paths don't match documentation
**Cause:** Running from wrong directory
**Fix:** `cd $UNIFYWEAVER_HOME` before execution

## References

- [Playbook Format Specification](../../../UnifyWeaver_Education-sandbox/book-workflow/playbook_format.md)
- [Workflow Philosophy](workflow_philosophy.md)
- [Test Runner Inference](../../TEST_RUNNER_INFERENCE.md)
