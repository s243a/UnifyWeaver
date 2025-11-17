# Skill: UnifyWeaver Environment Setup

## 1. Purpose

This skill enables agents to properly set up the Prolog environment for running UnifyWeaver scripts and playbooks.

## 2. When to Use

Use this skill when:
- Running Prolog scripts that use UnifyWeaver modules
- Executing playbooks that compile sources or targets
- Getting errors like "source_sink 'library(...)' does not exist"
- Getting errors like "Unknown procedure: ..."

## 3. Environment Requirements

### 3.1. Working Directory

**CRITICAL**: All UnifyWeaver commands must be run from the **project root directory**.

The project root contains:
- `src/unifyweaver/` - Core modules
- `playbooks/` - Playbook files
- `templates/` - Template files
- `scripts/` - Utility scripts

### 3.2. Module Loading

UnifyWeaver modules use **relative paths** from the project root. Scripts load modules like:

```prolog
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/python_source').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
```

**Important**:
- Paths are relative to the current working directory
- Do NOT use `library(unifyweaver/...)` syntax unless a proper init file is loaded
- The file search path is set up by `init.pl` or similar initialization files

## 4. Running Scripts

### 4.1. Direct Execution (Simple Method)

For standalone scripts that set up their own paths:

```bash
cd /path/to/UnifyWeaver  # Change to project root
swipl -g "consult('path/to/script.pl'), goal, halt"
```

### 4.2. With init.pl (Recommended for Scripts)

The **recommended approach** for running UnifyWeaver scripts is to use `init.pl`, which automatically sets up the environment.

#### What init.pl Does

The `init.pl` file in the project root:
1. Determines the project root from its own location
2. Sets up `file_search_path(unifyweaver, 'src/unifyweaver')`
3. Sets up `library_directory('src')` for module loading
4. Allows using `library(unifyweaver/...)` syntax in scripts

#### Using init.pl

**From project root** (most common):
```bash
cd /path/to/UnifyWeaver
swipl -f init.pl -g "goal, halt"
```

**Example - Loading and using a module**:
```bash
swipl -f init.pl -g "
    use_module(library(unifyweaver/sources)),
    format('Module loaded successfully~n', []),
    halt"
```

**Example - Interactive session**:
```bash
cd /path/to/UnifyWeaver
swipl -f init.pl
# Now you can use: use_module(library(unifyweaver/sources)).
```

**Example - Running a script that uses UnifyWeaver**:
```bash
swipl -f init.pl -g "
    consult('tmp/my_script.pl'),
    run_my_goal,
    halt"
```

#### Important Notes

- **Always run from project root**: `init.pl` must be in the project root directory
- **Use `-f init.pl`**: This loads the initialization before running your goal
- **No manual path setup needed**: init.pl handles all path configuration automatically

### 4.3. Inline Initialization (Preferred for Playbooks)

For reproducibility, playbooks should use inline initialization. This is the recommended method for non-interactive scripts.

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

### 4.4. Test Environment

For isolated testing, use the test environment initialization:

```bash
cd /path/to/UnifyWeaver
bash scripts/testing/init_testing.sh
cd scripts/testing/test_env
./unifyweaver.sh
```

This creates a self-contained environment with all modules copied.

## 5. Common Errors and Solutions

### Error: "source_sink 'library(unifyweaver/...)' does not exist"

**Cause**: Module path not in search path

**Solutions** (in order of preference):
1. **Use init.pl**: `swipl -f init.pl -g "use_module(library(unifyweaver/sources)), halt"`
2. Use relative paths: `'src/unifyweaver/...'` instead of `library(unifyweaver/...)`
3. Ensure running from project root directory

**Example that works**:
```bash
cd /path/to/UnifyWeaver
swipl -f init.pl -g "
    use_module(library(unifyweaver/sources)),
    halt"
```

### Error: "Unknown procedure: ..."

**Cause**: Module not loaded

**Solutions**:
1. Add `:- use_module('src/unifyweaver/...')` directive
2. Check that module file exists and exports the predicate
3. Verify working directory is project root

### Error: "Found file ... relative to the current working directory" (Warning)

**Cause**: Using deprecated relative path loading

**Solutions**:
- This is just a warning, not an error
- The code will still work
- To suppress: use absolute paths or proper init file

## 6. Best Practices for Playbooks

When creating playbooks that execute Prolog code:

1. **Set working directory first**:
   ```bash
   cd /path/to/UnifyWeaver
   ```

2. **Use relative module paths**:
   ```prolog
   :- use_module('src/unifyweaver/sources').
   ```

3. **Create temporary scripts in `tmp/` directory**:
   ```bash
   mkdir -p tmp
   cat > tmp/my_script.pl <<'EOF'
   ...
   EOF
   ```

4. **Run from project root**:
   ```bash
   swipl -g "consult('tmp/my_script.pl'), run, halt"
   ```

## 7. References

- Test environment setup: `scripts/testing/init_testing.sh`
- Template initialization: `templates/init_template.pl`
- Example playbooks: `playbooks/examples_library/`
