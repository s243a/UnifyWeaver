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

### 4.2. With Initialization (Advanced Method)

For interactive sessions or when using test environments:

```bash
cd /path/to/UnifyWeaver
swipl -f init.pl  # Load with initialization
```

The `init.pl` file sets up:
- File search paths for `library(unifyweaver/...)`
- Directory paths for module resolution
- Helper predicates and shortcuts

### 4.3. Test Environment

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

**Solutions**:
1. Use relative paths: `'src/unifyweaver/...'` instead of `library(unifyweaver/...)`
2. Or load `init.pl` first to set up library paths
3. Ensure running from project root directory

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
