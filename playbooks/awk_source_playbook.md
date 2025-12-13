# Playbook: AWK Foreign Function Binding

## Audience
This playbook demonstrates how to use AWK scripts/commands as data sources through UnifyWeaver's foreign function binding system.

## Overview
The `awk_source` plugin lets you declare Prolog predicates that execute AWK commands and return results. This enables AWK to act as a foreign function callable from Prolog.

## Key Differences from awk_advanced_playbook

| Aspect | awk_advanced | awk_source (this playbook) |
|--------|--------------|----------------------------|
| **Purpose** | Compile Prolog to AWK code | Call AWK as foreign function |
| **Direction** | Prolog → AWK compilation | AWK → Prolog binding |
| **Use Case** | Generate AWK scripts | Use existing AWK tools |
| **Module** | Advanced compiler | Dynamic source compiler |

## When to Use This Playbook

✅ **Use awk_source when:**
- You have existing AWK scripts/tools to call
- You want AWK to process data and return results
- You need foreign function binding for AWK
- You're integrating with AWK-based systems

❌ **Use awk_advanced when:**
- You want to compile Prolog predicates to AWK
- You're generating AWK code from Prolog

## Agent Inputs

Reference the following artifacts:
1. **Executable Records** – `playbooks/examples_library/awk_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/awk_source.pl`

## Execution Guidance

### Example 1: Basic AWK Command Binding

```bash
# Navigate to project root
cd /path/to/UnifyWeaver

# Extract and run example
perl scripts/extract_records.pl playbooks/examples_library/awk_source_examples.md \
    awk_command_basic > tmp/awk_command_basic.sh
chmod +x tmp/awk_command_basic.sh
bash tmp/awk_command_basic.sh
```

**Expected Output:**
```
Compiling AWK source: count_lines/1
Generated: tmp/count_lines.sh
Testing count_lines/1:
5
```

### Example 2: AWK File Binding

```bash
# Extract and run example
perl scripts/extract_records.pl playbooks/examples_library/awk_source_examples.md \
    awk_file_basic > tmp/awk_file_basic.sh
chmod +x tmp/awk_file_basic.sh
bash tmp/awk_file_basic.sh
```

**Expected Output:**
```
Compiling AWK source: extract_field/2
Generated: tmp/extract_field.sh
Testing extract_field/2 with field 2:
Alice
Bob
Charlie
```

### Example 3: AWK with Field Separator

```bash
# Extract and run example
perl scripts/extract_records.pl playbooks/examples_library/awk_source_examples.md \
    awk_separator > tmp/awk_separator.sh
chmod +x tmp/awk_separator.sh
bash tmp/awk_separator.sh
```

**Expected Output:**
```
Compiling AWK source: parse_csv/1
Generated: tmp/parse_csv.sh
Testing parse_csv/1:
id:1,name:Alice,score:85
id:2,name:Bob,score:92
id:3,name:Charlie,score:78
```

### Example 4: Query AWK Source Info

```bash
# Check AWK source plugin information
swipl -g "use_module('src/unifyweaver/sources/awk_source'), \
    awk_source:source_info(Info), \
    write(Info), nl, halt"
```

**Expected Output:**
```
info(name('AWK Source'),version('1.0.0'),description('Execute AWK commands as data sources'),supported_arities([1,2]))
```

## Architecture

### Dynamic Source Flow

```
┌──────────────────────────────────────────┐
│  Prolog Declaration                       │
│  :- dynamic_source(pred/1, awk, [        │
│      awk_command('...')                   │
│  ])                                       │
└──────────────────┬────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ awk_source Plugin   │
         │ - Validates config  │
         │ - Generates bash    │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Generated Bash Code │
         │ - Executes AWK      │
         │ - Returns results   │
         └─────────────────────┘
```

### Configuration Options

- `awk_command(Cmd)` - AWK command string to execute
- `awk_file(File)` - AWK script file to run
- `input_file(File)` - Optional input file
- `field_separator(Sep)` - Field separator (default: `:`)

## Expected Outcomes

✅ **Successful execution:**
- AWK sources compile to bash functions
- Functions execute AWK and return results
- Results integrate seamlessly with Prolog

❌ **Common errors:**
- Missing `awk_command` or `awk_file` in config
- AWK syntax errors in command/file
- Missing input files

## Integration with UnifyWeaver

### Combining with Other Sources

```prolog
% Use AWK for data extraction
:- dynamic_source(get_users/1, awk, [
    awk_command('{ print $1 }'),
    input_file('users.txt')
]).

% Use SQLite for lookups
:- dynamic_source(user_details/2, sqlite, [
    db_file('users.db'),
    query('SELECT * FROM users WHERE id = ?')
]).

% Combine in pipeline
process_users :-
    get_users(User),
    user_details(User, Details),
    format('~w: ~w~n', [User, Details]).
```

## See Also

- `playbooks/awk_advanced_playbook.md` - AWK code generation (inverse direction)
- `playbooks/bash_parallel_playbook.md` - Parallel AWK processing
- `playbooks/csv_data_source_playbook.md` - CSV processing (alternative to AWK)
- `src/unifyweaver/sources/awk_source.pl` - AWK source implementation

## Summary

**Key Concepts:**
- ✅ Foreign function binding for AWK
- ✅ Execute AWK commands from Prolog
- ✅ Reuse existing AWK scripts/tools
- ✅ Configurable input files and separators
- ✅ Supports arity 1 and 2 predicates

**Configuration:**
- `awk_command(Cmd)` or `awk_file(File)` (required)
- `input_file(File)` (optional)
- `field_separator(Sep)` (optional, default: `:`)

**When to Use:**
- Calling existing AWK scripts
- AWK-based data extraction
- Foreign function integration with AWK
