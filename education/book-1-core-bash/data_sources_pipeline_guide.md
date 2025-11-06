<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Data Sources Pipeline Guide

**Author:** Cline.bot (Claude Sonnet 4.5) & John William Creighton (@s243a)  
**Date:** October 16, 2025  
**Version:** 0.0.2

---

## Overview

UnifyWeaver's data sources system allows you to define external data sources in Prolog and compile them to efficient bash scripts. This guide covers the complete pipeline from definition to execution.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Loading Pattern](#module-loading-pattern)
3. [Source Definition](#source-definition)
4. [Compilation Process](#compilation-process)
5. [Pipeline Execution](#pipeline-execution)
6. [Complete Example](#complete-example)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### The Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│  User Code (Prolog)                     │
│  :- source(csv, users, [...])           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Public Interface                        │
│  src/unifyweaver/sources.pl             │
│  - Exports source/3                      │
│  - Registers with compiler               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Dynamic Source Compiler                 │
│  src/unifyweaver/core/                  │
│    dynamic_source_compiler.pl           │
│  - Plugin registry                       │
│  - Dispatch to plugins                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Source Plugins                          │
│  src/unifyweaver/sources/               │
│  - csv_source.pl                         │
│  - python_source.pl                      │
│  - http_source.pl                        │
│  - json_source.pl                        │
└─────────────────────────────────────────┘
```

### Key Components

**1. Public Interface (`src/unifyweaver/sources.pl`)**
- Exports `source/3` predicate
- Users call this to define sources
- Handles arity detection
- Registers with dynamic_source_compiler

**2. Dynamic Source Compiler**
- Maintains plugin registry
- Dispatches compilation requests
- Merges configuration options

**3. Source Plugins**
- Self-register via initialization
- Implement `compile_source/4`
- Generate bash code

---

## Module Loading Pattern

### The Problem We Solved

**Before v0.0.2:**
```prolog
:- use_module('src/unifyweaver/sources/csv_source').
:- use_module('src/unifyweaver/sources/http_source').
% ERROR: import/1: No permission to import 
%        http_source:validate_config/1 into user
%        (already imported from csv_source)
```

All plugins export predicates with the same names (`validate_config/1`, `compile_source/4`, etc.), causing import conflicts.

### The Solution: load_files with imports([])

**Fixed in v0.0.2:**
```prolog
% Load public interface (provides source/3)
:- use_module('src/unifyweaver/sources').

% Load plugins for side-effects ONLY (no imports)
:- load_files('src/unifyweaver/sources/csv_source', [imports([])]).
:- load_files('src/unifyweaver/sources/http_source', [imports([])]).
:- load_files('src/unifyweaver/sources/json_source', [imports([])]).
:- load_files('src/unifyweaver/sources/python_source', [imports([])]).
```

**Why This Works:**

1. **`load_files/2` with `imports([])`** explicitly prevents importing predicates
2. **Initialization directives still execute** - plugins register themselves
3. **No namespace pollution** - `user` module stays clean
4. **Module-qualified calls work** - `csv_source:compile_source(...)` still accessible

### Key Insight

The plugins need to be **loaded** (so initialization runs) but not **imported** (to avoid conflicts). `load_files` with `imports([])` gives us exactly this.

---

## Source Definition

### The source/3 Predicate

```prolog
source(Type, Name, Options)
```

**Parameters:**
- `Type` - Source plugin type (csv, python, http, json)
- `Name` - Predicate name to create
- `Options` - Configuration list

### Automatic Arity Detection

The system automatically determines predicate arity from:

1. **Explicit specification:**
   ```prolog
   :- source(csv, users, [csv_file('data.csv'), arity(3)]).
   ```

2. **Column list:**
   ```prolog
   :- source(csv, users, [
       csv_file('data.csv'),
       columns([id, name, role])  % Arity = 3
   ]).
   ```

3. **Header auto-detection:**
   ```prolog
   :- source(csv, users, [
       csv_file('data.csv'),
       has_header(true)  % Reads CSV to count columns
   ]).
   ```

4. **Default:**
   ```prolog
   :- source(python, script, [
       python_inline('...')  % Defaults to arity = 2
   ]).
   ```

### Example Source Definitions

#### CSV Source
```prolog
:- source(csv, users, [
    csv_file('examples/demo_users.csv'),
    has_header(true),
    delimiter(',')
]).
% Creates: users/4 (id, name, role, department)
```

#### Python Source
```prolog
:- source(python, analyze, [
    python_inline('
import sys
for line in sys.stdin:
    fields = line.strip().split(":")
    # Process and output
    print(f"{fields[0]}:{fields[1]}")
'),
    timeout(30)
]).
```

#### HTTP Source
```prolog
:- source(http, api_data, [
    url('https://api.example.com/data'),
    headers(['User-Agent: UnifyWeaver']),
    cache_duration(3600)
]).
```

---

## Compilation Process

### Step 1: Registration

When you call `source/3`:

```prolog
:- source(csv, users, [csv_file('data.csv'), has_header(true)]).
```

Internally:
```prolog
sources:source(csv, users, Options) :-
    determine_arity(Options, 4),  % Auto-detect: 4 columns
    register_dynamic_source(users/4, csv, Options).
    % Stores: dynamic_source_def(users/4, csv, Config)
```

### Step 2: Compilation

To compile a registered source to bash:

```prolog
?- compile_dynamic_source(users/4, [], BashCode).
```

This:
1. Looks up the source: `dynamic_source_def(users/4, csv, Config)`
2. Gets plugin module: `source_type_registry(csv, csv_source)`
3. Calls plugin: `csv_source:compile_source(users/4, Config, Options, BashCode)`

### Step 3: Bash Generation

The plugin generates bash code:

```bash
#!/bin/bash
# users - CSV source (arity 4)
# Columns: id, name, role, department

users() {
    local target_key="$1"
    
    if [[ -z "$target_key" ]]; then
        # Stream all rows
        awk -F"," '
        NR > 1 {
            gsub(/"/, "", $0)
            if (NF >= 4) print $1":"$2":"$3":"$4
        }
        ' examples/demo_users.csv
    else
        # Lookup mode
        awk -F"," -v key="$target_key" '
        NR > 1 {
            gsub(/"/, "", $0)
            if (NF >= 4 && $1 == key) print $1":"$2":"$3":"$4
        }
        ' examples/demo_users.csv
    fi
}

users_stream() {
    users
}
```

---

## Pipeline Execution

### Complete Pipeline Example

```prolog
% 1. Define source
:- source(csv, users, [
    csv_file('examples/demo_users.csv'),
    has_header(true)
]).

% 2. Compile to bash
compile_users_source :-
    compile_dynamic_source(users/4, [], BashCode),
    open('output/users.sh', write, Stream),
    write(Stream, BashCode),
    close(Stream),
    shell('chmod +x output/users.sh', _).

% 3. Create pipeline script
create_pipeline :-
    PipelineScript = '#!/bin/bash\n\
source output/users.sh\n\
\n\
echo "All users:"\n\
users_stream | while IFS=: read id name role dept; do\n\
    echo "  $name ($role) in $dept"\n\
done\n\
\n\
echo ""\n\
echo "Developers only:"\n\
users_stream | awk -F: \'$3 == "Developer"\' |\n\
    while IFS=: read id name role dept; do\n\
        echo "  $name"\n\
    done\n',
    
    open('output/run_pipeline.sh', write, Stream),
    write(Stream, PipelineScript),
    close(Stream),
    shell('chmod +x output/run_pipeline.sh', _).

% 4. Execute
execute_pipeline :-
    shell('bash output/run_pipeline.sh', Status),
    (Status = 0 -> writeln('✓ Success') ; writeln('✗ Failed')).

% 5. Main workflow
main :-
    compile_users_source,
    create_pipeline,
    execute_pipeline.
```

### Pipeline Operations

**Streaming:**
```bash
users_stream                    # Stream all records
```

**Filtering:**
```bash
users_stream | awk -F: '$3 == "Developer"'  # Filter by role
```

**Transformation:**
```bash
users_stream | while IFS=: read id name role dept; do
    echo "$name works in $dept"
done
```

**Aggregation:**
```bash
users_stream | awk -F: '{dept[$4]++} END {
    for (d in dept) print d": "dept[d]
}'
```

---

## Complete Example

See `examples/pipeline_demo.pl` for a complete working demonstration.

### Running the Demo

```bash
cd scripts/testing/test_env5
swipl -g main -t halt examples/pipeline_demo.pl
```

### What It Does

1. **Creates input data** - `examples/demo_users.csv` with 4 users
2. **Compiles source** - Generates `output/users.sh` with bash functions
3. **Builds pipeline** - Creates `output/run_pipeline.sh` with:
   - Stream all users
   - Filter developers
   - Count by department
4. **Executes pipeline** - Runs bash script and shows results
5. **Writes output** - Saves to `output/pipeline_results.txt`

### Expected Output

```
🎯 UnifyWeaver Data Sources Pipeline Demo
==============================================

📝 Step 1: Creating sample CSV data...
   ✓ Created examples/demo_users.csv with 4 users

🔨 Step 2: Compiling CSV source to bash...
   Compiling users/4...
   ✓ Generated output/users.sh

🚀 Step 3: Executing bash pipeline...
   ✓ Pipeline executed successfully

📊 Pipeline Results:
=== Pipeline Execution Results ===

1. All users:
  - Alice (Developer) in Engineering
  - Bob (Designer) in Design
  - Charlie (Manager) in Operations
  - Diana (Analyst) in Data

2. Developers only:
  - Alice

3. Users by department:
  - Operations: 1
  - Engineering: 1
  - Data: 1
  - Design: 1
```

---

## Troubleshooting

### Issue 1: "Unknown procedure: source/3"

**Problem:** source/3 not defined

**Solution:** Ensure you load the public interface:
```prolog
:- use_module('src/unifyweaver/sources').
```

### Issue 2: Import conflicts

**Problem:**
```
ERROR: import/1: No permission to import...
```

**Solution:** Use `load_files` with `imports([])`:
```prolog
:- load_files('src/unifyweaver/sources/csv_source', [imports([])]).
```

**NOT:**
```prolog
:- use_module('src/unifyweaver/sources/csv_source').  % Wrong!
```

### Issue 3: Source not found

**Problem:** 
```
Error: Source type X not registered
```

**Solution:** Ensure plugin is loaded:
```prolog
:- load_files('src/unifyweaver/sources/X_source', [imports([])]).
```

### Issue 4: Compilation fails

**Problem:** compile_dynamic_source/3 fails

**Check:**
1. Source is registered: `is_dynamic_source(name/arity)`
2. Config is valid: plugin's `validate_config/1`
3. Files exist if specified

### Issue 5: Pipeline produces no output

**Problem:** Bash script generates nothing

**Debug:**
```bash
# 1. Check generated bash script
cat output/users.sh

# 2. Test manually
bash output/users.sh
users_stream

# 3. Check file paths
ls -la examples/demo_users.csv

# 4. Test awk separately
awk -F"," 'NR > 1 {print}' examples/demo_users.csv
```

---

## Key Takeaways

1. **Module Loading Matters** - Use `load_files` with `imports([])` to avoid conflicts
2. **source/3 is the Interface** - Always go through the public interface
3. **Plugins Self-Register** - Initialization directives handle registration
4. **Compilation is Explicit** - Call `compile_dynamic_source/3` to generate bash
5. **Bash Execution is Separate** - Generated scripts execute independently

## Related Documentation

- **README.md** - Data source examples and quick start
- **context/source3_solution_summary.md** - Technical implementation details
- **examples/pipeline_demo.pl** - Complete working example
- **src/unifyweaver/sources/*.pl** - Plugin implementations

---

**Questions or issues?** Check the troubleshooting section or examine the working pipeline demo for reference.

---

## Navigation

**←** [Previous: Chapter 14: XML Source Plugin](14_xml_source_plugin.md) | [📖 Back to Book 1](README.md) | [Next: Appendix A: Recursion Patterns →](A_appendix_recursion_patterns.md)

