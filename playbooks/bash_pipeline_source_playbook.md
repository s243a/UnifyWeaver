# Playbook: Bash Pipeline Composer

## Audience
This playbook demonstrates composing multi-stage bash pipelines as UnifyWeaver data sources.

## Overview
The `bash_pipeline_source` plugin lets you declare Prolog predicates that execute complex bash pipeline chains (grep | awk | sort | uniq, etc.).

## When to Use

✅ **Use bash_pipeline when:**
- Composing multi-tool pipelines
- Need grep + awk + sort chains
- Want reusable pipeline abstractions
- Processing text with Unix tools

## Agent Inputs

1. **Executable Records** – `playbooks/examples_library/bash_pipeline_examples.md`
2. **Source Module** – `src/unifyweaver/sources/bash_pipeline_source.pl`

## Execution Guidance

### Example 1: Grep + AWK Pipeline

```bash
cd /path/to/UnifyWeaver

perl scripts/extract_records.pl playbooks/examples_library/bash_pipeline_examples.md \
    pipeline_basic > tmp/pipeline_basic.sh
chmod +x tmp/pipeline_basic.sh
bash tmp/pipeline_basic.sh
```

**Expected Output:**
```
Compiling bash pipeline: find_errors/1
Generated: tmp/find_errors.sh
Testing find_errors/1:
ERROR:Connection failed
ERROR:Timeout occurred
```

### Example 2: Multi-Stage Data Processing

```bash
perl scripts/extract_records.pl playbooks/examples_library/bash_pipeline_examples.md \
    pipeline_complex > tmp/pipeline_complex.sh
chmod +x tmp/pipeline_complex.sh
bash tmp/pipeline_complex.sh
```

**Expected Output:**
```
Compiling bash pipeline: top_sellers/1
Generated: tmp/top_sellers.sh
Testing top_sellers/1:
Widget:450
Gadget:250
```

### Example 3: Sort + Uniq Pipeline

```bash
perl scripts/extract_records.pl playbooks/examples_library/bash_pipeline_examples.md \
    pipeline_aggregate > tmp/pipeline_aggregate.sh
chmod +x tmp/pipeline_aggregate.sh
bash tmp/pipeline_aggregate.sh
```

**Expected Output:**
```
Compiling bash pipeline: unique_users/1
Generated: tmp/unique_users.sh
Testing unique_users/1:
alice
bob
charlie
```

## Configuration Options

- `stages(List)` - List of `stage(Tool, Script, Args)` (required)
- `input_file(File)` - Input file (default: stdin)
- `output_file(File)` - Output file (default: stdout)

## Stage Definitions

```prolog
% Grep stage
stage(grep, 'grep', ['-i', 'error'])

% AWK stage
stage(awk, 'awk', ['-F:', '{print $1}'])

% Sort stage
stage(sort, 'sort', ['-nr'])

% Uniq stage
stage(uniq, 'uniq', [])
```

## See Also

- `playbooks/awk_source_playbook.md` - AWK foreign functions
- `playbooks/bash_parallel_playbook.md` - Parallel bash execution

## Summary

**Key Concepts:**
- ✅ Compose multi-stage Unix pipelines
- ✅ Reusable pipeline abstractions
- ✅ Supports grep, awk, sort, uniq, sed, etc.
- ✅ Configurable input/output
