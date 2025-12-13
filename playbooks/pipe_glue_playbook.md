# Playbook: Pipe-Based Inter-Target Communication

## Audience
This playbook demonstrates pipe_glue for generating reader/writer code for Unix pipe communication between different targets.

## Overview
The `pipe_glue` module generates code for TSV and JSON pipe protocols, enabling targets to communicate via Unix pipes.

## When to Use

✅ **Use pipe_glue when:**
- Chaining different language targets via pipes
- Need TSV or JSON data exchange
- Building multi-stage pipelines
- Want producer-consumer patterns

## Example Usage

### TSV Pipe Communication

```prolog
:- use_module('src/unifyweaver/glue/pipe_glue').

% Generate AWK writer (producer)
Fields = [id, name, score].
?- generate_tsv_writer(awk, Fields, WriterCode).

% Generate Python reader (consumer)
?- generate_tsv_reader(python, Fields, ReaderCode).

% Use in pipeline:
% awk ... | python ...
```

### JSON Pipe Communication

```prolog
% Generate Go JSON writer
?- generate_json_writer(go, [id, name, score], WriterCode).

% Generate Rust JSON reader
?- generate_json_reader(rust, [id, name, score], ReaderCode).
```

### Pipeline Orchestration

```prolog
% Generate full pipeline script
Steps = [
    step(extract, awk, 'extract.awk', []),
    step(transform, python, 'transform.py', []),
    step(load, bash, 'load.sh', [])
].
?- generate_pipeline_script(Steps, [format(tsv)], Script).
```

## Supported Formats

- **TSV**: Tab-separated values (default)
- **JSON**: JSON lines format (one object per line)

## Supported Targets

AWK, Python, Bash, Go, Rust (readers and writers)

## See Also

- `playbooks/bash_pipeline_source_playbook.md` - Bash pipelines
- `playbooks/cross_target_glue_playbook.md` - Cross-language glue

## Summary

**Key Concepts:**
- ✅ Generate pipe readers/writers
- ✅ TSV and JSON formats
- ✅ Multi-target support
- ✅ Pipeline orchestration
