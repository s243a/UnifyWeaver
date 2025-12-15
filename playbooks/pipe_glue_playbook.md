# Playbook: Pipe-Based Inter-Target Communication

## Audience
This playbook is a high-level guide for coding agents. It demonstrates pipe_glue for generating reader/writer code for Unix pipe communication between different targets.

## Workflow Overview
Use pipe_glue for pipe-based communication:
1. Generate pipe writers (producers) for a target language
2. Generate pipe readers (consumers) for a target language
3. Chain targets via Unix pipes using TSV or JSON formats
4. Generate pipeline orchestration scripts

## Agent Inputs
Reference the following artifacts:
1. **Glue Module** – `src/unifyweaver/glue/pipe_glue.pl` contains pipe generation predicates
2. **Module Documentation** – See module header for API details

## Key Features

- TSV and JSON pipe protocols
- Multi-target support (AWK, Python, Bash, Go, Rust)
- Producer/consumer pattern
- Pipeline orchestration

## Execution Guidance

Consult the module for predicate usage:

```prolog
:- use_module('src/unifyweaver/glue/pipe_glue').

% Generate TSV writer (producer)
Fields = [id, name, score].
?- generate_tsv_writer(awk, Fields, WriterCode).

% Generate TSV reader (consumer)
?- generate_tsv_reader(python, Fields, ReaderCode).

% Generate JSON pipes
?- generate_json_writer(go, Fields, WriterCode).
?- generate_json_reader(rust, Fields, ReaderCode).

% Generate full pipeline
Steps = [
    step(extract, awk, 'extract.awk', []),
    step(transform, python, 'transform.py', [])
].
?- generate_pipeline_script(Steps, [format(tsv)], Script).
```

## Expected Outcome
- Pipe readers/writers generated for targets
- Data flows correctly through pipes
- Pipeline scripts orchestrate multi-stage processing
- TSV and JSON formats handled correctly

## Citations
[1] src/unifyweaver/glue/pipe_glue.pl
