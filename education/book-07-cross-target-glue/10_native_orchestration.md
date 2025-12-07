<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 10: Native Binary Orchestration

## Overview

Managing compiled binaries requires additional infrastructure: tracking compilations, checking for staleness, and orchestrating mixed pipelines. This chapter covers:

- Binary management
- Compilation on demand
- Cross-compilation
- Mixed pipelines (shell + native)
- Performance optimization

## Binary Management

### Registering Binaries

```prolog
register_binary(Pred/Arity, Target, Path, Options).
```

Example:
```prolog
% Register a compiled Go binary
register_binary(transform/2, go, './bin/transform', [
    optimized(true),
    compiled_at('2024-01-15T10:30:00')
]).

% Register a Rust binary
register_binary(aggregate/2, rust, './target/release/aggregate', []).
```

### Querying Binaries

```prolog
?- compiled_binary(transform/2, Target, Path).
Target = go,
Path = './bin/transform'.

?- compiled_binary(Pred, rust, Path).
Pred = aggregate/2,
Path = './target/release/aggregate'.
```

### Compilation on Demand

```prolog
compile_if_needed(Pred/Arity, Target, SourcePath, BinaryPath).
```

This predicate:
1. Checks if binary exists
2. Compares modification times
3. Compiles if source is newer
4. Registers the new binary

```prolog
?- compile_if_needed(transform/2, go, 'transform.go', BinaryPath).
BinaryPath = 'transform'.  % Compiled fresh or reused existing
```

### Staleness Detection

```prolog
% Check if recompilation needed
needs_recompile(Pred/Arity, Target, SourcePath) :-
    compiled_binary(Pred/Arity, Target, BinaryPath),
    exists_file(BinaryPath),
    exists_file(SourcePath),
    time_file(BinaryPath, BinTime),
    time_file(SourcePath, SrcTime),
    SrcTime > BinTime.
```

## Mixed Pipeline Generation

### Combining Shell and Native

```prolog
generate_native_pipeline(
    [
        step(parse, awk, 'parse.awk', []),
        step(transform, go, './transform', [parallel(4)]),
        step(filter, python, 'filter.py', []),
        step(aggregate, rust, './aggregate', [])
    ],
    [input('data.tsv'), output('result.tsv')],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

cat "data.tsv" \
    | awk -f "parse.awk" \
    | ./transform \
    | python3 "filter.py" \
    | ./aggregate \
    > "result.tsv"
```

### Automatic Binary Detection

The pipeline generator checks for binaries:

```prolog
% If binary doesn't exist, add build step
generate_native_pipeline(Steps, Options, Script) :-
    findall(S, (member(S, Steps), native_step(S), missing_binary(S)), Missing),
    (Missing \= [] ->
        generate_build_and_run(Steps, Missing, Options, Script)
    ;
        generate_run_only(Steps, Options, Script)
    ).
```

### Build-and-Run Script

```prolog
generate_native_pipeline(
    [step(transform, go, './transform', [source('transform.go')])],
    [auto_build(true)],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

# Build step (if needed)
if [ ! -f "./transform" ] || [ "transform.go" -nt "./transform" ]; then
    echo "Building transform.go..."
    go build -ldflags="-s -w" -o "./transform" "transform.go"
fi

# Run pipeline
cat input.tsv | ./transform > output.tsv
```

## Cross-Platform Orchestration

### Multi-Platform Build

```prolog
generate_multiplatform_build(
    [
        binary(transform, go, 'transform.go'),
        binary(aggregate, rust, 'src/main.rs')
    ],
    [linux-amd64, darwin-arm64, windows-amd64],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

PLATFORMS="linux-amd64 darwin-arm64 windows-amd64"
DIST_DIR="dist"

mkdir -p "$DIST_DIR"

for PLATFORM in $PLATFORMS; do
    OS=$(echo $PLATFORM | cut -d- -f1)
    ARCH=$(echo $PLATFORM | cut -d- -f2)

    echo "Building for $PLATFORM..."
    mkdir -p "$DIST_DIR/$PLATFORM"

    # Go binary
    GOOS=$OS GOARCH=$ARCH go build -ldflags="-s -w" \
        -o "$DIST_DIR/$PLATFORM/transform" transform.go

    # Rust binary (using cross)
    cross build --release --target ${PLATFORM/-/-unknown-}-gnu
    cp target/${PLATFORM/-/-unknown-}-gnu/release/aggregate \
        "$DIST_DIR/$PLATFORM/"
done

echo "Build complete!"
find "$DIST_DIR" -type f -executable
```

## Pipeline Optimization

### Buffering Configuration

```prolog
generate_native_pipeline(
    [
        step(fast_producer, go, './producer', [buffer(block(1048576))]),
        step(slow_consumer, rust, './consumer', [buffer(line)])
    ],
    [],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

# Use stdbuf for buffer control
stdbuf -oL ./producer \
    | stdbuf -iL ./consumer
```

### Parallel Fan-Out

```prolog
generate_native_pipeline(
    [
        step(split, bash, 'split.sh', []),
        parallel_steps([
            step(proc_a, go, './proc_a', []),
            step(proc_b, rust, './proc_b', []),
            step(proc_c, go, './proc_c', [])
        ]),
        step(merge, bash, 'merge.sh', [])
    ],
    [],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

# Create named pipes
mkfifo /tmp/pipe_a /tmp/pipe_b /tmp/pipe_c

# Start parallel processors
./proc_a < /tmp/pipe_a > /tmp/out_a &
./proc_b < /tmp/pipe_b > /tmp/out_b &
./proc_c < /tmp/pipe_c > /tmp/out_c &

# Split input to all pipes
bash split.sh | tee /tmp/pipe_a /tmp/pipe_b > /tmp/pipe_c

# Wait for completion
wait

# Merge results
bash merge.sh /tmp/out_a /tmp/out_b /tmp/out_c

# Cleanup
rm /tmp/pipe_* /tmp/out_*
```

### GNU Parallel Integration

```prolog
generate_native_pipeline(
    [
        step(process, go, './process', [
            gnu_parallel(true),
            jobs(16),
            chunk(10000)
        ])
    ],
    [input('huge.tsv')],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

cat "huge.tsv" \
    | parallel --pipe -j16 -N10000 ./process
```

## Resource Management

### Memory Limits

```prolog
generate_native_pipeline(
    [step(transform, go, './transform', [memory_limit('2G')])],
    [],
    Script
).
```

Generated (Linux):

```bash
#!/bin/bash
set -euo pipefail

# Set memory limit
systemd-run --scope -p MemoryMax=2G ./transform
```

### CPU Affinity

```prolog
generate_native_pipeline(
    [step(compute, rust, './compute', [cpu_cores([0,1,2,3])])],
    [],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

taskset -c 0,1,2,3 ./compute
```

## Error Handling

### Exit Code Propagation

```prolog
generate_native_pipeline(
    Steps,
    [error_handling(exit_codes)],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

# Capture exit codes
./step1 | ./step2 | ./step3

# Check all exit codes
PIPESTATUS_ALL="${PIPESTATUS[@]}"
for code in $PIPESTATUS_ALL; do
    if [ "$code" -ne 0 ]; then
        echo "Pipeline step failed with exit code $code" >&2
        exit $code
    fi
done
```

### Timeout Handling

```prolog
generate_native_pipeline(
    [step(slow, go, './slow', [timeout(300)])],  % 5 minutes
    [],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

timeout 300 ./slow || {
    echo "Step timed out after 300 seconds" >&2
    exit 124
}
```

## Monitoring

### Progress Tracking

```prolog
generate_native_pipeline(
    Steps,
    [monitor(progress)],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

# With pv for progress
cat input.tsv \
    | pv -l -N "input" \
    | ./transform \
    | pv -l -N "transform" \
    | ./aggregate
```

### Throughput Metrics

```prolog
generate_native_pipeline(
    Steps,
    [monitor(throughput)],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

START_TIME=$(date +%s.%N)

LINES=$(cat input.tsv | ./transform | ./aggregate | tee output.tsv | wc -l)

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)
RATE=$(echo "scale=2; $LINES / $DURATION" | bc)

echo "Processed $LINES records in ${DURATION}s (${RATE} rec/s)"
```

## Chapter Summary

- **Binary management** tracks compiled artifacts
- **Staleness detection** triggers recompilation
- **Mixed pipelines** combine shell and native
- **Parallel fan-out** distributes work
- **Resource limits** prevent runaway processes
- **Monitoring** tracks pipeline health

## Next Steps

In Chapter 11, we'll explore HTTP services:
- Server generation
- Client generation
- Distributed pipelines

## Exercises

1. **Binary management**: Write predicates to list all stale binaries in a project.

2. **Mixed pipeline**: Generate a pipeline with AWK → Go → Rust → Python stages.

3. **Parallel processing**: Generate a pipeline that processes 16 files in parallel.

4. **Resource limits**: Generate a pipeline with 1GB memory limit and 2-CPU affinity.

## Code Examples

See `examples/04-native-parallel/` for:
- `build.sh` - Multi-binary build
- `pipeline.sh` - Mixed orchestration
- `parallel.sh` - Fan-out example
- `monitor.sh` - Progress tracking

---

## Navigation

**←** [Previous: Chapter 9: Go and Rust Code Generation](09_native_code_gen) | [📖 Book 7: Cross-Target Glue](README) | [Next: Chapter 11: HTTP Services →](11_http_services)
