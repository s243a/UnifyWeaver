<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 9: Go and Rust Code Generation

## Overview

For performance-critical pipelines, Go and Rust provide compiled, type-safe alternatives to shell scripts. This chapter covers:

- Go pipe-compatible wrappers
- Rust pipe-compatible wrappers
- TSV and JSON modes
- Parallel processing
- Build script generation

## The Native Glue Module

Located at `src/unifyweaver/glue/native_glue.pl`:

```prolog
:- module(native_glue, [
    % Toolchain detection
    detect_go/1,
    detect_rust/1,
    detect_cargo/1,

    % Go code generation
    generate_go_pipe_main/3,
    generate_go_wrapper/4,
    generate_go_build_script/3,

    % Rust code generation
    generate_rust_pipe_main/3,
    generate_rust_wrapper/4,
    generate_rust_build_script/3,

    % Cross-compilation
    cross_compile_targets/1,
    generate_cross_compile/4
]).
```

## Toolchain Detection

### Detecting Go

```prolog
?- detect_go(Version).
Version = '1.21.0'.

?- detect_go(none).  % If not installed
```

### Detecting Rust

```prolog
?- detect_rust(Version).
Version = '1.75.0'.

?- detect_cargo(Version).
Version = '1.75.0'.
```

## Go Code Generation

### Basic TSV Processing

```prolog
?- generate_go_pipe_main(
    '    return fields',  % Just echo fields back
    [fields([name, age, city])],
    Code
).
```

Generated:

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
)

func process(fields []string) []string {
    return fields
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    // Increase buffer size for large lines
    buf := make([]byte, 0, 1024*1024)
    scanner.Buffer(buf, 10*1024*1024)

    for scanner.Scan() {
        line := scanner.Text()
        fields := strings.Split(line, "\t")
        result := process(fields)
        if result != nil {
            fmt.Println(strings.Join(result, "\t"))
        }
    }

    if err := scanner.Err(); err != nil {
        fmt.Fprintln(os.Stderr, "Error reading input:", err)
        os.Exit(1)
    }
}
```

### Go with Custom Logic

```prolog
generate_go_pipe_main(
    '
    // Filter records where age > 30
    age, _ := strconv.Atoi(fields[1])
    if age > 30 {
        return fields
    }
    return nil  // Skip this record
',
    [fields([name, age, city])],
    Code
).
```

### Go JSON Mode

```prolog
generate_go_pipe_main(
    '    return record',
    [format(json), fields([id, name, value])],
    Code
).
```

Generated:

```go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
)

type Record struct {
    Id    string `json:"id"`
    Name  string `json:"name"`
    Value string `json:"value"`
}

func process(record Record) *Record {
    return &record
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    buf := make([]byte, 0, 1024*1024)
    scanner.Buffer(buf, 10*1024*1024)

    for scanner.Scan() {
        var record Record
        if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
            fmt.Fprintln(os.Stderr, "JSON parse error:", err)
            continue
        }

        result := process(record)
        if result != nil {
            output, _ := json.Marshal(result)
            fmt.Println(string(output))
        }
    }
}
```

### Go Parallel Processing

```prolog
generate_go_pipe_main(
    '    return transform(fields)',
    [parallel(8), fields([name, value])],
    Code
).
```

Generated:

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
    "sync"
)

func process(fields []string) []string {
    return transform(fields)
}

func main() {
    lines := make(chan string, 10000)
    results := make(chan string, 10000)

    // Worker pool
    var wg sync.WaitGroup
    for i := 0; i < 8; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for line := range lines {
                fields := strings.Split(line, "\t")
                result := process(fields)
                if result != nil {
                    results <- strings.Join(result, "\t")
                }
            }
        }()
    }

    // Output goroutine
    var outputWg sync.WaitGroup
    outputWg.Add(1)
    go func() {
        defer outputWg.Done()
        for result := range results {
            fmt.Println(result)
        }
    }()

    // Read input
    scanner := bufio.NewScanner(os.Stdin)
    buf := make([]byte, 0, 1024*1024)
    scanner.Buffer(buf, 10*1024*1024)

    for scanner.Scan() {
        lines <- scanner.Text()
    }
    close(lines)

    wg.Wait()
    close(results)
    outputWg.Wait()
}
```

## Rust Code Generation

### Basic TSV Processing

```prolog
?- generate_rust_pipe_main(
    '    Some(fields.iter().map(|s| s.to_string()).collect())',
    [],
    Code
).
```

Generated:

```rust
use std::io::{self, BufRead, Write};

fn process(fields: &[&str]) -> Option<Vec<String>> {
    Some(fields.iter().map(|s| s.to_string()).collect())
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        match line {
            Ok(text) => {
                let fields: Vec<&str> = text.split('\t').collect();
                if let Some(result) = process(&fields) {
                    writeln!(stdout, "{}", result.join("\t")).unwrap();
                }
            }
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }
    }
}
```

### Rust with Custom Logic

```prolog
generate_rust_pipe_main(
    '
    // Parse and filter
    let age: i32 = fields[1].parse().unwrap_or(0);
    if age > 30 {
        Some(fields.iter().map(|s| s.to_string()).collect())
    } else {
        None
    }
',
    [],
    Code
).
```

### Rust JSON Mode

```prolog
generate_rust_pipe_main(
    '    Some(record)',
    [format(json), fields([id, name, value])],
    Code
).
```

Generated:

```rust
use std::io::{self, BufRead, Write};
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Deserialize, Serialize)]
struct Record {
    id: String,
    name: String,
    value: String,
}

fn process(record: Record) -> Option<Record> {
    Some(record)
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        match line {
            Ok(text) => {
                match serde_json::from_str::<Record>(&text) {
                    Ok(record) => {
                        if let Some(result) = process(record) {
                            let output = serde_json::to_string(&result).unwrap();
                            writeln!(stdout, "{}", output).unwrap();
                        }
                    }
                    Err(e) => {
                        eprintln!("JSON parse error: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }
    }
}
```

## Build Script Generation

### Go Build Script

```prolog
?- generate_go_build_script('main.go', [optimize(true)], Script).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

SOURCE="main.go"
OUTPUT="main"

echo "Building $SOURCE..."

go build -ldflags="-s -w" -o "$OUTPUT" "$SOURCE"

echo "Built: $OUTPUT"
ls -lh "$OUTPUT"
```

### Rust Build Script

```prolog
?- generate_rust_build_script('src/main.rs', [], Script).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

echo "Building Rust project..."

cargo build --release

echo "Built: target/release/main"
ls -lh target/release/main
```

## Cross-Compilation

### Available Targets

```prolog
?- cross_compile_targets(Targets).
Targets = [linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64].
```

### Go Cross-Compilation

```prolog
?- generate_cross_compile(
    go,
    'main.go',
    [linux-amd64, darwin-arm64, windows-amd64],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

SOURCE="main.go"
BASE="main"

echo "Cross-compiling $SOURCE..."

# Linux AMD64
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o "${BASE}-linux-amd64" "$SOURCE"
echo "Built: ${BASE}-linux-amd64"

# macOS ARM64
GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o "${BASE}-darwin-arm64" "$SOURCE"
echo "Built: ${BASE}-darwin-arm64"

# Windows AMD64
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o "${BASE}-windows-amd64.exe" "$SOURCE"
echo "Built: ${BASE}-windows-amd64.exe"

echo "Cross-compilation complete!"
ls -lh ${BASE}-*
```

### Rust Cross-Compilation

```prolog
?- generate_cross_compile(
    rust,
    'src/main.rs',
    [linux-amd64, linux-arm64],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

echo "Cross-compiling Rust project..."

# Add targets
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu

# Build for each target
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu

echo "Cross-compilation complete!"
ls -lh target/*/release/main
```

## Performance Characteristics

| Language | Startup | Throughput | Memory |
|----------|---------|------------|--------|
| AWK | ~1ms | ~100K rec/s | ~1MB |
| Python | ~50ms | ~50K rec/s | ~20MB |
| Go | ~5ms | ~500K rec/s | ~5MB |
| Rust | ~1ms | ~700K rec/s | ~2MB |

### When to Use Native

| Scenario | Recommendation |
|----------|----------------|
| < 10K records | AWK or Python (simplicity) |
| 10K - 1M records | Go or Rust if latency matters |
| > 1M records | Go or Rust (throughput) |
| CPU-intensive | Go/Rust (compiled) |
| Parallel processing | Go (goroutines) |
| Memory-constrained | Rust (zero-copy) |

## Chapter Summary

- **Go** provides easy parallelism with goroutines
- **Rust** provides memory safety and performance
- **TSV mode** is simple and universal
- **JSON mode** handles structured data
- **Cross-compilation** enables multi-platform distribution
- **Build scripts** automate compilation

## Next Steps

In Chapter 10, we'll cover native binary orchestration:
- Binary management
- Compilation on demand
- Mixed pipelines

## Exercises

1. **Go generator**: Generate a Go program that filters TSV records where column 2 contains "ERROR".

2. **Rust generator**: Generate a Rust program that parses JSON and doubles numeric values.

3. **Parallel Go**: Generate a Go program with 16 parallel workers for CPU-intensive processing.

4. **Cross-compile**: Generate a cross-compilation script for all 5 supported platforms.

## Code Examples

See `examples/04-native-parallel/` for:
- `filter.go` - Go TSV filter
- `transform.rs` - Rust transformation
- `parallel.go` - Parallel processing
- `build.sh` - Build automation

---

## Navigation

**←** [Previous: Chapter 8: IronPython Compatibility](08_ironpython_compat) | [📖 Book 7: Cross-Target Glue](README) | [Next: Chapter 10: Native Binary Orchestration →](10_native_orchestration)
