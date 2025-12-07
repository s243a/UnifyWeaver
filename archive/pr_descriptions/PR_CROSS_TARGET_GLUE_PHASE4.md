# PR: Cross-Target Glue Phase 4 - Native Binary Orchestration

## Title
feat: Add cross-target glue Phase 4 - native binary orchestration

## Summary

Phase 4 of the cross-target glue system implements binary orchestration for Go and Rust native targets. This enables high-performance data pipelines using compiled languages that can process millions of rows efficiently.

## Changes

### New Module: `src/unifyweaver/glue/native_glue.pl`

**Binary Management:**
- `register_binary/4` - Track compiled binaries for predicates
- `compiled_binary/3` - Query binary registry
- `compile_if_needed/4` - Compile on demand with caching (checks timestamps)

**Toolchain Detection:**
- `detect_go/1` - Find Go installation and parse version
- `detect_rust/1` - Find rustc installation
- `detect_cargo/1` - Find Cargo for Rust projects

**Go Code Generation:**

| Predicate | Description |
|-----------|-------------|
| `generate_go_pipe_main/3` | TSV/JSON streaming main() |
| `generate_go_parallel_main` | Worker pool with goroutines |
| `generate_go_wrapper/4` | Function wrapping for schemas |
| `generate_go_build_script/3` | Build with optimization flags |

Features:
- Configurable worker count (`parallel(N)`)
- Large buffer support (10MB scanner buffer)
- JSON mode with struct generation
- LTO and symbol stripping for release

**Rust Code Generation:**

| Predicate | Description |
|-----------|-------------|
| `generate_rust_pipe_main/3` | TSV/JSON streaming main() |
| `generate_rust_wrapper/4` | Function wrapping for schemas |
| `generate_rust_build_script/3` | Cargo build with profiles |
| `generate_rust_cargo_toml/3` | Project file generation |

Features:
- Serde integration for JSON mode
- Release profile with LTO
- Error handling for malformed input

**Cross-Compilation:**

```prolog
cross_compile_targets([
    target(linux, amd64, 'linux', 'amd64'),
    target(linux, arm64, 'linux', 'arm64'),
    target(darwin, amd64, 'darwin', 'amd64'),
    target(darwin, arm64, 'darwin', 'arm64'),
    target(windows, amd64, 'windows', 'amd64')
]).
```

- `generate_cross_compile/4` - Multi-platform build scripts
- Go: Uses GOOS/GOARCH environment variables
- Rust: Uses `rustup target add` and `--target` flag

**Pipeline Orchestration:**
- `generate_native_pipeline/3` - Mix native and scripted stages
- Supports: Go, Rust, AWK, Python, Bash
- Optional GNU parallel integration for streaming

### Integration Tests: `tests/integration/glue/test_native_glue.pl`

62 test assertions covering:
- Go code generation (9 tests)
- Rust code generation (8 tests)
- Go parallel generation (5 tests)
- JSON mode generation (12 tests)
- Build script generation (8 tests)
- Cross-compilation (9 tests)
- Pipeline orchestration (11 tests)

### Example: `examples/native-glue/`

High-performance 3-stage pipeline:

```
AWK (extract) → Go (transform, 8 workers) → Rust (aggregate)
```

**Architecture:**
```
┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐
│     AWK     │───▶│        Go           │───▶│    Rust     │
│   Extract   │    │     Transform       │    │  Aggregate  │
│             │    │   (8 goroutines)    │    │             │
└─────────────┘    └─────────────────────┘    └─────────────┘
```

**Files:**
- `high_perf_pipeline.pl` - Pipeline definition
- `README.md` - Usage and performance documentation

**Performance targets:**
- 10K rows: < 1s (startup-bound)
- 100K rows: ~2s (~50K rows/s)
- 1M rows: ~15s (~65K rows/s)

## Test Results

```
=== Native Glue Integration Tests ===

Test: Go code generation
  ✓ Go has package main
  ✓ Go imports bufio
  ✓ Go has process function
  ✓ Go splits on tab
  ✓ Go has large buffer
  ... (9 tests)

Test: Rust code generation
  ✓ Rust uses std::io
  ✓ Rust has process function
  ✓ Rust splits on tab
  ✓ Rust handles errors
  ... (8 tests)

Test: Go parallel code generation
  ✓ Go imports sync
  ✓ Go uses WaitGroup
  ✓ Go uses channels
  ✓ Go spawns 4 workers
  ✓ Go uses goroutines

Test: JSON mode code generation
  ✓ Go imports json
  ✓ Go has Record type
  ✓ Rust uses serde
  ✓ Rust derives Deserialize
  ... (12 tests)

Test: Build script generation
  ✓ Go build uses go build
  ✓ Go optimized strips symbols
  ✓ Rust build uses cargo
  ... (8 tests)

Test: Cross-compilation
  ✓ Go cross-compiles to Linux
  ✓ Go cross-compiles to macOS ARM
  ✓ Go Windows has .exe extension
  ✓ Rust has Linux target
  ... (9 tests)

Test: Pipeline orchestration
  ✓ Pipeline includes Go binary
  ✓ Pipeline includes Rust binary
  ✓ Pipeline includes AWK
  ✓ Pipeline uses pipes
  ... (11 tests)

All tests passed!
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    native_glue.pl                        │
├─────────────────────────────────────────────────────────┤
│  Binary Management                                       │
│  ├── register_binary/4                                  │
│  ├── compiled_binary/3                                  │
│  └── compile_if_needed/4                                │
├─────────────────────────────────────────────────────────┤
│  Toolchain Detection                                     │
│  ├── detect_go/1                                        │
│  ├── detect_rust/1                                      │
│  └── detect_cargo/1                                     │
├─────────────────────────────────────────────────────────┤
│  Go Code Generation                                      │
│  ├── generate_go_pipe_main/3                            │
│  ├── generate_go_parallel_main (internal)               │
│  ├── generate_go_wrapper/4                              │
│  └── generate_go_build_script/3                         │
├─────────────────────────────────────────────────────────┤
│  Rust Code Generation                                    │
│  ├── generate_rust_pipe_main/3                          │
│  ├── generate_rust_wrapper/4                            │
│  ├── generate_rust_build_script/3                       │
│  └── generate_rust_cargo_toml/3                         │
├─────────────────────────────────────────────────────────┤
│  Cross-Compilation                                       │
│  ├── cross_compile_targets/1                            │
│  └── generate_cross_compile/4                           │
├─────────────────────────────────────────────────────────┤
│  Pipeline Orchestration                                  │
│  └── generate_native_pipeline/3                         │
└─────────────────────────────────────────────────────────┘
```

## Relationship to Previous Phases

- **Phase 1**: Target registry and mapping (foundation)
- **Phase 2**: Shell integration (AWK ↔ Python ↔ Bash pipes)
- **Phase 3**: .NET integration (C# ↔ PowerShell ↔ IronPython)
- **Phase 4**: Native targets (Go ↔ Rust binary orchestration)

## Key Design Decisions

1. **Pipe-based I/O**: All native binaries read TSV/JSON from stdin, write to stdout
2. **Worker pool pattern**: Go parallel mode uses channels and WaitGroup
3. **Timestamp caching**: Binary recompilation only when source changes
4. **Cross-platform builds**: Single script generates all platform binaries

## Next Steps (Phase 5+)

- Network layer (HTTP/socket-based communication)
- Remote target invocation
- Error handling and retry mechanisms
- Monitoring and metrics collection
