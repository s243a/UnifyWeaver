# Native Glue Examples

Examples demonstrating UnifyWeaver's native binary orchestration for Go and Rust.

## High-Performance Data Pipeline

A three-stage pipeline optimized for processing millions of rows:

```
AWK (extract) → Go (transform, parallel) → Rust (aggregate)
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    High-Performance Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐ │
│  │     AWK     │───▶│        Go           │───▶│    Rust     │ │
│  │   Extract   │    │     Transform       │    │  Aggregate  │ │
│  │             │    │   (8 goroutines)    │    │             │ │
│  └─────────────┘    └─────────────────────┘    └─────────────┘ │
│                                                                  │
│   Fast text        Parallel processing      Memory-efficient    │
│   processing       with worker pool         streaming agg       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Combination?

| Stage | Language | Reason |
|-------|----------|--------|
| Extract | AWK | Fastest for simple text parsing, zero startup |
| Transform | Go | Excellent concurrency, goroutines for parallelism |
| Aggregate | Rust | Memory-efficient, zero-cost abstractions |

### Files

| File | Description |
|------|-------------|
| `high_perf_pipeline.pl` | Pipeline definition and code generator |
| Generated outputs: |
| `extract.awk` | AWK field extraction and filtering |
| `transform.go` | Go parallel transformation |
| `aggregate/` | Rust aggregation project |
| `build.sh` | Build all components |
| `run_pipeline.sh` | Execute the pipeline |
| `generate_data.sh` | Generate sample test data |
| `benchmark.sh` | Performance benchmark |

### Usage

```bash
# Generate the pipeline code
swipl high_perf_pipeline.pl

# Make scripts executable
chmod +x *.sh

# Build all components
./build.sh

# Run with sample data
./generate_data.sh | ./run_pipeline.sh

# Benchmark with 1M rows
./benchmark.sh
```

### Pipeline Stages

**Stage 1: AWK Extract**
- Input: CSV with header (timestamp, user_id, event_type, value)
- Output: TSV (timestamp, user_id, event_type, value)
- Filters out rows with negative values
- Near-zero startup time

**Stage 2: Go Transform (Parallel)**
- Processes records using 8 parallel goroutines
- Parses timestamps, extracts hour
- Categorizes values into buckets (low/medium/high)
- Applies log-scale normalization for large values
- Output: TSV with 6 fields

**Stage 3: Rust Aggregate**
- Groups by (user_id, hour)
- Computes: count, sum, avg, min, max
- Memory-efficient HashMap
- Streaming output

### Sample Input

```csv
timestamp,user_id,event_type,value
2025-01-15T10:30:00Z,user_042,click,150
2025-01-15T10:31:00Z,user_007,purchase,500
2025-01-15T14:45:00Z,user_042,view,25
```

### Sample Output

```
user_id	hour	count	sum	avg	min	max
user_042	10	1	112.04	112.04	112.04	112.04
user_007	10	1	169.90	169.90	169.90	169.90
user_042	14	1	25.00	25.00	25.00	25.00
```

### Performance Characteristics

| Rows | Expected Time | Throughput |
|------|---------------|------------|
| 10K | < 1s | Startup-bound |
| 100K | ~2s | ~50K rows/s |
| 1M | ~15s | ~65K rows/s |
| 10M | ~2min | ~80K rows/s |

*Note: Actual performance depends on hardware and data characteristics.*

### Key Optimizations

**AWK Stage:**
- Native field separator parsing
- Compiled regex patterns
- Zero memory allocation per row

**Go Stage:**
- Worker pool with buffered channels
- Pre-allocated scanner buffer (10MB)
- Minimal string allocations
- 8 parallel goroutines (configurable)

**Rust Stage:**
- HashMap with default hasher
- Stack-allocated Stats struct
- Buffered stdout writes
- LTO and single codegen unit for release

### Build Optimizations

```bash
# Go: Strip symbols, optimize for size
go build -ldflags="-s -w" -o transform transform.go

# Rust: LTO, single codegen unit
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

### Extending the Pipeline

To add a new stage:

1. Define the logic in the `.pl` file
2. Generate code using `native_glue` predicates:
   - `generate_go_pipe_main/3` for Go
   - `generate_rust_pipe_main/3` for Rust
3. Update the pipeline steps
4. Rebuild

Example: Adding a Go stage for deduplication:

```prolog
generate_dedup_go(Code) :-
    Logic = '
    // Skip duplicates based on first two fields
    key := fields[0] + ":" + fields[1]
    if seen[key] {
        return nil
    }
    seen[key] = true
    return fields
',
    generate_go_pipe_main(Logic, [], Code).
```

## Requirements

- Go 1.18+ (for generics if needed)
- Rust 1.70+ (for stable features)
- GNU AWK (gawk) or mawk

### Installation

**Debian/Ubuntu:**
```bash
sudo apt install golang-go rustc cargo gawk
```

**macOS:**
```bash
brew install go rust gawk
```

**Termux:**
```bash
pkg install golang rust gawk
```
