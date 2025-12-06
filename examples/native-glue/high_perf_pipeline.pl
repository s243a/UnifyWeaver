/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Example: High-Performance Data Pipeline with Go and Rust
 *
 * This example demonstrates a high-performance pipeline using:
 * 1. AWK: Quick field extraction and filtering
 * 2. Go: Parallel data transformation (with goroutines)
 * 3. Rust: Memory-efficient aggregation
 *
 * Designed for processing large datasets (millions of rows).
 */

:- use_module('../../src/unifyweaver/glue/native_glue').
:- use_module('../../src/unifyweaver/glue/shell_glue').

%% ============================================
%% Pipeline Definition
%% ============================================

%% Three-stage high-performance pipeline
pipeline_steps([
    step(extract, awk, 'extract.awk', []),
    step(transform, go, './transform', [parallel(8)]),
    step(aggregate, rust, './target/release/aggregate', [])
]).

%% ============================================
%% Stage 1: AWK Extraction
%% ============================================

%% Fast field extraction and filtering
%% Input: CSV with header
%% Output: TSV (timestamp, user_id, event_type, value)
generate_extract_awk(Code) :-
    Code = '#!/usr/bin/awk -f
# Stage 1: Extract and filter

BEGIN {
    FS = ","
    OFS = "\\t"
}

# Skip header
NR == 1 { next }

{
    timestamp = $1
    user_id = $2
    event_type = $3
    value = $4

    # Filter: Only process positive values
    if (value > 0) {
        print timestamp, user_id, event_type, value
    }
}
'.

%% ============================================
%% Stage 2: Go Transformation (Parallel)
%% ============================================

%% Parallel transformation with goroutines
%% - Parses timestamps
%% - Computes derived fields
%% - Handles high throughput via worker pool
generate_transform_go(Code) :-
    Logic = '
    // Parse input fields
    if len(fields) < 4 {
        return nil
    }

    timestamp := fields[0]
    userID := fields[1]
    eventType := fields[2]
    value := fields[3]

    // Parse value
    val, err := strconv.ParseFloat(value, 64)
    if err != nil {
        return nil
    }

    // Compute derived fields
    hour := timestamp[11:13]  // Extract hour from ISO timestamp
    bucket := "low"
    if val > 100 {
        bucket = "high"
    } else if val > 50 {
        bucket = "medium"
    }

    // Normalize value (log scale for large values)
    normalized := val
    if val > 1000 {
        normalized = 1000 + math.Log10(val-999)*100
    }

    return []string{
        timestamp,
        userID,
        eventType,
        fmt.Sprintf("%.2f", normalized),
        hour,
        bucket,
    }
',
    generate_go_parallel_main_with_imports(Logic, Code).

generate_go_parallel_main_with_imports(Logic, Code) :-
    format(atom(Code), '
package main

import (
    "bufio"
    "fmt"
    "math"
    "os"
    "strconv"
    "strings"
    "sync"
)

func process(fields []string) []string {
~w
}

func main() {
    // Configuration
    numWorkers := 8
    bufferSize := 10000

    // Channels
    lines := make(chan string, bufferSize)
    results := make(chan string, bufferSize)

    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for line := range lines {
                fields := strings.Split(line, "\\t")
                result := process(fields)
                if result != nil {
                    results <- strings.Join(result, "\\t")
                }
            }
        }()
    }

    // Output goroutine (preserves order via buffered channel)
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

    if err := scanner.Err(); err != nil {
        fmt.Fprintln(os.Stderr, "Error:", err)
        os.Exit(1)
    }
}
', [Logic]).

%% ============================================
%% Stage 3: Rust Aggregation
%% ============================================

%% Memory-efficient aggregation
%% - Groups by user_id and hour
%% - Computes sum, count, min, max
%% - Streaming output
generate_aggregate_rust(Code) :-
    Code = '
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

#[derive(Default)]
struct Stats {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn new(value: f64) -> Self {
        Stats {
            count: 1,
            sum: value,
            min: value,
            max: value,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn avg(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    // Aggregation map: (user_id, hour) -> Stats
    let mut aggregates: HashMap<(String, String), Stats> = HashMap::new();

    // Process input
    for line in stdin.lock().lines() {
        if let Ok(text) = line {
            let fields: Vec<&str> = text.split(\'\\t\').collect();
            if fields.len() >= 6 {
                let user_id = fields[1].to_string();
                let hour = fields[4].to_string();
                let value: f64 = fields[3].parse().unwrap_or(0.0);

                let key = (user_id, hour);
                aggregates
                    .entry(key)
                    .and_modify(|s| s.update(value))
                    .or_insert_with(|| Stats::new(value));
            }
        }
    }

    // Output aggregated results
    writeln!(stdout, "user_id\\thour\\tcount\\tsum\\tavg\\tmin\\tmax").unwrap();
    for ((user_id, hour), stats) in aggregates.iter() {
        writeln!(
            stdout,
            "{}\\t{}\\t{}\\t{:.2}\\t{:.2}\\t{:.2}\\t{:.2}",
            user_id, hour, stats.count, stats.sum, stats.avg(), stats.min, stats.max
        ).unwrap();
    }
}
'.

%% ============================================
%% Cargo.toml for Rust
%% ============================================

generate_cargo_toml(Toml) :-
    Toml = '[package]
name = "aggregate"
version = "0.1.0"
edition = "2021"

[dependencies]

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
'.

%% ============================================
%% Sample Data Generator
%% ============================================

generate_sample_data_script(Script) :-
    Script = '#!/bin/bash
# Generate sample CSV data

echo "timestamp,user_id,event_type,value"

for i in $(seq 1 10000); do
    # Random timestamp (today)
    hour=$(printf "%02d" $((RANDOM % 24)))
    minute=$(printf "%02d" $((RANDOM % 60)))
    second=$(printf "%02d" $((RANDOM % 60)))
    timestamp="2025-01-15T${hour}:${minute}:${second}Z"

    # Random user (1-100)
    user_id="user_$(printf "%03d" $((RANDOM % 100 + 1)))"

    # Random event type
    events=("click" "view" "purchase" "signup")
    event_type="${events[$((RANDOM % 4))]}"

    # Random value (-10 to 1000, some negative for filtering)
    value=$((RANDOM % 1010 - 10))

    echo "${timestamp},${user_id},${event_type},${value}"
done
'.

%% ============================================
%% Build Script
%% ============================================

generate_build_script(Script) :-
    Script = '#!/bin/bash
set -euo pipefail

echo "Building high-performance pipeline..."

# Make AWK script executable
chmod +x extract.awk
echo "  ✓ extract.awk ready"

# Build Go binary
echo "  Building Go transformer..."
go build -ldflags="-s -w" -o transform transform.go
echo "  ✓ transform built"

# Build Rust binary
echo "  Building Rust aggregator..."
cd aggregate && cargo build --release && cd ..
echo "  ✓ aggregate built"

echo ""
echo "Build complete! Run with:"
echo "  ./generate_data.sh | ./run_pipeline.sh"
'.

%% ============================================
%% Run Script
%% ============================================

generate_run_script(Script) :-
    Script = '#!/bin/bash
set -euo pipefail

# High-performance pipeline
# AWK (filter) -> Go (transform, 8 workers) -> Rust (aggregate)

./extract.awk \\
    | ./transform \\
    | ./aggregate/target/release/aggregate
'.

%% ============================================
%% Benchmark Script
%% ============================================

generate_benchmark_script(Script) :-
    Script = '#!/bin/bash
set -euo pipefail

echo "=== Pipeline Benchmark ==="
echo ""

# Generate test data
echo "Generating 1M rows of test data..."
for i in {1..100}; do
    ./generate_data.sh
done > /tmp/benchmark_data.csv
echo "  Data size: $(wc -l < /tmp/benchmark_data.csv) rows"
echo "  File size: $(du -h /tmp/benchmark_data.csv | cut -f1)"
echo ""

# Benchmark
echo "Running pipeline..."
time (cat /tmp/benchmark_data.csv | ./run_pipeline.sh > /tmp/benchmark_result.tsv)

echo ""
echo "Results:"
echo "  Output rows: $(wc -l < /tmp/benchmark_result.tsv)"
head -5 /tmp/benchmark_result.tsv

# Cleanup
rm -f /tmp/benchmark_data.csv /tmp/benchmark_result.tsv
'.

%% ============================================
%% Main: Generate All Files
%% ============================================

generate_all :-
    format('Generating high-performance pipeline...~n~n'),

    % Stage 1: AWK
    generate_extract_awk(AwkCode),
    open('extract.awk', write, S1),
    write(S1, AwkCode),
    close(S1),
    format('  Created: extract.awk~n'),

    % Stage 2: Go
    generate_transform_go(GoCode),
    open('transform.go', write, S2),
    write(S2, GoCode),
    close(S2),
    format('  Created: transform.go~n'),

    % Stage 3: Rust (create directory structure)
    make_directory_path('aggregate/src'),
    generate_aggregate_rust(RustCode),
    open('aggregate/src/main.rs', write, S3),
    write(S3, RustCode),
    close(S3),
    format('  Created: aggregate/src/main.rs~n'),

    generate_cargo_toml(CargoToml),
    open('aggregate/Cargo.toml', write, S4),
    write(S4, CargoToml),
    close(S4),
    format('  Created: aggregate/Cargo.toml~n'),

    % Scripts
    generate_sample_data_script(DataScript),
    open('generate_data.sh', write, S5),
    write(S5, DataScript),
    close(S5),
    format('  Created: generate_data.sh~n'),

    generate_build_script(BuildScript),
    open('build.sh', write, S6),
    write(S6, BuildScript),
    close(S6),
    format('  Created: build.sh~n'),

    generate_run_script(RunScript),
    open('run_pipeline.sh', write, S7),
    write(S7, RunScript),
    close(S7),
    format('  Created: run_pipeline.sh~n'),

    generate_benchmark_script(BenchScript),
    open('benchmark.sh', write, S8),
    write(S8, BenchScript),
    close(S8),
    format('  Created: benchmark.sh~n'),

    format('~nDone! Build and run:~n'),
    format('  chmod +x *.sh~n'),
    format('  ./build.sh~n'),
    format('  ./generate_data.sh | ./run_pipeline.sh~n').

%% ============================================
%% Main
%% ============================================

:- initialization(generate_all, main).
