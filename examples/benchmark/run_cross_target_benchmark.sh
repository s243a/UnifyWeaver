#!/bin/bash
# Cross-target effective distance benchmark: Haskell vs Rust vs Go
#
# Usage: ./run_cross_target_benchmark.sh [scale] [repetitions]
#   scale: 300 (default), 1k, 5k, 10k
#   repetitions: 3 (default)
#
# Requires: swipl, ghc, cargo, go

set -euo pipefail

SCALE="${1:-300}"
REPS="${2:-3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/benchmark/$SCALE"
BENCH_DIR="$PROJECT_ROOT/examples/benchmark"
OUTPUT_DIR="/tmp/uw_cross_target_bench_$SCALE"
REFERENCE="$DATA_DIR/reference_output.tsv"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Scale $SCALE not found at $DATA_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Cross-Target Effective Distance Benchmark"
echo "Scale: $SCALE  Repetitions: $REPS"
echo "=========================================="
echo ""

# --- Generate projects ---
echo "--- Generating projects ---"

# Haskell
HASKELL_DIR="$OUTPUT_DIR/haskell"
if [ ! -f "$HASKELL_DIR/src/Main.hs" ]; then
    LANG=C.utf8 LC_ALL=C.utf8 swipl -q -s "$BENCH_DIR/generate_wam_haskell_effective_distance_benchmark.pl" \
        -- "$DATA_DIR/facts.pl" "$HASKELL_DIR" 2>&1 | grep -v '^$' | head -5
    echo "  Haskell project generated"
else
    echo "  Haskell project cached"
fi

# Rust
RUST_DIR="$OUTPUT_DIR/rust"
if [ ! -f "$RUST_DIR/src/main.rs" ]; then
    if [ -f "$BENCH_DIR/generate_wam_rust_optimized_benchmark.pl" ]; then
        LANG=C.utf8 LC_ALL=C.utf8 swipl -q -s "$BENCH_DIR/generate_wam_rust_optimized_benchmark.pl" \
            -- "$DATA_DIR/facts.pl" "$RUST_DIR" 2>&1 | grep -v '^$' | head -5
        echo "  Rust project generated"
    else
        echo "  Rust generator not found, skipping"
        RUST_DIR=""
    fi
else
    echo "  Rust project cached"
fi

# Go
GO_DIR="$OUTPUT_DIR/go"
if [ ! -f "$GO_DIR/main.go" ]; then
    if [ -f "$BENCH_DIR/generate_wam_go_optimized_benchmark.pl" ]; then
        LANG=C.utf8 LC_ALL=C.utf8 swipl -q -s "$BENCH_DIR/generate_wam_go_optimized_benchmark.pl" \
            -- "$DATA_DIR/facts.pl" "$GO_DIR" 2>&1 | grep -v '^$' | head -5
        echo "  Go project generated"
    else
        echo "  Go generator not found, skipping"
        GO_DIR=""
    fi
else
    echo "  Go project cached"
fi

echo ""

# --- Build projects ---
echo "--- Building projects ---"

# Haskell
HASKELL_BIN="$HASKELL_DIR/bench"
if [ ! -f "$HASKELL_BIN" ]; then
    echo -n "  Building Haskell... "
    BUILD_START=$(date +%s%N)
    (cd "$HASKELL_DIR" && ghc --make src/Main.hs -isrc -outputdir build -o bench \
        -O2 -Wno-overlapping-patterns \
        -package parallel -package async \
        -package unordered-containers -package hashable \
        2>&1 | tail -1) || {
        echo "FAILED"
        HASKELL_BIN=""
    }
    BUILD_END=$(date +%s%N)
    BUILD_MS=$(( (BUILD_END - BUILD_START) / 1000000 ))
    echo "done (${BUILD_MS}ms)"
else
    echo "  Haskell binary cached"
fi

# Rust
RUST_BIN=""
if [ -n "${RUST_DIR:-}" ] && [ -f "$RUST_DIR/Cargo.toml" ]; then
    if [ ! -f "$RUST_DIR/target/release/bench" ] && [ ! -f "$RUST_DIR/target/release/wam-rust-bench" ] && [ ! -f "$RUST_DIR/target/release/main" ]; then
        echo -n "  Building Rust... "
        BUILD_START=$(date +%s%N)
        (cd "$RUST_DIR" && cargo build --release 2>&1 | tail -1) || {
            echo "FAILED"
        }
        BUILD_END=$(date +%s%N)
        BUILD_MS=$(( (BUILD_END - BUILD_START) / 1000000 ))
        echo "done (${BUILD_MS}ms)"
    else
        echo "  Rust binary cached"
    fi
    # Find the binary
    for name in bench wam-rust-bench main; do
        if [ -f "$RUST_DIR/target/release/$name" ]; then
            RUST_BIN="$RUST_DIR/target/release/$name"
            break
        fi
    done
fi

# Go
GO_BIN=""
if [ -n "${GO_DIR:-}" ] && [ -f "$GO_DIR/main.go" ]; then
    if [ ! -f "$GO_DIR/bench" ]; then
        echo -n "  Building Go... "
        BUILD_START=$(date +%s%N)
        (cd "$GO_DIR" && go build -o bench . 2>&1 | tail -1) || {
            echo "FAILED"
        }
        BUILD_END=$(date +%s%N)
        BUILD_MS=$(( (BUILD_END - BUILD_START) / 1000000 ))
        echo "done (${BUILD_MS}ms)"
    else
        echo "  Go binary cached"
    fi
    [ -f "$GO_DIR/bench" ] && GO_BIN="$GO_DIR/bench"
fi

echo ""

# --- Run benchmarks ---
echo "--- Running benchmarks ($REPS repetitions each) ---"
echo ""

run_benchmark() {
    local name="$1"
    local cmd="$2"
    local data_arg="$3"
    local times=()

    for i in $(seq 1 $REPS); do
        local stdout_file="$OUTPUT_DIR/${name}_run${i}.tsv"
        local stderr_file="$OUTPUT_DIR/${name}_run${i}.stderr"
        LANG=C.utf8 LC_ALL=C.utf8 eval "$cmd" "$data_arg" > "$stdout_file" 2> "$stderr_file" || true
        local qms=$(grep 'query_ms=' "$stderr_file" 2>/dev/null | sed 's/query_ms=//')
        local tms=$(grep 'total_ms=' "$stderr_file" 2>/dev/null | sed 's/total_ms=//')
        times+=("${qms:-?}")
        local rows=$(wc -l < "$stdout_file" 2>/dev/null || echo 0)
        rows=$((rows - 1))  # subtract header
    done

    # Report
    printf "  %-12s " "$name"
    printf "query_ms=["
    local first=true
    for t in "${times[@]}"; do
        $first || printf ", "
        printf "%s" "$t"
        first=false
    done
    printf "]  rows=$rows"

    # Check correctness against reference
    local last_stdout="$OUTPUT_DIR/${name}_run${REPS}.tsv"
    if [ -f "$REFERENCE" ] && [ -f "$last_stdout" ]; then
        local ref_rows=$(wc -l < "$REFERENCE")
        local out_rows=$(wc -l < "$last_stdout")
        if [ "$ref_rows" = "$out_rows" ]; then
            printf "  [OK rows match]"
        else
            printf "  [MISMATCH ref=$ref_rows out=$out_rows]"
        fi
    fi
    echo ""
}

if [ -n "${HASKELL_BIN:-}" ] && [ -f "$HASKELL_BIN" ]; then
    run_benchmark "Haskell" "$HASKELL_BIN" "$DATA_DIR"
fi

if [ -n "${RUST_BIN:-}" ] && [ -f "$RUST_BIN" ]; then
    run_benchmark "Rust" "$RUST_BIN" "$DATA_DIR"
fi

if [ -n "${GO_BIN:-}" ] && [ -f "$GO_BIN" ]; then
    run_benchmark "Go" "$GO_BIN" "$DATA_DIR"
fi

echo ""
echo "Results written to $OUTPUT_DIR/"
echo "=========================================="
