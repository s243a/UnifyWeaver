#!/usr/bin/env bash
# run_wam_cross_target_benchmark.sh — Generate and run WAM benchmarks across targets
#
# Usage:
#   ./run_wam_cross_target_benchmark.sh [scale] [reps]
#
# Requires: swipl, cargo/rustc
# Optional: go, python3 (for Go/Python targets — currently in progress)
#
# Toolchain env vars (adjust paths for your system):
#   RUST:  /tmp/cargo/bin/cargo
#   GO:    /tmp/go/bin/go
#   PYTHON: python3 (system)

set -euo pipefail

SCALE="${1:-300}"
REPS="${2:-3}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FACTS_DIR="$REPO_ROOT/data/benchmark/${SCALE}"
FACTS_FILE="$FACTS_DIR/facts.pl"
BENCH_DIR="/tmp/wam-bench"

# Toolchain paths (adjust for your environment)
export PATH="/tmp/go/bin:/tmp/cargo/bin:/tmp/dotnet:$PATH"
export GOPATH="${GOPATH:-/tmp/gopath}"
export GOMODCACHE="${GOMODCACHE:-/tmp/gomodcache}"
export RUSTUP_HOME="${RUSTUP_HOME:-/tmp/rustup}"
export CARGO_HOME="${CARGO_HOME:-/tmp/cargo}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/rust-target}"

echo "=== WAM Cross-Target Benchmark ==="
echo "Scale: $SCALE"
echo "Reps:  $REPS"
echo "Facts: $FACTS_FILE"
echo "Repo:  $REPO_ROOT"
echo ""

if [ ! -f "$FACTS_FILE" ]; then
    echo "ERROR: Facts file not found: $FACTS_FILE"
    exit 1
fi

cd "$REPO_ROOT"

# === Rust WAM Interpreter (Effective Distance) ===
echo "--- Rust WAM Interpreter (Effective Distance, accumulated) ---"
RUST_ED_DIR="$BENCH_DIR/rust-ed-${SCALE}"
mkdir -p "$RUST_ED_DIR"

echo "Generating Rust WAM project from Prolog source..."
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_effective_distance_benchmark.pl \
    -- "$FACTS_FILE" "$RUST_ED_DIR" accumulated 2>&1 || {
    echo "WARNING: Rust effective distance generation failed"
}

if [ -f "$RUST_ED_DIR/Cargo.toml" ]; then
    echo "Building Rust WAM..."
    cd "$RUST_ED_DIR"
    cargo build --release 2>&1 | tail -5

    RUST_BIN="./target/release/hybrid_ed_bench"
    if [ ! -x "$RUST_BIN" ]; then
        RUST_BIN="$CARGO_TARGET_DIR/release/hybrid_ed_bench"
    fi

    if [ -x "$RUST_BIN" ]; then
        echo "Running Rust WAM interpreter benchmark ($REPS reps)..."
        "$RUST_BIN" "$FACTS_DIR" "$REPS" 2>&1
    else
        echo "WARNING: Rust binary not found"
    fi
    cd "$REPO_ROOT"
else
    echo "WARNING: Rust project not generated"
fi

echo ""

# === Rust WAM Optimized (if generator exists) ===
if [ -f "examples/benchmark/generate_wam_rust_optimized_benchmark.pl" ]; then
    echo "--- Rust WAM Optimized ---"
    RUST_OPT_DIR="$BENCH_DIR/rust-opt-${SCALE}"
    mkdir -p "$RUST_OPT_DIR"

    echo "Generating Rust WAM optimized project..."
    LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_rust_optimized_benchmark.pl \
        -- "$FACTS_FILE" "$RUST_OPT_DIR" accumulated 2>&1 || {
        echo "WARNING: Rust optimized generation failed"
    }

    if [ -f "$RUST_OPT_DIR/Cargo.toml" ]; then
        echo "Building Rust WAM optimized..."
        cd "$RUST_OPT_DIR"
        cargo build --release 2>&1 | tail -5
        cd "$REPO_ROOT"

        RUST_OPT_BIN="$CARGO_TARGET_DIR/release/wam-rust-optimized-bench"
        if [ -x "$RUST_OPT_BIN" ]; then
            echo "Running Rust optimized benchmark..."
            "$RUST_OPT_BIN" "$FACTS_FILE" 2>&1
        else
            echo "WARNING: Rust optimized binary not found at $RUST_OPT_BIN"
        fi
    else
        echo "WARNING: Rust optimized project not generated"
    fi
    echo ""
fi

# === Python WAM (in progress) ===
echo "--- Python WAM (in progress) ---"
PYTHON_DIR="$BENCH_DIR/python-${SCALE}"
mkdir -p "$PYTHON_DIR"

echo "Generating Python WAM project..."
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_python_optimized_benchmark.pl \
    -- "$FACTS_FILE" "$PYTHON_DIR" accumulated 2>&1 || {
    echo "WARNING: Python generation failed"
}

if [ -f "$PYTHON_DIR/benchmark.py" ]; then
    echo "Running Python benchmark..."
    python3 "$PYTHON_DIR/benchmark.py" "$FACTS_FILE" 2>&1
elif [ -f "$PYTHON_DIR/main.py" ]; then
    echo "Running Python main.py..."
    cd "$PYTHON_DIR"
    python3 main.py 2>&1
    cd "$REPO_ROOT"
else
    echo "NOTE: Python WAM benchmark driver not yet connected"
fi

echo ""

# === Go WAM (in progress) ===
echo "--- Go WAM (in progress) ---"
GO_DIR="$BENCH_DIR/go-${SCALE}"
mkdir -p "$GO_DIR"

echo "Generating Go WAM project..."
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_go_optimized_benchmark.pl \
    -- "$FACTS_FILE" "$GO_DIR" accumulated 2>&1 || {
    echo "WARNING: Go generation failed"
}

if [ -f "$GO_DIR/go.mod" ]; then
    echo "Building Go WAM..."
    cd "$GO_DIR"
    go build ./... 2>&1 || {
        echo "WARNING: Go build failed"
    }
    cd "$REPO_ROOT"
    echo "NOTE: Go WAM builds but benchmark driver not yet wired up for fact loading"
else
    echo "WARNING: Go project not generated"
fi

echo ""
echo "=== Benchmark Complete ==="
echo ""
echo "Note: Haskell WAM benchmarks require a separate GHC environment."
echo "See examples/benchmark/generate_wam_haskell_optimized_benchmark.pl"
