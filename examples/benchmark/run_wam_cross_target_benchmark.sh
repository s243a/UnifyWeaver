#!/usr/bin/env bash
# run_wam_cross_target_benchmark.sh — Generate and run WAM benchmarks across targets
#
# Usage:
#   ./run_wam_cross_target_benchmark.sh [scale]
#
# Requires: swipl, go, cargo/rustc, python3
# Toolchain env vars (adjust paths for your system):
#   GO:    /tmp/go/bin/go
#   RUST:  /tmp/cargo/bin/cargo
#   PYTHON: python3 (system)

set -euo pipefail

SCALE="${1:-300}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FACTS_FILE="$REPO_ROOT/data/benchmark/${SCALE}/facts.pl"
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
echo "Facts: $FACTS_FILE"
echo "Repo:  $REPO_ROOT"
echo ""

if [ ! -f "$FACTS_FILE" ]; then
    echo "ERROR: Facts file not found: $FACTS_FILE"
    exit 1
fi

cd "$REPO_ROOT"

# === Rust WAM ===
echo "--- Rust WAM ---"
RUST_DIR="$BENCH_DIR/rust-${SCALE}"
mkdir -p "$RUST_DIR"

echo "Generating Rust WAM project..."
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_rust_optimized_benchmark.pl \
    -- "$FACTS_FILE" "$RUST_DIR" accumulated 2>&1 || {
    echo "WARNING: Rust generation failed"
}

if [ -f "$RUST_DIR/Cargo.toml" ]; then
    # Copy benchmark main.rs if not present
    if [ ! -f "$RUST_DIR/src/main.rs" ]; then
        echo "NOTE: No main.rs generated; Rust benchmark requires manual runner"
    fi

    echo "Building Rust WAM..."
    cd "$RUST_DIR"
    cargo build --release 2>&1 | tail -5
    cd "$REPO_ROOT"

    RUST_BIN="$CARGO_TARGET_DIR/release/wam-rust-optimized-bench"
    if [ -x "$RUST_BIN" ]; then
        echo "Running Rust benchmark..."
        "$RUST_BIN" "$FACTS_FILE" 2>&1
    else
        echo "WARNING: Rust binary not found at $RUST_BIN"
    fi
else
    echo "WARNING: Rust project not generated"
fi

echo ""

# === Python WAM ===
echo "--- Python WAM ---"
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
    echo "WARNING: No Python benchmark script found"
fi

echo ""

# === Go WAM ===
echo "--- Go WAM ---"
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
    # Note: Go runtime currently doesn't wire up fact loading
    echo "NOTE: Go WAM builds but fact loading is not yet implemented"
else
    echo "WARNING: Go project not generated"
fi

echo ""
echo "=== Benchmark Complete ==="
