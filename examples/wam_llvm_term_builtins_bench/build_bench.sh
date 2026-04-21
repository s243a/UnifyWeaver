#!/bin/bash
# Phase 0 build pipeline for the WAM-LLVM bench suite.
#
# Steps:
#   1. generate bench_suite.ll via generate_llvm_bench.pl
#   2. llc .ll -> .s (aarch64-linux-android)
#   3. clang .s + run_bench.c -> native binary
#
# Run from project root:
#   bash examples/wam_llvm_term_builtins_bench/build_bench.sh

set -euo pipefail

BENCH_DIR="examples/wam_llvm_term_builtins_bench"

for tool in swipl llc clang; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required tool '$tool' not found on PATH" >&2
        exit 1
    fi
done

echo "[1/3] generating LLVM IR..."
swipl "$BENCH_DIR/generate_llvm_bench.pl"

echo "[2/3] llc -> aarch64 assembly..."
# Android/Termux requires PIC for all executables (PIE).
llc --relocation-model=pic "$BENCH_DIR/bench_suite.ll" -o "$BENCH_DIR/bench_suite.s"

echo "[3/3] clang link with C driver..."
clang -O2 -fPIE -pie \
    "$BENCH_DIR/bench_suite.s" \
    "$BENCH_DIR/run_bench.c" \
    -o "$BENCH_DIR/bench_suite"

echo "built: $BENCH_DIR/bench_suite"
ls -la "$BENCH_DIR/bench_suite"
