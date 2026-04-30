#!/usr/bin/env bash
# run_effective_distance_comparison.sh — cross-target effective-distance
# benchmark runner. Compares the Go hybrid-WAM target against Rust WAM and
# optimized Prolog at a chosen scale.
#
# Usage:
#   ./run_effective_distance_comparison.sh [scale] [reps]
# Defaults: scale=300, reps=3.
#
# Each target is generated, built (if needed), and run REPS times. The
# stderr metrics line `query_ms=N` / `total_ms=N` is captured per rep,
# the median is reported, and the stdout result table is diffed against
# data/benchmark/<scale>/reference_output.tsv for correctness.
#
# Targets:
#   - prolog: optimized Prolog via generate_prolog_effective_distance_benchmark.pl
#   - rust:   Rust WAM via generate_wam_effective_distance_benchmark.pl
#   - go:     Go hybrid WAM via generate_wam_go_effective_distance_benchmark.pl
#
# Toolchains expected on PATH: swipl, cargo, go. Haskell (ghc/cabal)
# and .NET (dotnet) are not exercised here.

set -euo pipefail

SCALE="${1:-300}"
REPS="${2:-3}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FACTS_DIR="$REPO_ROOT/data/benchmark/${SCALE}"
FACTS_FILE="$FACTS_DIR/facts.pl"
REF_FILE="$FACTS_DIR/reference_output.tsv"
BENCH_ROOT="${BENCH_ROOT:-/tmp/uw-bench-cmp}"
RUST_TARGET_DIR="${CARGO_TARGET_DIR:-$BENCH_ROOT/rust-target}"

mkdir -p "$BENCH_ROOT"
cd "$REPO_ROOT"

if [ ! -f "$FACTS_FILE" ]; then
    echo "ERROR: missing facts file $FACTS_FILE" >&2
    exit 1
fi

echo "=== Effective-Distance Cross-Target Comparison ==="
echo "Scale=$SCALE  Reps=$REPS  Facts=$FACTS_FILE"
echo

# Pull `query_ms=N` from the stderr metric block. Take the last
# occurrence as a defensive measure in case a future driver change
# emits the metrics multiple times.
extract_metric() {
    local key="$1"
    local file="$2"
    grep -E "^${key}=" "$file" | tail -1 | cut -d= -f2 || echo ""
}

median() {
    # numeric median over stdin (one value per line)
    sort -n | awk '
        { vals[NR] = $1 }
        END {
            n = NR
            if (n == 0) { print ""; exit }
            if (n % 2 == 1) print vals[(n+1)/2]
            else printf "%d\n", (vals[n/2] + vals[n/2+1]) / 2
        }
    '
}

compare_to_ref() {
    local out="$1"
    local label="$2"
    if [ ! -f "$REF_FILE" ]; then
        echo "  [diff] no reference file at $REF_FILE — skipping"
        return
    fi
    local out_rows ref_rows
    out_rows=$(($(wc -l < "$out") - 1))
    ref_rows=$(($(wc -l < "$REF_FILE") - 1))
    if diff -q "$REF_FILE" "$out" >/dev/null 2>&1; then
        echo "  [diff] EXACT match against reference ($out_rows rows)"
    else
        echo "  [diff] MISMATCH: $label produced $out_rows rows; reference has $ref_rows rows"
        # Show a hint of how they differ. `diff` exits 1 when files differ;
        # under `set -euo pipefail` that propagates and kills the whole
        # script, so we explicitly soak up the failure with `|| true`.
        (diff "$REF_FILE" "$out" || true) | head -3 | sed 's/^/    /'
    fi
}

# ---------- Prolog ----------
PROLOG_DIR="$BENCH_ROOT/prolog-${SCALE}"
mkdir -p "$PROLOG_DIR"
echo "--- Optimized Prolog ---"
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_prolog_effective_distance_benchmark.pl \
    -- "$FACTS_FILE" "$PROLOG_DIR/bench.pl" accumulated >/dev/null 2>&1 \
    || { echo "  [gen] FAILED"; PROLOG_DIR=""; }

if [ -n "$PROLOG_DIR" ] && [ -f "$PROLOG_DIR/bench.pl" ]; then
    echo "  [gen] OK"
    PROLOG_QMS=()
    PROLOG_TMS=()
    for r in $(seq 1 "$REPS"); do
        # The generated bench.pl already declares
        #   :- initialization(run_benchmark, main).
        # so the toplevel goal we need is just `halt`. Passing
        # `-g run_benchmark` on top of that fired the entry point twice
        # and concatenated two output blocks into the TSV (and the
        # `query_ms=` line into stderr), tripping the diff against the
        # reference. -t halt alone is sufficient.
        LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -t halt "$PROLOG_DIR/bench.pl" \
            > "$PROLOG_DIR/run-${r}.tsv" 2> "$PROLOG_DIR/run-${r}.err"
        q=$(extract_metric query_ms "$PROLOG_DIR/run-${r}.err")
        t=$(extract_metric total_ms "$PROLOG_DIR/run-${r}.err")
        PROLOG_QMS+=("$q")
        PROLOG_TMS+=("$t")
        echo "  [rep ${r}] query_ms=$q  total_ms=$t"
    done
    PROLOG_QMED=$(printf '%s\n' "${PROLOG_QMS[@]}" | median)
    PROLOG_TMED=$(printf '%s\n' "${PROLOG_TMS[@]}" | median)
    compare_to_ref "$PROLOG_DIR/run-1.tsv" "Prolog"
fi
echo

# ---------- Rust ----------
RUST_DIR="$BENCH_ROOT/rust-${SCALE}"
mkdir -p "$RUST_DIR"
echo "--- Rust WAM ---"
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_effective_distance_benchmark.pl \
    -- "$FACTS_FILE" "$RUST_DIR" accumulated >/dev/null 2>&1 \
    || { echo "  [gen] FAILED"; RUST_DIR=""; }

if [ -n "$RUST_DIR" ] && [ -f "$RUST_DIR/Cargo.toml" ]; then
    echo "  [gen] OK"
    (cd "$RUST_DIR" && CARGO_TARGET_DIR="$RUST_TARGET_DIR" cargo build --release >/dev/null 2>&1) \
        && echo "  [build] OK" || echo "  [build] FAILED"
    RUST_BIN="$RUST_TARGET_DIR/release/hybrid_ed_bench"
    if [ -x "$RUST_BIN" ]; then
        RUST_QMS=()
        RUST_TMS=()
        for r in $(seq 1 "$REPS"); do
            "$RUST_BIN" "$FACTS_DIR" \
                > "$RUST_DIR/run-${r}.tsv" 2> "$RUST_DIR/run-${r}.err"
            q=$(extract_metric query_ms "$RUST_DIR/run-${r}.err")
            t=$(extract_metric total_ms "$RUST_DIR/run-${r}.err")
            RUST_QMS+=("$q")
            RUST_TMS+=("$t")
            echo "  [rep ${r}] query_ms=$q  total_ms=$t"
        done
        RUST_QMED=$(printf '%s\n' "${RUST_QMS[@]}" | median)
        RUST_TMED=$(printf '%s\n' "${RUST_TMS[@]}" | median)
        compare_to_ref "$RUST_DIR/run-1.tsv" "Rust"
    fi
fi
echo

# ---------- Go ----------
GO_DIR="$BENCH_ROOT/go-${SCALE}"
mkdir -p "$GO_DIR"
echo "--- Go hybrid WAM ---"
LANG=C.UTF-8 LC_ALL=C.UTF-8 swipl -q -s examples/benchmark/generate_wam_go_effective_distance_benchmark.pl \
    -- "$FACTS_FILE" "$GO_DIR" accumulated >/dev/null 2>&1 \
    || { echo "  [gen] FAILED"; GO_DIR=""; }

if [ -n "$GO_DIR" ] && [ -f "$GO_DIR/go.mod" ]; then
    echo "  [gen] OK"
    (cd "$GO_DIR" && go build ./... >/dev/null 2>&1) \
        && echo "  [build] OK" || echo "  [build] FAILED"
    GO_BIN="$GO_DIR/wam-go-effective-distance-bench"
    if [ -x "$GO_BIN" ]; then
        GO_QMS=()
        GO_TMS=()
        for r in $(seq 1 "$REPS"); do
            "$GO_BIN" \
                > "$GO_DIR/run-${r}.tsv" 2> "$GO_DIR/run-${r}.err"
            q=$(extract_metric query_ms "$GO_DIR/run-${r}.err")
            t=$(extract_metric total_ms "$GO_DIR/run-${r}.err")
            GO_QMS+=("$q")
            GO_TMS+=("$t")
            echo "  [rep ${r}] query_ms=$q  total_ms=$t"
        done
        GO_QMED=$(printf '%s\n' "${GO_QMS[@]}" | median)
        GO_TMED=$(printf '%s\n' "${GO_TMS[@]}" | median)
        compare_to_ref "$GO_DIR/run-1.tsv" "Go"
    fi
fi
echo

# ---------- Summary ----------
echo "=== Summary (median, ms) ==="
printf "%-10s %-12s %-12s %s\n" "target" "query_ms" "total_ms" "correctness"
printf "%-10s %-12s %-12s %s\n" "------" "--------" "--------" "-----------"

prolog_corr="(no run)"
[ -n "${PROLOG_QMED:-}" ] && {
    if diff -q "$REF_FILE" "$PROLOG_DIR/run-1.tsv" >/dev/null 2>&1; then prolog_corr="match"; else prolog_corr="MISMATCH"; fi
    printf "%-10s %-12s %-12s %s\n" "prolog"  "${PROLOG_QMED}" "${PROLOG_TMED}" "$prolog_corr"
}

rust_corr="(no run)"
[ -n "${RUST_QMED:-}" ] && {
    if diff -q "$REF_FILE" "$RUST_DIR/run-1.tsv" >/dev/null 2>&1; then rust_corr="match"; else rust_corr="MISMATCH"; fi
    printf "%-10s %-12s %-12s %s\n" "rust"    "${RUST_QMED}"   "${RUST_TMED}"   "$rust_corr"
}

go_corr="(no run)"
[ -n "${GO_QMED:-}" ] && {
    if diff -q "$REF_FILE" "$GO_DIR/run-1.tsv" >/dev/null 2>&1; then go_corr="match"; else go_corr="MISMATCH"; fi
    printf "%-10s %-12s %-12s %s\n" "go"      "${GO_QMED}"     "${GO_TMED}"     "$go_corr"
}
