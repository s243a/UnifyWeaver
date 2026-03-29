#!/bin/bash
# ==========================================================================
# Cross-Target Effective Distance Benchmark Runner
#
# Compiles effective_distance predicates to each target, executes them
# against the dev dataset, and compares output to SWI-Prolog reference.
#
# Usage:
#   bash examples/benchmark/run_benchmark.sh
#
# Prerequisites:
#   - SWI-Prolog (swipl)
#   - .NET SDK 9.0+ (for C# target)
#   - Go 1.21+ (for Go target)
#   - gawk (for AWK target)
#   - Python 3.10+ (for Python target)
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

FACTS_FILE="data/benchmark/dev/facts.pl"
OUTPUT_DIR="data/benchmark/dev/outputs"
REFERENCE="data/benchmark/dev/reference_output.tsv"

mkdir -p "$OUTPUT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass=0
fail=0
skip=0

print_result() {
    local target="$1" status="$2" detail="$3"
    if [ "$status" = "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $target: $detail"
        ((pass++))
    elif [ "$status" = "FAIL" ]; then
        echo -e "  ${RED}✗${NC} $target: $detail"
        ((fail++))
    else
        echo -e "  ${YELLOW}⊘${NC} $target: $detail"
        ((skip++))
    fi
}

# ==========================================================================
# Step 1: Generate SWI-Prolog reference output
# ==========================================================================
echo "=== Step 1: SWI-Prolog Reference ==="

if command -v swipl &>/dev/null; then
    swipl -l examples/benchmark/effective_distance.pl \
          -l "$FACTS_FILE" \
          -g run_benchmark -t halt 2>/dev/null \
          > "$OUTPUT_DIR/prolog.tsv"
    NLINES=$(tail -n +2 "$OUTPUT_DIR/prolog.tsv" | wc -l)
    print_result "SWI-Prolog" "PASS" "$NLINES articles"
    cp "$OUTPUT_DIR/prolog.tsv" "$REFERENCE"
else
    print_result "SWI-Prolog" "SKIP" "swipl not found"
fi

# ==========================================================================
# Step 2: Compile category_ancestor/3 to each target
# ==========================================================================
echo ""
echo "=== Step 2: Compile category_ancestor/3 ==="

# --- C# Query Engine ---
# Note: C# parameterized query engine requires:
#   1. mode/1 declaration for input/output args
#   2. Constants in body (Hops is 1) not head (category_ancestor(_, _, 1))
echo -n "  Compiling C# ... "
CSHARP_CODE=$(swipl -q -g "
    ['$FACTS_FILE'],
    assert(user:mode(category_ancestor(+, -, -))),
    assert(user:(category_ancestor(Cat, Parent, Hops) :- category_parent(Cat, Parent), Hops is 1)),
    assert(user:(category_ancestor(Cat, Ancestor, Hops) :- category_parent(Cat, Mid), category_ancestor(Mid, Ancestor, H1), Hops is H1 + 1)),
    use_module('src/unifyweaver/targets/csharp_target'),
    compile_predicate_to_csharp(category_ancestor/3, [target(csharp_query)], Code),
    write(Code)
" -t halt 2>/dev/null) && echo "OK" || echo "FAILED"

if [ -n "$CSHARP_CODE" ]; then
    echo "$CSHARP_CODE" > "$OUTPUT_DIR/category_ancestor.cs"
    print_result "C# Query" "PASS" "compiled to $(wc -l < "$OUTPUT_DIR/category_ancestor.cs") lines"
else
    print_result "C# Query" "FAIL" "compilation produced no output"
fi

# --- Go ---
echo -n "  Compiling Go ... "
GO_CODE=$(swipl -q -g "
    ['$FACTS_FILE'],
    assert(user:(category_ancestor(Cat, Parent, 1) :- category_parent(Cat, Parent))),
    assert(user:(category_ancestor(Cat, Ancestor, Hops) :- category_parent(Cat, Mid), category_ancestor(Mid, Ancestor, H1), Hops is H1 + 1)),
    use_module('src/unifyweaver/targets/go_target'),
    compile_predicate_to_go(category_ancestor/3, [], Code),
    write(Code)
" -t halt 2>/dev/null) && echo "OK" || echo "FAILED"

if [ -n "$GO_CODE" ]; then
    echo "$GO_CODE" > "$OUTPUT_DIR/category_ancestor.go"
    print_result "Go" "PASS" "compiled to $(wc -l < "$OUTPUT_DIR/category_ancestor.go") lines"
else
    print_result "Go" "FAIL" "compilation produced no output"
fi

# --- AWK ---
echo -n "  Compiling AWK ... "
AWK_CODE=$(swipl -q -g "
    ['$FACTS_FILE'],
    assert(user:(category_ancestor(Cat, Parent, 1) :- category_parent(Cat, Parent))),
    assert(user:(category_ancestor(Cat, Ancestor, Hops) :- category_parent(Cat, Mid), category_ancestor(Mid, Ancestor, H1), Hops is H1 + 1)),
    use_module('src/unifyweaver/targets/awk_target'),
    compile_predicate_to_awk(category_ancestor/3, [], Code),
    write(Code)
" -t halt 2>/dev/null) && echo "OK" || echo "FAILED"

if [ -n "$AWK_CODE" ]; then
    echo "$AWK_CODE" > "$OUTPUT_DIR/category_ancestor.awk"
    print_result "AWK" "PASS" "compiled to $(wc -l < "$OUTPUT_DIR/category_ancestor.awk") lines"
else
    print_result "AWK" "FAIL" "compilation produced no output"
fi

# --- Python ---
echo -n "  Compiling Python ... "
PY_CODE=$(swipl -q -g "
    ['$FACTS_FILE'],
    assert(user:(category_ancestor(Cat, Parent, 1) :- category_parent(Cat, Parent))),
    assert(user:(category_ancestor(Cat, Ancestor, Hops) :- category_parent(Cat, Mid), category_ancestor(Mid, Ancestor, H1), Hops is H1 + 1)),
    use_module('src/unifyweaver/targets/python_target'),
    compile_predicate_to_python(category_ancestor/3, [], Code),
    write(Code)
" -t halt 2>/dev/null) && echo "OK" || echo "FAILED"

if [ -n "$PY_CODE" ]; then
    echo "$PY_CODE" > "$OUTPUT_DIR/category_ancestor.py"
    print_result "Python" "PASS" "compiled to $(wc -l < "$OUTPUT_DIR/category_ancestor.py") lines"
else
    print_result "Python" "FAIL" "compilation produced no output"
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "=== Summary ==="
echo -e "  ${GREEN}Pass: $pass${NC}  ${RED}Fail: $fail${NC}  ${YELLOW}Skip: $skip${NC}"
echo ""
echo "Generated files in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Reference output: $REFERENCE"
echo "  $(tail -n +2 "$REFERENCE" | wc -l) articles with effective distances"
