#!/usr/bin/env bash
## runtime_test.sh — Compile and run generated validation output across available runtimes
##
## Usage: bash scripts/runtime_test.sh [--proot]
##   --proot   Also run tests in proot-distro debian (ghc, dotnet fsi, clojure, elixir)
##
## Prerequisites: Run validate_targets.pl first to generate output/validation/ files
##   swipl -g "consult('scripts/validate_targets'), run_validation" -t halt

DIR="output/validation"
PASS=0
FAIL=0
SKIP=0
ERRORS=""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

check_result() {
    local lang="$1" program="$2" expected="$3" actual="$4"
    actual=$(echo "$actual" | tr -d '[:space:]')
    expected=$(echo "$expected" | tr -d '[:space:]')
    if [ "$actual" = "$expected" ]; then
        printf "  ${GREEN}✓${NC} %-12s %-20s = %s\n" "$lang" "$program" "$actual"
        PASS=$((PASS + 1))
    else
        printf "  ${RED}✗${NC} %-12s %-20s expected %s, got '%s'\n" "$lang" "$program" "$expected" "$actual"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}\n  ${lang} ${program}: expected ${expected}, got '${actual}'"
    fi
}

skip_test() {
    printf "  ${YELLOW}⊘${NC} %-12s %s\n" "$1" "$2"
    SKIP=$((SKIP + 1))
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

# run_test <lang> <program> <expected> <command...>
run_test() {
    local lang="$1" program="$2" expected="$3"
    shift 3
    local result
    result=$("$@" 2>&1) || true
    check_result "$lang" "$program" "$expected" "$result"
}

# compile_and_test <lang> <program> <expected> <compile_cmd> -- <run_cmd>
compile_and_test() {
    local lang="$1" program="$2" expected="$3"
    shift 3
    local compile_args=()
    while [ "$1" != "--" ]; do compile_args+=("$1"); shift; done
    shift  # skip --
    if "${compile_args[@]}" 2>/dev/null; then
        local result
        result=$("$@" 2>&1) || true
        check_result "$lang" "$program" "$expected" "$result"
    else
        check_result "$lang" "$program" "$expected" "compile_error"
    fi
}

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  RUNTIME TESTING — Compile & Execute                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR not found. Run validate_targets.pl first."
    exit 1
fi

# Temp dir for compiled binaries
TMPDIR="$DIR/_bin"
mkdir -p "$TMPDIR"

# ============================================================================
# FACTORIAL(5) = 120
# ============================================================================
echo -e "${BOLD}--- factorial(5) = 120 ---${NC}"

has_cmd ruby    && run_test "Ruby"   "factorial(5)" "120" ruby "$DIR/factorial.rb" 5       || skip_test "Ruby" "not found"
has_cmd perl    && run_test "Perl"   "factorial(5)" "120" perl "$DIR/factorial.pl" 5       || skip_test "Perl" "not found"
has_cmd lua     && run_test "Lua"    "factorial(5)" "120" lua "$DIR/factorial.lua" 5       || skip_test "Lua" "not found"
has_cmd python3 && run_test "Python" "factorial(5)" "120" python3 "$DIR/factorial.jy.py" 5 || skip_test "Python" "not found"
has_cmd Rscript && run_test "R"      "factorial(5)" "120" Rscript "$DIR/factorial.R" 5     || skip_test "R" "not found"
has_cmd node    && run_test "Node"   "factorial(5)" "120" node "$DIR/factorial.ts" 5       || skip_test "Node" "not found"

if has_cmd gcc; then
    compile_and_test "C" "factorial(5)" "120" gcc -o "$TMPDIR/fac_c" "$DIR/factorial.c" -lm -- "$TMPDIR/fac_c" 5
else skip_test "C" "not found"; fi

if has_cmd g++; then
    compile_and_test "C++" "factorial(5)" "120" g++ -o "$TMPDIR/fac_cpp" "$DIR/factorial.cpp" -- "$TMPDIR/fac_cpp" 5
else skip_test "C++" "not found"; fi

if has_cmd javac; then
    JCLASS=$(grep -o 'class [A-Za-z_]*' "$DIR/factorial.java" 2>/dev/null | head -1 | awk '{print $2}')
    cp "$DIR/factorial.java" "$TMPDIR/${JCLASS}.java" 2>/dev/null || true
    compile_and_test "Java" "factorial(5)" "120" javac "$TMPDIR/${JCLASS}.java" -- java -cp "$TMPDIR" "$JCLASS" 5
else skip_test "Java" "not found"; fi

if has_cmd kotlinc; then
    compile_and_test "Kotlin" "factorial(5)" "120" kotlinc "$DIR/factorial.kt" -include-runtime -d "$TMPDIR/fac_kt.jar" -- java -jar "$TMPDIR/fac_kt.jar" 5
else skip_test "Kotlin" "not found"; fi

if has_cmd scalac; then
    mkdir -p "$TMPDIR/scala_cls"
    SCALA_LIB=$(find /data/data/com.termux/files/usr -name 'scala-library-*.jar' 2>/dev/null | head -1)
    SCALA3_LIB=$(find /data/data/com.termux/files/usr -name 'scala3-library_3-*.jar' 2>/dev/null | head -1)
    if [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
        compile_and_test "Scala" "factorial(5)" "120" scalac "$DIR/factorial.scala" -d "$TMPDIR/scala_cls" -- java -cp "$TMPDIR/scala_cls:$SCALA_LIB:$SCALA3_LIB" Main 5
    else skip_test "Scala" "runtime jars not found"; fi
else skip_test "Scala" "not found"; fi

# Elixir: skip on Termux (needs /system/bin/sh), test via --proot instead
skip_test "Elixir" "use --proot (needs /system/bin/sh)"

# ============================================================================
# FIB(10) = 55 (multicall)
# ============================================================================
echo ""
echo -e "${BOLD}--- fib(10) = 55 [multicall] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "fib(10)" "55" ruby "$DIR/fib.rb" 10       || true
has_cmd perl    && run_test "Perl"   "fib(10)" "55" perl "$DIR/fib.pl" 10       || true
has_cmd lua     && run_test "Lua"    "fib(10)" "55" lua "$DIR/fib.lua" 10       || true
has_cmd python3 && run_test "Python" "fib(10)" "55" python3 "$DIR/fib.jy.py" 10 || true
has_cmd node    && run_test "Node"   "fib(10)" "55" node "$DIR/fib.ts" 10       || true
has_cmd Rscript && run_test "R"      "fib(10)" "55" Rscript "$DIR/fib.R" 10     || true

# ============================================================================
# EVEN_ODD: is_even(4) = true (mutual recursion)
# ============================================================================
echo ""
echo -e "${BOLD}--- is_even(4) = true [mutual] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "is_even(4)" "true" ruby "$DIR/even_odd.rb" is_even 4        || true
has_cmd perl    && run_test "Perl"   "is_even(4)" "1"    perl "$DIR/even_odd.pl" is_even 4        || true
has_cmd lua     && run_test "Lua"    "is_even(4)" "true" lua "$DIR/even_odd.lua" is_even 4        || true
has_cmd python3 && run_test "Python" "is_even(4)" "True" python3 "$DIR/even_odd.jy.py" is_even 4  || true

# ============================================================================
# COUNT (tail recursion): count([1,2,3,4,5]) = 5
# ============================================================================
echo ""
echo -e "${BOLD}--- count([1,2,3,4,5]) = 5 [tail] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "count(5 items)" "5" ruby "$DIR/count.rb" 1,2,3,4,5       || true
has_cmd perl    && run_test "Perl"   "count(5 items)" "5" perl "$DIR/count.pl" 1,2,3,4,5       || true
has_cmd lua     && run_test "Lua"    "count(5 items)" "5" lua "$DIR/count.lua" 1,2,3,4,5       || true
has_cmd python3 && run_test "Python" "count(5 items)" "5" python3 "$DIR/count.jy.py" 1,2,3,4,5 || true
has_cmd node    && run_test "Node"   "count(5 items)" "5" node "$DIR/count.ts" 1,2,3,4,5       || true

# ============================================================================
# PROOT DEBIAN TESTS (optional)
# ============================================================================
USE_PROOT=false
for arg in "$@"; do
    [ "$arg" = "--proot" ] && USE_PROOT=true
done

if $USE_PROOT && has_cmd proot-distro; then
    PROOT_DIR="/data/data/com.termux/files/home/UnifyWeaver/context/gemini/UnifyWeaver/output/validation"
    proot_run() { proot-distro login debian -- bash -c "$1" 2>&1 | grep -v '^Warning:' | tail -1; }

    # Haskell (ghc)
    echo ""
    echo -e "${BOLD}  Haskell (ghc):${NC}"
    R=$(proot_run "cd $PROOT_DIR && ghc -o fac_hs factorial.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && ./fac_hs 5")
    check_result "Haskell" "factorial(5)" "120" "$R"
    R=$(proot_run "cd $PROOT_DIR && ghc -o fib_hs fib.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && ./fib_hs 10")
    check_result "Haskell" "fib(10)" "55" "$R"
    R=$(proot_run "cd $PROOT_DIR && ghc -o eo_hs even_odd.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && ./eo_hs is_even 4")
    check_result "Haskell" "is_even(4)" "True" "$R"
    R=$(proot_run "cd $PROOT_DIR && ghc -o count_hs count.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && ./count_hs 1,2,3,4,5")
    check_result "Haskell" "count(5 items)" "5" "$R"

    # F# (dotnet fsi) — rewrite [<EntryPoint>] to fsi.CommandLineArgs for script mode
    echo ""
    echo -e "${BOLD}  F# (dotnet fsi):${NC}"
    fsi_run() {
        local fsfile="$1"; shift
        local tmpfs="/tmp/_fsi_test.fsx"
        # Convert compiled F# (with [<EntryPoint>]) to fsi script mode
        proot-distro login debian -- bash -c "
            sed 's/\[<EntryPoint>\]//' '$fsfile' | \
            sed 's/let main argv =/let argv = fsi.CommandLineArgs.[1..] in ignore (/' | \
            sed 's/^    0$/    0)/' > $tmpfs && \
            ~/.dotnet/dotnet fsi $tmpfs $*" 2>&1 | grep -v '^Warning:\|^>' | tail -1
    }
    R=$(fsi_run "$PROOT_DIR/factorial.fs" 5)
    check_result "F#" "factorial(5)" "120" "$R"
    R=$(fsi_run "$PROOT_DIR/fib.fs" 10)
    check_result "F#" "fib(10)" "55" "$R"
    R=$(fsi_run "$PROOT_DIR/even_odd.fs" is_even 4)
    check_result "F#" "is_even(4)" "true" "$R"

    # Clojure
    echo ""
    echo -e "${BOLD}  Clojure:${NC}"
    CLJ_JAR="/usr/share/maven-repo/org/clojure/clojure/1.11.1/clojure-1.11.1.jar"
    SPEC_JAR=$(proot_run 'find /usr/share -name "spec.alpha-*.jar" 2>/dev/null | head -1') || true
    CORE_SPECS_JAR=$(proot_run 'find /usr/share -name "core.specs.alpha-*.jar" 2>/dev/null | head -1') || true
    CLJ_CP="$CLJ_JAR"
    [ -n "$SPEC_JAR" ] && CLJ_CP="$CLJ_CP:$SPEC_JAR"
    [ -n "$CORE_SPECS_JAR" ] && CLJ_CP="$CLJ_CP:$CORE_SPECS_JAR"

    R=$(proot_run "cd $PROOT_DIR && java -cp '$CLJ_CP' clojure.main factorial.clj 5")
    check_result "Clojure" "factorial(5)" "120" "$R"
    R=$(proot_run "cd $PROOT_DIR && java -cp '$CLJ_CP' clojure.main fib.clj 10")
    check_result "Clojure" "fib(10)" "55" "$R"
    R=$(proot_run "cd $PROOT_DIR && java -cp '$CLJ_CP' clojure.main even_odd.clj 4")
    check_result "Clojure" "is_even(4)" "true" "$R"

    # Elixir (proot)
    echo ""
    echo -e "${BOLD}  Elixir (proot debian):${NC}"
    R=$(proot_run "cd $PROOT_DIR && elixir factorial.ex 5")
    check_result "Elixir/pr" "factorial(5)" "120" "$R"
    R=$(proot_run "cd $PROOT_DIR && elixir fib.ex 10")
    check_result "Elixir/pr" "fib(10)" "55" "$R"
    R=$(proot_run "cd $PROOT_DIR && elixir even_odd.ex is_even 4")
    check_result "Elixir/pr" "is_even(4)" "true" "$R"

elif $USE_PROOT; then
    echo ""
    skip_test "proot" "proot-distro not found"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════╗"
printf "║  RESULTS: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}, ${YELLOW}%d skipped${NC}              ║\n" "$PASS" "$FAIL" "$SKIP"
echo "╚════════════════════════════════════════════════════════╝"

if [ $FAIL -gt 0 ]; then
    echo ""
    printf "${RED}Failures:${NC}${ERRORS}\n"
    exit 1
fi

# Cleanup
rm -rf "$TMPDIR" 2>/dev/null || true
