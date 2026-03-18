#!/usr/bin/env bash
## runtime_test.sh — Compile and run generated validation output across available runtimes
##
## Usage: bash scripts/runtime_test.sh [--proot]
##   --proot   Also run tests in proot-distro debian (ghc, dotnet fsi, clojure, elixir)
##             Only available on Termux.
##
## Supports: Termux (Android), Linux, WSL, macOS
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

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================
detect_environment() {
    if [ -d "/data/data/com.termux" ]; then
        ENV_TYPE="termux"
    elif grep -qi microsoft /proc/version 2>/dev/null; then
        ENV_TYPE="wsl"
    elif [ "$(uname)" = "Darwin" ]; then
        ENV_TYPE="macos"
    else
        ENV_TYPE="linux"
    fi
}

detect_environment

# Find Scala runtime jars (platform-dependent)
find_scala_jars() {
    if [ "$ENV_TYPE" = "termux" ]; then
        SCALA_LIB=$(find /data/data/com.termux/files/usr -name 'scala-library-*.jar' 2>/dev/null | head -1)
        SCALA3_LIB=$(find /data/data/com.termux/files/usr -name 'scala3-library_3-*.jar' 2>/dev/null | head -1)
    else
        # Standard Linux/WSL/macOS paths
        SCALA_LIB=$(find /usr/share /usr/local/share "$HOME/.sdkman" "$HOME/.coursier" 2>/dev/null -name 'scala-library-*.jar' 2>/dev/null | head -1)
        SCALA3_LIB=$(find /usr/share /usr/local/share "$HOME/.sdkman" "$HOME/.coursier" 2>/dev/null -name 'scala3-library_3-*.jar' 2>/dev/null | head -1)
    fi
}

# ============================================================================
# TEST HELPERS
# ============================================================================
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

# java_test <program_label> <expected> <source_file> <args...>
# Handles class name extraction and compilation
java_test() {
    local program="$1" expected="$2" source="$3"
    shift 3
    local jclass
    jclass=$(grep -o 'class [A-Za-z_]*' "$source" 2>/dev/null | head -1 | awk '{print $2}')
    if [ -z "$jclass" ]; then
        check_result "Java" "$program" "$expected" "no_class_found"
        return
    fi
    cp "$source" "$TMPDIR/${jclass}.java" 2>/dev/null || true
    compile_and_test "Java" "$program" "$expected" javac "$TMPDIR/${jclass}.java" -- java -cp "$TMPDIR" "$jclass" "$@"
}

# kotlin_test <program_label> <expected> <source_file> <jar_name> <args...>
kotlin_test() {
    local program="$1" expected="$2" source="$3" jar="$4"
    shift 4
    compile_and_test "Kotlin" "$program" "$expected" kotlinc "$source" -include-runtime -d "$TMPDIR/$jar" -- java -jar "$TMPDIR/$jar" "$@"
}

# scala_test <program_label> <expected> <source_file> <cls_dir> <args...>
scala_test() {
    local program="$1" expected="$2" source="$3" cls_dir="$4"
    shift 4
    mkdir -p "$TMPDIR/$cls_dir"
    compile_and_test "Scala" "$program" "$expected" scalac "$source" -d "$TMPDIR/$cls_dir" -- java -cp "$TMPDIR/$cls_dir:$SCALA_LIB:$SCALA3_LIB" Main "$@"
}

# rust_test <program_label> <expected> <source_file> <bin_name> <args...>
rust_test() {
    local program="$1" expected="$2" source="$3" bin="$4"
    shift 4
    compile_and_test "Rust" "$program" "$expected" rustc -o "$TMPDIR/$bin" "$source" -- "$TMPDIR/$bin" "$@"
}

# go_test <program_label> <expected> <source_file> <bin_name> <args...>
go_test() {
    local program="$1" expected="$2" source="$3" bin="$4"
    shift 4
    compile_and_test "Go" "$program" "$expected" go build -o "$TMPDIR/$bin" "$source" -- "$TMPDIR/$bin" "$@"
}

# pipe_test <lang> <program> <expected> <stdin_data> <command...>
# Like run_test but pipes stdin_data into the command
pipe_test() {
    local lang="$1" program="$2" expected="$3" stdin_data="$4"
    shift 4
    local result
    result=$(echo -e "$stdin_data" | "$@" 2>&1) || true
    check_result "$lang" "$program" "$expected" "$result"
}

# pipe_compile_test <lang> <program> <expected> <stdin_data> <compile_cmd> -- <run_cmd>
# Like compile_and_test but pipes stdin_data into the run command
pipe_compile_test() {
    local lang="$1" program="$2" expected="$3" stdin_data="$4"
    shift 4
    local compile_args=()
    while [ "$1" != "--" ]; do compile_args+=("$1"); shift; done
    shift  # skip --
    if "${compile_args[@]}" 2>/dev/null; then
        local result
        result=$(echo -e "$stdin_data" | "$@" 2>&1) || true
        check_result "$lang" "$program" "$expected" "$result"
    else
        check_result "$lang" "$program" "$expected" "compile_error"
    fi
}

# java_pkg_test <program_label> <expected> <source_file> <stdin_data> <args...>
# Like java_test but strips 'package ...' line and supports stdin piping
java_pkg_test() {
    local program="$1" expected="$2" source="$3" stdin_data="$4"
    shift 4
    local jclass
    jclass=$(grep -o 'class [A-Za-z_]*' "$source" 2>/dev/null | head -1 | awk '{print $2}')
    if [ -z "$jclass" ]; then
        check_result "Java" "$program" "$expected" "no_class_found"
        return
    fi
    # Strip package declaration for simple compilation
    sed '/^package /d' "$source" > "$TMPDIR/${jclass}.java" 2>/dev/null || true
    if javac "$TMPDIR/${jclass}.java" 2>/dev/null; then
        local result
        result=$(echo -e "$stdin_data" | java -cp "$TMPDIR" "$jclass" "$@" 2>&1) || true
        check_result "Java" "$program" "$expected" "$result"
    else
        check_result "Java" "$program" "$expected" "compile_error"
    fi
}

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  RUNTIME TESTING — Compile & Execute                   ║"
echo "╚════════════════════════════════════════════════════════╝"
printf "  Environment: ${BOLD}%s${NC}\n\n" "$ENV_TYPE"

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
    java_test "factorial(5)" "120" "$DIR/factorial.java" 5
else skip_test "Java" "not found"; fi

if has_cmd kotlinc; then
    kotlin_test "factorial(5)" "120" "$DIR/factorial.kt" "fac_kt.jar" 5
else skip_test "Kotlin" "not found"; fi

if has_cmd scalac; then
    find_scala_jars
    if [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
        scala_test "factorial(5)" "120" "$DIR/factorial.scala" "scala_fac" 5
    else skip_test "Scala" "runtime jars not found"; fi
else skip_test "Scala" "not found"; fi

if has_cmd rustc; then
    rust_test "factorial(5)" "120" "$DIR/factorial.rs" "fac_rs" 5
else skip_test "Rust" "not found"; fi

if has_cmd go; then
    go_test "factorial(5)" "120" "$DIR/factorial.go" "fac_go" 5
else skip_test "Go" "not found"; fi

# Elixir: on Termux needs proot (no /system/bin/sh), on other platforms try directly
if [ "$ENV_TYPE" = "termux" ]; then
    skip_test "Elixir" "use --proot (needs /system/bin/sh)"
elif has_cmd elixir; then
    run_test "Elixir" "factorial(5)" "120" elixir "$DIR/factorial.ex" 5
else
    skip_test "Elixir" "not found"
fi

# ============================================================================
# FIB(10) = 55 (multicall)
# ============================================================================
echo ""
echo -e "${BOLD}--- fib(10) = 55 [multicall] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "fib(10)" "55" ruby "$DIR/fib.rb" 10       || true
has_cmd perl    && run_test "Perl"   "fib(10)" "55" perl "$DIR/fib.pl" 10       || true
has_cmd lua     && run_test "Lua"    "fib(10)" "55" lua "$DIR/fib.lua" 10       || true
has_cmd python3 && run_test "Python" "fib(10)" "55" python3 "$DIR/fib.jy.py" 10 || true
has_cmd Rscript && run_test "R"      "fib(10)" "55" Rscript "$DIR/fib.R" 10     || true
has_cmd node    && run_test "Node"   "fib(10)" "55" node "$DIR/fib.ts" 10       || true

if has_cmd gcc; then
    compile_and_test "C" "fib(10)" "55" gcc -o "$TMPDIR/fib_c" "$DIR/fib.c" -lm -- "$TMPDIR/fib_c" 10
fi

if has_cmd g++; then
    compile_and_test "C++" "fib(10)" "55" g++ -o "$TMPDIR/fib_cpp" "$DIR/fib.cpp" -- "$TMPDIR/fib_cpp" 10
fi

if has_cmd javac; then
    java_test "fib(10)" "55" "$DIR/fib.java" 10
fi

if has_cmd kotlinc; then
    kotlin_test "fib(10)" "55" "$DIR/fib.kt" "fib_kt.jar" 10
fi

if has_cmd scalac && [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
    scala_test "fib(10)" "55" "$DIR/fib.scala" "scala_fib" 10
fi

if has_cmd rustc; then
    rust_test "fib(10)" "55" "$DIR/fib.rs" "fib_rs" 10
fi

if has_cmd go; then
    go_test "fib(10)" "55" "$DIR/fib.go" "fib_go" 10
fi

# ============================================================================
# EVEN_ODD: is_even(4) = true (mutual recursion)
# ============================================================================
echo ""
echo -e "${BOLD}--- is_even(4) = true [mutual] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "is_even(4)" "true" ruby "$DIR/even_odd.rb" is_even 4        || true
has_cmd perl    && run_test "Perl"   "is_even(4)" "1"    perl "$DIR/even_odd.pl" is_even 4        || true
has_cmd lua     && run_test "Lua"    "is_even(4)" "true" lua "$DIR/even_odd.lua" is_even 4        || true
has_cmd python3 && run_test "Python" "is_even(4)" "True" python3 "$DIR/even_odd.jy.py" is_even 4  || true
has_cmd Rscript && run_test "R"      "is_even(4)" "TRUE" Rscript "$DIR/even_odd.R" is_even 4     || true
has_cmd node    && run_test "Node"   "is_even(4)" "true" node "$DIR/even_odd.ts" is_even 4       || true

if has_cmd gcc; then
    compile_and_test "C" "is_even(4)" "1" gcc -o "$TMPDIR/eo_c" "$DIR/even_odd.c" -lm -- "$TMPDIR/eo_c" is_even 4
fi

if has_cmd g++; then
    compile_and_test "C++" "is_even(4)" "1" g++ -o "$TMPDIR/eo_cpp" "$DIR/even_odd.cpp" -- "$TMPDIR/eo_cpp" is_even 4
fi

if has_cmd javac; then
    java_test "is_even(4)" "true" "$DIR/even_odd.java" is_even 4
fi

if has_cmd kotlinc; then
    kotlin_test "is_even(4)" "true" "$DIR/even_odd.kt" "eo_kt.jar" is_even 4
fi

if has_cmd scalac && [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
    scala_test "is_even(4)" "true" "$DIR/even_odd.scala" "scala_eo" is_even 4
fi

if has_cmd rustc; then
    rust_test "is_even(4)" "1" "$DIR/even_odd.rs" "eo_rs" is_even 4
fi

if has_cmd go; then
    go_test "is_even(4)" "true" "$DIR/even_odd.go" "eo_go" is_even 4
fi

# ============================================================================
# COUNT (tail recursion): count([1,2,3,4,5]) = 5
# ============================================================================
echo ""
echo -e "${BOLD}--- count([1,2,3,4,5]) = 5 [tail] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "count(5 items)" "5" ruby "$DIR/count.rb" 1,2,3,4,5       || true
has_cmd perl    && run_test "Perl"   "count(5 items)" "5" perl "$DIR/count.pl" 1,2,3,4,5       || true
has_cmd lua     && run_test "Lua"    "count(5 items)" "5" lua "$DIR/count.lua" 1,2,3,4,5       || true
has_cmd python3 && run_test "Python" "count(5 items)" "5" python3 "$DIR/count.jy.py" 1,2,3,4,5 || true
has_cmd Rscript && run_test "R"      "count(5 items)" "5" Rscript "$DIR/count.R" 1,2,3,4,5     || true
has_cmd node    && run_test "Node"   "count(5 items)" "5" node "$DIR/count.ts" 1,2,3,4,5       || true

if has_cmd gcc; then
    compile_and_test "C" "count(5 items)" "5" gcc -o "$TMPDIR/cnt_c" "$DIR/count.c" -lm -- "$TMPDIR/cnt_c" 1,2,3,4,5
fi

if has_cmd g++; then
    compile_and_test "C++" "count(5 items)" "5" g++ -o "$TMPDIR/cnt_cpp" "$DIR/count.cpp" -- "$TMPDIR/cnt_cpp" 1,2,3,4,5
fi

if has_cmd javac; then
    java_test "count(5 items)" "5" "$DIR/count.java" 1,2,3,4,5
fi

if has_cmd kotlinc; then
    kotlin_test "count(5 items)" "5" "$DIR/count.kt" "cnt_kt.jar" 1,2,3,4,5
fi

if has_cmd scalac && [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
    scala_test "count(5 items)" "5" "$DIR/count.scala" "scala_cnt" 1,2,3,4,5
fi

if has_cmd rustc; then
    rust_test "count(5 items)" "5" "$DIR/count.rs" "cnt_rs" 1,2,3,4,5
fi

if has_cmd go; then
    go_test "count(5 items)" "5" "$DIR/count.go" "cnt_go" 1,2,3,4,5
fi

# ============================================================================
# LIST_SUM: list_sum([1,2,3,4,5]) = 15 (linear recursion, list fold)
# ============================================================================
echo ""
echo -e "${BOLD}--- list_sum([1,2,3,4,5]) = 15 [linear/list] ---${NC}"

has_cmd ruby    && run_test "Ruby"   "list_sum(5)" "15" ruby "$DIR/list_sum.rb" 1,2,3,4,5       || true
has_cmd perl    && run_test "Perl"   "list_sum(5)" "15" perl "$DIR/list_sum.pl" 1,2,3,4,5       || true
has_cmd lua     && run_test "Lua"    "list_sum(5)" "15" lua "$DIR/list_sum.lua" 1,2,3,4,5       || true
has_cmd python3 && run_test "Python" "list_sum(5)" "15" python3 "$DIR/list_sum.jy.py" 1,2,3,4,5 || true
has_cmd Rscript && run_test "R"      "list_sum(5)" "15" Rscript "$DIR/list_sum.R" 1,2,3,4,5     || true
has_cmd node    && run_test "Node"   "list_sum(5)" "15" node "$DIR/list_sum.ts" 1,2,3,4,5       || true

if has_cmd gcc; then
    compile_and_test "C" "list_sum(5)" "15" gcc -o "$TMPDIR/ls_c" "$DIR/list_sum.c" -lm -- "$TMPDIR/ls_c" 1,2,3,4,5
fi

if has_cmd g++; then
    compile_and_test "C++" "list_sum(5)" "15" g++ -o "$TMPDIR/ls_cpp" "$DIR/list_sum.cpp" -- "$TMPDIR/ls_cpp" 1,2,3,4,5
fi

if has_cmd javac; then
    java_test "list_sum(5)" "15" "$DIR/list_sum.java" 1,2,3,4,5
fi

if has_cmd kotlinc; then
    kotlin_test "list_sum(5)" "15" "$DIR/list_sum.kt" "ls_kt.jar" 1,2,3,4,5
fi

if has_cmd scalac && [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
    scala_test "list_sum(5)" "15" "$DIR/list_sum.scala" "scala_ls" 1,2,3,4,5
fi

if has_cmd rustc; then
    rust_test "list_sum(5)" "15" "$DIR/list_sum.rs" "ls_rs" 1,2,3,4,5
fi

if has_cmd go; then
    go_test "list_sum(5)" "15" "$DIR/list_sum.go" "ls_go" 1,2,3,4,5
fi

# ============================================================================
# ANCESTOR: ancestor(tom, dave) = tom:dave (transitive closure)
# ============================================================================
echo ""
echo -e "${BOLD}--- ancestor(tom,dave) = tom:dave [transitive] ---${NC}"

ANCESTOR_FACTS="tom:bob\nbob:charlie\ncharlie:dave"

has_cmd ruby    && pipe_test "Ruby"   "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" ruby "$DIR/ancestor.rb" tom dave        || true
has_cmd perl    && pipe_test "Perl"   "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" perl "$DIR/ancestor.pl" tom dave        || true
has_cmd lua     && pipe_test "Lua"    "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" lua "$DIR/ancestor.lua" tom dave        || true
has_cmd python3 && pipe_test "Python" "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" python3 "$DIR/ancestor.jy.py" tom dave || true
has_cmd Rscript && pipe_test "R"      "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" Rscript "$DIR/ancestor.R" tom dave     || true
if has_cmd node; then
    R=$(echo -e "$ANCESTOR_FACTS" | node "$DIR/ancestor.ts" tom dave 2>/dev/null) || true
    check_result "Node" "ancestor(tom,dave)" "tom:dave" "$R"
fi

if has_cmd gcc; then
    pipe_compile_test "C" "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" gcc -o "$TMPDIR/anc_c" "$DIR/ancestor.c" -lm -- "$TMPDIR/anc_c" tom dave
fi

if has_cmd g++; then
    pipe_compile_test "C++" "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" g++ -o "$TMPDIR/anc_cpp" "$DIR/ancestor.cpp" -- "$TMPDIR/anc_cpp" tom dave
fi

if has_cmd javac; then
    java_pkg_test "ancestor(tom,dave)" "tom:dave" "$DIR/ancestor.java" "$ANCESTOR_FACTS" tom dave
fi

if has_cmd kotlinc; then
    # Strip package declaration for Kotlin ancestor
    sed '/^package /d' "$DIR/ancestor.kt" > "$TMPDIR/ancestor_kt.kt" 2>/dev/null || true
    if kotlinc "$TMPDIR/ancestor_kt.kt" -include-runtime -d "$TMPDIR/anc_kt.jar" 2>/dev/null; then
        R=$(echo -e "$ANCESTOR_FACTS" | java -jar "$TMPDIR/anc_kt.jar" tom dave 2>&1) || true
        check_result "Kotlin" "ancestor(tom,dave)" "tom:dave" "$R"
    else
        check_result "Kotlin" "ancestor(tom,dave)" "tom:dave" "compile_error"
    fi
fi

if has_cmd scalac && [ -n "$SCALA_LIB" ] && [ -n "$SCALA3_LIB" ]; then
    sed '/^package /d' "$DIR/ancestor.scala" > "$TMPDIR/ancestor_sc.scala" 2>/dev/null || true
    mkdir -p "$TMPDIR/scala_anc"
    if scalac "$TMPDIR/ancestor_sc.scala" -d "$TMPDIR/scala_anc" 2>/dev/null; then
        R=$(echo -e "$ANCESTOR_FACTS" | java -cp "$TMPDIR/scala_anc:$SCALA_LIB:$SCALA3_LIB" ANCESTORQuery tom dave 2>&1) || true
        check_result "Scala" "ancestor(tom,dave)" "tom:dave" "$R"
    else
        check_result "Scala" "ancestor(tom,dave)" "tom:dave" "compile_error"
    fi
fi

if has_cmd rustc; then
    pipe_compile_test "Rust" "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" rustc -o "$TMPDIR/anc_rs" "$DIR/ancestor.rs" -- "$TMPDIR/anc_rs" tom dave
fi

if has_cmd go; then
    pipe_compile_test "Go" "ancestor(tom,dave)" "tom:dave" "$ANCESTOR_FACTS" go build -o "$TMPDIR/anc_go" "$DIR/ancestor.go" -- "$TMPDIR/anc_go" tom dave
fi

# ============================================================================
# TREE_FIB(10) = 55 (tree recursion — advanced output, limited targets)
# ============================================================================
ADVDIR="output/advanced"
if [ -d "$ADVDIR" ]; then
    echo ""
    echo -e "${BOLD}--- tree_fib(10) = 55 [tree recursion] ---${NC}"

    has_cmd lua && [ -f "$ADVDIR/tree_fib.lua" ] && run_test "Lua" "tree_fib(10)" "55" lua "$ADVDIR/tree_fib.lua" 10 || true

    if has_cmd gcc && [ -f "$ADVDIR/tree_fib.c" ]; then
        compile_and_test "C" "tree_fib(10)" "55" gcc -o "$TMPDIR/tfib_c" "$ADVDIR/tree_fib.c" -lm -- "$TMPDIR/tfib_c" 10
    fi

    # R tree_fib is a tree-sum (different pattern), skip — no numeric CLI

    # ============================================================================
    # FIB_DIRECT(10) = 55 (direct multicall — advanced output, limited targets)
    # ============================================================================
    echo ""
    echo -e "${BOLD}--- fib_direct(10) = 55 [direct multicall] ---${NC}"

    has_cmd lua && [ -f "$ADVDIR/fib_direct.lua" ] && run_test "Lua" "fib_direct(10)" "55" lua "$ADVDIR/fib_direct.lua" 10 || true

    if has_cmd gcc && [ -f "$ADVDIR/fib_direct.c" ]; then
        compile_and_test "C" "fib_direct(10)" "55" gcc -o "$TMPDIR/dfib_c" "$ADVDIR/fib_direct.c" -lm -- "$TMPDIR/dfib_c" 10
    fi

    has_cmd Rscript && [ -f "$ADVDIR/fib_direct.R" ] && run_test "R" "fib_direct(10)" "55" Rscript "$ADVDIR/fib_direct.R" 10 || true
fi

# ============================================================================
# PROOT DEBIAN TESTS (Termux only, optional)
# ============================================================================
USE_PROOT=false
for arg in "$@"; do
    [ "$arg" = "--proot" ] && USE_PROOT=true
done

if $USE_PROOT; then
    if [ "$ENV_TYPE" != "termux" ]; then
        echo ""
        skip_test "proot" "--proot is only available on Termux"
    elif ! has_cmd proot-distro; then
        echo ""
        skip_test "proot" "proot-distro not found"
    else
        PROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)/output/validation"
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
        R=$(proot_run "cd $PROOT_DIR && ghc -o anc_hs ancestor.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && echo -e 'tom:bob\nbob:charlie\ncharlie:dave' | ./anc_hs tom dave")
        check_result "Haskell" "ancestor(tom,dave)" "tom:dave" "$R"

        # Haskell advanced patterns (tree_fib, fib_direct)
        PROOT_ADV="$(cd "$(dirname "$0")/.." && pwd)/output/advanced"
        if [ -f "$PROOT_ADV/tree_fib.hs" ]; then
            R=$(proot_run "cd $PROOT_ADV && ghc -o tfib_hs tree_fib.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && ./tfib_hs 10")
            check_result "Haskell" "tree_fib(10)" "55" "$R"
        fi
        if [ -f "$PROOT_ADV/fib_direct.hs" ]; then
            R=$(proot_run "cd $PROOT_ADV && ghc -o dfib_hs fib_direct.hs -no-keep-hi-files -no-keep-o-files 2>/dev/null && ./dfib_hs 10")
            check_result "Haskell" "fib_direct(10)" "55" "$R"
        fi

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
        R=$(fsi_run "$PROOT_DIR/list_sum.fs" 1,2,3,4,5)
        check_result "F#" "list_sum(5)" "15" "$R"
        # F# ancestor: needs stdin piping — use direct fsi invocation
        R=$(proot-distro login debian -- bash -c "
            sed 's/\[<EntryPoint>\]//' '$PROOT_DIR/ancestor.fs' | \
            sed 's/let main argv =/let argv = fsi.CommandLineArgs.[1..] in ignore (/' | \
            sed 's/^    0$/    0)/' | \
            sed 's/^        1$/        1)/' > /tmp/_fsi_anc.fsx && \
            echo -e 'tom:bob\nbob:charlie\ncharlie:dave' | ~/.dotnet/dotnet fsi /tmp/_fsi_anc.fsx tom dave" 2>&1 | grep -v '^Warning:\|^>' | tail -1)
        check_result "F#" "ancestor(tom,dave)" "tom:dave" "$R"
        # F# advanced patterns
        if [ -f "$PROOT_ADV/tree_fib.fs" ]; then
            R=$(fsi_run "$PROOT_ADV/tree_fib.fs" 10)
            check_result "F#" "tree_fib(10)" "55" "$R"
        fi
        if [ -f "$PROOT_ADV/fib_direct.fs" ]; then
            R=$(fsi_run "$PROOT_ADV/fib_direct.fs" 10)
            check_result "F#" "fib_direct(10)" "55" "$R"
        fi

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
        R=$(proot_run "cd $PROOT_DIR && java -cp '$CLJ_CP' clojure.main list_sum.clj 1,2,3,4,5")
        check_result "Clojure" "list_sum(5)" "15" "$R"
        R=$(proot_run "cd $PROOT_DIR && echo -e 'tom:bob\nbob:charlie\ncharlie:dave' | java -cp '$CLJ_CP' clojure.main ancestor.clj tom dave")
        check_result "Clojure" "ancestor(tom,dave)" "tom:dave" "$R"

        # Elixir (proot)
        echo ""
        echo -e "${BOLD}  Elixir (proot debian):${NC}"
        R=$(proot_run "cd $PROOT_DIR && elixir factorial.ex 5")
        check_result "Elixir/pr" "factorial(5)" "120" "$R"
        R=$(proot_run "cd $PROOT_DIR && elixir fib.ex 10")
        check_result "Elixir/pr" "fib(10)" "55" "$R"
        R=$(proot_run "cd $PROOT_DIR && elixir even_odd.ex is_even 4")
        check_result "Elixir/pr" "is_even(4)" "true" "$R"
        R=$(proot_run "cd $PROOT_DIR && elixir list_sum.ex 1,2,3,4,5")
        check_result "Elixir/pr" "list_sum(5)" "15" "$R"
        R=$(proot_run "cd $PROOT_DIR && echo -e 'tom:bob\nbob:charlie\ncharlie:dave' | elixir ancestor.ex tom dave")
        check_result "Elixir/pr" "ancestor(tom,dave)" "tom:dave" "$R"
        # Elixir advanced patterns
        if [ -f "$PROOT_ADV/tree_fib.exs" ]; then
            R=$(proot_run "cd $PROOT_ADV && elixir tree_fib.exs 10")
            check_result "Elixir/pr" "tree_fib(10)" "55" "$R"
        fi
        if [ -f "$PROOT_ADV/fib_direct.exs" ]; then
            R=$(proot_run "cd $PROOT_ADV && elixir fib_direct.exs 10")
            check_result "Elixir/pr" "fib_direct(10)" "55" "$R"
        fi
    fi
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
