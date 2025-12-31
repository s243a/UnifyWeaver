#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 John William Creighton (s243a)
#
# Bridge Integration Tests for CI
#
# Usage:
#   ./test_bridges.sh --all           # Test all bridges
#   ./test_bridges.sh --jvm           # Test JVM bridges (jpype, jpy)
#   ./test_bridges.sh --dotnet        # Test .NET bridges (pythonnet, csnakes)
#   ./test_bridges.sh --ffi           # Test FFI bridges (rust-ffi-go)
#   ./test_bridges.sh --rust          # Test Rust bridges (pyo3)
#   ./test_bridges.sh --ruby          # Test Ruby bridges (pycall)
#   ./test_bridges.sh pythonnet jpype # Test specific bridges

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RPYC_SERVER="$PROJECT_ROOT/examples/rpyc-integration/rpyc_server.py"
RPYC_PORT=18812
RPYC_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0
SKIPPED=0
declare -a FAILED_BRIDGES=()

#######################################
# Logging functions
#######################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=$((FAILED + 1))
    FAILED_BRIDGES+=("$1")
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    SKIPPED=$((SKIPPED + 1))
}

log_section() {
    echo
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
}

#######################################
# RPyC Server Management
#######################################

start_rpyc_server() {
    log_info "Starting RPyC server on port $RPYC_PORT..."

    # Check if already running
    if nc -z localhost $RPYC_PORT 2>/dev/null; then
        log_info "RPyC server already running on port $RPYC_PORT"
        return 0
    fi

    # Start server in background
    python3 "$RPYC_SERVER" --port "$RPYC_PORT" > /tmp/rpyc_server.log 2>&1 &
    RPYC_PID=$!

    # Wait for server to start
    local retries=10
    while ! nc -z localhost $RPYC_PORT 2>/dev/null; do
        ((retries--))
        if [ $retries -le 0 ]; then
            log_fail "Failed to start RPyC server"
            cat /tmp/rpyc_server.log
            exit 1
        fi
        sleep 0.5
    done

    log_info "RPyC server started (PID: $RPYC_PID)"
}

stop_rpyc_server() {
    if [ -n "$RPYC_PID" ]; then
        log_info "Stopping RPyC server (PID: $RPYC_PID)..."
        kill "$RPYC_PID" 2>/dev/null || true
        wait "$RPYC_PID" 2>/dev/null || true
        RPYC_PID=""
    fi
}

#######################################
# Dependency Checking
#######################################

check_dotnet() {
    if command -v dotnet &> /dev/null; then
        local version=$(dotnet --version 2>/dev/null | head -1)
        log_info ".NET SDK found: $version"
        return 0
    fi
    return 1
}

check_java() {
    if command -v java &> /dev/null; then
        local version=$(java -version 2>&1 | head -1)
        log_info "Java found: $version"
        return 0
    fi
    return 1
}

check_gradle() {
    if command -v gradle &> /dev/null || [ -f "./gradlew" ]; then
        return 0
    fi
    return 1
}

check_rust() {
    if command -v cargo &> /dev/null; then
        local version=$(cargo --version 2>/dev/null)
        log_info "Cargo found: $version"
        return 0
    fi
    return 1
}

check_ruby() {
    if command -v ruby &> /dev/null; then
        local version=$(ruby --version 2>/dev/null)
        log_info "Ruby found: $version"
        return 0
    fi
    return 1
}

check_go() {
    if command -v go &> /dev/null; then
        local version=$(go version 2>/dev/null)
        log_info "Go found: $version"
        return 0
    fi
    return 1
}

#######################################
# Bridge Test Functions
#######################################

test_pythonnet() {
    log_section "Testing Python.NET Bridge"
    cd "$SCRIPT_DIR/pythonnet"

    if ! check_dotnet; then
        log_skip "pythonnet: .NET SDK not found"
        return 0
    fi

    if ! pip3 show pythonnet &>/dev/null; then
        log_skip "pythonnet: pythonnet package not installed"
        return 0
    fi

    log_info "Building and running Python.NET test..."
    if timeout 120 dotnet run > /tmp/pythonnet_test.log 2>&1; then
        if grep -qE "PASSED|success|All tests|4\.0" /tmp/pythonnet_test.log; then
            cat /tmp/pythonnet_test.log
            log_pass "pythonnet"
        else
            log_fail "pythonnet"
            cat /tmp/pythonnet_test.log
        fi
    else
        log_fail "pythonnet (execution failed or timed out)"
        cat /tmp/pythonnet_test.log
    fi
}

test_csnakes() {
    log_section "Testing CSnakes Bridge"
    cd "$SCRIPT_DIR/csnakes"

    if ! check_dotnet; then
        log_skip "csnakes: .NET SDK not found"
        return 0
    fi

    # CSnakes requires .NET 9+
    local major_version=$(dotnet --version 2>/dev/null | cut -d. -f1)
    if [ "$major_version" -lt 9 ] 2>/dev/null; then
        log_skip "csnakes: Requires .NET 9+ (found: $major_version)"
        return 0
    fi

    log_info "Building and running CSnakes test..."
    if timeout 120 dotnet run > /tmp/csnakes_test.log 2>&1; then
        if grep -qE "PASSED|success|Connected|All tests" /tmp/csnakes_test.log; then
            cat /tmp/csnakes_test.log
            log_pass "csnakes"
        else
            log_fail "csnakes"
            cat /tmp/csnakes_test.log
        fi
    else
        log_fail "csnakes (execution failed or timed out)"
        cat /tmp/csnakes_test.log
    fi
}

test_jpype() {
    log_section "Testing JPype Bridge"
    cd "$SCRIPT_DIR/jpype"

    if ! check_java; then
        log_skip "jpype: Java not found"
        return 0
    fi

    if ! pip3 show jpype1 &>/dev/null; then
        log_skip "jpype: jpype1 package not installed"
        return 0
    fi

    log_info "Running JPype Python test..."
    if timeout 60 python3 rpyc_client.py > /tmp/jpype_test.log 2>&1; then
        if grep -qE "PASSED|success|All tests|4\.0|3\.0" /tmp/jpype_test.log; then
            cat /tmp/jpype_test.log
            log_pass "jpype"
        else
            log_fail "jpype"
            cat /tmp/jpype_test.log
        fi
    else
        log_fail "jpype (execution failed or timed out)"
        cat /tmp/jpype_test.log
    fi
}

test_jpy() {
    log_section "Testing jpy Bridge"
    cd "$SCRIPT_DIR/jpy"

    if ! check_java; then
        log_skip "jpy: Java not found"
        return 0
    fi

    if ! pip3 show jpy &>/dev/null; then
        log_skip "jpy: jpy package not installed"
        return 0
    fi

    log_info "Running jpy Python test..."
    if timeout 60 python3 rpyc_client.py > /tmp/jpy_test.log 2>&1; then
        if grep -qE "PASSED|success|All tests|4\.0|3\.0" /tmp/jpy_test.log; then
            cat /tmp/jpy_test.log
            log_pass "jpy"
        else
            log_fail "jpy"
            cat /tmp/jpy_test.log
        fi
    else
        log_fail "jpy (execution failed or timed out)"
        cat /tmp/jpy_test.log
    fi
}

test_pyo3() {
    log_section "Testing PyO3 Bridge"
    cd "$SCRIPT_DIR/pyo3"

    if ! check_rust; then
        log_skip "pyo3: Rust/Cargo not found"
        return 0
    fi

    log_info "Building PyO3 test..."
    if ! cargo build --release 2>&1 | tee /tmp/pyo3_build.log; then
        log_fail "pyo3 (build failed)"
        cat /tmp/pyo3_build.log
        return 0
    fi

    log_info "Running PyO3 test..."
    if timeout 60 ./target/release/rpyc-client > /tmp/pyo3_test.log 2>&1; then
        if grep -qE "PASSED|success|All tests|4\.0|sqrt" /tmp/pyo3_test.log; then
            cat /tmp/pyo3_test.log
            log_pass "pyo3"
        else
            log_fail "pyo3"
            cat /tmp/pyo3_test.log
        fi
    else
        log_fail "pyo3 (execution failed or timed out)"
        cat /tmp/pyo3_test.log
    fi
}

test_pycall() {
    log_section "Testing PyCall.rb Bridge"
    cd "$SCRIPT_DIR/pycall"

    if ! check_ruby; then
        log_skip "pycall: Ruby not found"
        return 0
    fi

    # Check if pycall gem is installed
    if ! gem list pycall 2>/dev/null | grep -q pycall; then
        log_skip "pycall: pycall gem not installed"
        return 0
    fi

    log_info "Running PyCall.rb test..."
    if timeout 60 ruby rpyc_client.rb > /tmp/pycall_test.log 2>&1; then
        if grep -qE "PASSED|success|All tests|4\.0|3\.0" /tmp/pycall_test.log; then
            cat /tmp/pycall_test.log
            log_pass "pycall"
        else
            log_fail "pycall"
            cat /tmp/pycall_test.log
        fi
    else
        log_fail "pycall (execution failed or timed out)"
        cat /tmp/pycall_test.log
    fi
}

test_rust_ffi_go() {
    log_section "Testing Rust FFI (Go) Bridge"
    cd "$SCRIPT_DIR/rust-ffi-go"

    if ! check_rust; then
        log_skip "rust-ffi-go: Rust/Cargo not found"
        return 0
    fi

    if ! check_go; then
        log_skip "rust-ffi-go: Go not found"
        return 0
    fi

    log_info "Building Rust FFI library..."
    if ! cargo build --release 2>&1 | tee /tmp/rust_ffi_build.log; then
        log_fail "rust-ffi-go (Rust build failed)"
        cat /tmp/rust_ffi_build.log
        return 0
    fi

    # Copy library
    cp target/release/librpyc_bridge.so . 2>/dev/null || true

    log_info "Building Go example..."
    if ! CGO_ENABLED=1 go build -o rpyc_example main.go 2>&1 | tee /tmp/go_build.log; then
        log_fail "rust-ffi-go (Go build failed)"
        cat /tmp/go_build.log
        return 0
    fi

    log_info "Running Rust FFI Go test..."
    # Run test with timeout, capture output, then check
    if timeout 60 bash -c 'LD_LIBRARY_PATH=. ./rpyc_example' > /tmp/rust_ffi_test.log 2>&1; then
        if grep -q "All tests passed" /tmp/rust_ffi_test.log; then
            cat /tmp/rust_ffi_test.log
            log_pass "rust-ffi-go"
        else
            log_fail "rust-ffi-go"
            cat /tmp/rust_ffi_test.log
        fi
    else
        log_fail "rust-ffi-go (execution failed or timed out)"
        cat /tmp/rust_ffi_test.log
    fi
}

test_rust_ffi_node() {
    log_section "Testing Rust FFI (Node.js) Bridge"
    cd "$SCRIPT_DIR/rust-ffi-node"

    if ! check_rust; then
        log_skip "rust-ffi-node: Rust/Cargo not found"
        return 0
    fi

    if ! command -v node &> /dev/null; then
        log_skip "rust-ffi-node: Node.js not found"
        return 0
    fi

    local node_version=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
    if [ "$node_version" -lt 18 ] 2>/dev/null; then
        log_skip "rust-ffi-node: Requires Node.js 18+ (found: $node_version)"
        return 0
    fi

    log_info "Node.js found: $(node --version)"

    # Build Rust library if not exists
    if [ ! -f "librpyc_bridge.so" ]; then
        log_info "Building Rust FFI library..."
        if ! (cd ../rust-ffi-go && cargo build --release && cp target/release/librpyc_bridge.so ../rust-ffi-node/) 2>&1 | tee /tmp/rust_ffi_node_build.log; then
            log_fail "rust-ffi-node (Rust build failed)"
            cat /tmp/rust_ffi_node_build.log
            return 0
        fi
    fi

    # Install npm dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing npm dependencies..."
        if ! npm install 2>&1 | tee /tmp/npm_install.log; then
            log_fail "rust-ffi-node (npm install failed)"
            cat /tmp/npm_install.log
            return 0
        fi
    fi

    log_info "Running Node.js FFI test..."
    if timeout 60 npm run test > /tmp/rust_ffi_node_test.log 2>&1; then
        if grep -q "All tests passed" /tmp/rust_ffi_node_test.log; then
            cat /tmp/rust_ffi_node_test.log
            log_pass "rust-ffi-node"
        else
            log_fail "rust-ffi-node"
            cat /tmp/rust_ffi_node_test.log
        fi
    else
        log_fail "rust-ffi-node (execution failed or timed out)"
        cat /tmp/rust_ffi_node_test.log
    fi
}

#######################################
# Bridge Group Functions
#######################################

test_all() {
    test_pythonnet
    test_csnakes
    test_jpype
    test_jpy
    test_pyo3
    test_pycall
    test_rust_ffi_go
    test_rust_ffi_node
}

test_jvm() {
    test_jpype
    test_jpy
}

test_dotnet() {
    test_pythonnet
    test_csnakes
}

test_ffi() {
    test_rust_ffi_go
    test_rust_ffi_node
}

test_rust() {
    test_pyo3
}

test_ruby() {
    test_pycall
}

#######################################
# Main
#######################################

print_usage() {
    echo "Usage: $0 [OPTIONS] [BRIDGES...]"
    echo
    echo "Options:"
    echo "  --all       Test all bridges"
    echo "  --jvm       Test JVM bridges (jpype, jpy)"
    echo "  --dotnet    Test .NET bridges (pythonnet, csnakes)"
    echo "  --ffi       Test FFI bridges (rust-ffi-go, rust-ffi-node)"
    echo "  --rust      Test Rust bridges (pyo3)"
    echo "  --ruby      Test Ruby bridges (pycall)"
    echo "  --node      Test Node.js bridges (rust-ffi-node)"
    echo "  --help      Show this help message"
    echo
    echo "Bridges:"
    echo "  pythonnet, csnakes, jpype, jpy, pyo3, pycall, rust-ffi-go, rust-ffi-node"
    echo
    echo "Examples:"
    echo "  $0 --all                    # Test all bridges"
    echo "  $0 --jvm --rust             # Test JVM and Rust bridges"
    echo "  $0 pythonnet jpype          # Test specific bridges"
}

print_summary() {
    log_section "Test Summary"
    echo
    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo

    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}Failed bridges:${NC}"
        for bridge in "${FAILED_BRIDGES[@]}"; do
            echo "  - $bridge"
        done
        echo
    fi

    if [ $FAILED -eq 0 ] && [ $PASSED -gt 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    elif [ $FAILED -gt 0 ]; then
        echo -e "${RED}Some tests failed.${NC}"
    else
        echo -e "${YELLOW}No tests were run.${NC}"
    fi
}

cleanup() {
    stop_rpyc_server
}

trap cleanup EXIT

main() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        print_usage
        exit 0
    fi

    log_section "Python Bridge Integration Tests"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Script dir: $SCRIPT_DIR"

    # Start RPyC server
    start_rpyc_server

    # Parse arguments and run tests
    for arg in "$@"; do
        case "$arg" in
            --all)
                test_all
                ;;
            --jvm)
                test_jvm
                ;;
            --dotnet)
                test_dotnet
                ;;
            --ffi)
                test_ffi
                ;;
            --rust)
                test_rust
                ;;
            --ruby)
                test_ruby
                ;;
            --node)
                test_rust_ffi_node
                ;;
            pythonnet)
                test_pythonnet
                ;;
            csnakes)
                test_csnakes
                ;;
            jpype)
                test_jpype
                ;;
            jpy)
                test_jpy
                ;;
            pyo3)
                test_pyo3
                ;;
            pycall)
                test_pycall
                ;;
            rust-ffi-go)
                test_rust_ffi_go
                ;;
            rust-ffi-node)
                test_rust_ffi_node
                ;;
            *)
                log_fail "Unknown bridge or option: $arg"
                print_usage
                exit 1
                ;;
        esac
    done

    # Print summary
    print_summary

    # Exit with failure if any tests failed
    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
}

main "$@"
