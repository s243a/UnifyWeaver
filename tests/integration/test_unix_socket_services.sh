#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_unix_socket_services.sh - Integration tests for Client-Server Phase 2
# Tests Unix socket service compilation for Python, Go, and Rust targets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

info() {
    echo -e "${YELLOW}INFO${NC}: $1"
}

run_prolog() {
    timeout 30 swipl -g "$1" -t "halt(1)" 2>&1
    return $?
}

echo "==========================================="
echo "Client-Server Phase 2: Unix Socket Services"
echo "Integration Tests"
echo "==========================================="
echo ""

# Test 1: Service validation helper predicates work
info "Test 1: Service validation helper predicates work"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), get_service_transport(service(test, [transport(unix_socket('/tmp/test.sock'))], []), T), T = unix_socket('/tmp/test.sock'), halt(0)" >/dev/null; then
    pass "get_service_transport extracts unix_socket"
else
    fail "get_service_transport failed"
fi

# Test 2: is_cross_process_service identifies unix socket services
info "Test 2: is_cross_process_service identifies unix socket services"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), is_cross_process_service(service(test, [transport(unix_socket('/tmp/test.sock'))], [])), halt(0)" >/dev/null; then
    pass "is_cross_process_service works for unix socket"
else
    fail "is_cross_process_service failed"
fi

# Test 3: In-process service is not cross-process
info "Test 3: In-process service is not cross-process"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), \\+ is_cross_process_service(service(test, [], [])), halt(0)" >/dev/null; then
    pass "In-process service correctly not cross-process"
else
    fail "In-process service incorrectly identified as cross-process"
fi

# Test 4: get_service_protocol extracts protocol
info "Test 4: get_service_protocol extracts protocol"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), get_service_protocol(service(test, [protocol(jsonl)], []), P), P = jsonl, halt(0)" >/dev/null; then
    pass "get_service_protocol works"
else
    fail "get_service_protocol failed"
fi

# Test 5: Python Unix socket service compilation
info "Test 5: Python Unix socket service compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(echo, [transport(unix_socket('/tmp/echo.sock'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'socket.AF_UNIX'), sub_string(Code, _, _, _, 'start_server'), halt(0)" >/dev/null; then
    pass "Python Unix socket service compiles"
else
    fail "Python Unix socket service compilation failed"
fi

# Test 6: Python Unix socket service has JSONL protocol
info "Test 6: Python Unix socket service has JSONL protocol"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(echo, [transport(unix_socket('/tmp/echo.sock'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_id'), sub_string(Code, _, _, _, '_payload'), sub_string(Code, _, _, _, 'json.loads'), halt(0)" >/dev/null; then
    pass "Python Unix socket has JSONL protocol"
else
    fail "Python Unix socket JSONL protocol missing"
fi

# Test 7: Python Unix socket client compilation
info "Test 7: Python Unix socket client compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_unix_socket_client_python(echo, '/tmp/echo.sock', Code), sub_string(Code, _, _, _, 'EchoClient'), sub_string(Code, _, _, _, 'connect'), halt(0)" >/dev/null; then
    pass "Python Unix socket client compiles"
else
    fail "Python Unix socket client compilation failed"
fi

# Test 8: Go Unix socket service compilation
info "Test 8: Go Unix socket service compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_service_to_go(service(echo, [transport(unix_socket('/tmp/echo.sock'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'net.Listen'), sub_string(Code, _, _, _, 'StartServer'), halt(0)" >/dev/null; then
    pass "Go Unix socket service compiles"
else
    fail "Go Unix socket service compilation failed"
fi

# Test 9: Go Unix socket service has JSONL protocol
info "Test 9: Go Unix socket service has JSONL protocol"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_service_to_go(service(echo, [transport(unix_socket('/tmp/echo.sock'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_id'), sub_string(Code, _, _, _, '_payload'), sub_string(Code, _, _, _, 'json.Unmarshal'), halt(0)" >/dev/null; then
    pass "Go Unix socket has JSONL protocol"
else
    fail "Go Unix socket JSONL protocol missing"
fi

# Test 10: Go Unix socket client compilation
info "Test 10: Go Unix socket client compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_unix_socket_client_go(echo, '/tmp/echo.sock', Code), sub_string(Code, _, _, _, 'EchoClient'), sub_string(Code, _, _, _, 'Connect'), halt(0)" >/dev/null; then
    pass "Go Unix socket client compiles"
else
    fail "Go Unix socket client compilation failed"
fi

# Test 11: Rust Unix socket service compilation
info "Test 11: Rust Unix socket service compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_service_to_rust(service(echo, [transport(unix_socket('/tmp/echo.sock'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'UnixListener'), sub_string(Code, _, _, _, 'start_server'), halt(0)" >/dev/null; then
    pass "Rust Unix socket service compiles"
else
    fail "Rust Unix socket service compilation failed"
fi

# Test 12: Rust Unix socket service has JSONL protocol
info "Test 12: Rust Unix socket service has JSONL protocol"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_service_to_rust(service(echo, [transport(unix_socket('/tmp/echo.sock'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_id'), sub_string(Code, _, _, _, '_payload'), sub_string(Code, _, _, _, 'serde_json'), halt(0)" >/dev/null; then
    pass "Rust Unix socket has JSONL protocol"
else
    fail "Rust Unix socket JSONL protocol missing"
fi

# Test 13: Rust Unix socket client compilation
info "Test 13: Rust Unix socket client compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_unix_socket_client_rust(echo, '/tmp/echo.sock', Code), sub_string(Code, _, _, _, 'EchoClient'), sub_string(Code, _, _, _, 'connect'), halt(0)" >/dev/null; then
    pass "Rust Unix socket client compiles"
else
    fail "Rust Unix socket client compilation failed"
fi

# Test 14: Stateful Unix socket service (Python)
info "Test 14: Stateful Unix socket service (Python)"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(counter, [transport(unix_socket('/tmp/counter.sock')), stateful(true)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'stateful=True'), sub_string(Code, _, _, _, '_lock'), halt(0)" >/dev/null; then
    pass "Python stateful Unix socket service compiles"
else
    fail "Python stateful Unix socket service failed"
fi

# Test 15: Stateful Unix socket service (Go)
info "Test 15: Stateful Unix socket service (Go)"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_service_to_go(service(counter, [transport(unix_socket('/tmp/counter.sock')), stateful(true)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'StatefulService'), sub_string(Code, _, _, _, 'sync.Mutex'), halt(0)" >/dev/null; then
    pass "Go stateful Unix socket service compiles"
else
    fail "Go stateful Unix socket service failed"
fi

# Test 16: Stateful Unix socket service (Rust)
info "Test 16: Stateful Unix socket service (Rust)"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_service_to_rust(service(counter, [transport(unix_socket('/tmp/counter.sock')), stateful(true)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'RwLock'), sub_string(Code, _, _, _, 'state_get'), halt(0)" >/dev/null; then
    pass "Rust stateful Unix socket service compiles"
else
    fail "Rust stateful Unix socket service failed"
fi

# Test 17: Service with custom timeout
info "Test 17: Service with custom timeout"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(fast, [transport(unix_socket('/tmp/fast.sock')), timeout(5000)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '5000'), halt(0)" >/dev/null; then
    pass "Custom timeout included in generated code"
else
    fail "Custom timeout not found in generated code"
fi

# Test 18: In-process service still works (regression test)
info "Test 18: In-process service still works (regression test)"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(echo, [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'class EchoService'), \\+ sub_string(Code, _, _, _, 'socket.AF_UNIX'), halt(0)" >/dev/null; then
    pass "In-process service compilation still works"
else
    fail "In-process service compilation broken"
fi

echo ""
echo "==========================================="
echo -e "Results: ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}"
echo "==========================================="

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
exit 0
