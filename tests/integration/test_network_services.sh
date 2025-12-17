#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_network_services.sh - Integration tests for Client-Server Phase 3
# Tests TCP and HTTP network service compilation for Python, Go, and Rust targets

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
echo "Client-Server Phase 3: Network Services"
echo "Integration Tests"
echo "==========================================="
echo ""

# Test 1: is_network_service identifies TCP services
info "Test 1: is_network_service identifies TCP services"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), is_network_service(service(test, [transport(tcp('0.0.0.0', 8080))], [])), halt(0)" >/dev/null; then
    pass "is_network_service works for TCP"
else
    fail "is_network_service failed for TCP"
fi

# Test 2: is_network_service identifies HTTP services
info "Test 2: is_network_service identifies HTTP services"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), is_network_service(service(test, [transport(http('/api'))], [])), halt(0)" >/dev/null; then
    pass "is_network_service works for HTTP"
else
    fail "is_network_service failed for HTTP"
fi

# Test 3: TCP service is not cross-process (it's network)
info "Test 3: TCP service is network, not cross-process"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), \\+ is_cross_process_service(service(test, [transport(tcp('0.0.0.0', 8080))], [])), halt(0)" >/dev/null; then
    pass "TCP service correctly categorized as network"
else
    fail "TCP service incorrectly categorized"
fi

# Test 4: Python TCP service compilation
info "Test 4: Python TCP service compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_tcp_service_python(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'socket.AF_INET'), sub_string(Code, _, _, _, 'start_server'), halt(0)" >/dev/null; then
    pass "Python TCP service compiles"
else
    fail "Python TCP service compilation failed"
fi

# Test 5: Python TCP service has JSONL protocol
info "Test 5: Python TCP service has JSONL protocol"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_tcp_service_python(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_id'), sub_string(Code, _, _, _, '_payload'), halt(0)" >/dev/null; then
    pass "Python TCP has JSONL protocol"
else
    fail "Python TCP JSONL protocol missing"
fi

# Test 6: Python HTTP service compilation
info "Test 6: Python HTTP service compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_http_service_python(service(rest, [transport(http('/api/v1'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'HTTPServer'), sub_string(Code, _, _, _, 'do_GET'), halt(0)" >/dev/null; then
    pass "Python HTTP service compiles"
else
    fail "Python HTTP service compilation failed"
fi

# Test 7: Python HTTP service handles REST methods
info "Test 7: Python HTTP service handles REST methods"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_http_service_python(service(rest, [transport(http('/api/v1'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'do_POST'), sub_string(Code, _, _, _, 'do_PUT'), sub_string(Code, _, _, _, 'do_DELETE'), halt(0)" >/dev/null; then
    pass "Python HTTP handles REST methods"
else
    fail "Python HTTP REST methods missing"
fi

# Test 8: Go TCP service compilation
info "Test 8: Go TCP service compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_tcp_service_go(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'net.Listen'), sub_string(Code, _, _, _, 'StartServer'), halt(0)" >/dev/null; then
    pass "Go TCP service compiles"
else
    fail "Go TCP service compilation failed"
fi

# Test 9: Go TCP service has JSONL protocol
info "Test 9: Go TCP service has JSONL protocol"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_tcp_service_go(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_id'), sub_string(Code, _, _, _, 'json.Unmarshal'), halt(0)" >/dev/null; then
    pass "Go TCP has JSONL protocol"
else
    fail "Go TCP JSONL protocol missing"
fi

# Test 10: Go HTTP service compilation
info "Test 10: Go HTTP service compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_http_service_go(service(rest, [transport(http('/api/v1'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'http.Server'), sub_string(Code, _, _, _, 'ServeHTTP'), halt(0)" >/dev/null; then
    pass "Go HTTP service compiles"
else
    fail "Go HTTP service compilation failed"
fi

# Test 11: Go HTTP service handles REST methods
info "Test 11: Go HTTP service handles REST methods"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_http_service_go(service(rest, [transport(http('/api/v1'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'Method'), sub_string(Code, _, _, _, 'POST'), halt(0)" >/dev/null; then
    pass "Go HTTP handles REST methods"
else
    fail "Go HTTP REST methods missing"
fi

# Test 12: Rust TCP service compilation
info "Test 12: Rust TCP service compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_tcp_service_rust(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'TcpListener'), sub_string(Code, _, _, _, 'start_server'), halt(0)" >/dev/null; then
    pass "Rust TCP service compiles"
else
    fail "Rust TCP service compilation failed"
fi

# Test 13: Rust TCP service has JSONL protocol
info "Test 13: Rust TCP service has JSONL protocol"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_tcp_service_rust(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_id'), sub_string(Code, _, _, _, 'serde_json'), halt(0)" >/dev/null; then
    pass "Rust TCP has JSONL protocol"
else
    fail "Rust TCP JSONL protocol missing"
fi

# Test 14: Rust HTTP service compilation
info "Test 14: Rust HTTP service compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_http_service_rust(service(rest, [transport(http('/api/v1'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'tiny_http'), sub_string(Code, _, _, _, 'start_server'), halt(0)" >/dev/null; then
    pass "Rust HTTP service compiles"
else
    fail "Rust HTTP service compilation failed"
fi

# Test 15: Rust HTTP service handles REST methods
info "Test 15: Rust HTTP service handles REST methods"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_http_service_rust(service(rest, [transport(http('/api/v1'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'Method'), sub_string(Code, _, _, _, 'Post'), halt(0)" >/dev/null; then
    pass "Rust HTTP handles REST methods"
else
    fail "Rust HTTP REST methods missing"
fi

# Test 16: TCP client compilation (Python)
info "Test 16: Python TCP client compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_tcp_client_python(api, '0.0.0.0', 8080, Code), sub_string(Code, _, _, _, 'ApiClient'), sub_string(Code, _, _, _, 'connect'), halt(0)" >/dev/null; then
    pass "Python TCP client compiles"
else
    fail "Python TCP client compilation failed"
fi

# Test 17: HTTP client compilation (Python)
info "Test 17: Python HTTP client compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_http_client_python(rest, '/api/v1', Code), sub_string(Code, _, _, _, 'RestClient'), sub_string(Code, _, _, _, 'get'), halt(0)" >/dev/null; then
    pass "Python HTTP client compiles"
else
    fail "Python HTTP client compilation failed"
fi

# Test 18: TCP client compilation (Go)
info "Test 18: Go TCP client compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_tcp_client_go(api, '0.0.0.0', 8080, Code), sub_string(Code, _, _, _, 'ApiClient'), sub_string(Code, _, _, _, 'Connect'), halt(0)" >/dev/null; then
    pass "Go TCP client compiles"
else
    fail "Go TCP client compilation failed"
fi

# Test 19: HTTP client compilation (Go)
info "Test 19: Go HTTP client compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_http_client_go(rest, '/api/v1', Code), sub_string(Code, _, _, _, 'RestClient'), sub_string(Code, _, _, _, 'Get'), halt(0)" >/dev/null; then
    pass "Go HTTP client compiles"
else
    fail "Go HTTP client compilation failed"
fi

# Test 20: TCP client compilation (Rust)
info "Test 20: Rust TCP client compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_tcp_client_rust(api, '0.0.0.0', 8080, Code), sub_string(Code, _, _, _, 'ApiClient'), sub_string(Code, _, _, _, 'TcpStream'), halt(0)" >/dev/null; then
    pass "Rust TCP client compiles"
else
    fail "Rust TCP client compilation failed"
fi

# Test 21: HTTP client compilation (Rust)
info "Test 21: Rust HTTP client compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_http_client_rust(rest, '/api/v1', Code), sub_string(Code, _, _, _, 'RestClient'), sub_string(Code, _, _, _, 'reqwest'), halt(0)" >/dev/null; then
    pass "Rust HTTP client compiles"
else
    fail "Rust HTTP client compilation failed"
fi

# Test 22: Stateful TCP service (Python)
info "Test 22: Stateful TCP service (Python)"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_tcp_service_python(service(counter, [transport(tcp('0.0.0.0', 8080)), stateful(true)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'stateful=True'), halt(0)" >/dev/null; then
    pass "Python stateful TCP service compiles"
else
    fail "Python stateful TCP service failed"
fi

# Test 23: Stateful HTTP service (Go)
info "Test 23: Stateful HTTP service (Go)"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_http_service_go(service(counter, [transport(http('/api')), stateful(true)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'sync.RWMutex'), halt(0)" >/dev/null; then
    pass "Go stateful HTTP service compiles"
else
    fail "Go stateful HTTP service failed"
fi

# Test 24: Service dispatch through compile_service_to_python (TCP)
info "Test 24: Service dispatch through compile_service_to_python (TCP)"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(api, [transport(tcp('0.0.0.0', 8080))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'socket.AF_INET'), halt(0)" >/dev/null; then
    pass "Python service dispatch works for TCP"
else
    fail "Python service dispatch failed for TCP"
fi

# Test 25: Service dispatch through compile_service_to_go (HTTP)
info "Test 25: Service dispatch through compile_service_to_go (HTTP)"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_service_to_go(service(rest, [transport(http('/api'))], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'http.Server'), halt(0)" >/dev/null; then
    pass "Go service dispatch works for HTTP"
else
    fail "Go service dispatch failed for HTTP"
fi

# Test 26: In-process service still works (regression test)
info "Test 26: In-process service still works (regression test)"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(echo, [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'class EchoService'), \\+ sub_string(Code, _, _, _, 'socket.AF_INET'), halt(0)" >/dev/null; then
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
