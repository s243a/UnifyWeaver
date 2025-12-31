#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# test_in_process_services.sh - Integration tests for Client-Server Phase 1
# Tests in-process service definitions and call_service pipeline stage

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

echo "=========================================="
echo "Client-Server Phase 1: In-Process Services"
echo "Integration Tests"
echo "=========================================="
echo ""

# Test 1: Service validation module loads correctly
info "Test 1: Service validation module loads correctly"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), halt(0)" >/dev/null; then
    pass "Service validation module loads"
else
    fail "Service validation module failed to load"
fi

# Test 2: Valid service definition is accepted
info "Test 2: Valid service definition is accepted"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), is_valid_service(service(echo, [receive(X), respond(X)])), halt(0)" >/dev/null; then
    pass "Valid simple service accepted"
else
    fail "Valid simple service rejected"
fi

# Test 3: Service with options validates correctly
info "Test 3: Service with options validates correctly"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), is_valid_service(service(counter, [stateful(true), timeout(5000)], [receive(X), respond(X)])), halt(0)" >/dev/null; then
    pass "Service with options validated"
else
    fail "Service with options validation failed"
fi

# Test 4: Service operations are validated
info "Test 4: Service operations are validated"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), is_valid_service_operation(receive(X)), is_valid_service_operation(respond(Y)), is_valid_service_operation(respond_error(not_found)), is_valid_service_operation(state_get(count, V)), is_valid_service_operation(state_put(count, 0)), is_valid_service_operation(call_service(other_service, Req, Resp)), halt(0)" >/dev/null; then
    pass "All service operations validated"
else
    fail "Service operation validation failed"
fi

# Test 5: call_service stage validates in pipeline
info "Test 5: call_service stage validates in pipeline"
if run_prolog "use_module('src/unifyweaver/core/pipeline_validation'), is_valid_stage(call_service(my_service, request_field, response_field)), is_valid_stage(call_service(my_service, request_field, response_field, [timeout(5000), retry(3)])), halt(0)" >/dev/null; then
    pass "call_service stage validates"
else
    fail "call_service validation failed"
fi

# Test 6: call_service options are validated
info "Test 6: call_service options are validated"
if run_prolog "use_module('src/unifyweaver/core/pipeline_validation'), is_valid_call_service_option(timeout(5000)), is_valid_call_service_option(retry(3)), is_valid_call_service_option(retry_delay(100)), is_valid_call_service_option(fallback(default)), halt(0)" >/dev/null; then
    pass "call_service options validated"
else
    fail "call_service options validation failed"
fi

# Test 7: Python service compilation
info "Test 7: Python service compilation"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), compile_service_to_python(service(echo, [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'class EchoService'), sub_string(Code, _, _, _, 'def call'), halt(0)" >/dev/null; then
    pass "Python service compilation works"
else
    fail "Python service compilation failed"
fi

# Test 8: Go service compilation
info "Test 8: Go service compilation"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), compile_service_to_go(service(echo, [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'type EchoService struct'), sub_string(Code, _, _, _, 'func (s *EchoService) Call'), halt(0)" >/dev/null; then
    pass "Go service compilation works"
else
    fail "Go service compilation failed"
fi

# Test 9: Rust service compilation
info "Test 9: Rust service compilation"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), compile_service_to_rust(service(echo, [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'pub struct EchoService'), sub_string(Code, _, _, _, 'impl Service for EchoService'), halt(0)" >/dev/null; then
    pass "Rust service compilation works"
else
    fail "Rust service compilation failed"
fi

# Test 10: Python enhanced helpers include service infrastructure
info "Test 10: Python enhanced helpers include service infrastructure"
if run_prolog "use_module('src/unifyweaver/targets/python_target'), enhanced_pipeline_helpers(Code), sub_string(Code, _, _, _, '_services = {}'), sub_string(Code, _, _, _, 'class Service'), halt(0)" >/dev/null; then
    pass "Python service infrastructure included"
else
    fail "Python service infrastructure missing"
fi

# Test 11: Go enhanced helpers include service infrastructure
info "Test 11: Go enhanced helpers include service infrastructure"
if run_prolog "use_module('src/unifyweaver/targets/go_target'), go_enhanced_helpers(Code), sub_string(Code, _, _, _, 'type Service interface'), sub_string(Code, _, _, _, 'var services'), halt(0)" >/dev/null; then
    pass "Go service infrastructure included"
else
    fail "Go service infrastructure missing"
fi

# Test 12: Rust enhanced helpers include service infrastructure
info "Test 12: Rust enhanced helpers include service infrastructure"
if run_prolog "use_module('src/unifyweaver/targets/rust_target'), rust_enhanced_helpers(std_thread, Code), sub_string(Code, _, _, _, 'pub trait Service'), sub_string(Code, _, _, _, 'SERVICES'), halt(0)" >/dev/null; then
    pass "Rust service infrastructure included"
else
    fail "Rust service infrastructure missing"
fi

# Test 13: Invalid service rejected
info "Test 13: Invalid service rejected"
if run_prolog "use_module('src/unifyweaver/core/service_validation'), validate_service(service(123, []), Errors), Errors \\= [], halt(0)" >/dev/null; then
    pass "Invalid service correctly rejected"
else
    fail "Invalid service should be rejected"
fi

echo ""
echo "=========================================="
echo -e "Results: ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}"
echo "=========================================="

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
exit 0
