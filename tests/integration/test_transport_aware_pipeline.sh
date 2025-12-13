#!/bin/bash
# test_transport_aware_pipeline.sh - Tests for transport-aware pipeline compilation
# Tests all three transports: pipe, direct, http
# Includes both meta-interpreter tests and low-level tests

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_transport_test"
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

cleanup() {
    rm -rf "$OUTPUT_DIR"
}

setup() {
    cleanup
    mkdir -p "$OUTPUT_DIR"
}

# ==================================================================
# Section 1: Low-Level Transport Tests
# ==================================================================

# Test 1: infer_transport_from_targets - dotnet family
test_infer_transport_dotnet() {
    log_info "Test 1: infer_transport_from_targets - .NET family"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        infer_transport_from_targets(csharp, powershell, T),
        format('transport:~w~n', [T]),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "transport:direct"; then
        log_pass "csharp -> powershell returns 'direct'"
    else
        log_fail "Expected 'direct', got: $OUTPUT"
    fi
}

# Test 2: infer_transport_from_targets - cross family
test_infer_transport_cross_family() {
    log_info "Test 2: infer_transport_from_targets - cross family"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        infer_transport_from_targets(bash, python, T),
        format('transport:~w~n', [T]),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "transport:pipe"; then
        log_pass "bash -> python returns 'pipe'"
    else
        log_fail "Expected 'pipe', got: $OUTPUT"
    fi
}

# Test 3: step_pair_transport
test_step_pair_transport() {
    log_info "Test 3: step_pair_transport"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        S1 = step(s1, csharp, 'a.cs', []),
        S2 = step(s2, powershell, 'b.ps1', []),
        step_pair_transport(S1, S2, T),
        format('transport:~w~n', [T]),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "transport:direct"; then
        log_pass "step_pair_transport works correctly"
    else
        log_fail "Expected 'direct', got: $OUTPUT"
    fi
}

# Test 4: group_steps_by_transport - mixed targets
test_group_steps_mixed() {
    log_info "Test 4: group_steps_by_transport - mixed targets"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        Steps = [
            step(s1, bash, 'a.sh', []),
            step(s2, awk, 'b.awk', []),
            step(s3, csharp, 'c.cs', []),
            step(s4, powershell, 'd.ps1', []),
            step(s5, python, 'e.py', [])
        ],
        group_steps_by_transport(Steps, Groups),
        length(Groups, NumGroups),
        format('num_groups:~w~n', [NumGroups]),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "num_groups:3"; then
        log_pass "Mixed targets split into 3 groups"
    else
        log_fail "Expected 3 groups, got: $OUTPUT"
    fi
}

# ==================================================================
# Section 2: Transport-Specific Generation Tests
# ==================================================================

# Test 5: Pipe transport generates shell script
test_pipe_transport_generation() {
    log_info "Test 5: Pipe transport generates shell script"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        use_module('src/unifyweaver/glue/shell_glue'),
        Steps = [
            step(s1, bash, 'a.sh', []),
            step(s2, python, 'b.py', [])
        ],
        generate_pipeline_for_groups([group(pipe, Steps)], [], Code),
        (sub_atom(Code, _, _, _, '#!/bin/bash') -> format('has_shebang:true~n') ; format('has_shebang:false~n')),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "has_shebang:true"; then
        log_pass "Pipe transport generates bash script"
    else
        log_fail "Expected bash shebang, got: $OUTPUT"
    fi
}

# Test 6: Direct transport generates C# code
test_direct_transport_generation() {
    log_info "Test 6: Direct transport generates C# code"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        use_module('src/unifyweaver/glue/dotnet_glue'),
        Steps = [
            step(enrich, csharp, 'a.cs', []),
            step(aggregate, powershell, 'b.ps1', [])
        ],
        generate_pipeline_for_groups([group(direct, Steps)], [], Code),
        (sub_atom(Code, _, _, _, 'PowerShellBridge') -> format('has_bridge:true~n') ; format('has_bridge:false~n')),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "has_bridge:true"; then
        log_pass "Direct transport generates C# with PowerShellBridge"
    else
        log_fail "Expected PowerShellBridge, got: $OUTPUT"
    fi
}

# Test 7: HTTP transport generates Python with requests
test_http_transport_generation() {
    log_info "Test 7: HTTP transport generates Python with requests"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        use_module('src/unifyweaver/glue/network_glue'),
        Steps = [
            step(ml_predict, python, 'http://ml:8080/predict', [])
        ],
        generate_pipeline_for_groups([group(http, Steps)], [language(python)], Code),
        (sub_atom(Code, _, _, _, 'requests.post') -> format('has_requests:true~n') ; format('has_requests:false~n')),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "has_requests:true"; then
        log_pass "HTTP transport generates Python with requests.post"
    else
        log_fail "Expected requests.post, got: $OUTPUT"
    fi
}

# ==================================================================
# Section 3: Meta-Interpreter Integration Tests
# ==================================================================

# Test 8: compile_goal_to_pipeline basic usage
test_meta_interpreter_basic() {
    log_info "Test 8: compile_goal_to_pipeline basic usage"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/meta_test.pl" << 'EOF'
:- use_module('src/unifyweaver/core/compiler_driver').
:- use_module('src/unifyweaver/core/target_mapping').
:- use_module('src/unifyweaver/glue/shell_glue').

% Define a simple goal
process_data(Input, Output) :-
    fetch(Input, Raw),
    transform(Raw, Transformed),
    store(Transformed, Output).

% Declare targets
:- declare_target(fetch/2, bash, [file('fetch.sh')]).
:- declare_target(transform/2, python, [file('transform.py')]).
:- declare_target(store/2, awk, [file('store.awk')]).

test :-
    compile_goal_to_pipeline(process_data(_, _), [input('data.csv')], Script, Steps),
    length(Steps, NumSteps),
    format('num_steps:~w~n', [NumSteps]),
    (sub_atom(Script, _, _, _, 'python3') -> format('has_python:true~n') ; format('has_python:false~n')).
EOF

    OUTPUT=$(swipl -g "consult('output_transport_test/meta_test.pl'), test" -t halt 2>&1)

    if echo "$OUTPUT" | grep -q "num_steps:3" && echo "$OUTPUT" | grep -q "has_python:true"; then
        log_pass "Meta-interpreter infers 3 steps and generates pipeline"
    else
        log_fail "Meta-interpreter test failed: $OUTPUT"
    fi
}

# Test 9: compile_goal_to_pipeline with mixed transports
test_meta_interpreter_mixed_transports() {
    log_info "Test 9: compile_goal_to_pipeline with mixed transports"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/mixed_transport_test.pl" << 'EOF'
:- use_module('src/unifyweaver/core/compiler_driver').
:- use_module('src/unifyweaver/core/target_mapping').
:- use_module('src/unifyweaver/glue/shell_glue').
:- use_module('src/unifyweaver/glue/dotnet_glue').

% Define a goal with mixed targets
analytics_pipeline(Input, Output) :-
    ingest(Input, Raw),
    enrich(Raw, Enriched),
    aggregate(Enriched, Stats),
    export(Stats, Output).

% Declare targets - mixing shell and .NET
:- declare_target(ingest/2, bash, [file('ingest.sh')]).
:- declare_target(enrich/2, csharp, [file('Enricher.cs')]).
:- declare_target(aggregate/2, powershell, [file('aggregate.ps1')]).
:- declare_target(export/2, python, [file('export.py')]).

test :-
    compile_goal_to_pipeline(analytics_pipeline(_, _), [], Script, Steps),
    length(Steps, NumSteps),
    format('num_steps:~w~n', [NumSteps]).
EOF

    OUTPUT=$(swipl -g "consult('output_transport_test/mixed_transport_test.pl'), test" -t halt 2>&1)

    if echo "$OUTPUT" | grep -q "num_steps:4"; then
        log_pass "Mixed transport pipeline infers 4 steps"
    else
        log_fail "Mixed transport test failed: $OUTPUT"
    fi
}

# ==================================================================
# Section 4: Multi-Transport Orchestration Tests
# ==================================================================

# Test 10: Multi-group orchestration combines scripts
test_multi_group_orchestration() {
    log_info "Test 10: Multi-group orchestration"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        use_module('src/unifyweaver/glue/shell_glue'),
        use_module('src/unifyweaver/glue/dotnet_glue'),
        
        % Create groups with different transports
        Groups = [
            group(pipe, [step(s1, bash, 'a.sh', []), step(s2, awk, 'b.awk', [])]),
            group(direct, [step(s3, csharp, 'c.cs', []), step(s4, powershell, 'd.ps1', [])])
        ],
        generate_pipeline_for_groups(Groups, [], Script),
        atom_length(Script, Len),
        (Len > 100 -> format('generated:true~n') ; format('generated:false~n')),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "generated:true"; then
        log_pass "Multi-group orchestration generates combined script"
    else
        log_fail "Multi-group orchestration failed: $OUTPUT"
    fi
}

# Test 11: Boundary nodes appear in both groups
test_boundary_nodes() {
    log_info "Test 11: Boundary nodes in adjacent groups"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/glue/goal_inference'),
        Steps = [
            step(s1, bash, 'a.sh', []),
            step(s2, csharp, 'c.cs', []),
            step(s3, powershell, 'd.ps1', []),
            step(s4, python, 'e.py', [])
        ],
        group_steps_by_transport(Steps, Groups),
        
        % Check that boundary node (s2) appears in first group
        Groups = [Group1|_],
        Group1 = group(_, Steps1),
        (member(step(s2, _, _, _), Steps1) -> format('s2_in_g1:true~n') ; format('s2_in_g1:false~n')),
        halt(0)
    " 2>&1)

    if echo "$OUTPUT" | grep -q "s2_in_g1:true"; then
        log_pass "Boundary node appears in first group"
    else
        log_fail "Boundary node check failed: $OUTPUT"
    fi
}

# ==================================================================
# Main test runner
# ==================================================================

main() {
    echo "=============================================="
    echo "  Transport-Aware Pipeline Tests"
    echo "=============================================="
    echo ""

    setup

    echo ""
    echo "--- Low-Level Transport Tests ---"
    test_infer_transport_dotnet
    test_infer_transport_cross_family
    test_step_pair_transport
    test_group_steps_mixed

    echo ""
    echo "--- Transport-Specific Generation Tests ---"
    test_pipe_transport_generation
    test_direct_transport_generation
    test_http_transport_generation

    echo ""
    echo "--- Meta-Interpreter Integration Tests ---"
    test_meta_interpreter_basic
    test_meta_interpreter_mixed_transports

    echo ""
    echo "--- Multi-Transport Orchestration Tests ---"
    test_multi_group_orchestration
    test_boundary_nodes

    echo ""
    echo "=============================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=============================================="

    cleanup

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
