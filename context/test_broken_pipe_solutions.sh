#!/bin/bash
# Test script: Broken Pipe Solutions
# Location: context/test_broken_pipe_solutions.sh
#
# This script demonstrates three approaches to handling broken pipes
# when a producer writes faster than a consumer that exits early.

echo "=== Testing Broken Pipe Solutions ==="
echo "This simulates the ancestor_all | grep -q issue"
echo

# Simulate a slow producer (like ancestor_all doing BFS)
slow_producer() {
    local target="$1"
    echo "Producer started..." >&2
    for i in {1..20}; do
        echo "line_$i"
        sleep 0.05  # Simulate computation time
        if [[ "$i" -eq 5 && "$target" == "line_5" ]]; then
            echo "Producer: Found target early at iteration $i!" >&2
        fi
    done
    echo "Producer finished all 20 lines" >&2
}

# Test 1: Baseline (broken pipe) - Current problematic approach
test_baseline() {
    echo "--- Test 1: Baseline (pipe to grep -q) - BROKEN PIPE EXPECTED ---"
    echo "Command: slow_producer 'line_5' | grep -q 'line_5'"
    echo "Expected: Broken pipe error when grep exits early"
    echo
    
    slow_producer "line_5" 2>&1 | grep -q "line_5"
    status=$?
    echo "Exit status: $status (0=found, 1=not found)"
    echo "☝️  You should see 'Broken pipe' error above"
    echo
}

# Test 2: Tee approach - Safe but continues full computation
test_tee() {
    echo "--- Test 2: Tee Approach - NO BROKEN PIPE ---"
    echo "Command: slow_producer 'line_5' | tee >(grep -q 'line_5' && touch flag) >/dev/null"
    echo "Expected: No broken pipe, but producer runs all 20 iterations"
    echo
    
    tmpflag="/tmp/test_tee_$$"
    slow_producer "line_5" 2>&1 | tee >(grep -q "line_5" && touch "$tmpflag") >/dev/null
    
    if [[ -f "$tmpflag" ]]; then
        echo "✅ Match found!"
        rm -f "$tmpflag"
        status=0
    else
        echo "❌ Match not found"
        rm -f "$tmpflag" 
        status=1
    fi
    echo "Exit status: $status"
    echo "☝️  No broken pipe error, producer completed all iterations"
    echo
}

# Test 3: Early exit simulation - Efficient and safe
test_early_exit() {
    echo "--- Test 3: Early Exit (producer stops itself) - OPTIMAL ---"
    echo "Command: producer_with_early_exit 'line_5'"
    echo "Expected: No broken pipe, producer stops when target found"
    echo
    
    producer_with_early_exit() {
        local target="$1"
        echo "Producer started..." >&2
        for i in {1..20}; do
            echo "line_$i"
            if [[ "$target" == "line_$i" ]]; then
                echo "Producer: Found target, exiting early at iteration $i!" >&2
                return 0
            fi
            sleep 0.05
        done
        echo "Producer finished all 20 lines (target not found)" >&2
        return 1
    }
    
    producer_with_early_exit "line_5" >/dev/null 2>&1
    status=$?
    echo "Exit status: $status"
    echo "☝️  No broken pipe, producer stopped efficiently at iteration 5"
    echo
}

# Test 4: Tee with downstream piping capability
test_tee_with_downstream() {
    echo "--- Test 4: Tee with Downstream Output - PRACTICAL SOLUTION ---"
    echo "Command: Uses tee but produces output that can be piped downstream"
    echo "Expected: No broken pipe + output suitable for further processing"
    echo
    
    ancestor_check_with_tee() {
        local start="$1"
        local target="$2"
        local tmpflag="/tmp/ancestor_found_$$"
        
        # Use slow_producer to simulate ancestor_all
        slow_producer "$target" 2>/dev/null | tee >(grep -q "$target" && touch "$tmpflag") >/dev/null
        
        if [[ -f "$tmpflag" ]]; then
            echo "$start:$target"  # This can be piped downstream!
            rm -f "$tmpflag"
            return 0
        else
            rm -f "$tmpflag"
            return 1
        fi
    }
    
    # Test the function
    result=$(ancestor_check_with_tee "a" "line_5")
    status=$?
    echo "Output: '$result'"
    echo "Exit status: $status"
    echo "☝️  This output can be piped to other commands safely"
    
    # Demonstrate piping capability
    echo "Piping test:"
    ancestor_check_with_tee "a" "line_5" | sed 's/line_5/FOUND_IT/' | tr ':' '→'
    echo
}

# Run all tests with timing
echo "🧪 Starting tests (this will take ~4 seconds)..."
echo

time test_baseline
time test_tee  
time test_early_exit
test_tee_with_downstream

echo "=== Performance Summary ==="
echo "📊 Iterations performed:"
echo "  Baseline:   ~5 (broken pipe interrupts)"
echo "  Tee:        20 (inefficient but safe)"
echo "  Early exit: 5 (optimal)"
echo
echo "🚀 For UnifyWeaver:"
echo "  - Tee approach: Safe for Gemini CLI, prevents broken pipe"
echo "  - Early exit:   More efficient, requires template changes"
echo "  - Recommendation: Start with tee, optimize with early exit later"
