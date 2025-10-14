#!/bin/bash
# Test script: EOF Signaling Approach for Early Termination
# Testing if we can signal downstream consumers to stop reading early

echo "=== Testing EOF Signaling for Efficiency ==="
echo

# Simulate the slow producer again
slow_producer() {
    local target="$1"
    echo "Producer started..." >&2
    for i in {1..20}; do
        echo "line_$i"
        sleep 0.05
        if [[ "$i" -eq 5 && "$target" == "line_5" ]]; then
            echo "Producer: Found target at iteration $i!" >&2
        fi
    done
    echo "Producer finished all 20 lines" >&2
}

# Test 1: Current tee approach (baseline for comparison)
test_tee_baseline() {
    echo "--- Test 1: Current Tee Approach (Full Computation) ---"
    tmpflag="/tmp/test_tee_$$"
    
    time (slow_producer "line_5" 2>&1 | tee >(grep -q "line_5" && touch "$tmpflag") >/dev/null)
    
    if [[ -f "$tmpflag" ]]; then
        echo "✅ Found: line_5"
        rm -f "$tmpflag"
    fi
    echo "☝️  Producer ran all 20 iterations (inefficient)"
    echo
}

# Test 2: EOF signaling approach
test_eof_signaling() {
    echo "--- Test 2: EOF Signaling Approach ---"
    tmpflag="/tmp/test_eof_$$"
    
    ancestor_check_with_eof() {
        local target="$1"
        
        # Use tee to prevent broken pipe
        slow_producer "$target" 2>/dev/null | tee >(grep -q "$target" && touch "$tmpflag") >/dev/null
        
        if [[ -f "$tmpflag" ]]; then
            echo "FOUND:$target"
            echo -e '\004'  # EOF character (Ctrl-D, ASCII 4)
            rm -f "$tmpflag"
            return 0
        else
            echo "NOT_FOUND"
            rm -f "$tmpflag" 
            return 1
        fi
    }
    
    # Test with downstream consumer that stops on EOF
    downstream_consumer() {
        while IFS= read -r line; do
            if [[ "$line" == $'\004' ]]; then
                echo "Downstream: Received EOF, stopping"
                break
            fi
            echo "Downstream processed: $line"
        done
    }
    
    time (ancestor_check_with_eof "line_5" | downstream_consumer)
    echo "☝️  Still runs full producer (EOF doesn't help efficiency)"
    echo
}

# Test 3: What if we could signal the producer?
test_producer_signaling() {
    echo "--- Test 3: Producer Signaling (Hypothetical) ---"
    echo "This would require complex process management..."
    
    # This is what we'd WANT but it's complex to implement reliably
    producer_with_signaling() {
        local target="$1"
        local signal_file="/tmp/stop_signal_$$"
        
        echo "Producer started..." >&2
        for i in {1..20}; do
            # Check if we should stop
            if [[ -f "$signal_file" ]]; then
                echo "Producer: Received stop signal at iteration $i!" >&2
                rm -f "$signal_file"
                return 0
            fi
            
            echo "line_$i"
            sleep 0.05
        done
        echo "Producer finished all 20 lines" >&2
    }
    
    echo "This approach would need:"
    echo "  - Background processes"  
    echo "  - Signal files or pipes"
    echo "  - Race condition handling"
    echo "  - Cleanup on failure"
    echo "☝️  Complex but could be efficient"
    echo
}

# Test 4: The real issue - timing
test_timing_issue() {
    echo "--- Test 4: The Timing Issue ---"
    echo "The core problem: tee approach doesn't save time because:"
    echo
    echo "Timeline:"
    echo "  1. ancestor_all starts BFS"
    echo "  2. tee splits output to grep and /dev/null" 
    echo "  3. grep finds match at iteration 5, exits"
    echo "  4. /dev/null keeps consuming (prevents SIGPIPE)"
    echo "  5. ancestor_all continues to iteration 20"
    echo "  6. THEN we check flag and produce output"
    echo
    echo "The EOF after step 6 doesn't help - work is already done!"
    echo
}

# Run tests
test_tee_baseline
test_eof_signaling  
test_producer_signaling
test_timing_issue

echo "=== Analysis ==="
echo "💡 EOF signaling doesn't solve efficiency because:"
echo "   - Producer has already completed when we send EOF"
echo "   - Computational work is already done"  
echo "   - EOF only affects downstream consumers, not producers"
echo
echo "🎯 For true efficiency, we need:"
echo "   - Producer to stop itself (early exit in ancestor_all)"
echo "   - OR producer/consumer communication (complex)"
echo
echo "🚀 Recommendation:"
echo "   - Phase 1: Implement tee approach (safe, prevents SIGPIPE)"
echo "   - Phase 2: Add early exit to ancestor_all (efficient)"
echo "   - EOF signaling: Not helpful for this use case"
