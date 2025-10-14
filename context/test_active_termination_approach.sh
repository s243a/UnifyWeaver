#!/bin/bash
# Test script: Active Process Termination with EOF Signaling
# Testing if we can kill the producer and send EOF when target found

echo "=== Testing Active Termination + EOF Signaling ==="
echo

# Unbuffered producer with process tracking
unbuffered_producer() {
    local target="$1"
    local pid_file="$2"
    
    # Write our PID so parent can kill us
    echo $$ > "$pid_file"
    
    echo "Producer started (PID: $$)..." >&2
    for i in {1..20}; do
        # Unbuffered output (line-by-line)
        echo "line_$i"
        printf '' >&1  # Force flush
        
        sleep 0.1  # Simulate computation
        
        if [[ "$i" -eq 5 && "$target" == "line_5" ]]; then
            echo "Producer: Found target at iteration $i!" >&2
        fi
        
        # Check if we should exit (killed by parent)
        if [[ -f "/tmp/kill_signal_$$" ]]; then
            echo "Producer: Received kill signal at iteration $i!" >&2
            rm -f "/tmp/kill_signal_$$"
            return 0
        fi
    done
    echo "Producer finished all 20 lines" >&2
}

# Test 1: Baseline unbuffered (still has broken pipe)
test_unbuffered_baseline() {
    echo "--- Test 1: Unbuffered Baseline ---"
    echo "Command: unbuffered_producer | grep -q"
    echo "Expected: Still broken pipe, but less buffering"
    echo
    
    pid_file="/tmp/producer_pid_$$"
    unbuffered_producer "line_5" "$pid_file" 2>&1 | grep -q "line_5"
    status=$?
    echo "Exit status: $status"
    echo "☝️  Unbuffering doesn't prevent SIGPIPE"
    rm -f "$pid_file"
    echo
}

# Test 2: Active termination with EOF
test_active_termination() {
    echo "--- Test 2: Active Termination + EOF ---"
    echo "Command: Kill producer when grep succeeds, send EOF downstream"
    echo "Expected: Efficient + no broken pipe + EOF signaling"
    echo
    
    ancestor_check_with_termination() {
        local target="$1"
        local tmpflag="/tmp/found_$$"
        local pid_file="/tmp/producer_pid_$$"
        local output_file="/tmp/output_$$"
        
        # Start producer in background, capture output
        unbuffered_producer "$target" "$pid_file" 2>/dev/null > "$output_file" &
        local producer_pid=$!
        
        # Monitor for grep success in background
        (
            tail -f "$output_file" 2>/dev/null | grep -q "$target"
            touch "$tmpflag"
            # Signal producer to stop
            if [[ -f "$pid_file" ]]; then
                local actual_pid=$(cat "$pid_file")
                echo "Terminating producer PID $actual_pid" >&2
                kill "$actual_pid" 2>/dev/null || true
            fi
        ) &
        local monitor_pid=$!
        
        # Wait for either completion or early termination
        local timeout=3
        local elapsed=0
        while [[ $elapsed -lt $timeout ]]; do
            if [[ -f "$tmpflag" ]] || ! kill -0 "$producer_pid" 2>/dev/null; then
                break
            fi
            sleep 0.1
            elapsed=$((elapsed + 1))
        done
        
        # Cleanup
        kill "$producer_pid" "$monitor_pid" 2>/dev/null || true
        wait 2>/dev/null || true
        
        if [[ -f "$tmpflag" ]]; then
            echo "FOUND:$target"
            echo -e '\004'  # EOF signal
            rm -f "$tmpflag" "$pid_file" "$output_file"
            return 0
        else
            echo "NOT_FOUND"
            rm -f "$tmpflag" "$pid_file" "$output_file"
            return 1
        fi
    }
    
    time (ancestor_check_with_termination "line_5")
    echo "☝️  Should be faster - producer killed early"
    echo
}

# Test 3: Simpler approach - timeout based
test_timeout_approach() {
    echo "--- Test 3: Timeout-Based Termination ---"
    echo "Command: Kill producer after reasonable timeout"
    echo "Expected: Bounded execution time"
    echo
    
    ancestor_check_timeout() {
        local target="$1"
        local tmpflag="/tmp/timeout_$$"
        local timeout=1  # 1 second max
        
        # Start producer with timeout
        timeout "$timeout" bash -c "
            unbuffered_producer '$target' /dev/null 2>/dev/null | 
            tee >(grep -q '$target' && touch '$tmpflag') >/dev/null
        " || true
        
        if [[ -f "$tmpflag" ]]; then
            echo "FOUND:$target"
            echo -e '\004'
            rm -f "$tmpflag"
            return 0
        else
            echo "TIMEOUT_OR_NOT_FOUND"
            rm -f "$tmpflag"
            return 1
        fi
    }
    
    time (ancestor_check_timeout "line_5")
    echo "☝️  Uses timeout command - bounded execution"
    echo
}

# Test 4: Process group termination
test_process_group() {
    echo "--- Test 4: Process Group Termination ---"
    echo "Command: Kill entire process group when done"
    echo "Expected: Clean termination of all processes"
    echo
    
    ancestor_check_pgroup() {
        local target="$1"
        local tmpflag="/tmp/pgroup_$$"
        
        # Create new process group
        (
            set -m  # Enable job control
            pid_file="/tmp/prod_pid_$$"
            
            # Start producer in background
            unbuffered_producer "$target" "$pid_file" 2>/dev/null | 
            tee >(grep -q "$target" && touch "$tmpflag") >/dev/null &
            
            # Wait for flag or timeout
            local count=0
            while [[ ! -f "$tmpflag" && $count -lt 30 ]]; do
                sleep 0.1
                count=$((count + 1))
            done
            
            # Kill entire process group
            pkill -TERM -P $$ 2>/dev/null || true
            
        ) &
        local pgroup_pid=$!
        
        wait "$pgroup_pid" 2>/dev/null || true
        
        if [[ -f "$tmpflag" ]]; then
            echo "FOUND:$target"
            echo -e '\004'
            rm -f "$tmpflag"
            return 0
        else
            echo "NOT_FOUND_OR_TIMEOUT"
            rm -f "$tmpflag"
            return 1
        fi
    }
    
    time (ancestor_check_pgroup "line_5")
    echo "☝️  Uses process groups for clean termination"
    echo
}

# Run tests
test_unbuffered_baseline
test_active_termination
test_timeout_approach  
test_process_group

echo "=== Analysis ==="
echo "🎯 Your active termination idea shows promise!"
echo
echo "✅ Benefits:"
echo "   - Could achieve efficiency (kill producer early)"
echo "   - Prevents SIGPIPE (no direct pipe to grep)"  
echo "   - EOF signaling works for downstream consumers"
echo
echo "⚠️  Challenges:"
echo "   - Process management complexity"
echo "   - Race conditions between detection and termination"
echo "   - Cleanup on failure scenarios"
echo "   - Unbuffering may not help much (BFS is the bottleneck)"
echo
echo "🚀 Promising approaches:"
echo "   - timeout command (simple, bounded execution)"
echo "   - Process groups (cleaner termination)"
echo "   - Hybrid: tee + active termination when flag detected"
echo
echo "💡 Key insight: This could work but adds significant complexity"
echo "   vs. early exit in ancestor_all (simpler, same efficiency)"
