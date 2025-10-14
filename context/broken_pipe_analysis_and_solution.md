# Broken Pipe Analysis and Solution

**Date:** October 14, 2025  
**Issue:** SIGPIPE warnings in UnifyWeaver tests affecting Gemini CLI streaming  
**Status:** ✅ SOLUTION IDENTIFIED - Tee Approach Ready for Implementation  

## Problem Summary

### Root Cause
The broken pipe warnings occur in generated bash scripts when:
1. `ancestor_all()` performs BFS traversal, writing results via `echo`
2. Output is piped to `grep -q` which exits immediately after finding first match
3. `ancestor_all()` continues running and tries to `echo` more results
4. Pipe is closed → **SIGPIPE** → "Broken pipe" error message

### Location
- **File:** `output/core_tests/ancestor.sh` (line 36)
- **Function:** `ancestor_check()` calls `ancestor_all "$start" | grep -q "pattern"`
- **Tests:** Called from `tests/core/test_compiler_driver.pl` generated test runner

### Why This Matters
User's concern: **SIGPIPE errors could interfere with Gemini CLI streaming**
- Gemini CLI uses streaming responses
- Broken pipe errors sent to stderr could corrupt data streams
- Need clean, signal-free solution

## Solution Analysis

### Approaches Evaluated

| Approach | Broken Pipe | Efficiency | Complexity | Stream Safe |
|----------|-------------|------------|------------|-------------|
| **Suppress stderr** | Hidden | ⚡ Fast | ✅ Simple | ❌ Hides real errors |
| **Temp file** | ✅ None | 🐌 Slow | ⚠️ Medium | ✅ Safe |
| **trap '' PIPE** | ⚠️ Ignored | ⚡ Fast | ✅ Simple | ⚠️ Signal still occurs |
| **tee approach** | ✅ None | 🐌 Full computation | ⚠️ Medium | ✅ Safe |
| **Early exit** | ✅ None | ⚡ Optimal | ⚠️ Template changes | ✅ Safe |

### Recommended Solution: Tee Approach

**Why tee approach wins:**
- ✅ **No SIGPIPE** - keeps one output path open to prevent pipe break
- ✅ **Stream safe** - no error signals that could affect Gemini CLI
- ✅ **Pipeable output** - can still pipe results downstream
- ✅ **Minimal template changes** - doesn't require algorithmic redesign
- ⚠️ **Efficiency trade-off** - computes full graph (can optimize later)

## Implementation

### Current Problematic Code
```bash
ancestor_check() {
    local start="$1" 
    local target="$2"
    ancestor_all "$start" | grep -q "^$start:$target$"  # ← SIGPIPE here
}
```

### Proposed Tee Solution
```bash
ancestor_check() {
    local start="$1"
    local target="$2"
    local tmpflag="/tmp/ancestor_found_$$"
    
    # Tee prevents broken pipe
    ancestor_all "$start" | tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null
    
    if [[ -f "$tmpflag" ]]; then
        echo "$start:$target"  # Can be piped downstream
        rm -f "$tmpflag"
        return 0
    else
        rm -f "$tmpflag"
        return 1
    fi
}
```

### How Tee Works
```
ancestor_all → pipe → tee → stdout → /dev/null
                       ↓
                   >(grep -q)
```

1. `ancestor_all` writes to tee
2. `tee` duplicates to two paths:
   - Process substitution `>(grep -q)` - exits when found
   - stdout to `/dev/null` - stays open
3. When grep exits, main pipe stays open → **no SIGPIPE**
4. Status captured via temp flag file
5. Clean output can be piped downstream

## Test Validation

### Test Script Created
**Location:** `context/test_broken_pipe_solutions.sh`  
**Status:** ✅ Ready to run

**Test Cases:**
1. **Baseline** - Demonstrates broken pipe (current behavior)
2. **Tee approach** - Shows no broken pipe with full computation
3. **Early exit** - Shows optimal efficiency (for comparison)
4. **Tee with downstream** - Proves piping capability works

**To validate:** Run `./context/test_broken_pipe_solutions.sh`

### Expected Results
- Test 1: Shows broken pipe error (current problem)
- Test 2: No errors, completes all iterations (safe but slower)
- Test 3: No errors, stops early (optimal comparison)
- Test 4: Demonstrates downstream piping works

## Implementation Plan

### Phase 1: Apply Tee Solution ⭐ (Recommended Immediate)
**Target:** Template that generates `ancestor_check()` function  
**Location:** Likely in `src/unifyweaver/core/advanced/` modules  
**Change:** Replace direct pipe to grep with tee + flag approach  
**Benefits:** Immediate fix, safe for Gemini CLI, minimal disruption  

### Phase 2: Optimize with Early Exit (Future)
**Target:** BFS algorithm in transitive closure templates  
**Change:** Add target parameter to `ancestor_all`, exit when found  
**Benefits:** Better performance, cleaner semantics  
**Timing:** When performance becomes critical  

## Separate Issue: Firewall Handling

**Different Problem:** Tests don't handle firewall violations gracefully
- If firewall blocks bash backend → compilation fails
- Tests still try to execute non-existent scripts
- Need better test flow: check compilation success before execution

**Recommendation:** 
1. Fix broken pipe first (main issue)
2. Address firewall test handling separately
3. Bash should probably be "always allowed" until alternatives exist

## Branch Status Clarification

**Current Work:**
- **feature/fold-pattern-integration** - Cline.bot's completed fold pattern work
- Gemini's broken pipe fix (stderr suppression) was also on this branch
- My analysis and tee solution are new additions

**Recommendation:**
1. Commit current analysis and test script
2. Implement tee solution 
3. Replace Gemini's stderr suppression with proper tee fix

## Files Modified

1. **Created:** `context/test_broken_pipe_solutions.sh` - Validation test
2. **Created:** `context/broken_pipe_analysis_and_solution.md` - This document
3. **To modify:** Template file that generates ancestor_check (TBD - need to locate)

## Efficiency Analysis: EOF Signaling

**User's Question:** Can we use EOF signaling downstream to improve efficiency?

**Answer:** ❌ **No** - EOF signaling doesn't solve the efficiency problem.

**Test Script:** `context/test_eof_signaling_approach.sh` demonstrates why.

### Timeline Issue
```
1. ancestor_all starts BFS (expensive computation)
2. tee splits output to grep and /dev/null
3. grep finds match at iteration 5, exits
4. /dev/null keeps consuming (prevents SIGPIPE) 
5. ancestor_all continues to iteration 20 ← STILL INEFFICIENT
6. THEN we check flag and produce output
7. EOF sent downstream ← Too late! Work already done
```

**Key Insight:** EOF only affects downstream consumers, not the producer. The expensive BFS computation still runs to completion.

### Efficiency Solutions Ranked

| Approach | Prevents SIGPIPE | Efficiency | Complexity |
|----------|------------------|------------|------------|
| **tee + EOF** | ✅ Yes | 🐌 Still full computation | ⚠️ Medium |
| **tee only** | ✅ Yes | 🐌 Full computation | ⚠️ Medium |  
| **Early exit** | ✅ Yes | ⚡ Stops when found | ⚠️ Medium |
| **Active termination** | ✅ Yes | ⚡ Kill producer early | ❌ Complex |
| **timeout command** | ✅ Yes | ⚡ Bounded execution | ✅ Simple |

## Recommended Approach

**Phase 1: Tee Solution (Immediate)** ⭐
- Solves SIGPIPE problem for Gemini CLI
- Safe and reliable
- Accept efficiency trade-off for now

**Phase 2: Early Exit Optimization (Future)**
- Modify `ancestor_all` to accept target parameter
- Exit when target found
- True efficiency gain

## Advanced Analysis: Active Termination

**User's Follow-up:** Can we reduce buffering and actively kill the producer when target found?

**Answer:** ✅ **Yes** - This is a sophisticated approach that could achieve both efficiency and safety!

**Test Script:** `context/test_active_termination_approach.sh` explores this concept.

### Active Termination Concept
```
1. Start unbuffered_producer in background
2. Monitor output with grep in separate process  
3. When grep succeeds → kill producer immediately
4. Send EOF downstream → inform consumers to stop
5. Clean termination, no SIGPIPE, early exit achieved!
```

### Promising Variations

**Option A: Process Killing**
- Track producer PID, kill when target found
- Complex but potentially very efficient

**Option B: timeout Command** ⭐ (Simplest)
```bash
timeout 1s ancestor_all "$start" | tee >(grep -q "$target" && touch flag) >/dev/null
```
- Bounded execution time
- Simple implementation
- Good balance of safety and efficiency

**Option C: Process Groups**
- Clean termination of all related processes
- Better cleanup on failure

### Updated Efficiency Ranking

| Approach | SIGPIPE | Efficiency | Complexity | Recommended |
|----------|---------|------------|------------|-------------|
| **timeout + tee** | ✅ None | ⚡ Bounded | ✅ Simple | ⭐ **Yes** |
| **Early exit** | ✅ None | ⚡ Optimal | ⚠️ Medium | ✅ Future |
| **Active termination** | ✅ None | ⚡ Very good | ❌ Complex | ⚠️ Advanced |
| **tee only** | ✅ None | 🐌 Full compute | ⚠️ Medium | ✅ Fallback |

## Next Steps

1. **Test all approaches:** Run three test scripts to compare
   - `./context/test_broken_pipe_solutions.sh` (original analysis)
   - `./context/test_eof_signaling_approach.sh` (EOF signaling)  
   - `./context/test_active_termination_approach.sh` (active termination)

2. **Consider timeout approach:** Simple and effective hybrid solution

3. **Implementation priority:**
   - **Immediate:** timeout + tee (efficient + safe)
   - **Future:** Early exit in ancestor_all (optimal)
   - **Advanced:** Full active termination (if needed)

**Priority:** High - affects Gemini CLI reliability  
**Effort:** Medium - template changes required  
**Risk:** Low - tee approach is well-established Unix pattern  
**Efficiency:** Accept trade-off now, optimize later
