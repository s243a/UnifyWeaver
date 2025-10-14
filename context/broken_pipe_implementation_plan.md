# Broken Pipe Solution Implementation Plan

**Date:** October 14, 2025  
**Branch Strategy:** New branch `feature/fix-broken-pipe-signals`  
**Priority:** High - Affects Gemini CLI streaming reliability  

## Executive Summary

Implement timeout + tee approach to eliminate SIGPIPE warnings that could interfere with Gemini CLI streaming. This provides a good balance of safety, efficiency, and implementation simplicity.

## Target Files for Modification

### Primary Target: Transitive Closure Templates
**Location:** `src/unifyweaver/core/advanced/`

The broken pipe occurs in generated `ancestor_check()` functions. Need to locate the template/code generation that produces:

```bash
ancestor_check() {
    ancestor_all "$start" | grep -q "^$start:$target$"  # ← Causes SIGPIPE
}
```

**Likely files to examine:**
- `advanced_recursive_compiler.pl` - Main dispatcher
- Files that handle transitive closure compilation
- Template system files that generate bash code

### Secondary: Test Framework
**Location:** `tests/core/test_compiler_driver.pl`

Currently generates test calls like:
```bash
ancestor a c >/dev/null  # ← Where SIGPIPE manifests
```

## Implementation Phases

### Phase 1: Timeout + Tee Solution ⭐ (Immediate)

**Goal:** Safe, bounded-time execution without SIGPIPE

**Implementation:**
```bash
ancestor_check() {
    local start="$1"
    local target="$2"
    local tmpflag="/tmp/ancestor_found_$$"
    local timeout_duration="5s"  # Configurable
    
    # Timeout prevents infinite execution, tee prevents SIGPIPE
    timeout "$timeout_duration" ancestor_all "$start" | 
    tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null
    
    if [[ -f "$tmpflag" ]]; then
        echo "$start:$target"
        rm -f "$tmpflag"
        return 0
    else
        rm -f "$tmpflag"
        return 1
    fi
}
```

**Benefits:**
- ✅ No SIGPIPE (safe for Gemini CLI streaming)
- ✅ Bounded execution time (no infinite loops)
- ✅ Simple implementation
- ✅ Can pipe output downstream
- ⚠️ Still computes more than optimal (acceptable trade-off)

### Phase 2: Early Exit Optimization (Future)

**Goal:** Optimal efficiency by stopping BFS when target found

**Implementation:** Modify `ancestor_all()` to accept target parameter:
```bash
ancestor_all() {
    local start="$1"
    local target="$2"  # Optional - exit early if provided
    
    # ... BFS logic ...
    
    while [[ -s "$queue_file" ]]; do
        # ... process queue ...
        echo "$start:$to"
        
        # Early exit when target found
        if [[ -n "$target" && "$to" == "$target" ]]; then
            cleanup_and_exit 0
        fi
    done
}
```

**Benefits:**
- ✅ Optimal efficiency (stops immediately when found)
- ✅ No SIGPIPE
- ⚠️ Requires more extensive template changes

## Detailed Implementation Steps

### Step 1: Locate Code Generation ⏱️ (30 minutes)

1. **Find template source:**
   ```bash
   grep -r "ancestor_check" src/
   grep -r "grep -q" src/
   grep -r "ancestor_all.*|" src/
   ```

2. **Identify generation mechanism:**
   - Template files
   - Code generation predicates
   - Variable substitution patterns

3. **Trace call path:**
   - How does `compile_recursive` create transitive closure?
   - Which module handles `ancestor/2` pattern?

### Step 2: Implement Timeout + Tee Solution ⏱️ (2 hours)

1. **Modify template/generator:**
   - Replace direct pipe to grep with timeout + tee pattern
   - Add temporary flag file logic
   - Ensure proper cleanup

2. **Add configuration:**
   - Timeout duration (default: 5s)
   - Flag file prefix (default: `/tmp/`)
   - Make configurable via options

3. **Update tests:**
   - Verify no SIGPIPE warnings
   - Confirm functionality intact
   - Test with various timeout values

### Step 3: Integration Testing ⏱️ (1 hour)

1. **Run existing test suite:**
   ```bash
   swipl -s run_all_tests.pl -g "main, halt."
   ```

2. **Verify no broken pipe warnings:**
   - Check stderr output
   - Confirm all tests pass
   - Test with different ancestor relationships

3. **Performance benchmarking:**
   - Compare timing with old approach
   - Verify timeout bounds are respected

### Step 4: Documentation and Cleanup ⏱️ (30 minutes)

1. **Update code comments:**
   - Explain timeout + tee approach
   - Document configuration options
   - Add performance characteristics

2. **Clean up Gemini's changes:**
   - Remove `2>/dev/null` stderr suppression
   - Replace with proper solution

## Configuration Options

### Template Variables
```prolog
% New options for transitive closure compilation
compile_transitive_closure(Pred/Arity, Options, BashCode) :-
    % Extract timeout setting
    (member(timeout(Duration), Options) -> 
        TimeoutOpt = Duration 
    ; TimeoutOpt = '5s'),
    
    % Extract temp directory
    (member(temp_dir(TempDir), Options) -> 
        TempOpt = TempDir 
    ; TempOpt = '/tmp'),
    
    % Generate code with timeout + tee
    generate_timeout_tee_code(Pred/Arity, TimeoutOpt, TempOpt, BashCode).
```

### Runtime Options
```prolog
% Usage examples
compile_recursive(ancestor/2, [timeout('10s')], Code).
compile_recursive(ancestor/2, [temp_dir('/var/tmp')], Code).
```

## Testing Strategy

### Unit Tests
```bash
# Test timeout functionality
./test_timeout_bounds
