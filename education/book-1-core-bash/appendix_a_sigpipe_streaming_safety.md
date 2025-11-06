<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Appendix A: SIGPIPE and Streaming Safety

**Educational Topic:** Unix pipes, streaming safety, and production reliability  
**Real-World Problem:** How broken pipes can corrupt data streams in CLI applications  
**Case Study:** UnifyWeaver's solution for Gemini CLI streaming compatibility  

---

## Introduction

This appendix explores a real production issue encountered in UnifyWeaver: SIGPIPE warnings that could interfere with Gemini CLI streaming responses. This case study demonstrates Unix systems programming concepts, the subtleties of pipe handling, and engineering approaches to streaming safety.

---

## The Problem: What is SIGPIPE?

### Background: Unix Pipes and Signals

In Unix systems, pipes connect the output of one process to the input of another:

```bash
producer | consumer
```

**Normal flow:**
1. `producer` writes data to the pipe
2. `consumer` reads data from the pipe  
3. Both processes coordinate through the pipe

**Broken pipe scenario:**
1. `producer` is generating data
2. `consumer` exits early (e.g., `grep -q` finds first match)
3. `producer` tries to write to a closed pipe
4. Kernel sends SIGPIPE signal to `producer`
5. By default, SIGPIPE terminates the producer process

### SIGPIPE in Generated Scripts

UnifyWeaver generates bash scripts for recursive predicates like:

```bash
# Find if 'target' is reachable from 'start'
ancestor_check() {
    local start="$1"
    local target="$2"
    
    # This pattern causes SIGPIPE:
    ancestor_all "$start" | grep -q "^$start:$target$"
    #                     ↑
    # grep -q exits after first match
    # ancestor_all keeps running and hits broken pipe
}
```

**Timeline of the problem:**
1. `ancestor_all` starts BFS traversal (expensive operation)
2. Output flows through pipe to `grep -q`
3. `grep -q` finds match on iteration 3, **exits immediately**
4. `ancestor_all` continues to iteration 20, tries to output result
5. **SIGPIPE** - pipe is broken, error message goes to stderr
6. Error message: `bash: line 1: echo: write error: Broken pipe`

### Why This Matters for CLI Streaming

**The Core Issue:** Error messages to stderr can corrupt streaming data in CLI applications.

**Gemini CLI Context:**
- Gemini CLI uses streaming responses for real-time interaction
- SIGPIPE errors appear in stderr mixed with normal output
- Streaming parsers expect clean data, not error messages
- Broken pipe warnings could cause parsing failures or data corruption

**General CLI Streaming:**
- Many modern CLI tools use streaming JSON, structured output
- Error messages in stderr break structured data assumptions
- Production scripts need "signal-quiet" operation
- Unpredictable errors reduce CLI tool reliability

---

## Analysis: Solution Approaches Evaluated

### Approach 1: Stderr Suppression (❌ Rejected)

**Implementation:**
```bash
ancestor_all "$start" | grep -q "^$start:$target$" 2>/dev/null
```

**Why rejected:**
- Hides real errors, not just SIGPIPE warnings
- Poor engineering practice - masking symptoms
- Debug information lost when actual problems occur
- Doesn't solve root cause (signals still occur)

### Approach 2: Signal Trapping (⚠️ Limited)

**Implementation:**
```bash
trap '' PIPE  # Ignore SIGPIPE
ancestor_all "$start" | grep -q "^$start:$target$"
trap - PIPE   # Restore default handling
```

**Limitations:**
- Signals still occur, just ignored
- Process still receives signal (overhead)
- Can mask legitimate pipe errors
- Complex to implement correctly in templates

### Approach 3: Temporary Files (✅ Safe but Slower)

**Implementation:**
```bash
ancestor_check() {
    local temp_file="/tmp/ancestor_$$"
    ancestor_all "$start" > "$temp_file"
    grep -q "^$start:$target$" "$temp_file"
    local result=$?
    rm -f "$temp_file"
    return $result
}
```

**Trade-offs:**
- ✅ No SIGPIPE (no pipes involved)
- ✅ Safe and reliable
- ❌ Slower (full computation + disk I/O)
- ❌ Doesn't scale to large datasets
- ⚠️ Temporary file management complexity

### Approach 4: Timeout + Tee Solution (⭐ Chosen)

**Implementation:**
```bash
ancestor_check() {
    local start="$1"
    local target="$2"
    local tmpflag="/tmp/ancestor_found_$$"
    local timeout_duration="5s"
    
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

**How it works:**
1. **`timeout "5s"`** - Bounds execution time (safety feature)
2. **`tee`** - Duplicates output to two destinations:
   - `>(grep -q ...)` - Process substitution that exits when match found
   - `>/dev/null` - **Stays open**, preventing SIGPIPE
3. **`tmpflag`** - Communicates grep result via filesystem
4. **Clean output** - Can still be piped downstream if needed

**Why this approach wins:**
- ✅ **No SIGPIPE** - main pipe stays open via `/dev/null`
- ✅ **Bounded execution** - timeout prevents infinite loops
- ✅ **Simple implementation** - uses standard Unix tools
- ✅ **Proven pattern** - `tee` is well-established for this purpose
- ✅ **Production safe** - handles edge cases properly
- ⚠️ **Efficiency trade-off** - still computes full result (acceptable)

---

## Implementation: Engineering Considerations

### Template System Integration

UnifyWeaver uses a template system for code generation. The SIGPIPE fix required updating templates in two locations:

**Files Modified:**
1. `src/unifyweaver/core/template_system.pl` - Main transitive closure template
2. `src/unifyweaver/core/recursive_compiler.pl` - Descendant-specific template

**Template Variable Design:**
```bash
# Template with placeholders
{{pred}}_check() {
    local tmpflag="/tmp/{{pred}}_found_$$"
    timeout "{{timeout_duration}}" {{pred}}_all "$start" | 
    tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null
    # ... rest of implementation
}
```

**Configuration Options:**
- `timeout_duration` - Configurable timeout (default: "5s")  
- `tmpflag` pattern - Process-specific temporary files
- Error handling and cleanup logic

### Process Substitution Details

**Process Substitution Syntax:**
```bash
tee >(grep -q "pattern" && touch flag) >/dev/null
```

**What happens:**
1. `>(...)` creates a temporary FIFO (named pipe)
2. `tee` writes to both the FIFO and stdout
3. `grep -q` reads from FIFO, exits when match found
4. FIFO closes, but main stdout (to `/dev/null`) stays open
5. No SIGPIPE because producer still has open output stream

**Bash Version Compatibility:**
- Process substitution requires bash 2.0+ (available on all modern systems)
- Alternative implementations possible for older shells if needed

### Testing and Validation

**Test Strategy:**
```bash
# Before fix - check for SIGPIPE warnings
swipl -s run_all_tests.pl -g "main, halt." 2>&1 | grep -i "broken pipe"

# After fix - should find none
swipl -s run_all_tests.pl -g "main, halt." 2>&1 | grep -i "broken pipe" || echo "No warnings found!"
```

**Results:**
- **Before:** Multiple "Broken pipe" warnings in test output
- **After:** Zero SIGPIPE warnings, all tests pass
- **Performance:** Acceptable overhead for safety improvement
- **Compatibility:** No functional changes for end users

---

## Future Optimizations: Early Exit Patterns

The timeout + tee solution fixes the streaming safety issue but still computes full results. Future optimizations could improve efficiency:

### Phase 2: Algorithm-Level Early Exit

**Concept:** Modify the BFS algorithm itself to stop when target found.

**Current:**
```bash
ancestor_all() {
    # BFS that finds ALL reachable nodes
    while [[ -s "$queue_file" ]]; do
        # ... traverse all nodes
        echo "$start:$node"  # Output every relationship
    done
}
```

**Optimized:**
```bash
ancestor_all() {
    local target="$2"  # Optional target parameter
    while [[ -s "$queue_file" ]]; do
        # ... BFS logic
        echo "$start:$node"
        
        # Early exit when specific target found
        if [[ -n "$target" && "$node" == "$target" ]]; then
            cleanup_and_exit 0
        fi
    done
}
```

**Benefits:**
- ✅ **True efficiency** - stops immediately when target found
- ✅ **No SIGPIPE** - no premature pipe closure
- ⚠️ **More complex** - requires algorithm modifications

### Phase 3: Active Termination

**Advanced Concept:** Kill producer process when consumer has enough data.

**Implementation complexity:**
- Track producer process IDs
- Signal handling between processes  
- Race condition management
- Error recovery for partial results

**Use cases:** High-frequency streaming scenarios where every microsecond matters.

---

## Educational Exercises

### Exercise 1: Reproduce the Problem

**Setup:**
```bash
# Create a simple producer that outputs many lines
#!/bin/bash
producer() {
    for i in {1..100}; do
        echo "line_$i"
        sleep 0.01  # Small delay to show the issue
    done
}

# This will cause SIGPIPE
producer | head -5
```

**Observe:** The "Broken pipe" error message when producer tries to write after head exits.

### Exercise 2: Test Solutions

**Stderr suppression (poor solution):**
```bash
producer | head -5 2>/dev/null
```

**Tee solution (good solution):**
```bash
producer | tee >(head -5 > results.txt) >/dev/null
```

**Compare:** Which approach is more maintainable and debuggable?

### Exercise 3: Streaming JSON Corruption

**Problem simulation:**
```bash
# Simulate streaming JSON with errors
json_stream() {
    echo '{"data": ['
    for i in {1..10}; do
        echo "  {\"id\": $i},"
    done
    echo ']}'
}

# This corrupts the JSON stream with error messages
json_stream | head -3
```

**Solution:** Apply the tee pattern to keep JSON clean.

---

## Real-World Applications

### CLI Tool Design Principles

1. **Separate error channels** - Never mix errors with data output
2. **Handle signals gracefully** - Don't let SIGPIPE crash your tools  
3. **Test with pipes** - Many tools will be used in pipelines
4. **Document streaming behavior** - Users need to know what to expect

### Streaming API Best Practices

**Good streaming output:**
```bash
# Clean, parseable output
api_tool --format=json | jq '.results[]'
api_tool --format=csv | cut -d, -f1,3
```

**Bad streaming output:**
```bash
# Mixed errors and data - breaks automation
api_tool 2>&1 | parser  # stderr mixed with stdout
```

### Production Monitoring

**Monitor for SIGPIPE in production:**
```bash
# Log analysis for broken pipes
grep -i "broken pipe" /var/log/app.log

# System-wide signal monitoring
dmesg | grep -i "killed by signal"
```

---

## Summary

The SIGPIPE issue in UnifyWeaver demonstrates several important concepts:

### Unix Systems Programming
- **Pipe mechanics** - How processes communicate via pipes
- **Signal handling** - How the kernel manages process communication failures
- **Process substitution** - Advanced bash features for complex data flows

### Engineering Principles  
- **Root cause analysis** - Don't just suppress symptoms
- **Trade-off evaluation** - Balance safety, efficiency, and complexity
- **Production readiness** - Consider all failure modes and edge cases

### Streaming Safety
- **Clean error handling** - Keep errors separate from data
- **Robust pipe patterns** - Use techniques that handle early exits gracefully  
- **Testing methodology** - Validate with realistic failure scenarios

### Key Takeaway

**The timeout + tee solution demonstrates that elegant engineering solutions often combine multiple simple tools (timeout, tee, process substitution) to solve complex problems safely and efficiently.**

This approach provides:
- ✅ **Immediate reliability** - Fixes the streaming issue now
- ✅ **Future optimization path** - Algorithm improvements can build on this foundation  
- ✅ **Educational value** - Shows proper Unix programming patterns
- ✅ **Production safety** - Handles edge cases and provides bounds

The lesson: When building production systems, prioritize reliability and safety first, then optimize for efficiency. The timeout + tee pattern achieves both goals with minimal complexity.

---

## Further Reading

- **Advanced Bash Scripting Guide** - Process substitution and pipe handling
- **Unix Network Programming** - Signal handling and process communication  
- **The Art of Unix Programming** - Philosophy of robust tool design
- **SRE Handbook** - Production reliability patterns

**Related UnifyWeaver Documentation:**
- Chapter 8: Template System Architecture  
- Chapter 9: Advanced Recursion Patterns
- Constraint System Documentation
- Firewall and Security Architecture

---

*This appendix documents a real production issue encountered October 14, 2025, and the engineering solution developed to address streaming safety concerns in CLI applications.*

---

## Navigation

**←** [Previous: Appendix A: Recursion Patterns](A_appendix_recursion_patterns.md) | [📖 Back to Book 1](README.md) | [Next: Appendix C: Fold Pattern Deep Dive →](appendix_b_fold_pattern_deep_dive.md)

