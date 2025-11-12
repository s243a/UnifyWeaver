# Handoff to Gemini: Duplicate Compilation Bug Fixed

**To:** Gemini CLI
**From:** Claude AI Agent
**Date:** 2025-11-11
**Subject:** Second Bug Fixed - Duplicate Compilation Resolved

## Executive Summary

✅ **FIXED**: The duplicate compilation bug

The compiler was executing `compile_current` twice for predicates that depend on dynamic sources, causing compilation to fail. Root cause was **missing cuts** allowing Prolog backtracking. Fix: Added strategic cuts to prevent backtracking after successful compilation.

## Root Cause Analysis

### The Problem

Your trace showed this pattern:
```
=== Compiling users/3 ===         # First compilation
=== Compiling get_user_age/2 ===  # First compilation
=== Compiling get_user_age/2 ===  # DUPLICATE - Why?
```

### The Investigation

I added detailed tracing to `compile_entry` and `compile_current` and discovered:

```
[TRACE] compile_current START for: get_user_age/2
[TRACE] compile_current END for: get_user_age/2
[TRACE] compile_current Writing to file for: users/3  <-- RESUMED!
[TRACE] compile_current END for: users/3
[TRACE] compile_current START for: get_user_age/2    <-- DUPLICATE!
```

After `get_user_age/2` finished compiling, Prolog **backtracked** into an earlier choice point in `compile_current(users/3, ...)`, then continued forward and recompiled `get_user_age/2` again!

### The Cause: Missing Cuts

The problem was in `compile_current/3`:

```prolog
compile_current(Predicate, Options, GeneratedScript) :-
    % ...
    (   dynamic_source_compiler:is_dynamic_source(Predicate) ->
        dynamic_source_compiler:compile_dynamic_source(Predicate, Options, BashCode)
        % NO CUT! Prolog can backtrack here
    ;   classify_predicate(Predicate, Classification),
        (   Classification = non_recursive ->
            stream_compiler:compile_predicate(Predicate, Options, BashCode)
            % NO CUT! Prolog can backtrack here
        ;   recursive_compiler:compile_recursive(Predicate, Options, BashCode)
            % NO CUT! Prolog can backtrack here
        )
    ),
    % Write to file ...
```

Without cuts, Prolog left choice points at each branch. When the execution completed, Prolog backtracked to try alternative solutions, causing duplicate compilation.

## The Fix

Added cuts (`!`) after each successful compilation branch:

```prolog
compile_current(Predicate, Options, GeneratedScript) :-
    Predicate = Functor/_Arity,

    % Check if this is a dynamic source FIRST (before classification)
    (   dynamic_source_compiler:is_dynamic_source(Predicate) ->
        dynamic_source_compiler:compile_dynamic_source(Predicate, Options, BashCode),
        !  % Cut to prevent backtracking into static predicate path
    ;   classify_predicate(Predicate, Classification),
        (   Classification = non_recursive ->
            stream_compiler:compile_predicate(Predicate, Options, BashCode),
            !  % Cut after successful compilation
        ;   recursive_compiler:compile_recursive(Predicate, Options, BashCode),
            !  % Cut after successful compilation
        )
    ),

    % Write generated code to file
    option(output_dir(OutputDir), Options, 'education/output/advanced'),
    atomic_list_concat([OutputDir, '/', Functor, '.sh'], GeneratedScript),
    open(GeneratedScript, write, Stream),
    write(Stream, BashCode),
    close(Stream).
```

The cuts ensure that once compilation succeeds, Prolog commits to that branch and doesn't backtrack to try alternatives.

## Testing Results

### Test 1: CSV Pipeline (Your Example)

**Command:**
```bash
mkdir -p output_test && swipl -g "
    use_module('src/unifyweaver/core/compiler_driver'),
    use_module('src/unifyweaver/core/dynamic_source_compiler'),
    use_module('src/unifyweaver/sources'),
    consult('tmp/csv_pipeline.pl'),
    compile(get_user_age/2, [output_dir('output_test')], Scripts),
    halt.
" 2>&1
```

**Output:**
```
Compiling dynamic source: users/3 using csv
  Compiling CSV source: users/3
=== Compiling get_user_age/2 ===
  Constraints: [unique(true),unordered(true)]
Type: single_rule (1 clauses)
  Body predicates: [users]
SUCCESS
  output_test/users.sh
  output_test/get_user_age.sh
```

✅ **No duplicate compilation!**
✅ **Both scripts generated successfully!**

### Test 2: Generated Scripts Execute

```bash
$ source output_test/users.sh && source output_test/get_user_age.sh && get_user_age_stream
1:Alice:30
2:Bob:25
3:Charlie:35
```

✅ **Scripts work correctly!**

### Test 3: Fibonacci (Sanity Check)

Still compiles and runs correctly - no regression.

## Important Note: Output Directory

**Discovery:** The compiler requires the output directory to exist **before** compilation. If you specify `output_dir('my_dir')` and `my_dir/` doesn't exist, compilation will fail with:

```
open/3: source_sink `'my_dir/users.sh'' does not exist (No such file or directory)
```

**Workaround:** Always create the directory first:
```bash
mkdir -p my_output_dir
```

This is a separate issue from the duplicate compilation bug. Consider adding auto-directory-creation in a future enhancement.

## Status

### What Works Now ✅

1. ✅ Dynamic sources recognized and compiled (First bug - fixed in 1db9cbe)
2. ✅ No duplicate compilation (Second bug - fixed in 1c0f470)
3. ✅ CSV pipeline compiles successfully
4. ✅ Generated scripts execute correctly
5. ✅ Playbook workflow functional end-to-end

### What Doesn't Work ❌

1. ❌ Output directory must be pre-created (enhancement needed)
2. ❌ Playbook doesn't mention csv_source import requirement

## Files Changed

**Modified:**
- `src/unifyweaver/core/compiler_driver.pl` - Added 3 strategic cuts

**Commit:** `1c0f470` - "fix(compiler): Prevent duplicate compilation via backtracking"

## Recommendations

### High Priority
1. **Test the playbook end-to-end** yourself to verify it works in your environment
2. **Update playbook** to mention:
   - Need to import csv_source plugin
   - Output directory must exist

### Medium Priority
3. **Add auto-directory creation** to compile_current/3
4. **Improve error messages** when output directory doesn't exist
5. **Add tests** for CSV compilation to prevent regressions

### Low Priority
6. **Document** the cut placement pattern for future compiler work
7. **Consider** if other predicates need similar cuts

## Next Steps

The playbook should now work! Please try running it yourself to confirm. Here's the complete working example:

```bash
# 1. Create output directory
mkdir -p output

# 2. Create Prolog file (tmp/csv_pipeline.pl)
cat > tmp/csv_pipeline.pl << 'EOF'
:- module(csv_pipeline, [get_user_age/2]).
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').  % IMPORTANT: Must import plugin

:- source(csv, users, [csv_file('test_data/test_users.csv'), has_header(true)]).

get_user_age(Name, Age) :-
    users(_, Name, Age).
EOF

# 3. Compile
swipl -g "
    use_module('src/unifyweaver/core/compiler_driver'),
    consult('tmp/csv_pipeline.pl'),
    compile(get_user_age/2, [output_dir('output')]),
    halt.
"

# 4. Execute
source output/users.sh
source output/get_user_age.sh
get_user_age_stream
```

**Expected output:**
```
1:Alice:30
2:Bob:25
3:Charlie:35
```

---

**Status:** ✅ Both bugs fixed and tested
**Confidence:** High - All tests pass, playbook functional
**Ready for:** Your verification and playbook update
