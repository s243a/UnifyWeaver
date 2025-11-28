# Handoff to Claude: The Duplicate Compilation Bug

**To:** Claude AI Agent
**From:** Gemini CLI
**Date:** 2025-11-11
**Subject:** Deeper Compiler Bug Found: Duplicate Compilation of Dependent Predicates

## 1. Executive Summary

Thank you for your previous fix (`1db9cbe`). Your analysis was correct, and integrating the dynamic source compiler into the driver was the right solution for the first bug.

However, fixing that issue has allowed us to uncover a second, more subtle bug: **when a predicate depends on a dynamic data source, the dependent predicate is compiled twice, causing the entire process to fail before it can write the output files.**

## 2. History of the Problem

This has been a difficult debugging process. Here is the full history:

1.  **Initial Failure:** My first attempt to run the `csv_data_source_playbook` failed with `ERROR: No clauses found for users/3`.
2.  **Your Correct Fix:** You correctly diagnosed that the `compiler_driver` was unaware of dynamic sources. You implemented a fix in commit `1db9cbe` to check for and compile dynamic sources first.
3.  **Environment Hell:** My subsequent failures were my own fault, caused by a desynchronized local branch and incorrect Prolog syntax. These issues are now resolved, and I can confirm your fix is active in my environment.
4.  **The New Failure:** With a clean environment and the correct code, I ran the test one last time.

## 3. The Current Failure: Duplicate Compilation

This is the crucial evidence.

### My Command:
```bash
swipl -g "
    use_module('src/unifyweaver/core/compiler_driver'),
    use_module('src/unifyweaver/core/dynamic_source_compiler'),
    use_module('src/unifyweaver/sources'),
    consult('tmp/csv_pipeline.pl'),
    compile(get_user_age/2, [output_dir('output')]),
    halt.
" 2>&1
```

### The Output Log:
```
...
Compiling dynamic source: users/3 using csv  # <-- SUCCESS! Your fix is working.
  Compiling CSV source: users/3
=== Compiling get_user_age/2 ===             # <-- First compilation attempt
  ...
  Body predicates: [users]
=== Compiling get_user_age/2 ===             # <-- DUPLICATE! Why is it running again?
  ...
  Body predicates: [users]
ERROR: ... : false                          # <-- Final failure
```

As you can see, `users/3` is correctly identified and compiled as a dynamic source. However, the `compiler_driver` then proceeds to compile `get_user_age/2` **twice**. This second, redundant compilation appears to be what causes the final error and prevents the `ls -l` command from finding the generated files.

## 4. Root Cause Hypothesis

My theory is that there is a flaw in the `compile_entry/3` or `compile_dependencies/3` logic within `compiler_driver.pl`.

The process seems to be:
1.  `compile(get_user_age/2, ...)` is called.
2.  `find_dependencies` correctly identifies `users/3` as a dependency.
3.  The `compile_dependencies` loop is entered for `users/3`.
4.  Inside this loop, `compile_current(users/3, ...)` is called, which correctly invokes the dynamic source compiler.
5.  **Here is the likely bug:** Something about this process is causing the original predicate, `get_user_age/2`, to be added to the compilation plan a second time. The `compiled/1` predicate that should prevent this is likely failing or being cleared incorrectly in this specific scenario.

## 5. Your Task

Please investigate the `compiler_driver.pl` to determine why a predicate that depends on a dynamic source is being processed twice. The goal is to eliminate this duplicate compilation so that the process can complete successfully and write the final `.sh` files to disk.

Thank you for your help in tracking down this complex issue.
