# C# Generator Mode Playbook (Fibonacci & Derived Columns)

This playbook shows how to use **generator mode** to emit standalone C# with a local fixpoint solver. It pairs with `playbooks/csharp_query_playbook.md`, which is limited to datalog-style joins/filters. Recursive arithmetic (e.g., Fibonacci) works here but not in query mode.

## Audience

- AI agents executing automated tests
- Developers exploring generator mode for recursive predicates

## Prerequisites

- SWI-Prolog with Janus (`library(janus)`).
- .NET SDK in PATH (`dotnet`).
- From the repo root: `src/unifyweaver/targets/csharp_target.pl` available.

## Workflow Overview

1. Define Prolog predicates (including recursive ones)
2. Compile to C# using `compile_predicate_to_csharp/3` with `mode(generator)` option
3. Generated code uses fixpoint solver for recursive evaluation

## Agent Inputs

Extract the example script from the examples library:

**For Bash (Linux/macOS):**
```bash
perl scripts/utils/extract_records.pl \
  "playbooks/examples_library/csharp_examples.md" \
  "unifyweaver.execution.csharp_fib_generator"
```

**For PowerShell (Windows/Cross-platform):**
```bash
perl scripts/utils/extract_records.pl \
  "playbooks/examples_library/csharp_examples.md" \
  "unifyweaver.execution.csharp_fib_generator_ps"
```

## Execution Guidance

### Step 1: Navigate to Repository Root

```bash
cd /root/UnifyWeaver
```

### Step 2: Create Temporary Directory

```bash
mkdir -p tmp/csharp_fib_project
```

### Step 3: Write the Prolog Source (Fibonacci)

```bash
cat > tmp/fib_generator.pl <<'EOF'
% Fibonacci sequence - requires generator mode (recursive calls with computed args)
:- multifile fib/2.
:- dynamic fib/2.

fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
EOF
```

### Step 4: Write the SWIPL Goal File

```bash
cat > tmp/swipl_fib_goal.pl <<'GOAL'
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- consult('tmp/fib_generator.pl').
:- use_module(library(csharp_target)).
:- compile_predicate_to_csharp(fib/2, [mode(generator)], CSharpCode),
   open('tmp/csharp_fib_project/fib.cs', write, Stream),
   write(Stream, CSharpCode),
   close(Stream).
:- halt.
GOAL
```

### Step 5: Execute Compilation

```bash
swipl -l tmp/swipl_fib_goal.pl
```

### Step 6: Verify Output

```bash
cat tmp/csharp_fib_project/fib.cs
```

## Expected Output

The generated C# file should contain:
- A `fib` method implementing the Fibonacci sequence
- Fixpoint solver logic for recursive evaluation
- Base cases for `fib(0, 0)` and `fib(1, 1)`

Example structure:
```csharp
// Generated code will include:
// - Relation definitions for fib/2
// - Recursive evaluation using fixpoint iteration
// - Proper handling of arithmetic expressions
```

## Quick Patterns Reference

### Derived Column (Works in Both Query & Generator)

```prolog
num_pair(1,2).
num_pair(3,4).
sum_pair(X, Y, Sum) :- num_pair(X, Y), Sum is X + Y.
```

Generator call:
```prolog
?- csharp_target:compile_predicate_to_csharp(sum_pair/3, [mode(generator)], Code).
```

### Recursive Arithmetic (Fibonacci) — Generator OK, Query Mode Will Reject

```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
```

Generator call:
```prolog
?- csharp_target:compile_predicate_to_csharp(fib/2, [mode(generator)], Code).
```

## Platform-Specific Notes

### Linux/macOS
- Use bash script directly
- Ensure `swipl` is in PATH

### Windows
- Use PowerShell version from examples library
- SWI-Prolog installer adds to PATH automatically
- Use forward slashes in paths for cross-platform compatibility

## Troubleshooting

### "Query mode rejected recursive predicate"
- Use `[mode(generator)]` option instead of default query mode
- Generator mode handles recursive calls with computed arguments

### "File not created"
- Check that `tmp/` directory exists
- Verify SWI-Prolog can find the library paths
- Run `swipl` interactively to debug

### "Module not found"
- Ensure you're running from repo root
- Verify `src/unifyweaver/targets/csharp_target.pl` exists

## Notes

- Aggregates in generator mode: `aggregate_all/3` (count/sum/min/max/set/bag) and grouped `aggregate_all/4` (sum/min/max/set/bag/count) are supported.
- Indexing defaults ON (per-relation, arg0/arg1 buckets); disable with `enable_indexing(false)` if needed.
- Builtins/negation that reference only bound vars are evaluated early to prune work; order is otherwise preserved.
