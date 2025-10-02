<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# Advanced Recursion Compiler

This directory contains advanced recursion pattern detection and compilation modules for UnifyWeaver.

## Quick Start

```prolog
% Load the advanced compiler
?- use_module('advanced/advanced_recursive_compiler').

% Define a tail-recursive predicate
?- assertz((count([], Acc, Acc))),
   assertz((count([_|T], Acc, N) :- Acc1 is Acc + 1, count(T, Acc1, N))).

% Compile it
?- compile_advanced_recursive(count/3, [], BashCode).

% Or compile a mutual recursion group
?- compile_predicate_group([is_even/1, is_odd/1], [], BashCode).
```

## Modules

| Module | Purpose | Key Predicates |
|--------|---------|----------------|
| `advanced_recursive_compiler.pl` | Main orchestrator | `compile_advanced_recursive/3`, `compile_predicate_group/3` |
| `call_graph.pl` | Build dependency graphs | `build_call_graph/2`, `get_dependencies/2` |
| `scc_detection.pl` | Find mutual recursion | `find_sccs/2`, `topological_order/2` |
| `pattern_matchers.pl` | Detect patterns | `is_tail_recursive_accumulator/2`, `is_linear_recursive_streamable/1` |
| `tail_recursion.pl` | Compile tail recursion | `compile_tail_recursion/3` |
| `linear_recursion.pl` | Compile linear recursion | `compile_linear_recursion/3` |
| `mutual_recursion.pl` | Compile mutual groups | `compile_mutual_recursion/3` |
| `test_advanced.pl` | Test suite | `test_all_advanced/0`, `test_all/0` |

## Testing

```prolog
?- [test_advanced].
?- test_all_advanced.  % Run all tests
```

Generates bash scripts in `output/advanced/`.

## Integration

The advanced compiler is automatically used by `recursive_compiler.pl` via an optional hook:

```prolog
compile_recursive(Pred/Arity, Options, BashCode) :-
    % ... try basic patterns ...
    ;   % Try advanced patterns
        catch(
            advanced_recursive_compiler:compile_advanced_recursive(
                Pred/Arity, Options, BashCode
            ),
            _,
            fail
        )
    ;   % Fall back to memoization
        ...
```

## Pattern Examples

### Tail Recursion
```prolog
sum([], Acc, Acc).
sum([H|T], Acc, Sum) :-
    Acc1 is Acc + H,
    sum(T, Acc1, Sum).
```

### Linear Recursion
```prolog
length([], 0).
length([_|T], N) :-
    length(T, N1),
    N is N1 + 1.
```

### Mutual Recursion
```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```

## Documentation

See [`docs/ADVANCED_RECURSION.md`](../../../../docs/ADVANCED_RECURSION.md) for detailed architecture and design documentation.
