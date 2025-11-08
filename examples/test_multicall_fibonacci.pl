:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_multicall_fibonacci.pl - Test multi-call linear recursion detection
%
% Tests that fibonacci (with 2 recursive calls) is now detected as
% linear recursive streamable with memoization, instead of fold pattern.

:- use_module('../src/unifyweaver/core/advanced/pattern_matchers').

% Fibonacci - should now be detected as linear recursive (multi-call)
% Previously required forbid_linear_recursion to use fold pattern
% Now with multi-call support, this should use linear recursion + memoization
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% Tribonacci - 3 recursive calls (even more interesting!)
trib(0, 0).
trib(1, 1).
trib(2, 1).
trib(N, T) :-
    N > 2,
    N1 is N - 1,
    N2 is N - 2,
    N3 is N - 3,
    trib(N1, T1),
    trib(N2, T2),
    trib(N3, T3),
    T is T1 + T2 + T3.

% Counter-example: Tree sum - uses structural decomposition (should NOT be linear)
tree_sum([], 0).
tree_sum([V, L, R], S) :-
    tree_sum(L, LS),
    tree_sum(R, RS),
    S is V + LS + RS.

% Counter-example: Shared variable across calls (should NOT be linear)
bad_shared(X, R1, R2) :-
    X > 0,
    bad_shared(X, R1, _),  % Same variable X in both calls!
    bad_shared(X, _, R2).

%% Test predicates
test_fibonacci_detection :-
    format('~n=== Testing Multi-Call Linear Recursion Detection ===~n', []),
    format('~nTest 1: Fibonacci (2 recursive calls)~n', []),
    (   is_linear_recursive_streamable(fib/2)
    ->  format('  ✓ fib/2 detected as linear recursive (multi-call)~n', [])
    ;   format('  ✗ FAIL: fib/2 NOT detected as linear recursive~n', []),
        fail
    ),

    format('~nTest 2: Tribonacci (3 recursive calls)~n', []),
    (   is_linear_recursive_streamable(trib/2)
    ->  format('  ✓ trib/2 detected as linear recursive (multi-call)~n', [])
    ;   format('  ✗ FAIL: trib/2 NOT detected as linear recursive~n', []),
        fail
    ),

    format('~nTest 3: Tree sum (structural args - should NOT be linear)~n', []),
    (   \+ is_linear_recursive_streamable(tree_sum/2)
    ->  format('  ✓ tree_sum/2 correctly NOT detected as linear~n', [])
    ;   format('  ✗ FAIL: tree_sum/2 incorrectly detected as linear~n', []),
        fail
    ),

    format('~nTest 4: Bad shared (shared variable - should NOT be linear)~n', []),
    (   \+ is_linear_recursive_streamable(bad_shared/3)
    ->  format('  ✓ bad_shared/3 correctly NOT detected as linear~n', [])
    ;   format('  ✗ FAIL: bad_shared/3 incorrectly detected as linear~n', []),
        fail
    ),

    format('~n=== All Tests Passed ✓ ===~n', []).

test_distinct_args :-
    format('~n=== Testing Distinct Args Check ===~n', []),

    % Test fibonacci body
    format('~nTest: Fibonacci args are distinct~n', []),
    Body = (N > 1, N1 is N - 1, N2 is N - 2, fib(N1, F1), fib(N2, F2), F is F1 + F2),
    (   recursive_calls_have_distinct_args(Body, fib)
    ->  format('  ✓ Fibonacci args N1, N2 are distinct~n', [])
    ;   format('  ✗ FAIL: Fibonacci args not recognized as distinct~n', []),
        fail
    ),

    % TODO: Add test for shared variable detection (requires refining collect_recursive_call)
    % For now, we focus on the positive case (distinct variables)

    format('~n=== Distinct Args Tests Passed ✓ ===~n', []).

test_call_count :-
    format('~n=== Testing Call Count Extraction ===~n', []),

    % Test fibonacci call count
    format('~nTest: Fibonacci call count~n', []),
    get_recursive_call_count(fib/2, FibCount),
    (   FibCount =:= 2
    ->  format('  ✓ fib/2 has 2 recursive calls~n', [])
    ;   format('  ✗ FAIL: Expected 2 calls, got ~w~n', [FibCount]),
        fail
    ),

    % Test tribonacci call count
    format('~nTest: Tribonacci call count~n', []),
    get_recursive_call_count(trib/2, TribCount),
    (   TribCount =:= 3
    ->  format('  ✓ trib/2 has 3 recursive calls~n', [])
    ;   format('  ✗ FAIL: Expected 3 calls, got ~w~n', [TribCount]),
        fail
    ),

    % Test tree_sum call count
    format('~nTest: Tree sum call count~n', []),
    get_recursive_call_count(tree_sum/2, TreeCount),
    (   TreeCount =:= 2
    ->  format('  ✓ tree_sum/2 has 2 recursive calls~n', [])
    ;   format('  ✗ FAIL: Expected 2 calls, got ~w~n', [TreeCount]),
        fail
    ),

    % Test multi_call_info for fibonacci
    format('~nTest: Fibonacci multi_call_info~n', []),
    get_multi_call_info(fib/2, multi_call(Count, IsLinear, HasDistinct, IsPrecomp)),
    (   Count =:= 2, IsLinear = true, HasDistinct = true, IsPrecomp = true
    ->  format('  ✓ fib/2 info: multi_call(2, true, true, true)~n', [])
    ;   format('  ✗ FAIL: Got multi_call(~w, ~w, ~w, ~w)~n', [Count, IsLinear, HasDistinct, IsPrecomp]),
        fail
    ),

    % Test multi_call_info for tribonacci
    format('~nTest: Tribonacci multi_call_info~n', []),
    get_multi_call_info(trib/2, multi_call(Count2, IsLinear2, HasDistinct2, IsPrecomp2)),
    (   Count2 =:= 3, IsLinear2 = true, HasDistinct2 = true, IsPrecomp2 = true
    ->  format('  ✓ trib/2 info: multi_call(3, true, true, true)~n', [])
    ;   format('  ✗ FAIL: Got multi_call(~w, ~w, ~w, ~w)~n', [Count2, IsLinear2, HasDistinct2, IsPrecomp2]),
        fail
    ),

    % Test multi_call_info for tree_sum (should NOT be linear)
    format('~nTest: Tree sum multi_call_info (should NOT be linear)~n', []),
    get_multi_call_info(tree_sum/2, multi_call(Count3, IsLinear3, _HasDistinct3, _IsPrecomp3)),
    (   Count3 =:= 2, IsLinear3 = false
    ->  format('  ✓ tree_sum/2 correctly NOT linear: multi_call(2, false, ...)~n', [])
    ;   format('  ✗ FAIL: Got multi_call(~w, ~w, ...)~n', [Count3, IsLinear3]),
        fail
    ),

    format('~n=== Call Count Tests Passed ✓ ===~n', []).

main :-
    test_distinct_args,
    test_call_count,
    test_fibonacci_detection,
    format('~n=== All Multi-Call Linear Recursion Tests Complete ===~n', []),
    halt(0).

main :-
    format('~n✗ Tests failed~n', []),
    halt(1).

:- initialization(main, main).
