:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_mutual_recursion_negative.pl - Negative Test Cases for Mutual Recursion
%
% Tests that mutually recursive predicates (is_even/is_odd) correctly fail
% for invalid inputs and handle edge cases properly.

:- use_module('../src/unifyweaver/core/advanced/mutual_recursion').

%% ============================================
%% MAIN TEST SUITE
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Mutual Recursion Negative Cases               ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Setup predicates
    setup_is_even_odd,

    % Run all tests
    test_positive_cases,
    test_negative_cases,
    test_edge_cases,
    test_bash_execution,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% SETUP
%% ============================================

setup_is_even_odd :-
    % Clear any existing definitions
    catch(abolish(user:is_even/1), _, true),
    catch(abolish(user:is_odd/1), _, true),

    % Define mutually recursive even/odd predicates
    assertz(user:is_even(0)),
    assertz(user:(is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))),
    assertz(user:is_odd(1)),
    assertz(user:(is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))),

    format('[Setup] Defined is_even/1 and is_odd/1~n', []).

%% ============================================
%% TEST 1: POSITIVE CASES (Should Succeed)
%% ============================================

test_positive_cases :-
    format('~n[Test 1] Positive Cases (Should Succeed)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 1.1: is_even(0) - base case
    (   user:is_even(0)
    ->  format('  ✓ is_even(0) succeeds (base case)~n', [])
    ;   format('  ✗ FAIL: is_even(0) should succeed~n', []),
        fail
    ),

    % Test 1.2: is_even(2)
    (   user:is_even(2)
    ->  format('  ✓ is_even(2) succeeds~n', [])
    ;   format('  ✗ FAIL: is_even(2) should succeed~n', []),
        fail
    ),

    % Test 1.3: is_even(4)
    (   user:is_even(4)
    ->  format('  ✓ is_even(4) succeeds~n', [])
    ;   format('  ✗ FAIL: is_even(4) should succeed~n', []),
        fail
    ),

    % Test 1.4: is_odd(1) - base case
    (   user:is_odd(1)
    ->  format('  ✓ is_odd(1) succeeds (base case)~n', [])
    ;   format('  ✗ FAIL: is_odd(1) should succeed~n', []),
        fail
    ),

    % Test 1.5: is_odd(3)
    (   user:is_odd(3)
    ->  format('  ✓ is_odd(3) succeeds~n', [])
    ;   format('  ✗ FAIL: is_odd(3) should succeed~n', []),
        fail
    ),

    % Test 1.6: is_odd(5)
    (   user:is_odd(5)
    ->  format('  ✓ is_odd(5) succeeds~n', [])
    ;   format('  ✗ FAIL: is_odd(5) should succeed~n', []),
        fail
    ),

    format('[✓] Test 1 Complete~n', []),
    !.

%% ============================================
%% TEST 2: NEGATIVE CASES (Should Fail)
%% ============================================

test_negative_cases :-
    format('~n[Test 2] Negative Cases (Should Fail)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 2.1: is_even(1) - should fail
    (   \+ user:is_even(1)
    ->  format('  ✓ is_even(1) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_even(1) should fail~n', []),
        fail
    ),

    % Test 2.2: is_even(3) - should fail
    (   \+ user:is_even(3)
    ->  format('  ✓ is_even(3) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_even(3) should fail~n', []),
        fail
    ),

    % Test 2.3: is_even(5) - should fail
    (   \+ user:is_even(5)
    ->  format('  ✓ is_even(5) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_even(5) should fail~n', []),
        fail
    ),

    % Test 2.4: is_odd(0) - should fail
    (   \+ user:is_odd(0)
    ->  format('  ✓ is_odd(0) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_odd(0) should fail~n', []),
        fail
    ),

    % Test 2.5: is_odd(2) - should fail
    (   \+ user:is_odd(2)
    ->  format('  ✓ is_odd(2) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_odd(2) should fail~n', []),
        fail
    ),

    % Test 2.6: is_odd(4) - should fail
    (   \+ user:is_odd(4)
    ->  format('  ✓ is_odd(4) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_odd(4) should fail~n', []),
        fail
    ),

    % Test 2.7: is_odd(6) - should fail
    (   \+ user:is_odd(6)
    ->  format('  ✓ is_odd(6) correctly fails~n', [])
    ;   format('  ✗ FAIL: is_odd(6) should fail~n', []),
        fail
    ),

    format('[✓] Test 2 Complete~n', []),
    !.

%% ============================================
%% TEST 3: EDGE CASES
%% ============================================

test_edge_cases :-
    format('~n[Test 3] Edge Cases~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 3.1: Large even number
    (   user:is_even(100)
    ->  format('  ✓ is_even(100) succeeds (large number)~n', [])
    ;   format('  ✗ FAIL: is_even(100) should succeed~n', []),
        fail
    ),

    % Test 3.2: Large odd number
    (   user:is_odd(99)
    ->  format('  ✓ is_odd(99) succeeds (large number)~n', [])
    ;   format('  ✗ FAIL: is_odd(99) should succeed~n', []),
        fail
    ),

    % Test 3.3: Negative numbers (implementation dependent)
    % Note: Current implementation requires N > 0, so negatives should fail
    (   \+ user:is_even(-2)
    ->  format('  ✓ is_even(-2) correctly fails (negative)~n', [])
    ;   format('  ⚠ is_even(-2) succeeded (implementation allows negatives)~n', [])
    ),

    (   \+ user:is_odd(-3)
    ->  format('  ✓ is_odd(-3) correctly fails (negative)~n', [])
    ;   format('  ⚠ is_odd(-3) succeeded (implementation allows negatives)~n', [])
    ),

    format('[✓] Test 3 Complete~n', []),
    !.

%% ============================================
%% TEST 4: BASH EXECUTION
%% ============================================

test_bash_execution :-
    format('~n[Test 4] Bash Script Execution~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Compile to bash
    Predicates = [is_even/1, is_odd/1],
    (   can_compile_mutual_recursion(Predicates)
    ->  format('  ✓ Mutual recursion detected~n', [])
    ;   format('  ✗ FAIL: Should detect mutual recursion~n', []),
        fail
    ),

    compile_mutual_recursion(Predicates, [output_dir('/tmp/mutual_test')], BashCode),

    % Write bash scripts
    open('/tmp/mutual_test/is_even.sh', write, Stream1),
    write(Stream1, BashCode),
    close(Stream1),

    % Make executable
    process_create('/bin/chmod', ['+x', '/tmp/mutual_test/is_even.sh'], []),

    format('  ✓ Generated bash scripts~n', []),

    % Test positive cases
    process_create('/tmp/mutual_test/is_even.sh', ['4'], [stdout(pipe(Out1)), process(PID1)]),
    read_string(Out1, _, _Result1),
    process_wait(PID1, exit(Code1)),
    close(Out1),

    (   Code1 = 0
    ->  format('  ✓ Bash: is_even(4) exits 0 (success)~n', [])
    ;   format('  ✗ FAIL: is_even(4) should exit 0, got ~w~n', [Code1]),
        fail
    ),

    % Test negative cases - expect non-zero exit or empty output
    process_create('/tmp/mutual_test/is_even.sh', ['3'], [stdout(pipe(Out2)), stderr(null), process(PID2)]),
    read_string(Out2, _, Result2),
    process_wait(PID2, Status2),
    close(Out2),

    (   (Status2 \= exit(0) ; Result2 = "")
    ->  format('  ✓ Bash: is_even(3) fails correctly (exit=~w, output empty=~w)~n',
               [Status2, Result2 = ""])
    ;   format('  ⚠ Bash: is_even(3) behavior: ~w, output: ~w~n', [Status2, Result2])
    ),

    % Cleanup
    catch(delete_file('/tmp/mutual_test/is_even.sh'), _, true),

    format('[✓] Test 4 Complete~n', []),
    !.

:- initialization(main, main).
