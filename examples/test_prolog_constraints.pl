:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_prolog_constraints.pl - Test constraint handling for Prolog target
%
% Demonstrates:
% - Tabling for SWI-Prolog unique constraint
% - Wrapper for GNU Prolog unique constraint
% - Configurable failure modes
% - Constraint satisfaction checking

:- use_module('../src/unifyweaver/targets/prolog_target').
:- use_module('../src/unifyweaver/targets/prolog_dialects').
:- use_module('../src/unifyweaver/targets/prolog_constraints').

%% ============================================
%% TEST PREDICATES
%% ============================================

% Simple transitive closure - perfect for tabling
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(pat, jim).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Factorial - can have unique constraint
factorial(0, 1) :- !.
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.

%% ============================================
%% TESTS
%% ============================================

test_constraint_satisfaction :-
    writeln('=== Test 1: Constraint Satisfaction Checking ==='),

    constraint_satisfied(unique(true), swi, SwiMethod),
    format('  SWI-Prolog + unique: ~w~n', [SwiMethod]),

    constraint_satisfied(unique(true), gnu, GnuMethod),
    format('  GNU Prolog + unique: ~w~n', [GnuMethod]),

    constraint_satisfied(unordered(true), swi, UnorderedMethod),
    format('  Any + unordered: ~w~n', [UnorderedMethod]),

    (   SwiMethod = tabling, GnuMethod = wrapper
    ->  writeln('  ✓ PASS: Correct methods detected')
    ;   writeln('  ✗ FAIL: Unexpected methods')
    ),
    writeln('').

test_swi_tabling :-
    writeln('=== Test 2: SWI-Prolog Tabling for unique(true) ==='),

    % Generate with unique constraint for SWI
    generate_prolog_script(
        [ancestor/2],
        [
            dialect(swi),
            constraints([unique(true)]),
            entry_point(test_ancestor)
        ],
        Code
    ),

    % Check for tabling directive
    (   sub_atom(Code, _, _, _, ':- table ancestor/2')
    ->  writeln('  ✓ PASS: Tabling directive generated'),
        format('  Generated code preview:~n'),
        split_string(Code, '\n', '', Lines),
        length(Lines, NumLines),
        Take is min(15, NumLines),
        length(Preview, Take),
        append(Preview, _, Lines),
        maplist(format('    ~s~n'), Preview)
    ;   writeln('  ✗ FAIL: No tabling directive found')
    ),
    writeln('').

test_gnu_wrapper :-
    writeln('=== Test 3: GNU Prolog Wrapper for unique(true) ==='),

    % Generate with unique constraint for GNU
    generate_prolog_script(
        [factorial/2],
        [
            dialect(gnu),
            constraints([unique(true)]),
            entry_point(test_factorial)
        ],
        Code
    ),

    % Check for wrapper pattern
    (   sub_atom(Code, _, _, _, 'factorial_impl'),
        sub_atom(Code, _, _, _, 'setof')
    ->  writeln('  ✓ PASS: Wrapper code generated'),
        format('  Generated code preview:~n'),
        split_string(Code, '\n', '', Lines),
        length(Lines, NumLines),
        Take is min(20, NumLines),
        length(Preview, Take),
        append(Preview, _, Lines),
        maplist(format('    ~s~n'), Preview)
    ;   writeln('  ✗ FAIL: No wrapper pattern found')
    ),
    writeln('').

test_no_constraint :-
    writeln('=== Test 4: No Constraint (Verbatim Copy) ==='),

    % Generate without constraints
    generate_prolog_script(
        [ancestor/2],
        [
            dialect(swi),
            entry_point(test)
        ],
        Code
    ),

    % Should NOT have tabling directive
    (   \+ sub_atom(Code, _, _, _, ':- table')
    ->  writeln('  ✓ PASS: No tabling directive (verbatim copy)'),
        format('  Code contains ancestor/2 clauses without modification~n')
    ;   writeln('  ✗ FAIL: Unexpected tabling directive')
    ),
    writeln('').

test_failure_modes :-
    writeln('=== Test 5: Constraint Failure Modes ==='),

    % Test warn mode
    writeln('  Testing warn mode...'),
    set_constraint_failure_mode(warn),
    get_constraint_failure_mode(Mode1),
    format('    Current mode: ~w~n', [Mode1]),

    % Test fail mode (default)
    writeln('  Testing fail mode...'),
    set_constraint_failure_mode(fail),
    get_constraint_failure_mode(Mode2),
    format('    Current mode: ~w~n', [Mode2]),

    (   Mode2 = fail
    ->  writeln('  ✓ PASS: Failure modes configurable')
    ;   writeln('  ✗ FAIL: Unexpected mode')
    ),
    writeln('').

test_constraint_modes :-
    writeln('=== Test 6: Constraint Handling Modes ==='),

    % Set to native mode
    set_constraint_mode(unique, native),
    get_constraint_mode(unique, Mode1),
    format('  unique mode: ~w~n', [Mode1]),

    % Set to wrapper mode
    set_constraint_mode(unique, wrapper),
    get_constraint_mode(unique, Mode2),
    format('  unique mode: ~w~n', [Mode2]),

    % Set back to native
    set_constraint_mode(unique, native),

    (   Mode1 = native, Mode2 = wrapper
    ->  writeln('  ✓ PASS: Constraint modes configurable')
    ;   writeln('  ✗ FAIL: Unexpected modes')
    ),
    writeln('').

test_multiple_constraints :-
    writeln('=== Test 7: Multiple Constraints ==='),

    % Generate with multiple constraints
    generate_prolog_script(
        [ancestor/2],
        [
            dialect(swi),
            constraints([unique(true), unordered(true)]),
            entry_point(test)
        ],
        Code
    ),

    % Should have tabling for unique
    (   sub_atom(Code, _, _, _, ':- table ancestor/2')
    ->  writeln('  ✓ PASS: Multiple constraints handled'),
        writeln('    - unique(true) → tabling'),
        writeln('    - unordered(true) → natural (no modification)')
    ;   writeln('  ✗ FAIL: Tabling not generated')
    ),
    writeln('').

test_real_world_example :-
    writeln('=== Test 8: Real-World Example ==='),
    writeln(''),
    writeln('  Scenario: Compile ancestor/2 with uniqueness guarantee'),
    writeln(''),

    % SWI-Prolog version
    writeln('  Option A: SWI-Prolog (development)'),
    set_prolog_default(swi),
    generate_prolog_script(
        [ancestor/2],
        [
            dialect(swi),
            constraints([unique(true)]),
            entry_point(main)
        ],
        SwiCode
    ),
    write_prolog_script(SwiCode, 'output/ancestor_swi.pl', [dialect(swi)]),
    writeln('    ✓ Generated: output/ancestor_swi.pl (with tabling)'),

    % GNU Prolog version
    writeln(''),
    writeln('  Option B: GNU Prolog (production)'),
    set_prolog_default(gnu),
    generate_prolog_script(
        [ancestor/2],
        [
            dialect(gnu),
            constraints([unique(true)]),
            entry_point(main),
            compile(false)  % Don't auto-compile for test
        ],
        GnuCode
    ),
    write_prolog_script(GnuCode, 'output/ancestor_gnu.pl', [dialect(gnu)]),
    writeln('    ✓ Generated: output/ancestor_gnu.pl (with wrapper)'),

    writeln(''),
    writeln('  ✓ Both versions enforce unique(true) using dialect-appropriate methods'),
    writeln('').

%% ============================================
%% MAIN
%% ============================================

main :-
    writeln(''),
    writeln('╔════════════════════════════════════════╗'),
    writeln('║  Prolog Constraint Handling Tests     ║'),
    writeln('╚════════════════════════════════════════╝'),
    writeln(''),

    % Ensure output directory exists
    (   exists_directory('output')
    ->  true
    ;   make_directory('output')
    ),

    % Run tests
    test_constraint_satisfaction,
    test_swi_tabling,
    test_gnu_wrapper,
    test_no_constraint,
    test_failure_modes,
    test_constraint_modes,
    test_multiple_constraints,
    test_real_world_example,

    writeln('╔════════════════════════════════════════╗'),
    writeln('║  All Tests Complete                    ║'),
    writeln('╚════════════════════════════════════════╝'),
    writeln(''),
    writeln('Generated files:'),
    writeln('  - output/ancestor_swi.pl (SWI-Prolog with tabling)'),
    writeln('  - output/ancestor_gnu.pl (GNU Prolog with wrapper)'),
    writeln(''),
    writeln('Try running:'),
    writeln('  swipl output/ancestor_swi.pl'),
    writeln('  gprolog --consult-file output/ancestor_gnu.pl'),
    writeln('').

:- initialization(main, main).
