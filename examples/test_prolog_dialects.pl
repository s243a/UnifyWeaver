:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_prolog_dialects.pl - Test Prolog dialect system
%
% This test demonstrates:
% - Generating scripts for different Prolog dialects
% - SWI-Prolog interpreted execution
% - GNU Prolog compilation with gplc
% - Dialect-specific optimizations

:- use_module('../src/unifyweaver/targets/prolog_target').
:- use_module('../src/unifyweaver/targets/prolog_dialects').

%% ============================================
%% TEST PREDICATES
%% ============================================

% Simple factorial - good for both dialects
factorial(0, 1) :- !.
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.

% Tail-recursive sum - optimal for GNU Prolog compilation
sum_list(List, Sum) :-
    sum_list(List, 0, Sum).

sum_list([], Acc, Acc).
sum_list([H|T], Acc, Sum) :-
    Acc1 is Acc + H,
    sum_list(T, Acc1, Sum).

% Test predicate that uses both
test_math :-
    factorial(5, F),
    format('Factorial of 5: ~w~n', [F]),
    sum_list([1,2,3,4,5], Sum),
    format('Sum of [1,2,3,4,5]: ~w~n', [Sum]).

%% ============================================
%% TESTS
%% ============================================

test_swi_prolog :-
    writeln('=== Test 1: SWI-Prolog Dialect ==='),

    % Generate script for SWI-Prolog
    generate_prolog_script(
        [factorial/2, sum_list/2, sum_list/3, test_math/0],
        [
            dialect(swi),
            entry_point(test_math)
        ],
        Code
    ),

    % Write to file
    write_prolog_script(Code, 'output/test_swi.pl', [dialect(swi)]),

    writeln('  ✓ Generated output/test_swi.pl'),
    writeln('  Run with: ./output/test_swi.pl'),
    writeln('').

test_gnu_prolog :-
    writeln('=== Test 2: GNU Prolog Dialect ==='),

    % Generate script for GNU Prolog
    generate_prolog_script(
        [factorial/2, sum_list/2, sum_list/3, test_math/0],
        [
            dialect(gnu),
            entry_point(test_math)
        ],
        Code
    ),

    % Write to file
    write_prolog_script(Code, 'output/test_gnu.pl', [dialect(gnu)]),

    writeln('  ✓ Generated output/test_gnu.pl'),
    writeln('  Run interpreted: ./output/test_gnu.pl'),
    writeln('  Or compile first: gplc output/test_gnu.pl -o output/test_gnu'),
    writeln('  Then run: ./output/test_gnu'),
    writeln('').

test_gnu_prolog_compiled :-
    writeln('=== Test 3: GNU Prolog with Auto-Compilation ==='),

    % Generate and compile script
    generate_prolog_script(
        [factorial/2, sum_list/2, sum_list/3, test_math/0],
        [
            dialect(gnu),
            entry_point(test_math),
            compile(true)
        ],
        Code
    ),

    % Write and compile
    write_prolog_script(Code, 'output/test_gnu_compiled.pl',
                       [dialect(gnu), compile(true)]),

    writeln('  ✓ Generated and compiled output/test_gnu_compiled'),
    writeln('  Run binary: ./output/test_gnu_compiled'),
    writeln('').

test_dialect_capabilities :-
    writeln('=== Test 4: Dialect Capabilities ==='),

    dialect_capabilities(swi, SwiCaps),
    format('  SWI-Prolog: ~w~n', [SwiCaps]),

    dialect_capabilities(gnu, GnuCaps),
    format('  GNU Prolog: ~w~n', [GnuCaps]),

    writeln('').

test_dialect_validation :-
    writeln('=== Test 5: Dialect Validation ==='),

    % Check compatibility
    validate_for_dialect(swi, [factorial/2, sum_list/3], SwiIssues),
    format('  SWI-Prolog issues: ~w~n', [SwiIssues]),

    validate_for_dialect(gnu, [factorial/2, sum_list/3], GnuIssues),
    format('  GNU Prolog issues: ~w~n', [GnuIssues]),

    writeln('').

test_dialect_recommendation :-
    writeln('=== Test 6: Dialect Recommendation ==='),

    % Get recommendation for tail-recursive predicates
    recommend_dialect([sum_list/3], Dialect, Reason),
    format('  Recommended dialect: ~w~n', [Dialect]),
    format('  Reason: ~w~n', [Reason]),

    writeln('').

%% ============================================
%% MAIN
%% ============================================

main :-
    writeln(''),
    writeln('╔════════════════════════════════════════╗'),
    writeln('║  Prolog Dialect System Tests          ║'),
    writeln('╚════════════════════════════════════════╝'),
    writeln(''),

    % Ensure output directory exists
    (   exists_directory('output')
    ->  true
    ;   make_directory('output')
    ),

    % Run tests
    test_dialect_capabilities,
    test_dialect_validation,
    test_dialect_recommendation,
    test_swi_prolog,
    test_gnu_prolog,

    % Only compile if gplc is available
    (   dialect_available(gnu)
    ->  test_gnu_prolog_compiled
    ;   writeln('  ⚠ Skipping GNU Prolog compilation (gplc not available)')
    ),

    writeln(''),
    writeln('╔════════════════════════════════════════╗'),
    writeln('║  All Tests Complete                    ║'),
    writeln('╚════════════════════════════════════════╝'),
    writeln('').

:- initialization(main, main).
