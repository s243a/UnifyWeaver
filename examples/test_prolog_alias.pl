:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_prolog_alias.pl - Test Prolog dialect alias configuration
%
% This test demonstrates the configurable 'prolog' alias that can expand to
% different dialect strategies.

:- use_module('../src/unifyweaver/targets/prolog_dialects').

%% ============================================
%% TEST PREDICATES
%% ============================================

% Simple factorial
factorial(0, 1) :- !.
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.

%% ============================================
%% TESTS
%% ============================================

test_default_expansion :-
    writeln('=== Test 1: Default Expansion (swi) ==='),

    expand_prolog_alias(prolog, Dialects1),
    format('  prolog expands to: ~w~n', [Dialects1]),

    (   Dialects1 = [swi]
    ->  writeln('  ✓ PASS: Default is swi')
    ;   writeln('  ✗ FAIL: Expected [swi]')
    ),
    writeln('').

test_set_gnu_default :-
    writeln('=== Test 2: Set GNU as Default ==='),

    set_prolog_default(gnu),
    expand_prolog_alias(prolog, Dialects2),
    format('  prolog expands to: ~w~n', [Dialects2]),

    (   Dialects2 = [gnu]
    ->  writeln('  ✓ PASS: Changed to gnu')
    ;   writeln('  ✗ FAIL: Expected [gnu]')
    ),
    writeln('').

test_gnu_fallback_swi :-
    writeln('=== Test 3: GNU with SWI Fallback ==='),

    set_prolog_default(gnu_fallback_swi),
    expand_prolog_alias(prolog, Dialects3),
    format('  prolog expands to: ~w~n', [Dialects3]),

    (   Dialects3 = [gnu, swi]
    ->  writeln('  ✓ PASS: Expands to [gnu, swi]')
    ;   writeln('  ✗ FAIL: Expected [gnu, swi]')
    ),
    writeln('').

test_swi_fallback_gnu :-
    writeln('=== Test 4: SWI with GNU Fallback ==='),

    set_prolog_default(swi_fallback_gnu),
    expand_prolog_alias(prolog, Dialects4),
    format('  prolog expands to: ~w~n', [Dialects4]),

    (   Dialects4 = [swi, gnu]
    ->  writeln('  ✓ PASS: Expands to [swi, gnu]')
    ;   writeln('  ✗ FAIL: Expected [swi, gnu]')
    ),
    writeln('').

test_custom_list :-
    writeln('=== Test 5: Custom Dialect List ==='),

    set_prolog_default([gnu, swi]),
    expand_prolog_alias(prolog, Dialects5),
    format('  prolog expands to: ~w~n', [Dialects5]),

    (   Dialects5 = [gnu, swi]
    ->  writeln('  ✓ PASS: Custom list preserved')
    ;   writeln('  ✗ FAIL: Expected [gnu, swi]')
    ),
    writeln('').

test_concrete_dialect_passthrough :-
    writeln('=== Test 6: Concrete Dialects Pass Through ==='),

    expand_prolog_alias(swi, Dialects6a),
    expand_prolog_alias(gnu, Dialects6b),

    format('  swi expands to: ~w~n', [Dialects6a]),
    format('  gnu expands to: ~w~n', [Dialects6b]),

    (   Dialects6a = [swi], Dialects6b = [gnu]
    ->  writeln('  ✓ PASS: Concrete dialects unchanged')
    ;   writeln('  ✗ FAIL: Expected [swi] and [gnu]')
    ),
    writeln('').

test_get_current_default :-
    writeln('=== Test 7: Get Current Default ==='),

    set_prolog_default(gnu_fallback_swi),
    get_prolog_default(Strategy),
    format('  Current strategy: ~w~n', [Strategy]),

    (   Strategy = gnu_fallback_swi
    ->  writeln('  ✓ PASS: Retrieved correct strategy')
    ;   writeln('  ✗ FAIL: Expected gnu_fallback_swi')
    ),
    writeln('').

test_preferences_integration :-
    writeln('=== Test 8: Integration with Preferences System ==='),
    writeln(''),
    writeln('  Example 1: Simple fallback'),
    writeln('  prefer([bash]),'),
    writeln('  fallback_order([prolog])  % Will expand to [swi] by default'),
    writeln(''),
    writeln('  Example 2: GNU Prolog fallback'),
    set_prolog_default(gnu_fallback_swi),
    writeln('  set_prolog_default(gnu_fallback_swi),'),
    writeln('  prefer([bash]),'),
    writeln('  fallback_order([prolog])  % Will expand to [gnu, swi]'),
    writeln(''),
    writeln('  Example 3: Direct dialect specification'),
    writeln('  prefer([bash]),'),
    writeln('  fallback_order([prolog_gnu, prolog_swi])  % Explicit'),
    writeln(''),
    writeln('  ✓ Users can choose: generic "prolog" or specific dialects'),
    writeln('').

test_usage_scenarios :-
    writeln('=== Test 9: Common Usage Scenarios ==='),
    writeln(''),

    % Scenario 1: Development
    writeln('  Scenario 1: Development (fast iteration)'),
    set_prolog_default(swi),
    expand_prolog_alias(prolog, Dev),
    format('    prolog → ~w (interpreted, full features)~n', [Dev]),
    writeln(''),

    % Scenario 2: Production
    writeln('  Scenario 2: Production (performance)'),
    set_prolog_default(gnu),
    expand_prolog_alias(prolog, Prod),
    format('    prolog → ~w (compiled binaries)~n', [Prod]),
    writeln(''),

    % Scenario 3: Best-effort
    writeln('  Scenario 3: Best-effort (try compilation, fall back to interpreted)'),
    set_prolog_default(gnu_fallback_swi),
    expand_prolog_alias(prolog, BestEffort),
    format('    prolog → ~w (compile if possible, interpret otherwise)~n', [BestEffort]),
    writeln(''),

    writeln('  ✓ Flexible configuration for different environments'),
    writeln('').

%% ============================================
%% MAIN
%% ============================================

main :-
    writeln(''),
    writeln('╔════════════════════════════════════════╗'),
    writeln('║  Prolog Dialect Alias Tests           ║'),
    writeln('╚════════════════════════════════════════╝'),
    writeln(''),

    % Run all tests
    test_default_expansion,
    test_set_gnu_default,
    test_gnu_fallback_swi,
    test_swi_fallback_gnu,
    test_custom_list,
    test_concrete_dialect_passthrough,
    test_get_current_default,
    test_preferences_integration,
    test_usage_scenarios,

    writeln('╔════════════════════════════════════════╗'),
    writeln('║  All Tests Complete                    ║'),
    writeln('╚════════════════════════════════════════╝'),
    writeln('').

:- initialization(main, main).
