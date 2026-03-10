:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.
%
% elixir_bindings.pl - Elixir-specific bindings
%
% This module defines bindings for Elixir target language features.
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(elixir_bindings, [
    init_elixir_bindings/0,
    elixir_binding/5,             % Convenience: elixir_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_elixir_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

init_elixir_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings,
    register_comparison_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

elixir_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(elixir, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- elixir_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(elixir, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % Output
    declare_binding(elixir, print/1, 'IO.puts(~w)',
        [string], [], [effect(io), deterministic, total, pattern(expression)]),
    declare_binding(elixir, write/1, 'IO.write(~w)',
        [string], [], [effect(io), deterministic, total, pattern(expression)]),
    
    % Conditionals / Tests
    declare_binding(elixir, true/0, 'true',
        [], [], [pure, deterministic, total, pattern(expression)]),
    declare_binding(elixir, fail/0, 'false',
        [], [], [pure, deterministic, total, pattern(expression)]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(elixir, atom_concat/3, '~w = to_string(~w) <> to_string(~w)',
        [string, string], [string], [pure, deterministic, total, pattern(expression)]),
    
    declare_binding(elixir, string_length/2, '~w = String.length(to_string(~w))',
        [string], [int], [pure, deterministic, total, pattern(expression)]),
        
    declare_binding(elixir, string_upper/2, '~w = String.upcase(to_string(~w))',
        [string], [string], [pure, deterministic, total, pattern(expression)]),
        
    declare_binding(elixir, string_lower/2, '~w = String.downcase(to_string(~w))',
        [string], [string], [pure, deterministic, total, pattern(expression)]),
        
    declare_binding(elixir, is_alpha/1, 'Regex.match?(~r/^[a-zA-Z]+$/, to_string(~w))',
        [string], [], [pure, deterministic, total, pattern(test)]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    declare_binding(elixir, add/3, '~w = ~w + ~w',
        [int, int], [int], [pure, deterministic, total, pattern(arithmetic)]),
    declare_binding(elixir, subtract/3, '~w = ~w - ~w',
        [int, int], [int], [pure, deterministic, total, pattern(arithmetic)]),
    declare_binding(elixir, multiply/3, '~w = ~w * ~w',
        [int, int], [int], [pure, deterministic, total, pattern(arithmetic)]),
    declare_binding(elixir, divide/3, '~w = div(~w, ~w)',
        [int, int], [int], [pure, deterministic, total, pattern(arithmetic)]),
    declare_binding(elixir, mod/3, '~w = rem(~w, ~w)',
        [int, int], [int], [pure, deterministic, total, pattern(arithmetic)]).

% ============================================================================
% COMPARISON OPERATION BINDINGS
% ============================================================================

register_comparison_bindings :-
    declare_binding(elixir, (>)/2, '~w > ~w',
        [any, any], [], [pure, deterministic, total, pattern(test)]),
    declare_binding(elixir, (<)/2, '~w < ~w',
        [any, any], [], [pure, deterministic, total, pattern(test)]),
    declare_binding(elixir, (>=)/2, '~w >= ~w',
        [any, any], [], [pure, deterministic, total, pattern(test)]),
    declare_binding(elixir, (=<)/2, '~w <= ~w',
        [any, any], [], [pure, deterministic, total, pattern(test)]),
    declare_binding(elixir, (=:=)/2, '~w === ~w',
        [any, any], [], [pure, deterministic, total, pattern(test)]),
    declare_binding(elixir, (=\=)/2, '~w !== ~w',
        [any, any], [], [pure, deterministic, total, pattern(test)]).

% ============================================================================
% TESTS
% ============================================================================

test_elixir_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Elixir Bindings Tests                 ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),
    
    init_elixir_bindings,
    format('[✓] Elixir bindings initialized~n~n', []),
    
    (   elixir_binding(string_length/2, '~w = String.length(to_string(~w))', _, _, _)
    ->  format('[✓] string_length/2 binding exists~n', [])
    ;   format('[✗] string_length/2 binding missing~n', []), fail
    ),
    
    (   elixir_binding(add/3, '~w = ~w + ~w', _, _, _)
    ->  format('[✓] add/3 binding exists~n', [])
    ;   format('[✗] add/3 binding missing~n', []), fail
    ),
    
    findall(P, elixir_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total Elixir bindings: ~w~n', [Count]).
