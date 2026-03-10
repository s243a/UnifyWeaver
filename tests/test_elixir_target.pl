:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_elixir_target.pl - Unit tests for the Elixir target

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/elixir_target').
:- use_module('../src/unifyweaver/bindings/elixir_bindings').

:- begin_tests(elixir_target).

% Setup facts
:- dynamic user:myparent/2.
user:myparent(tom, bob).
user:myparent(bob, jim).

test('elixir target info loading') :-
    elixir_target:target_info(Info),
    Info.name == "Elixir",
    Info.family == beam.

test('elixir facts export') :-
    elixir_target:compile_facts_to_elixir(myparent, 2, Code),
    sub_atom(Code, _, _, _, '@facts'),
    sub_atom(Code, _, _, _, '{"tom", "bob"}'),
    sub_atom(Code, _, _, _, 'def stream, do: Stream.map(@facts'),
    % Module name should be CamelCase: Myparent
    sub_atom(Code, _, _, _, 'defmodule Generated.Myparent'),
    !.

test('snake_to_camel conversion') :-
    elixir_target:snake_to_camel(elix_greet, 'ElixGreet'),
    elixir_target:snake_to_camel(my_parent, 'MyParent'),
    elixir_target:snake_to_camel(hello, 'Hello'),
    !.

test('elixir simple rule compilation') :-
    assertz((user:elix_greet(_Name, _Result) :- true)),
    elixir_target:compile_rules_to_elixir(elix_greet/2, [], Code),
    % Function name exists in generated code
    sub_atom(Code, _, _, _, 'def elix_greet('),
    % Module name should be CamelCase
    sub_atom(Code, _, _, _, 'defmodule Generated.ElixGreet'),
    retractall(user:elix_greet(_, _)),
    !.

test('elixir simple rule with ground args') :-
    assertz((user:elix_check(hello, _X) :- true)),
    assertz((user:elix_check(world, _Y) :- true)),
    elixir_target:compile_rules_to_elixir(elix_check/2, [], Code),
    % Should have pattern-matched "hello" literal in function head
    sub_atom(Code, _, _, _, '"hello"'),
    sub_atom(Code, _, _, _, '"world"'),
    retractall(user:elix_check(_, _)),
    !.

test('elixir mutual recursion compilation') :-
    assertz((user:elix_even(0) :- true)),
    assertz((user:elix_odd(1) :- true)),
    elixir_target:compile_mutual_recursion_elixir([elix_even/1, elix_odd/1], [], Code),
    sub_atom(Code, _, _, _, 'MutualGroup'),
    sub_atom(Code, _, _, _, 'def elix_even('),
    sub_atom(Code, _, _, _, 'def elix_odd('),
    retractall(user:elix_even(_)),
    retractall(user:elix_odd(_)),
    !.

test('elixir pipeline output') :-
    assertz((user:elix_proc(_In, _Out) :- true)),
    elixir_target:compile_predicate_to_elixir(elix_proc/2, [pipeline_input(true)], Code),
    sub_atom(Code, _, _, _, 'Jason.decode!'),
    sub_atom(Code, _, _, _, '|> Stream.reject(&is_nil/1)'),
    retractall(user:elix_proc(_, _)),
    !.

test('elixir generator output') :-
    assertz((user:elix_gen(_In, _Out) :- true)),
    elixir_target:compile_predicate_to_elixir(elix_gen/2, [generator_mode(true)], Code),
    sub_atom(Code, _, _, _, 'def generate(init_record) do'),
    sub_atom(Code, _, _, _, 'Stream.unfold('),
    retractall(user:elix_gen(_, _)),
    !.

test('elixir bindings initialization') :-
    elixir_bindings:init_elixir_bindings.

% --- Negative / edge-case tests ---

test('elixir empty predicate generates fallback') :-
    % A predicate with no clauses should still produce valid code
    elixir_target:compile_simple_mode_elixir(no_clauses, 1, [], Code),
    sub_atom(Code, _, _, _, 'def process(_), do: nil'),
    sub_atom(Code, _, _, _, 'defmodule Generated.NoClauses'),
    !.

test('snake_to_camel single char') :-
    elixir_target:snake_to_camel(x, 'X'),
    !.

test('snake_to_camel multiple underscores') :-
    elixir_target:snake_to_camel(a_b_c_d, 'ABCD'),
    !.

test('elixir mutual recursion empty predicate list') :-
    elixir_target:compile_mutual_recursion_elixir([], [], Code),
    sub_atom(Code, _, _, _, 'No predicates found'),
    !.

:- end_tests(elixir_target).
