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
    !.

test('elixir simple rule compilation') :-
    assertz((user:elix_greet(_Name, _Result) :- true)),
    elixir_target:compile_rules_to_elixir(elix_greet/2, [], Code),
    % Function name exists in generated code
    sub_atom(Code, _, _, _, 'def elix_greet('),
    % Module wrapper present
    sub_atom(Code, _, _, _, 'defmodule Generated.Elix_greet'),
    retractall(user:elix_greet(_, _)),
    !.

test('elixir mutual recursion compilation wrapper') :-
    elixir_target:compile_mutual_recursion_elixir([even/1, odd/1], [], Code),
    sub_atom(Code, _, _, _, 'MutualGroup'),
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

:- end_tests(elixir_target).
