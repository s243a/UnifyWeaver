% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_template_sections.pl - Tests for the mustache-section additions
% to template_system.pl ({{#flag}}/{{/flag}} and {{^flag}}/{{/flag}}).
%
% Standalone runner: each test gets its own scope via copy_term so the
% variables don't leak between cases.
%
% Usage:
%   swipl -g run_tests -t halt tests/core/test_template_sections.pl

:- use_module('../../src/unifyweaver/core/template_system').

:- dynamic pass_count/1, fail_count/1.

run_test(Name, Goal) :-
    format("Test ~w: ", [Name]),
    copy_term(Goal, FreshGoal),
    (   catch(call(FreshGoal), E,
              (format("ERROR ~w~n", [E]), fail))
    ->  writeln("PASS"), bump_pass
    ;   writeln("FAIL"), bump_fail
    ).

bump_pass :-
    retract(pass_count(N)),
    N1 is N + 1,
    assertz(pass_count(N1)).

bump_fail :-
    retract(fail_count(N)),
    N1 is N + 1,
    assertz(fail_count(N1)).

reset_counters :-
    retractall(pass_count(_)),
    retractall(fail_count(_)),
    assertz(pass_count(0)),
    assertz(fail_count(0)).

run_tests :-
    reset_counters,
    writeln('=== Template Engine: Section Tests ==='),

    % --- Section blocks (truthy) ---
    run_test('section_truthy_keeps_body', (
        render_template('A{{#flag}}B{{/flag}}C', [flag=true], R),
        sub_string(R, _, _, _, 'ABC')
    )),
    run_test('section_falsy_strips_body', (
        render_template('A{{#flag}}B{{/flag}}C', [flag=false], R),
        sub_string(R, _, _, _, 'AC'),
        \+ sub_string(R, _, _, _, 'B')
    )),
    run_test('section_missing_key_strips_body', (
        render_template('A{{#absent}}B{{/absent}}C', [], R),
        sub_string(R, _, _, _, 'AC'),
        \+ sub_string(R, _, _, _, 'B')
    )),
    run_test('section_empty_string_is_falsy', (
        render_template('A{{#flag}}B{{/flag}}C', [flag=""], R),
        sub_string(R, _, _, _, 'AC'),
        \+ sub_string(R, _, _, _, 'B')
    )),
    run_test('section_zero_is_falsy', (
        render_template('A{{#flag}}B{{/flag}}C', [flag=0], R),
        sub_string(R, _, _, _, 'AC'),
        \+ sub_string(R, _, _, _, 'B')
    )),
    run_test('section_empty_list_is_falsy', (
        render_template('A{{#flag}}B{{/flag}}C', [flag=[]], R),
        sub_string(R, _, _, _, 'AC'),
        \+ sub_string(R, _, _, _, 'B')
    )),

    % --- Inverted sections ---
    run_test('inverted_falsy_keeps_body', (
        render_template('A{{^flag}}B{{/flag}}C', [flag=false], R),
        sub_string(R, _, _, _, 'ABC')
    )),
    run_test('inverted_truthy_strips_body', (
        render_template('A{{^flag}}B{{/flag}}C', [flag=true], R),
        sub_string(R, _, _, _, 'AC'),
        \+ sub_string(R, _, _, _, 'B')
    )),
    run_test('inverted_missing_key_keeps_body', (
        render_template('A{{^absent}}B{{/absent}}C', [], R),
        sub_string(R, _, _, _, 'ABC')
    )),

    % --- Substitutions inside sections ---
    run_test('substitution_inside_truthy_section', (
        render_template('Hello {{#greet}}{{name}}{{/greet}}!', [greet=true, name='World'], R),
        sub_string(R, _, _, _, 'Hello World!')
    )),
    run_test('substitution_inside_falsy_section_dropped', (
        render_template('Hello {{#greet}}{{name}}{{/greet}}!', [greet=false, name='World'], R),
        sub_string(R, _, _, _, 'Hello !'),
        \+ sub_string(R, _, _, _, 'World')
    )),

    % --- Nested differently-named sections ---
    run_test('nested_both_truthy', (
        render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=true, b=true], R),
        sub_string(R, _, _, _, '[AB]')
    )),
    run_test('nested_outer_truthy_inner_falsy', (
        render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=true, b=false], R),
        sub_string(R, _, _, _, '[A]'),
        \+ sub_string(R, _, _, _, 'B')
    )),
    run_test('nested_outer_falsy', (
        render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=false, b=true], R),
        sub_string(R, _, _, _, '[]'),
        \+ sub_string(R, _, _, _, 'A'),
        \+ sub_string(R, _, _, _, 'B')
    )),

    % --- Backward compatibility ---
    run_test('plain_substitution_still_works', (
        render_template('plain {{x}} plain {{y}} plain', [x='X', y='Y'], R),
        sub_string(R, _, _, _, 'plain X plain Y plain')
    )),
    run_test('template_with_no_template_syntax_passes_through', (
        render_template('just literal text', [unused=true], R),
        sub_string(R, _, _, _, 'just literal text')
    )),

    % --- Edge cases ---
    run_test('only_section_truthy', (
        render_template('{{#k}}hi{{/k}}', [k=true], R),
        sub_string(R, _, _, _, 'hi')
    )),
    run_test('only_section_falsy_yields_empty', (
        render_template('{{#k}}hi{{/k}}', [k=false], R),
        atom_string(R, "") ; R == ""
    )),

    pass_count(P), fail_count(F),
    format("~n=== ~w passed, ~w failed ===~n", [P, F]),
    (   F =:= 0 -> true ; halt(1) ).
