:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_javascript_lowered.pl
%
% Tier-2 JS WAM lowered emitter: functions / mixed / interpreter emit
% modes, direct-function shape, Node vs SWI parity, determinism.
%
%   swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl

:- module(test_wam_javascript_lowered, [test_wam_javascript_lowered/0]).

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3,
               javascript_wam_resolve_emit_mode/2]).
:- use_module('../src/unifyweaver/targets/wam_javascript_lowered_emitter',
              [wam_javascript_lowerable/3,
               wam_javascript_explain_lower/3,
               js_lowered_func_name/2]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:hello/1.
:- dynamic user:probe_lowered_hello/0.
:- dynamic user:color/2.
:- dynamic user:probe_color_det/0.
:- dynamic user:age/2.
:- dynamic user:pick/2.
:- dynamic user:char_idx/4.
:- dynamic user:probe_char_idx/0.
:- dynamic user:memo_fact/1.
:- dynamic user:probe_memo/0.
:- dynamic user:wrap_tail/1.
:- dynamic user:wrap_helper/1.

install_lowered_preds :-
    retractall(user:hello/1),
    retractall(user:probe_lowered_hello),
    retractall(user:color/2),
    retractall(user:probe_color_det),
    retractall(user:age/2),
    retractall(user:pick/2),
    assertz(user:hello(world)),
    assertz((user:probe_lowered_hello :-
        hello(world), deterministic)),
    assertz(user:color(red, 1)),
    assertz(user:color(green, 2)),
    assertz(user:color(blue, 3)),
    assertz((user:probe_color_det :-
        color(green, X), write(X), nl, X == 2, deterministic)),
    assertz(user:age(alice, 30)),
    assertz(user:age(bob, 25)),
    assertz((user:pick(a, X) :- X = apple)),
    assertz((user:pick(b, X) :- X = banana)),
    retractall(user:char_idx/4),
    retractall(user:probe_char_idx/0),
    retractall(user:memo_fact/1),
    retractall(user:probe_memo/0),
    retractall(user:wrap_tail/1),
    retractall(user:wrap_helper/1),
    % T4+ITE list recursion (first_char_index/4 shape).
    assertz(user:char_idx([], _, _, -1)),
    assertz((user:char_idx([C|Cs], T, I, Index) :-
        (   C == T
        ->  Index = I
        ;   I1 is I + 1,
            char_idx(Cs, T, I1, Index)
        ))),
    assertz((user:probe_char_idx :-
        char_idx([a,b,=,c], =, 0, I), write(I), nl, I =:= 2)),
    % Ground fact: interned after first success; two callers do not alias.
    assertz(user:memo_fact(g(a, [1, 2, 3]))),
    assertz((user:probe_memo :-
        memo_fact(X), memo_fact(Y),
        X = g(a, [1, 2, 3]),
        Y = g(a, [1, 2, 3]),
        X == Y,
        write(ok), nl)),
    % Last-goal Execute of a different predicate: must stay interpreted.
    assertz(user:wrap_helper(ok)),
    assertz((user:wrap_tail(X) :- wrap_helper(X))).

read_generated_js(Dir, Text) :-
    directory_file_path(Dir, 'js', JsDir),
    directory_file_path(JsDir, 'generated_program.js', Path),
    read_file_to_string(Path, Text, []).

run_node_args(Dir, Args, Exit, Out) :-
    directory_file_path(Dir, 'js', JsDir),
    process_create(path(node), ['generated_program.js'|Args],
        [cwd(JsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(Exit)),
    atomic_list_concat([OS, ES], Out).

node_succeeded(Out) :-
    split_string(Out, "\n", " \t\r", Lines0),
    exclude([L]>>(L == ""), Lines0, Lines),
    last(Lines, Last),
    (Last == "true" ; Last == "true\n").

test_wam_javascript_lowered :-
    run_tests(js_wam_lowered_standalone).

:- begin_tests(js_wam_lowered_standalone).

test(resolve_emit_mode) :-
    javascript_wam_resolve_emit_mode([], interpreter),
    javascript_wam_resolve_emit_mode([emit_mode(functions)], functions),
    javascript_wam_resolve_emit_mode([emit_mode(mixed)], mixed),
    javascript_wam_resolve_emit_mode([emit_mode(mixed([p/1]))], mixed([p/1])).

test(func_name) :-
    js_lowered_func_name(color/2, lowered_color_2),
    js_lowered_func_name(hello/1, lowered_hello_1).

test(hello_is_deterministic, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(hello/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:hello/1, Wam, Reason),
    assertion(Reason == deterministic).

test(color_is_clause_chain, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(color/2,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:color/2, Wam, Reason),
    assertion(Reason == clause_chain).

test(functions_direct_and_swi_parity, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_standalone_fn',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:probe_lowered_hello/0,
         user:color/2, user:probe_color_det/0, user:pick/2],
        [emit_mode(functions)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_hello_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_color_2")),
    assertion(sub_string(Code, _, _, _, "T5 first-argument dispatch")),
    assertion(sub_string(Code, _, _, _, "return lowered_color_2(shared_program, state) === true")),
    run_node_args(Dir, ['hello/1', 'world'], HExit, HOut),
    assertion(HExit =:= 0),
    assertion(node_succeeded(HOut)),
    run_node_args(Dir, ['probe_lowered_hello/0'], DExit, DOut),
    assertion(DExit =:= 0),
    assertion(node_succeeded(DOut)),
    run_node_args(Dir, ['color/2', 'green'], CExit, COut),
    assertion(CExit =:= 0),
    assertion(node_succeeded(COut)),
    assertion(sub_string(COut, _, _, _, "2")),
    run_node_args(Dir, ['probe_color_det/0'], PExit, POut),
    assertion(PExit =:= 0),
    assertion(node_succeeded(POut)),
    run_node_args(Dir, ['pick/2', 'b'], PkExit, PkOut),
    assertion(PkExit =:= 0),
    assertion(node_succeeded(PkOut)),
    assertion(sub_string(PkOut, _, _, _, "banana")).

test(mixed_only_named, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_standalone_mixed',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:color/2, user:age/2, user:pick/2],
        [emit_mode(mixed([color/2]))], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_color_2")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_hello_1")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_age_2")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_pick_2")),
    run_node_args(Dir, ['color/2', 'green'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)).

test(interpreter_unchanged, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_standalone_interp',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:color/2],
        [emit_mode(interpreter)], Dir),
    read_generated_js(Dir, Code),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_")),
    assertion(sub_string(Code, _, _, _, "Runtime.run_predicate(shared_program")),
    run_node_args(Dir, ['hello/1', 'world'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)).

test(t4_ite_char_idx_node_vs_swi, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(char_idx/4,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:char_idx/4, Wam, Reason),
    assertion(Reason == multi_clause_n),
    Dir = 'output/js_wam_lowered_t4_ite',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:char_idx/4, user:probe_char_idx/0],
        [emit_mode(functions)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_char_idx_4")),
    assertion(sub_string(Code, _, _, _, "T4 all-clauses inline")),
    run_node_args(Dir, ['probe_char_idx/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "2")).

test(ground_fact_memo_independent, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(memo_fact/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:memo_fact/1, Wam, Reason),
    assertion(Reason == deterministic),
    Dir = 'output/js_wam_lowered_memo',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:memo_fact/1, user:probe_memo/0],
        [emit_mode(functions)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "ground_memo")),
    assertion(sub_string(Code, _, _, _, "Trail-safety")),
    run_node_args(Dir, ['probe_memo/0'], Exit1, Out1),
    assertion(Exit1 =:= 0),
    assertion(node_succeeded(Out1)),
    assertion(sub_string(Out1, _, _, _, "ok")),
    run_node_args(Dir, ['probe_memo/0'], Exit2, Out2),
    assertion(Exit2 =:= 0),
    assertion(node_succeeded(Out2)),
    assertion(sub_string(Out2, _, _, _, "ok")).

test(execute_other_stays_interpreted, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(wrap_tail/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_explain_lower(user:wrap_tail/1, Wam, Decision),
    assertion(Decision = fallback(_)),
    \+ wam_javascript_lowerable(user:wrap_tail/1, Wam, _).

test(mixed_auto_lowers_eligible, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_mixed_auto',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:wrap_tail/1, user:wrap_helper/1],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_hello_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_helper_1")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_wrap_tail_1")),
    assertion(sub_string(Code, _, _, _, "wamjs lower fallback: wrap_tail/1")),
    run_node_args(Dir, ['hello/1', 'world'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)).

:- end_tests(js_wam_lowered_standalone).
