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
:- dynamic user:probe_wrap_sub/0.
:- dynamic user:wrap_entry/1.
:- dynamic user:after_call/1.
:- dynamic user:probe_cont_exec/0.
:- dynamic user:probe_nested_exec/0.
:- dynamic user:call_sub/3.
:- dynamic user:probe_call_sub/0.
:- dynamic user:proj_head/2.
:- dynamic user:probe_proj_head/0.
:- dynamic user:cands/1.
:- dynamic user:cand/1.
:- dynamic user:sel/2.
:- dynamic user:need_one/1.
:- dynamic user:probe_search_cp/0.
:- dynamic user:marker_cut/1.
:- dynamic user:need_after_cut/1.
:- dynamic user:probe_neck_cut_cps/0.

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
    retractall(user:wrap_sub/3),
    retractall(user:probe_wrap_sub/0),
    retractall(user:wrap_entry/1),
    retractall(user:after_call/1),
    retractall(user:probe_cont_exec/0),
    retractall(user:probe_nested_exec/0),
    retractall(user:call_sub/3),
    retractall(user:probe_call_sub/0),
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
    % A named compound on A1 is duplicated by switch_on_structure; the
    % classifier collapses identical ground copies onto the memo path.
    assertz(user:memo_fact(g(a, [1, 2, 3]))),
    assertz((user:probe_memo :-
        memo_fact(X), memo_fact(Y),
        X = g(a, [1, 2, 3]),
        Y = g(a, [1, 2, 3]),
        X == Y,
        write(ok), nl)),
    % Last-goal Execute of a different *user* predicate: lowers via
    % emit_execute (lowered-fn tail call, or execute_user_isolated).
    assertz(user:wrap_helper(ok)),
    assertz((user:wrap_tail(X) :- wrap_helper(X))),
    % Nested Execute + Call-then-continue (corpus test 1 shape).
    assertz((user:wrap_entry(X) :- wrap_tail(X))),
    assertz((user:after_call(X) :- wrap_entry(X), X == ok)),
    assertz((user:probe_cont_exec :- after_call(X), write(X), nl)),
    assertz((user:probe_nested_exec :- wrap_entry(X), write(X), nl, X == ok)),
    % Last-goal Execute of a JS WAM builtin: lowers via op_builtin return.
    assertz((user:wrap_sub(S, N, Sub) :- sub_string(S, 0, N, _, Sub))),
    assertz((user:probe_wrap_sub :- wrap_sub("abcd", 2, S), write(S), nl, S == "ab")),
    % Call (not Execute) of sub_string/5 then continue — corpus starts_with/2 shape.
    assertz((user:call_sub(S, N, Sub) :- sub_string(S, 0, N, _, Sub), true)),
    assertz((user:probe_call_sub :- call_sub("abcd", 2, S), write(S), nl, S == "ab")),
    % Projection, not a unit ground fact: intern-once would pin A2 to the
    % first call's H (packages/2 catalog accessors).
    retractall(user:proj_head/2),
    retractall(user:probe_proj_head/0),
    assertz((user:proj_head(pair(H, _), H))),
    assertz((user:probe_proj_head :-
        proj_head(pair(a, x), X),
        proj_head(pair(b, y), Y),
        write(X), write(' '), write(Y), nl,
        X == a, Y == b)),
    % Search that must backtrack: cand enumerates via member/2. Lowering
    % sel/need_one would isolate that member with cp=0 and never reach 1.
    retractall(user:cands/1),
    retractall(user:cand/1),
    retractall(user:sel/2),
    retractall(user:need_one/1),
    retractall(user:probe_search_cp/0),
    assertz(user:cands([2, 1])),
    assertz((user:cand(X) :- cands(L), member(X, L))),
    assertz((user:sel(classic, X) :- cand(X))),
    assertz((user:need_one(X) :- sel(classic, X), X =:= 1)),
    assertz((user:probe_search_cp :- need_one(X), write(X), nl, X =:= 1)),
    % A lowered helper with a neck cut (satisfies/2 shape) must not wipe
    % the caller's member/2 choice points. state.cps = [] used to make
    % need_after_cut fail even though 1 is in the list.
    retractall(user:marker_cut/1),
    retractall(user:need_after_cut/1),
    retractall(user:probe_neck_cut_cps/0),
    assertz((user:marker_cut(X) :- X = X, !)),
    assertz((user:need_after_cut(X) :- cands(L), member(X, L), marker_cut(X), X =:= 1)),
    assertz((user:probe_neck_cut_cps :- need_after_cut(X), write(X), nl, X =:= 1)).

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
    assertion((sub_string(Code, _, _, _, "T4 nil/cons dispatch")
               ; sub_string(Code, _, _, _, "T4 all-clauses inline"))),
    assertion(sub_string(Code, _, _, _, "Runtime.op_get_list")),
    assertion(sub_string(Code, _, _, _, "Runtime.op_builtin")),
    assertion((sub_string(Code, _, _, _, "capture_a_regs")
               ; sub_string(Code, _, _, _, "snapshot_lite"))),
    assertion(sub_string(Code, _, _, _, "term_is_nil")),
    assertion(sub_string(Code, _, _, _,
                         "if (lowered_char_idx_4(program, state) !== true)")),
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

test(execute_builtin_lowers, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(wrap_sub/3,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:wrap_sub/3, Wam, Reason),
    assertion(Reason == deterministic),
    Dir = 'output/js_wam_lowered_exec_builtin',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:wrap_sub/3, user:probe_wrap_sub/0],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_sub_3")),
    assertion(sub_string(Code, _, _, _, "op_builtin")),
    run_node_args(Dir, ['probe_wrap_sub/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "ab")).

test(call_builtin_op_builtin, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_call_builtin',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:call_sub/3, user:probe_call_sub/0],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_call_sub_3")),
    assertion(sub_string(Code, _, _, _, "op_builtin")),
    assertion(\+ sub_string(Code, _, _, _, 'lowered_dispatch["sub_string/5"]')),
    run_node_args(Dir, ['probe_call_sub/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "ab")).

test(mixed_auto_lowers_eligible, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_mixed_auto',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:hello/1, user:wrap_tail/1, user:wrap_helper/1, user:wrap_sub/3],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_hello_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_helper_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_tail_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_sub_3")),
    assertion(\+ sub_string(Code, _, _, _, "wamjs lower fallback: wrap_sub/3")),
    run_node_args(Dir, ['hello/1', 'world'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)).

test(execute_of_user_lowers, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(wrap_tail/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:wrap_tail/1, Wam, Reason),
    assertion((Reason == deterministic ; Reason == multi_clause_n)).

test(execute_user_continuation_integrity, [setup(install_lowered_preds)]) :-
    % Nested Execute (wrap_entry → wrap_tail → wrap_helper) plus
    % Call-then-continue (after_call). This is the shape that broke
    % corpus test 1 when Execute set cp=0 without restoring it.
    Dir = 'output/js_wam_lowered_exec_user',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:wrap_helper/1, user:wrap_tail/1, user:wrap_entry/1,
         user:after_call/1, user:probe_cont_exec/0, user:probe_nested_exec/0],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_tail_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_entry_1")),
    assertion(sub_string(Code, _, _, _, "function lowered_after_call_1")),
    assertion(sub_string(Code, _, _, _,
                         "return lowered_wrap_tail_1(program, state) === true")
              ; sub_string(Code, _, _, _, "execute_user_isolated")
              ; sub_string(Code, _, _, _, "return _lf(program, state) === true")),
    run_node_args(Dir, ['probe_nested_exec/0'], NExit, NOut),
    assertion(NExit =:= 0),
    assertion(node_succeeded(NOut)),
    assertion(sub_string(NOut, _, _, _, "ok")),
    run_node_args(Dir, ['probe_cont_exec/0'], CExit, COut),
    assertion(CExit =:= 0),
    assertion(node_succeeded(COut)),
    assertion(sub_string(COut, _, _, _, "ok")).

test(execute_user_interpreted_callee, [setup(install_lowered_preds)]) :-
    % wrap_helper stays interpreted; wrap_tail lowers and Execute must
    % isolate without stealing the Call-then-continue CP.
    Dir = 'output/js_wam_lowered_exec_isolated',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:wrap_helper/1, user:wrap_tail/1, user:wrap_entry/1,
         user:after_call/1, user:probe_cont_exec/0],
        [emit_mode(mixed([wrap_tail/1, wrap_entry/1, after_call/1, probe_cont_exec/0]))], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_wrap_tail_1")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_wrap_helper_1")),
    assertion(sub_string(Code, _, _, _, "execute_user_isolated")),
    run_node_args(Dir, ['probe_cont_exec/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "ok")).

test(projection_not_ground_fact_memo, [setup(install_lowered_preds)]) :-
    compile_predicate_to_wam_text(proj_head/2,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_lowerable(user:proj_head/2, Wam, Reason),
    assertion(Reason == deterministic),
    Dir = 'output/js_wam_lowered_proj_head',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:proj_head/2, user:probe_proj_head/0],
        [emit_mode(functions)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_proj_head_2")),
    assertion(\+ sub_string(Code, _, _, _,
                            "Lowered: proj_head/2 (deterministic ground fact")),
    run_node_args(Dir, ['probe_proj_head/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "a b")).

test(t4_does_not_steal_search_cps, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_search_cp',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:cands/1, user:cand/1, user:sel/2, user:need_one/1, user:probe_search_cp/0],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_sel_2")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_need_one_1")),
    assertion(sub_string(Code, _, _, _,
                         "naked member/2 (or callee) needs interpreter choice points")),
    run_node_args(Dir, ['probe_search_cp/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "1")).

test(neck_cut_does_not_steal_caller_cps, [setup(install_lowered_preds)]) :-
    Dir = 'output/js_wam_lowered_neck_cut',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:cands/1, user:marker_cut/1, user:need_after_cut/1,
         user:probe_neck_cut_cps/0],
        [emit_mode(mixed)], Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, "function lowered_marker_cut_1")),
    assertion(\+ sub_string(Code, _, _, _, "function lowered_need_after_cut_1")),
    run_node_args(Dir, ['probe_neck_cut_cps/0'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "1")).

:- end_tests(js_wam_lowered_standalone).
