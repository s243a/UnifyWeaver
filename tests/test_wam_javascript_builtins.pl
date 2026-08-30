:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_javascript_builtins.pl
%
% Compiles JS WAM probes (findall, functor, arg, =.., copy_term, \+,
% call, bagof, setof, aggregate_all) plus the shared classic-program
% fixtures, runs them under Node, and checks SWI-Prolog answers.
%
%   swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl

:- module(test_wam_javascript_builtins, [test_wam_javascript_builtins/0]).

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3]).
:- use_module('wam_conformance_fixtures',
              [conformance_program/2, conformance_query/4]).

:- dynamic user:probe_findall/0.
:- dynamic user:probe_functor/0.
:- dynamic user:probe_arg/0.
:- dynamic user:probe_univ/0.
:- dynamic user:probe_copy_term/0.
:- dynamic user:probe_naf/0.
:- dynamic user:probe_call/0.
:- dynamic user:probe_bagof/0.
:- dynamic user:probe_setof/0.
:- dynamic user:probe_bagof_group/0.
:- dynamic user:probe_bagof_anon/0.
:- dynamic user:probe_bagof_exist/0.
:- dynamic user:probe_setof_exist/0.
:- dynamic user:probe_bagof_empty/0.
:- dynamic user:probe_setof_mixed/0.
:- dynamic user:probe_setof_group/0.
:- dynamic user:age/2.
:- dynamic user:probe_agg_count/0.
:- dynamic user:probe_agg_sum/0.
:- dynamic user:probe_agg_bag/0.
:- dynamic user:probe_agg_set/0.

install_probes :-
    retractall(user:probe_findall),
    retractall(user:probe_functor),
    retractall(user:probe_arg),
    retractall(user:probe_univ),
    retractall(user:probe_copy_term),
    retractall(user:probe_naf),
    retractall(user:probe_call),
    retractall(user:probe_bagof),
    retractall(user:probe_setof),
    retractall(user:probe_bagof_group),
    retractall(user:probe_bagof_anon),
    retractall(user:probe_bagof_exist),
    retractall(user:probe_setof_exist),
    retractall(user:probe_bagof_empty),
    retractall(user:probe_setof_mixed),
    retractall(user:probe_setof_group),
    retractall(user:age/2),
    retractall(user:probe_agg_count),
    retractall(user:probe_agg_sum),
    retractall(user:probe_agg_bag),
    retractall(user:probe_agg_set),
    assertz((user:probe_findall :-
        findall(X, member(X, [1,2,3]), L), write(L), nl, L == [1,2,3])),
    assertz((user:probe_functor :-
        functor(foo(a,b), N, A), write(N), write(' '), write(A), nl,
        N == foo, A == 2)),
    assertz((user:probe_arg :-
        arg(2, foo(a,b), X), write(X), nl, X == b)),
    assertz((user:probe_univ :-
        foo(a,b) =.. L, write(L), nl, L == [foo,a,b])),
    assertz((user:probe_copy_term :-
        copy_term(f(X,X), C), write(C), nl,
        C = f(Y,Y), X \== Y)),
    assertz((user:probe_naf :-
        \+ member(9, [1,2,3]))),
    assertz((user:probe_call :-
        call(member(2, [1,2,3])))),
    assertz(user:age(alice, 30)),
    assertz(user:age(bob, 25)),
    assertz(user:age(carol, 30)),
    assertz((user:probe_bagof :-
        bagof(X, member(X, [1,2,1]), L), write(L), nl, L == [1,2,1])),
    assertz((user:probe_setof :-
        setof(X, member(X, [1,2,1]), L), write(L), nl, L == [1,2])),
    assertz((user:probe_bagof_group :-
        findall(N-Cs, bagof(C, age(N, C), Cs), All),
        write(All), nl,
        All == [alice-[30], bob-[25], carol-[30]])),
    assertz((user:probe_bagof_anon :-
        findall(Cs, bagof(C, age(_, C), Cs), All),
        write(All), nl,
        All == [[30], [25], [30]])),
    assertz((user:probe_bagof_exist :-
        bagof(C, N^age(N, C), Cs), write(Cs), nl, Cs == [30, 25, 30])),
    assertz((user:probe_setof_exist :-
        setof(C, N^age(N, C), Cs), write(Cs), nl, Cs == [25, 30])),
    assertz((user:probe_bagof_empty :-
        \+ bagof(x, fail, _))),
    assertz((user:probe_setof_mixed :-
        setof(X, member(X, [foo, 1, bar, 1, z(a), 3.5]), L),
        write(L), nl,
        L == [1, 3.5, bar, foo, z(a)])),
    assertz((user:probe_setof_group :-
        findall(N-Cs, setof(C, age(N, C), Cs), All),
        write(All), nl,
        All == [alice-[30], bob-[25], carol-[30]])),
    assertz((user:probe_agg_count :-
        aggregate_all(count, member(_, [1,2,3]), N), write(N), nl, N == 3)),
    assertz((user:probe_agg_sum :-
        aggregate_all(sum(X), member(X, [1,2,3]), N), write(N), nl, N == 6)),
    assertz((user:probe_agg_bag :-
        aggregate_all(bag(X), member(X, [1,2,1]), L), write(L), nl, L == [1,2,1])),
    assertz((user:probe_agg_set :-
        aggregate_all(set(X), member(X, [1,2,1]), L), write(L), nl,
        (L == [1,2] ; L == [2,1]))).

probe_preds([
    user:probe_findall/0,
    user:probe_functor/0,
    user:probe_arg/0,
    user:probe_univ/0,
    user:probe_copy_term/0,
    user:probe_naf/0,
    user:probe_call/0,
    user:probe_bagof/0,
    user:probe_setof/0,
    user:probe_bagof_group/0,
    user:probe_bagof_anon/0,
    user:probe_bagof_exist/0,
    user:probe_setof_exist/0,
    user:probe_bagof_empty/0,
    user:probe_setof_mixed/0,
    user:probe_setof_group/0,
    user:age/2,
    user:probe_agg_count/0,
    user:probe_agg_sum/0,
    user:probe_agg_bag/0,
    user:probe_agg_set/0
]).

:- dynamic user:ctw_js/0.

install_conformance_wrappers(Map) :-
    retractall(user:ctw_js),
    findall(q(K,A,E), conformance_query(_, K, A, E), Queries),
    synth(Queries, 1, [], Map).

synth([], _, Acc, Map) :- reverse(Acc, Map).
synth([q(K,A,E)|Rest], I, Acc, Map) :-
    atomic_list_concat([ctw_js_, I], WName),
    atomic_list_concat([PredName|_], '/', K),
    atom_string(Pred, PredName),
    Goal =.. [Pred|A],
    assertz(user:(WName :- Goal)),
    I1 is I + 1,
    synth(Rest, I1, [ctw(K,A,E,WName)|Acc], Map).

conformance_preds(Preds) :-
    findall(P, (conformance_program(_, Ps), member(P, Ps)), Preds).

run_node(Dir, Key, Exit, Out) :-
    directory_file_path(Dir, 'js', JsDir),
    process_create(path(node), ['generated_program.js', Key],
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

compile_js_project(Preds, Dir) :-
    Dir = 'output/js_wam_builtin_probes',
    make_directory_path(Dir),
    write_wam_javascript_project(Preds, [emit_mode(interpreter)], Dir).

test_wam_javascript_builtins :-
    run_tests(js_wam_builtins).

:- begin_tests(js_wam_builtins).

test(compile_and_probes, [setup(install_probes)]) :-
    probe_preds(Probes),
    conformance_preds(CPreds),
    install_conformance_wrappers(Map),
    findall(W/0, member(ctw(_,_,_,W), Map), Wrappers),
    append([Probes, CPreds, Wrappers], All),
    compile_js_project(All, Dir),
    forall(member(user:Name/0, Probes),
           (   format(atom(Key), '~w/0', [Name]),
               atom_string(Key, KeyStr),
               run_node(Dir, KeyStr, Exit, Out),
               format('PROBE ~w exit=~w~n~w~n', [Name, Exit, Out]),
               assertion(Exit =:= 0),
               assertion(node_succeeded(Out))
           )),
    forall(member(ctw(K,A,E,W), Map),
           (   format(atom(Key), '~w/0', [W]),
               atom_string(Key, KeyStr),
               run_node(Dir, KeyStr, _Exit, Out),
               (   E == true
               ->  (   node_succeeded(Out)
                   ->  true
                   ;   format('CONFORMANCE FAIL ~w ~w expected true got ~w~n',
                              [K, A, Out]),
                       fail
                   )
               ;   (   node_succeeded(Out)
                   ->  format('CONFORMANCE FAIL ~w ~w expected false got ~w~n',
                              [K, A, Out]),
                       fail
                   ;   true
                   )
               )
           )).

:- end_tests(js_wam_builtins).
