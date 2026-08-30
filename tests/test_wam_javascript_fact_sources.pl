:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_javascript_fact_sources.pl
%
% Lightweight file-backed P/2 facts for the JS WAM (Lua-style
% javascript_wam_fact_sources/1). CSV/TSV and JSONL are read by Node
% with fs only. Answers must match SWI with the same triples.
%
%   swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl

:- module(test_wam_javascript_fact_sources, [test_wam_javascript_fact_sources/0]).

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3]).

:- dynamic user:js_fs_edge/2.
:- dynamic user:js_fs_has/2.
:- dynamic user:js_fs_probe/0.
:- dynamic user:js_fs_inline/2.

install_fs_preds :-
    retractall(user:js_fs_edge/2),
    retractall(user:js_fs_has/2),
    retractall(user:js_fs_probe),
    retractall(user:js_fs_inline/2),
    assertz((user:js_fs_has(X, Y) :- user:js_fs_edge(X, Y))),
    assertz((user:js_fs_probe :-
        findall(Y, js_fs_edge(a, Y), L), L == [b, c],
        js_fs_edge(x, z),
        js_fs_edge(1, 2),
        \+ js_fs_edge(a, z),
        js_fs_has(a, c),
        write(ok), nl)),
    assertz(user:js_fs_inline(red, 1)),
    assertz(user:js_fs_inline(green, 2)).

write_csv_fixture(Path) :-
    setup_call_cleanup(
        open(Path, write, S),
        ( writeln(S, '# name,neighbour'),
          writeln(S, 'a,b'),
          writeln(S, 'a,c'),
          writeln(S, 'x,z'),
          writeln(S, '1,2')
        ),
        close(S)).

write_tsv_fixture(Path) :-
    setup_call_cleanup(
        open(Path, write, S),
        ( writeln(S, 'a	b'),
          writeln(S, 'a	c'),
          writeln(S, 'x	z'),
          writeln(S, '1	2')
        ),
        close(S)).

write_jsonl_fixture(Path) :-
    setup_call_cleanup(
        open(Path, write, S),
        ( writeln(S, '["a","b"]'),
          writeln(S, '{"a1":"a","a2":"c"}'),
          writeln(S, '{"args":["x","z"]}'),
          writeln(S, '[1,2]')
        ),
        close(S)).

swi_oracle :-
    findall(Y, (member(Y, [b, c])), L), L == [b, c],
    memberchk(z, [z]),
    1 =:= 1,
    \+ member(z, [b, c]).

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

read_generated_js(Dir, Text) :-
    directory_file_path(Dir, 'js', JsDir),
    directory_file_path(JsDir, 'generated_program.js', Path),
    read_file_to_string(Path, Text, []).

test_wam_javascript_fact_sources :-
    run_tests(js_wam_fact_sources).

:- begin_tests(js_wam_fact_sources).

test(csv_fact_source, [setup(install_fs_preds)]) :-
    assertion(swi_oracle),
    Dir = 'output/js_wam_fact_csv',
    make_directory_path(Dir),
    directory_file_path(Dir, 'edges.csv', Csv),
    write_csv_fixture(Csv),
    write_wam_javascript_project(
        [user:js_fs_edge/2, user:js_fs_has/2, user:js_fs_probe/0],
        [javascript_wam_fact_sources([source(js_fs_edge/2, file(Csv))])],
        Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, 'I.CallFactStream("js_fs_edge/2", 2)')),
    assertion(sub_string(Code, _, _, _, '"js_fs_edge/2": { path:')),
    assertion(\+ sub_string(Code, _, _, _, 'I.CallFactStream("js_fs_has/2"')),
    run_node_args(Dir, ['js_fs_probe/0'], ProbeExit, ProbeOut),
    assertion(ProbeExit =:= 0),
    assertion(node_succeeded(ProbeOut)),
    assertion(sub_string(ProbeOut, _, _, _, "ok")),
    run_node_args(Dir, ['js_fs_edge/2', 'a', 'b'], AbExit, AbOut),
    assertion(AbExit =:= 0),
    assertion(node_succeeded(AbOut)),
    run_node_args(Dir, ['js_fs_edge/2', 'a', 'c'], AcExit, AcOut),
    assertion(AcExit =:= 0),
    assertion(node_succeeded(AcOut)),
    run_node_args(Dir, ['js_fs_edge/2', 'a', 'z'], AzExit, AzOut),
    assertion(AzExit =:= 1),
    assertion(\+ node_succeeded(AzOut)),
    run_node_args(Dir, ['js_fs_edge/2', '1', '2'], NExit, NOut),
    assertion(NExit =:= 0),
    assertion(node_succeeded(NOut)),
    run_node_args(Dir, ['js_fs_has/2', 'a', 'c'], HasExit, HasOut),
    assertion(HasExit =:= 0),
    assertion(node_succeeded(HasOut)).

test(tsv_and_jsonl_fact_source, [setup(install_fs_preds)]) :-
    TsvDir = 'output/js_wam_fact_tsv',
    make_directory_path(TsvDir),
    directory_file_path(TsvDir, 'edges.tsv', Tsv),
    write_tsv_fixture(Tsv),
    write_wam_javascript_project(
        [user:js_fs_edge/2, user:js_fs_probe/0],
        [javascript_wam_fact_sources([source(user:js_fs_edge/2, file(Tsv))])],
        TsvDir),
    run_node_args(TsvDir, ['js_fs_probe/0'], TsvExit, TsvOut),
    assertion(TsvExit =:= 0),
    assertion(node_succeeded(TsvOut)),
    JsonDir = 'output/js_wam_fact_jsonl',
    make_directory_path(JsonDir),
    directory_file_path(JsonDir, 'edges.jsonl', Jsonl),
    write_jsonl_fixture(Jsonl),
    write_wam_javascript_project(
        [user:js_fs_edge/2, user:js_fs_probe/0],
        [js_fact_sources([source(js_fs_edge/2, file(Jsonl))])],
        JsonDir),
    read_generated_js(JsonDir, JCode),
    assertion(sub_string(JCode, _, _, _, '.jsonl')),
    run_node_args(JsonDir, ['js_fs_probe/0'], JExit, JOut),
    assertion(JExit =:= 0),
    assertion(node_succeeded(JOut)).

test(no_option_stays_inline, [setup(install_fs_preds)]) :-
    Dir = 'output/js_wam_fact_inline',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:js_fs_inline/2],
        [emit_mode(interpreter)], Dir),
    read_generated_js(Dir, Code),
    assertion(\+ sub_string(Code, _, _, _, 'I.CallFactStream("js_fs_inline/2"')),
    assertion(sub_string(Code, _, _, _, 'const fact_sources = {')),
    run_node_args(Dir, ['js_fs_inline/2', 'green'], Exit, Out),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    assertion(sub_string(Out, _, _, _, "2")).

:- end_tests(js_wam_fact_sources).
