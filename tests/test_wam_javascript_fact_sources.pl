:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_javascript_fact_sources.pl
%
% Lightweight file-backed P/2 facts for the JS WAM (Lua-style
% javascript_wam_fact_sources/1). CSV/TSV and JSONL are read by Node
% with fs only (D27 file(Path) is unchanged). GP-LMDB adds:
%   source(P/2, indexed(Prefix))  — backend B, Prefix.data + Prefix.idx
%   source(P/2, lmdb(Dir))        — backend A, opt-in npm `lmdb`
% Answers must match SWI with the same triples.
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
:- dynamic user:js_idx_edge/2.
:- dynamic user:js_idx_probe_bound/0.
:- dynamic user:js_idx_probe_unbound_len/0.
:- dynamic user:js_idx_probe_filter/0.
:- dynamic user:js_idx_probe_types/0.
:- dynamic user:js_idx_probe_shared/0.
:- dynamic user:js_lmdb_edge/2.
:- dynamic user:js_lmdb_probe_filter/0.
:- dynamic user:js_lmdb_probe_missing/0.

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

n_xs(N, Atom) :-
    length(Cs, N),
    maplist(=(0'x), Cs),
    atom_codes(Atom, Cs).

idx_pad(Pad) :-
    n_xs(180, Pad).

install_idx_preds :-
    retractall(user:js_idx_edge(_, _)),
    retractall(user:js_idx_probe_bound),
    retractall(user:js_idx_probe_unbound_len),
    retractall(user:js_idx_probe_filter),
    retractall(user:js_idx_probe_types),
    retractall(user:js_idx_probe_shared),
    retractall(user:js_lmdb_edge(_, _)),
    retractall(user:js_lmdb_probe_filter),
    retractall(user:js_lmdb_probe_missing),
    idx_pad(Pad),
    forall(between(0, 4999, I),
           (   format(atom(K), 'k~d', [I]),
               format(atom(V), 'v~d_~w', [I, Pad]),
               assertz(user:js_idx_edge(K, V))
           )),
    assertz(user:js_idx_edge(probekey, alpha)),
    assertz(user:js_idx_edge(probekey, beta)),
    assertz(user:js_idx_edge(42, 99)),
    assertz(user:js_idx_edge("strkey", atomval)),
    assertz((user:js_idx_probe_bound :-
        findall(Y, js_idx_edge(k2500, Y), L),
        write(bound), write(L), nl,
        L = [One], atom_concat(v2500, _, One))),
    assertz((user:js_idx_probe_unbound_len :-
        findall(1, js_idx_edge(_, _), L),
        length(L, N),
        write(unbound_len), write(N), nl,
        N =:= 5004)),
    assertz((user:js_idx_probe_filter :-
        js_idx_edge(probekey, alpha),
        \+ js_idx_edge(probekey, zzz),
        findall(Y, js_idx_edge(probekey, Y), L),
        L == [alpha, beta],
        write(ok), nl)),
    assertz((user:js_idx_probe_types :-
        js_idx_edge(42, 99),
        js_idx_edge("strkey", atomval),
        write(ok), nl)),
    assertz((user:js_idx_probe_shared :-
        findall(K-V, js_idx_edge(K, V), All),
        length(All, 5004),
        findall(Y, js_idx_edge(probekey, Y), L), L == [alpha, beta],
        js_idx_edge(42, 99),
        js_idx_edge("strkey", atomval),
        write(shared_ok), nl)),
    assertz((user:js_lmdb_edge(X, Y) :- user:js_idx_edge(X, Y))),
    assertz((user:js_lmdb_probe_filter :-
        js_lmdb_edge(probekey, alpha),
        \+ js_lmdb_edge(probekey, zzz),
        findall(Y, js_lmdb_edge(probekey, Y), L),
        L == [alpha, beta],
        js_lmdb_edge(42, 99),
        js_lmdb_edge("strkey", atomval),
        write(ok), nl)),
    assertz((user:js_lmdb_probe_missing :- js_lmdb_edge(probekey, alpha))).

write_idx_tsv(Path) :-
    idx_pad(Pad),
    setup_call_cleanup(
        open(Path, write, S),
        (   forall(between(0, 4999, I),
                   format(S, 'k~d\tv~d_~w~n', [I, I, Pad])),
            writeln(S, 'probekey	alpha'),
            writeln(S, 'probekey	beta'),
            writeln(S, '42	99'),
            writeln(S, '"strkey"	atomval')
        ),
        close(S)).

uw_script(Name, Abs) :-
    atom_concat('scripts/js_wam/', Name, Rel),
    absolute_file_name(Rel, Abs, [access(read)]).

run_node_cwd(Dir, Args, EnvPairs, Exit, Out, Err) :-
    directory_file_path(Dir, 'js', JsDir),
    maplist(env_assign, EnvPairs, Assigns),
    atomic_list_concat(Assigns, ' ', Prefix),
    quote_node_args(Args, Quoted),
    atomic_list_concat(Quoted, ' ', ArgStr),
    (   Prefix == ""
    ->  format(string(Cmd), 'node ~w', [ArgStr])
    ;   format(string(Cmd), '~w node ~w', [Prefix, ArgStr])
    ),
    process_create(path(bash), ['-lc', Cmd],
        [cwd(JsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(Exit)),
    Out = OS, Err = ES.

env_assign(Name=Value, Assign) :-
    format(atom(Assign), '~w=~w', [Name, Value]).

quote_node_args([], []).
quote_node_args([A|Rest], [Q|QRest]) :-
    format(atom(Q), "'~w'", [A]),
    quote_node_args(Rest, QRest).

run_builder(Script, Args, Exit, Out) :-
    process_create(path(node), [Script|Args],
        [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(Exit)),
    atomic_list_concat([OS, ES], Out).

lmdb_pkg_prefix('/tmp/uw-lmdb-pkg').

lmdb_available :-
    catch(ensure_lmdb_pkg, _, fail),
    lmdb_pkg_prefix(P),
    directory_file_path(P, 'node_modules', NM),
    format(atom(Cmd), "NODE_PATH='~w' node -e 'require(\"lmdb\")'", [NM]),
    process_create(path(bash), ['-lc', Cmd],
        [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, _),
    close(O), close(E),
    process_wait(Pid, exit(0)).

ensure_lmdb_pkg :-
    lmdb_pkg_prefix(P),
    make_directory_path(P),
    directory_file_path(P, 'node_modules/lmdb', Mod),
    (   exists_directory(Mod)
    ->  true
    ;   process_create(path(npm), ['install', '--prefix', P, 'lmdb'],
            [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
        read_string(O, _, _), read_string(E, _, ES),
        close(O), close(E),
        process_wait(Pid, exit(Code)),
        (Code =:= 0 -> true ; throw(error(lmdb_npm_install(Code, ES), _)))
    ).

parse_fact_io_stats(Err, Bytes, DataSize) :-
    split_string(Err, "\n", " \t\r", Lines),
    member(Line, Lines),
    sub_string(Line, _, _, _, "fact_io bytes_read="),
    split_string(Line, " ", "", Parts),
    member(BPart, Parts), sub_string(BPart, 0, _, _, "bytes_read="),
    sub_string(BPart, 11, _, 0, BStr), number_string(Bytes, BStr),
    member(DPart, Parts), sub_string(DPart, 0, _, _, "data_size="),
    sub_string(DPart, 10, _, 0, DStr), number_string(DataSize, DStr),
    !.

test(indexed_store_bound_unbound_filter, [setup(install_idx_preds)]) :-
    Dir = 'output/js_wam_fact_indexed',
    make_directory_path(Dir),
    directory_file_path(Dir, 'edges.tsv', Tsv),
    directory_file_path(Dir, 'edges_store', Store),
    write_idx_tsv(Tsv),
    uw_script('uw_fact_index.js', Script),
    run_builder(Script, ['build', Tsv, Store], BExit, BOut),
    assertion(BExit =:= 0),
    assertion(sub_string(BOut, _, _, _, "uw_fact_index")),
    write_wam_javascript_project(
        [user:js_idx_edge/2, user:js_idx_probe_bound/0,
         user:js_idx_probe_unbound_len/0, user:js_idx_probe_filter/0,
         user:js_idx_probe_types/0, user:js_idx_probe_shared/0],
        [javascript_wam_fact_sources([source(js_idx_edge/2, indexed(Store))])],
        Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, 'kind: "indexed"')),
    assertion(sub_string(Code, _, _, _, 'I.CallFactStream("js_idx_edge/2", 2)')),
    run_node_args(Dir, ['js_idx_probe_bound/0'], BoundExit, BoundOut),
    assertion(BoundExit =:= 0),
    assertion(node_succeeded(BoundOut)),
    assertion(sub_string(BoundOut, _, _, _, "v2500")),
    run_node_args(Dir, ['js_idx_probe_unbound_len/0'], UExit, UOut),
    assertion(UExit =:= 0),
    assertion(node_succeeded(UOut)),
    assertion(sub_string(UOut, _, _, _, "unbound_len5004")),
    run_node_args(Dir, ['js_idx_probe_filter/0'], FExit, FOut),
    assertion(FExit =:= 0),
    assertion(node_succeeded(FOut)),
    assertion(sub_string(FOut, _, _, _, "ok")),
    run_node_args(Dir, ['js_idx_probe_types/0'], TExit, TOut),
    assertion(TExit =:= 0),
    assertion(node_succeeded(TOut)),
    findall(Y, user:js_idx_edge(k2500, Y), SWIBound),
    assertion(SWIBound = [_]),
    findall(1, user:js_idx_edge(_, _), SWIAll),
    length(SWIAll, 5004),
    findall(Y, user:js_idx_edge(probekey, Y), SWIFilt),
    assertion(SWIFilt == [alpha, beta]).

test(indexed_store_bytes_read_proof, [setup(install_idx_preds)]) :-
    Dir = 'output/js_wam_fact_indexed',
    directory_file_path(Dir, 'edges_store', Store),
    write_wam_javascript_project(
        [user:js_idx_edge/2, user:js_idx_probe_bound/0],
        [javascript_wam_fact_sources([source(js_idx_edge/2, indexed(Store))])],
        Dir),
    run_node_cwd(Dir, ['generated_program.js', 'js_idx_probe_bound/0'],
                 ['UW_FACT_IO_STATS'='1'], Exit, Out, Err),
    assertion(Exit =:= 0),
    assertion(node_succeeded(Out)),
    parse_fact_io_stats(Err, Bytes, DataSize),
    assertion(DataSize > 100000),
    assertion(Bytes > 200),
    assertion(Bytes * 20 < DataSize),
    assertion(Bytes < 16384),
    format(user_error, '~n[bytes-read proof] bytes_read=~w data_size=~w~n[bytes-read proof raw stderr]~n~w~n',
           [Bytes, DataSize, Err]).

test(lmdb_missing_package_is_loud, [setup(install_idx_preds)]) :-
    Dir = 'output/js_wam_fact_lmdb_missing',
    make_directory_path(Dir),
    write_wam_javascript_project(
        [user:js_lmdb_edge/2, user:js_lmdb_probe_missing/0],
        [javascript_wam_fact_sources([source(js_lmdb_edge/2, lmdb('/tmp/uw-no-such-lmdb'))])],
        Dir),
    read_generated_js(Dir, Code),
    assertion(sub_string(Code, _, _, _, 'kind: "lmdb"')),
    run_node_cwd(Dir, ['generated_program.js', 'js_lmdb_probe_missing/0'],
                 ['UW_LMDB_FORCE_MISSING'='1'], Exit, Out, Err),
    assertion(Exit =\= 0),
    atomic_list_concat([Out, Err], Combined),
    assertion(sub_string(Combined, _, _, _, "npm install lmdb")),
    assertion(sub_string(Combined, _, _, _, "not used as a fallback")),
    assertion(sub_string(Combined, _, _, _, "lmdb(")).

run_lmdb_builder(Tsv, LmdbDir, Exit, Combined) :-
    uw_script('uw_fact_lmdb.js', Script),
    lmdb_pkg_prefix(P),
    directory_file_path(P, 'node_modules', NM),
    format(atom(Cmd), "NODE_PATH='~w' node '~w' build '~w' '~w'",
           [NM, Script, Tsv, LmdbDir]),
    process_create(path(bash), ['-lc', Cmd],
        [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS), read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(Exit)),
    atomic_list_concat([OS, ES], Combined).

test(lmdb_store_when_available,
     [setup(install_idx_preds), condition(lmdb_available)]) :-
    Dir = 'output/js_wam_fact_lmdb',
    make_directory_path(Dir),
    directory_file_path(Dir, 'edges.tsv', Tsv),
    directory_file_path(Dir, 'edges.lmdb', LmdbDir),
    write_idx_tsv(Tsv),
    run_lmdb_builder(Tsv, LmdbDir, BExit, BOut),
    assertion(BExit =:= 0),
    assertion(sub_string(BOut, _, _, _, "uw_fact_lmdb")),
    write_wam_javascript_project(
        [user:js_lmdb_edge/2, user:js_lmdb_probe_filter/0],
        [javascript_wam_fact_sources([source(js_lmdb_edge/2, lmdb(LmdbDir))])],
        Dir),
    lmdb_pkg_prefix(P),
    directory_file_path(P, 'node_modules', NM),
    run_node_cwd(Dir, ['generated_program.js', 'js_lmdb_probe_filter/0'],
                 ['NODE_PATH'=NM], FExit, FOut, FErr),
    assertion(FExit =:= 0),
    assertion(node_succeeded(FOut)),
    assertion(sub_string(FOut, _, _, _, "ok") ; sub_string(FErr, _, _, _, "ok")).

test(indexed_and_lmdb_shared_semantics, [setup(install_idx_preds)]) :-
    DirB = 'output/js_wam_fact_indexed',
    make_directory_path(DirB),
    directory_file_path(DirB, 'edges.tsv', TsvB),
    directory_file_path(DirB, 'edges_store', StoreB),
    write_idx_tsv(TsvB),
    uw_script('uw_fact_index.js', ScriptB),
    run_builder(ScriptB, ['build', TsvB, StoreB], IdxExit, _),
    assertion(IdxExit =:= 0),
    write_wam_javascript_project(
        [user:js_idx_edge/2, user:js_idx_probe_shared/0],
        [javascript_wam_fact_sources([source(js_idx_edge/2, indexed(StoreB))])],
        DirB),
    run_node_args(DirB, ['js_idx_probe_shared/0'], BExit, BOut),
    assertion(BExit =:= 0),
    assertion(node_succeeded(BOut)),
    assertion(sub_string(BOut, _, _, _, "shared_ok")),
    findall(Y, user:js_idx_edge(probekey, Y), SWI),
    assertion(SWI == [alpha, beta]),
    format(user_error, '~n[shared-semantics] SWI probekey -> ~q~n', [SWI]),
    format(user_error, '[shared-semantics] backend B stdout:~n~w', [BOut]),
    (   lmdb_available
    ->  DirA = 'output/js_wam_fact_lmdb',
        make_directory_path(DirA),
        directory_file_path(DirA, 'edges.tsv', TsvA),
        directory_file_path(DirA, 'edges.lmdb', LmdbDir),
        write_idx_tsv(TsvA),
        run_lmdb_builder(TsvA, LmdbDir, LExit, _),
        assertion(LExit =:= 0),
        write_wam_javascript_project(
            [user:js_idx_edge/2, user:js_idx_probe_shared/0],
            [javascript_wam_fact_sources([source(js_idx_edge/2, lmdb(LmdbDir))])],
            DirA),
        lmdb_pkg_prefix(P),
        directory_file_path(P, 'node_modules', NM),
        run_node_cwd(DirA, ['generated_program.js', 'js_idx_probe_shared/0'],
                     ['NODE_PATH'=NM], AExit, AOut, AErr),
        assertion(AExit =:= 0),
        assertion(node_succeeded(AOut)),
        assertion(sub_string(AOut, _, _, _, "shared_ok")),
        format(user_error, '[shared-semantics] backend A stdout:~n~w', [AOut]),
        format(user_error, '[shared-semantics] backend A stderr:~n~w', [AErr])
    ;   format(user_error, '[shared-semantics] backend A skipped (lmdb package not loadable)~n', [])
    ).

:- end_tests(js_wam_fact_sources).
