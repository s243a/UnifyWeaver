:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-pass persistent cache, phase 5: the LMDB BACKEND. A cache
% store declared `backend "lmdb"` is durable in a real LMDB file rather than
% the portable flat file. The plawk build links the C LMDB runtime
% (src/unifyweaver/targets/wam_cache_lmdb.c) and -llmdb when (and only when)
% a program uses an lmdb-backed store; the file backend links neither. The
% cache handle is still the in-memory assoc table (eager materialisation):
% load reads all pairs from LMDB into the table, commit writes them back.
%
% Requires liblmdb at build+run time (apt-get install liblmdb-dev); the test
% is skipped when clang cannot link -llmdb.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_lmdb_available :-
    catch(( tmp_c(C, Bin),
            setup_call_cleanup(open(C, write, S),
                write(S, "#include <lmdb.h>\nint main(void){MDB_env*e;return mdb_env_create(&e);}\n"),
                close(S)),
            format(atom(Cmd), 'clang -w ~w -o ~w -llmdb 2>/dev/null', [C, Bin]),
            process_create(path(sh), ['-c', Cmd], [process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

tmp_c(C, Bin) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_lmdb_probe.c', C),
    directory_file_path(Tmp, 'uw_lmdb_probe', Bin).

:- begin_tests(plawk_cache_lmdb).

% `backend "lmdb"` parses to a cache_table with the lmdb backend; the
% default (no backend clause) is `file`.
test(lmdb_backend_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"h.lmdb\" backend \"lmdb\") { declare c }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n",
        program([begin([cache_table(c, "h.lmdb", lmdb)])], _, _)),
    plawk_parse_string(
        "BEGIN cache(\"h.db\") { declare c }\n{ c[$1]++ }\n",
        program([begin([cache_table(c, "h.db", file)])], _, _)),
    !.

% End to end: an lmdb-backed histogram persists and accumulates across runs,
% and writes a real LMDB file (data + -lock sidecar).
test(lmdb_histogram_persists, [condition(clang_lmdb_available)]) :-
    cl_dir(Dir),
    directory_file_path(Dir, 'h.lmdb', Store),
    directory_file_path(Dir, 'h.lmdb-lock', Lock),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    ( exists_file(Lock) -> delete_file(Lock) ; true ),
    format(string(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare c }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n", [Store]),
    build_bin(Dir, 'hist', Src, Bin),
    directory_file_path(Dir, 'in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, "a\na\nb\n"), close(SI)),
    run_sorted(Bin, [Input], S1), assertion(S1 == ["a 2", "b 1"]),
    run_sorted(Bin, [Input], S2), assertion(S2 == ["a 4", "b 2"]),
    % A genuine LMDB store was created (single-file: data + lock sidecar).
    assertion(exists_file(Store)),
    assertion(exists_file(Lock)),
    !.

% The store written by one compiled binary is read by a SEPARATELY compiled
% binary: LMDB durability across rebuilds (pre-population).
test(lmdb_store_survives_rebuild, [condition(clang_lmdb_available)]) :-
    cl_dir(Dir),
    directory_file_path(Dir, 'shared.lmdb', Store),
    directory_file_path(Dir, 'shared.lmdb-lock', Lock),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    ( exists_file(Lock) -> delete_file(Lock) ; true ),
    format(string(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare c }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n", [Store]),
    directory_file_path(Dir, 'seed.txt', SeedIn),
    setup_call_cleanup(open(SeedIn, write, S0, [encoding(utf8)]),
        write(S0, "x\nx\nx\n"), close(S0)),
    build_bin(Dir, 'writer', Src, WriterBin),
    run_sorted(WriterBin, [SeedIn], _),
    build_bin(Dir, 'reader', Src, ReaderBin),
    directory_file_path(Dir, 'more.txt', MoreIn),
    setup_call_cleanup(open(MoreIn, write, S1, [encoding(utf8)]),
        write(S1, "x\n"), close(S1)),
    run_sorted(ReaderBin, [MoreIn], S), assertion(S == ["x 4"]),
    !.

:- end_tests(plawk_cache_lmdb).

% --- helpers ---------------------------------------------------------------

cl_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_cache_lmdb', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_bin(Dir, Name, Src, Bin) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0).

run_sorted(Bin, Args, Sorted) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
