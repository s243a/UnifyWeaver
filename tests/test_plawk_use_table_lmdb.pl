:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.7, phase 8.8): the
% `use` reader over the LMDB BACKEND. The file-backend counterpart is
% test_plawk_use_table.pl. `use NAME` attaches to an existing store without
% re-stating its columns -- the plawk build reads the store's persisted schema
% and expands `use NAME` into the cache_table + cache_schema a matching
% `declare` would produce. On the file backend the schema is a plain header the
% build reads directly; on LMDB it lives under a key inside the B-tree, so the
% build compiles and runs a small liblmdb probe (wam_cache_lmdb_schema.c) to
% extract it. This test proves a durable LMDB row store written by a `declare`
% writer is read back by a separate `use` reader (no re-declare).
%
% Requires liblmdb at build+run time (apt-get install liblmdb-dev); skipped
% when clang cannot link -llmdb.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

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
    directory_file_path(Tmp, 'uw_lmdb_use_probe.c', C),
    directory_file_path(Tmp, 'uw_lmdb_use_probe', Bin).

:- begin_tests(plawk_use_table_lmdb).

% End to end: a `declare` writer populates a durable LMDB store; a separate
% `use` reader (NO declare(cols)) reads it back by column name and by position.
% The reader's schema is read from the LMDB store at build time via the probe.
test(use_reads_existing_lmdb_store, [condition(clang_lmdb_available)]) :-
    udir(Dir),
    store(Dir, 'u.lmdb', Store),
    format(atom(WSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare t(a str, b str) }\n\c
         pass records of t as r { print r[\"a\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'luw', WSrc, WBin, WIn, "x 10\ny 20\n"),
    run_sorted(WBin, WIn, _),                    % populate + commit schema
    format(atom(RSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\") { use t }\n\c
         pass records of t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass rows of t as r { print r[1] }\n", [Store]),
    build(Dir, 'lur', RSrc, RBin, RIn, "x 10\ny 20\n"),
    run_sorted(RBin, RIn, S),
    assertion(S == ["x", "x 10", "y", "y 20"]),
    !.

% `use` on a missing LMDB store is a compile error (exit 2): no schema to read.
test(use_missing_lmdb_store_errors, [condition(clang_lmdb_available)]) :-
    udir(Dir),
    store(Dir, 'nope.lmdb', Store),
    format(atom(RSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\") { use t }\n\c
         pass records of t as r { print r[\"a\"] }\n\c
         pass rows of t as r { print r[1] }\n", [Store]),
    directory_file_path(Dir, 'lum.plawk', Prog),
    setup_call_cleanup(open(Prog, write, W, [encoding(utf8)]),
        write(W, RSrc), close(W)),
    directory_file_path(Dir, 'lum_bin', Bin),
    cli([build, Prog, '-o', Bin], 2),            % compile error: no store/schema
    !.

:- end_tests(plawk_use_table_lmdb).

% --- helpers ---------------------------------------------------------------

udir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_use_table_lmdb', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Fresh single-file lmdb store: remove the data file and its -lock sidecar.
store(Dir, Name, Store) :-
    directory_file_path(Dir, Name, Store),
    atom_concat(Store, '-lock', Lock),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    ( exists_file(Lock) -> delete_file(Lock) ; true ).

build(Dir, Name, Src, Bin, In, Input) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)).

run_sorted(Bin, In, Sorted) :-
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
