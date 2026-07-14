:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.4): DURABLE
% ROWS over the LMDB BACKEND. The file-backend counterpart is
% test_plawk_row_durable.pl; this is the same durability contract for a store
% declared `backend "lmdb"`. A row value is a process-local atom id, so the
% plain i64 lmdb path (which persists the id) cannot restore a row in a fresh
% process. The byte-valued lmdb helpers (wam_cache_{commit,load}_lmdb_str)
% persist the row BYTES instead -- commit resolves each id to its bytes and
% mdb_puts them under the i64 key; load re-interns the bytes to a valid
% this-process id. The declared schema is stored under a distinguished key and
% validated on open, so a mismatched reopen fails cleanly (exit 3).
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
    directory_file_path(Tmp, 'uw_lmdb_rows_probe.c', C),
    directory_file_path(Tmp, 'uw_lmdb_rows_probe', Bin).

:- begin_tests(plawk_row_durable_lmdb).

% Reader pass first (reads the committed store), writer pass second. Run 1:
% empty lmdb store -> reader silent; writer writes x/y and commits the row
% BYTES. Run 2: reader sees the durable rows from run 1 -- which only works if
% bytes, not the stale process-local id, were persisted to LMDB.
test(rows_persist_across_runs, [condition(clang_lmdb_available)]) :-
    ldir(Dir),
    store(Dir, 'dur.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare t(a str, b str) }\n\c
         pass records of t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'ldur', Src, Bin, In, "x 10\ny 20\n"),
    run_sorted(Bin, In, S1),
    assertion(S1 == []),                        % run 1: store was empty
    run_sorted(Bin, In, S2),
    assertion(S2 == ["x 10", "y 20"]),          % run 2: durable rows restored
    !.

% Positional durability over LMDB: bytes survive, read back with `rows of`.
test(rows_persist_positional, [condition(clang_lmdb_available)]) :-
    ldir(Dir),
    store(Dir, 'durp.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare t(a str, b str) }\n\c
         pass rows of t as r { print r[1], r[2] }\n\c
         pass { t[$1] = $0 }\n", [Store]),
    build(Dir, 'ldurp', Src, Bin, In, "p 1\nq 2\n"),
    run_sorted(Bin, In, S1), assertion(S1 == []),
    run_sorted(Bin, In, S2), assertion(S2 == ["p 1", "q 2"]),
    !.

% Self-describing lmdb store: the schema is persisted under its own key, so a
% reopen with a DIFFERENT declared schema fails cleanly (exit 3) rather than
% mis-reading field offsets. Run 1 writes schema (a,b); the second program
% opens it with schema (a,c).
test(schema_mismatch_errors, [condition(clang_lmdb_available)]) :-
    ldir(Dir),
    store(Dir, 'scm.lmdb', Store),
    format(atom(SrcA),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare t(a str, b str) }\n\c
         pass records of t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'lscma', SrcA, BinA, InA, "x 10\n"),
    run_sorted(BinA, InA, _),                   % run 1: populate + commit schema
    format(atom(SrcB),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare t(a str, c str) }\n\c
         pass records of t as r { print r[\"a\"], r[\"c\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'lscmb', SrcB, BinB, InB, "x 10\n"),
    process_create(BinB, [InB], [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Code)),
    assertion(Code == 3),                       % schema mismatch -> clean fail
    !.

:- end_tests(plawk_row_durable_lmdb).

% --- helpers ---------------------------------------------------------------

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_row_durable_lmdb', Dir),
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
