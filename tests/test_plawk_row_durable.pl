:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.4):
% DURABLE ROWS across runs. A row value is an atom-registry id, which is
% process-local -- the plain i64 cache (which persists the id) cannot restore
% a row in a fresh process. The byte-valued cache variant persists the row
% BYTES for a str-valued (row) table: commit resolves each value id to its
% bytes; load re-interns them to a valid this-process id. So a row committed
% by one run is readable by a later run.
%
% Tested with a reader-pass-then-writer-pass program run twice: run 1's store
% is empty (reader prints nothing), the writer populates + commits; run 2's
% reader sees the rows run 1 committed -- which only works if the row bytes,
% not the (stale) id, were persisted.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_row_durable).

% Reader pass first (reads the committed store), writer pass second
% (repopulates from input). Run 1: empty store -> reader silent; writer
% writes x/y. Run 2: reader sees the durable rows from run 1.
test(rows_persist_across_runs, [condition(clang_available)]) :-
    ddir(Dir),
    directory_file_path(Dir, 'dur.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(a str, b str) }\n\c
         pass records of t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'dur', Src, Bin, In, "x 10\ny 20\n"),
    run_sorted(Bin, In, S1),
    assertion(S1 == []),                        % run 1: store was empty
    run_sorted(Bin, In, S2),
    assertion(S2 == ["x 10", "y 20"]),          % run 2: durable rows restored
    !.

% Positional durability: the row bytes survive, read back with `rows of`.
test(rows_persist_positional, [condition(clang_available)]) :-
    ddir(Dir),
    directory_file_path(Dir, 'durp.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(a str, b str) }\n\c
         pass rows of t as r { print r[1], r[2] }\n\c
         pass { t[$1] = $0 }\n", [Store]),
    build(Dir, 'durp', Src, Bin, In, "p 1\nq 2\n"),
    run_sorted(Bin, In, S1), assertion(S1 == []),
    run_sorted(Bin, In, S2), assertion(S2 == ["p 1", "q 2"]),
    !.

% Self-describing store: the schema is persisted, so reopening the same store
% with a DIFFERENT declared schema fails cleanly (exit 3) rather than silently
% mis-reading field offsets. Run 1 writes with schema (a,b); a second program
% opening it with schema (a,c) mismatches.
test(schema_mismatch_errors, [condition(clang_available)]) :-
    ddir(Dir),
    directory_file_path(Dir, 'scm.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(SrcA),
        "BEGIN cache(\"~w\") { declare t(a str, b str) }\n\c
         pass records of t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'scma', SrcA, BinA, InA, "x 10\n"),
    run_sorted(BinA, InA, _),                   % run 1: populate + commit schema
    format(atom(SrcB),
        "BEGIN cache(\"~w\") { declare t(a str, c str) }\n\c
         pass records of t as r { print r[\"a\"], r[\"c\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'scmb', SrcB, BinB, InB, "x 10\n"),
    process_create(BinB, [InB], [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Code)),
    assertion(Code == 3),                       % schema mismatch -> clean fail
    !.

:- end_tests(plawk_row_durable).

% --- helpers ---------------------------------------------------------------

ddir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_row_durable', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

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
