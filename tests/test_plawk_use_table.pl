:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.7, phase 8.8): the
% `use` reader -- the `select`. `BEGIN cache("path") { use NAME }` attaches to
% an EXISTING store without re-stating its columns: the plawk build reads the
% store's persisted schema header (§8.7) and expands `use NAME` into the same
% cache_table + cache_schema a matching `declare NAME(cols)` would produce. So
% a reader queries a durable store by column name (`records of`) without
% duplicating the schema, and the runtime schema check still guards drift.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_use_table).

% `use NAME` parses to a cache_use begin action.
test(use_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"s.db\") { use t }\n\c
         pass records of t as r { print r[\"a\"] }\n\c
         pass rows of t as r { print r[1] }\n",
        program_passes([begin([cache_use(t, "s.db", file)])], _, _)),
    !.

% End to end: a writer program populates a store with schema (a,b); a
% separate `use` reader (NO declare(cols)) reads it back by column name and by
% position. The reader's schema comes from the store, resolved at build time.
test(use_reads_existing_store, [condition(clang_available)]) :-
    udir(Dir),
    directory_file_path(Dir, 'u.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    % writer: declare schema, populate rows, commit
    format(atom(WSrc),
        "BEGIN cache(\"~w\") { declare t(a str, b str) }\n\c
         pass records of t as r { print r[\"a\"] }\n\c
         pass { t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'uw', WSrc, WBin, WIn, "x 10\ny 20\n"),
    run_sorted(WBin, WIn, _),                    % populate + commit schema
    % use reader: no declare(cols); schema read from the store at build time
    format(atom(RSrc),
        "BEGIN cache(\"~w\") { use t }\n\c
         pass records of t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass rows of t as r { print r[1] }\n", [Store]),
    build(Dir, 'ur', RSrc, RBin, RIn, "x 10\ny 20\n"),
    run_sorted(RBin, RIn, S),
    assertion(S == ["x", "x 10", "y", "y 20"]),
    !.

% `use` on a missing store is a compile error (exit 2): the schema cannot be
% read.
test(use_missing_store_errors, [condition(clang_available)]) :-
    udir(Dir),
    directory_file_path(Dir, 'nope.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(RSrc),
        "BEGIN cache(\"~w\") { use t }\n\c
         pass records of t as r { print r[\"a\"] }\n\c
         pass rows of t as r { print r[1] }\n", [Store]),
    directory_file_path(Dir, 'um.plawk', Prog),
    setup_call_cleanup(open(Prog, write, W, [encoding(utf8)]),
        write(W, RSrc), close(W)),
    directory_file_path(Dir, 'um_bin', Bin),
    cli([build, Prog, '-o', Bin], 2),            % compile error: no store/schema
    !.

:- end_tests(plawk_use_table).

% --- helpers ---------------------------------------------------------------

udir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_use_table', Dir),
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
