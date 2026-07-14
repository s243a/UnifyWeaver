:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-table stores (PLAWK_MULTITABLE_IMPLEMENTATION_PLAN.md, phase 8.9
% PR 2): per-store table grouping + the backend rule (PLAWK_CACHE_BACKENDS.md).
% When several declared tables share ONE cache store path, the backend decides:
% a `file` store (class A) is single-table -> a permanent compile error; an
% `lmdb` store (class B) will route each table to its own named sub-DB, but that
% routing is a later PR, so for now a multi-table lmdb store is a clear "not
% yet" compile error rather than a silent overwrite into the unnamed DB.
% In-memory tables (no cache backing) are unaffected -- multiple of them, or one
% backed table plus bare in-memory ones, remain fine (PR 1).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_multitable_store).

% Two tables in one FILE store -> class-A compile error (exit 2). The check
% runs before codegen/clang, so this needs no clang.
test(file_multitable_is_compile_error) :-
    sdir(Dir),
    directory_file_path(Dir, 'mf.db', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare orders(k str, v str); declare customers(k str, v str) }\n\c
         pass records of orders as r { print r[\"k\"] }\n\c
         pass records of customers as r { print r[\"k\"] }\n", [Store]),
    build_status(Dir, 'mf', Src, St),
    assertion(St == 2),
    !.

% Two tables in one LMDB store -> "not yet" compile error (exit 2) until the
% sub-DB routing (PR 3) lands.
test(lmdb_multitable_is_not_yet_error) :-
    sdir(Dir),
    directory_file_path(Dir, 'ml.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare orders(k str, v str); declare customers(k str, v str) }\n\c
         pass records of orders as r { print r[\"k\"] }\n\c
         pass records of customers as r { print r[\"k\"] }\n", [Store]),
    build_status(Dir, 'ml', Src, St),
    assertion(St == 2),
    !.

% Multiple IN-MEMORY tables (no cache) are not a store at all -> still builds
% and runs (the PR-1 capability is unaffected by the PR-2 check).
test(in_memory_multitable_still_builds, [condition(clang_available)]) :-
    sdir(Dir),
    Src = "pass { a[$1] = $0 ; b[$1] = $0 }\n\c
           pass rows of a as r { print r[1] }\n\c
           pass rows of b as r { print r[2] }\n",
    run_sorted(Dir, 'mm', Src, "k v\n", S),
    assertion(S == ["k", "v"]),
    !.

% One backed table plus a bare in-memory table: the store has a single table,
% so no error; the in-memory table lives alongside.
test(one_backed_plus_inmem_builds, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'ms.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare orders(k str, v str) }\n\c
         pass { orders[$1] = row($1, $2) ; b[$1] = $0 }\n\c
         pass records of orders as r { print r[\"k\"] }\n\c
         pass rows of b as r { print r[2] }\n", [Store]),
    run_sorted(Dir, 'ms', Src, "x 10\ny 20\n", S),
    assertion(S == ["10", "20", "x", "y"]),
    !.

:- end_tests(plawk_multitable_store).

% --- helpers ---------------------------------------------------------------

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multitable_store', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build and return the CLI exit status (no assertion on it).
build_status(Dir, Name, Src, Status) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

run_sorted(Dir, Name, Src, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
