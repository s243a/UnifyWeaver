:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk namespaced cache stores (PLAWK_MULTITABLE_IMPLEMENTATION_PLAN.md, phase
% 8.9 PR 4; PLAWK_MULTIPASS_CACHE.md §3.7): `cache("db" as ns) { declare orders
% ... }` makes `ns` a namespace, so its tables are referenced `ns.orders` and
% never collide with global names. A namespaced table's internal name is the
% dotted atom `ns.orders`; its LOCAL part (`orders`) is the sub-DB it routes to,
% so `as ns` asks for named sub-DBs (a store uses sub-DBs even with one table).
% The file backend (class A) is single-table, so a namespaced file store is a
% compile error.
%
% LMDB cases require liblmdb at build+run time; skipped when clang cannot link
% -llmdb.

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
    directory_file_path(Tmp, 'uw_ns_lmdb_probe.c', C),
    directory_file_path(Tmp, 'uw_ns_lmdb_probe', Bin).

:- begin_tests(plawk_namespace).

% --- parse -----------------------------------------------------------------

% `cache("db" as ns)` qualifies each declared table to the dotted atom
% `ns.table`; `ns.table` references parse to the same atom.
test(namespace_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"db\" backend \"lmdb\" as shop) { declare orders(k str, v str) }\n\c
         pass records of shop.orders as r { print r[\"k\"] }\n\c
         pass rows of shop.orders as r { print r[1] }\n",
        program_passes(
            [begin([cache_table('shop.orders', "db", lmdb),
                    cache_schema('shop.orders', [col(k, str), col(v, str)])])],
            [pass_records(var(r), var('shop.orders'), _),
             pass_rows(var(r), var('shop.orders'), _)],
            [])),
    !.

% A namespaced write target (`ns.t[$k] = ...`) parses to the qualified name.
test(namespace_write_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"db\" as ns) { declare t(a str, b str) }\n\c
         pass { ns.t[$1] = row($1, $2) }\n\c
         pass rows of ns.t as r { print r[1] }\n",
        program_passes(
            [begin([cache_table('ns.t', "db", file), cache_schema('ns.t', _)])],
            [pass([rule(always, [set_row_cons(var('ns.t'), field(1), _)])]),
             pass_rows(var(r), var('ns.t'), _)],
            [])),
    !.

% A bare (un-namespaced) program is unchanged: no dot, plain atom names.
test(bare_names_unchanged) :-
    plawk_parse_string(
        "pass { t[$1] = $0 }\npass rows of t as r { print r[1] }\n",
        program_passes([],
            [pass([rule(always, [set_row(var(t), field(1))])]),
             pass_rows(var(r), var(t), _)],
            [])),
    !.

% --- end to end ------------------------------------------------------------

% A namespaced lmdb store: two `ns.table` tables, durable across runs, each in
% its own sub-DB (named by the LOCAL part). Readers first, writer last.
test(namespace_lmdb_durable, [condition(clang_lmdb_available)]) :-
    ndir(Dir),
    store(Dir, 'shop.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\" as shop) { declare orders(k str, v str); declare customers(k str, v str) }\n\c
         pass records of shop.orders as r { print r[\"k\"], r[\"v\"] }\n\c
         pass records of shop.customers as r { print r[\"k\"], r[\"v\"] }\n\c
         pass { shop.orders[$1] = row($1, $2); shop.customers[$1] = row($2, $1) }\n", [Store]),
    build(Dir, 'shop', Src, Bin, In, "a 1\nb 2\n"),
    run_sorted(Bin, In, S1),
    assertion(S1 == []),
    run_sorted(Bin, In, S2),
    assertion(S2 == ["1 a", "2 b", "a 1", "b 2"]),
    !.

% A single-table namespaced lmdb store also persists (namespace -> sub-DB even
% with one table).
test(namespace_single_table_durable, [condition(clang_lmdb_available)]) :-
    ndir(Dir),
    store(Dir, 'one.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\" as ns) { declare t(a str, b str) }\n\c
         pass records of ns.t as r { print r[\"a\"], r[\"b\"] }\n\c
         pass { ns.t[$1] = row($1, $2) }\n", [Store]),
    build(Dir, 'one', Src, Bin, In, "x 10\ny 20\n"),
    run_sorted(Bin, In, S1), assertion(S1 == []),
    run_sorted(Bin, In, S2), assertion(S2 == ["x 10", "y 20"]),
    !.

% A namespaced FILE store is a compile error (exit 2): class A is single-table,
% so it cannot hold named sub-tables.
test(namespace_file_is_compile_error) :-
    ndir(Dir),
    directory_file_path(Dir, 'nf.db', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" as ns) { declare t(a str, b str) }\n\c
         pass rows of ns.t as r { print r[1] }\n\c
         pass rows of ns.t as r { print r[2] }\n", [Store]),
    build_status(Dir, 'nf', Src, St),
    assertion(St == 2),
    !.

:- end_tests(plawk_namespace).

% --- helpers ---------------------------------------------------------------

ndir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_namespace', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

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

build_status(Dir, Name, Src, Status) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

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
