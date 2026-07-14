:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `use` over a namespaced multi-table store (PLAWK_MULTITABLE_IMPLEMENTATION_PLAN.md
% phase 8.9 PR 5; PLAWK_MULTIPASS_CACHE.md §3.7): attach to a multi-table LMDB
% store WITHOUT re-stating any columns. A namespaced `use orders` (under `as
% ns`, so the internal name is `ns.orders`) reads its schema from the named
% sub-DB `orders` -- the build runs the liblmdb schema probe with the sub-DB
% name. So a reader queries each table of a multi-table store by column name
% with no `declare`. Exercised with THREE-column tables to confirm the schema
% read is not limited to two columns.
%
% Requires liblmdb at build+run time; skipped when clang cannot link -llmdb.

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
    directory_file_path(Tmp, 'uw_usens_lmdb_probe.c', C),
    directory_file_path(Tmp, 'uw_usens_lmdb_probe', Bin).

:- begin_tests(plawk_use_namespace).

% A `declare` writer populates a namespaced multi-table LMDB store with two
% THREE-column tables; a separate `use` reader (NO declare) attaches to both via
% the namespace and reads them back by column name -- the schema for each comes
% from its own named sub-DB at build time. Distinct data per table proves each
% `use` reads the right sub-DB.
test(use_reads_namespaced_multitable_3col, [condition(clang_lmdb_available)]) :-
    udir(Dir),
    store(Dir, 'shop.lmdb', Store),
    format(atom(WSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\" as shop) { declare orders(id str, cust str, amt str); declare items(sku str, name str, qty str) }\n\c
         pass records of shop.orders as r { print r[\"id\"] }\n\c
         pass { shop.orders[$1] = row($1, $2, $3); shop.items[$1] = row($1, $2, $3) }\n", [Store]),
    build(Dir, 'nsw', WSrc, WBin, WIn, "o1 alice 100\ns1 widget 7\n"),
    run_sorted(WBin, WIn, _),                    % populate + commit schemas
    format(atom(RSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\" as shop) { use orders; use items }\n\c
         pass records of shop.orders as r { print r[\"cust\"], r[\"amt\"] }\n\c
         pass records of shop.items as r { print r[\"name\"], r[\"qty\"] }\n", [Store]),
    build(Dir, 'nsr', RSrc, RBin, RIn, "o1 alice 100\ns1 widget 7\n"),
    run_sorted(RBin, RIn, S),
    % orders rows: (cust, amt) of each stored row; items rows: (name, qty).
    % Both tables hold the two input records ($1,$2,$3), so:
    %   orders: "alice 100", "widget 7"  ; items: "alice 100", "widget 7"
    assertion(S == ["alice 100", "alice 100", "widget 7", "widget 7"]),
    !.

% Column arithmetic over a `use`d 3-column table: the third column divided by a
% constant, proving the schema-read columns are fully usable, not just printed.
test(use_namespaced_3col_arithmetic, [condition(clang_lmdb_available)]) :-
    udir(Dir),
    store(Dir, 'arith.lmdb', Store),
    format(atom(WSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\" as ns) { declare t(k str, a str, b str) }\n\c
         pass records of ns.t as r { print r[\"k\"] }\n\c
         pass { ns.t[$1] = row($1, $2, $3) }\n", [Store]),
    build(Dir, 'aw', WSrc, WBin, WIn, "x 10 4\ny 9 3\n"),
    run_sorted(WBin, WIn, _),
    format(atom(RSrc),
        "BEGIN cache(\"~w\" backend \"lmdb\" as ns) { use t }\n\c
         pass records of ns.t as r { print r[\"k\"], r[\"a\"] / r[\"b\"] }\n\c
         pass rows of ns.t as r { print r[1] }\n", [Store]),
    build(Dir, 'ar', RSrc, RBin, RIn, "x 10 4\ny 9 3\n"),
    run_sorted(RBin, RIn, S),
    % pass 1: 10/4 = 2.5, 9/3 = 3 (by name) ; pass 2: keys by position
    assertion(S == ["x", "x 2.5", "y", "y 3"]),
    !.

:- end_tests(plawk_use_namespace).

% --- helpers ---------------------------------------------------------------

udir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_use_namespace', Dir),
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
