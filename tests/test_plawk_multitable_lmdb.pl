:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-table stores (PLAWK_MULTITABLE_IMPLEMENTATION_PLAN.md, phase 8.9
% PR 3): durable multiple tables in one LMDB store via NAMED SUB-DBs. A store
% with two or more declared tables routes each to its own sub-DB (named by the
% plawk table name) -- mdb_env_set_maxdbs + a named mdb_dbi_open -- so the
% tables are isolated and durable, and each named sub-DB carries its own schema
% entry (validated independently on open). The unnamed default DB is the
% catalog and holds no plawk data. Single-table stores keep the unnamed-DB path
% (byte-compatible with existing stores).
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
    directory_file_path(Tmp, 'uw_mt_lmdb_probe.c', C),
    directory_file_path(Tmp, 'uw_mt_lmdb_probe', Bin).

:- begin_tests(plawk_multitable_lmdb).

% Two ROW tables in one lmdb store persist across runs, each in its own sub-DB.
% Readers first, writer last: run 1 both readers silent (empty sub-DBs), writer
% commits; run 2 each reader sees its own durable rows -- and orders/customers
% stay distinct, which only holds if they are in separate sub-DBs.
test(two_row_tables_persist, [condition(clang_lmdb_available)]) :-
    mdir(Dir),
    store(Dir, 'rows.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare orders(k str, v str); declare customers(k str, v str) }\n\c
         pass records of orders as r { print r[\"k\"], r[\"v\"] }\n\c
         pass records of customers as r { print r[\"k\"], r[\"v\"] }\n\c
         pass { orders[$1] = row($1, $2); customers[$1] = row($2, $1) }\n", [Store]),
    build(Dir, 'mtr', Src, Bin, In, "a 1\nb 2\n"),
    run_sorted(Bin, In, S1),
    assertion(S1 == []),
    run_sorted(Bin, In, S2),
    assertion(S2 == ["1 a", "2 b", "a 1", "b 2"]),
    !.

% Two i64 counter tables in one lmdb store accumulate independently across runs.
test(two_i64_tables_accumulate, [condition(clang_lmdb_available)]) :-
    mdir(Dir),
    store(Dir, 'ctrs.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare ca; declare cb }\n\c
         pass { ca[$1]++ ; cb[$2]++ }\n\c
         pass over ca as k { print k, ca[k] }\n\c
         pass over cb as k { print k, cb[k] }\n", [Store]),
    build(Dir, 'mti', Src, Bin, In, "a x\na y\nb x\n"),
    run_sorted(Bin, In, S1),
    assertion(S1 == ["a 2", "b 1", "x 2", "y 1"]),
    run_sorted(Bin, In, S2),
    assertion(S2 == ["a 4", "b 2", "x 4", "y 2"]),
    !.

% Per-sub-DB schema validation: reopening the store with a DIFFERENT schema for
% one of the named tables fails cleanly (exit 3) -- the schema lives inside each
% sub-DB, so it is checked per table.
test(per_subdb_schema_mismatch, [condition(clang_lmdb_available)]) :-
    mdir(Dir),
    store(Dir, 'scm.lmdb', Store),
    format(atom(SrcA),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare orders(k str, v str); declare customers(k str, v str) }\n\c
         pass records of orders as r { print r[\"k\"] }\n\c
         pass records of customers as r { print r[\"k\"] }\n\c
         pass { orders[$1] = row($1, $2); customers[$1] = row($2, $1) }\n", [Store]),
    build(Dir, 'mtsa', SrcA, BinA, InA, "a 1\n"),
    run_sorted(BinA, InA, _),
    format(atom(SrcB),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare orders(k str, w str); declare customers(k str, v str) }\n\c
         pass records of orders as r { print r[\"k\"] }\n\c
         pass records of customers as r { print r[\"k\"] }\n\c
         pass { orders[$1] = row($1, $2); customers[$1] = row($2, $1) }\n", [Store]),
    build(Dir, 'mtsb', SrcB, BinB, InB, "a 1\n"),
    process_create(BinB, [InB], [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Code)),
    assertion(Code == 3),
    !.

:- end_tests(plawk_multitable_lmdb).

% --- helpers ---------------------------------------------------------------

mdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multitable_lmdb', Dir),
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
