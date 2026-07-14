:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-table stores (PLAWK_MULTITABLE_IMPLEMENTATION_PLAN.md, phase 8.9
% PR 1): the multi-pass driver generalised from ONE shared table to N. Until
% now a program that referenced two tables was rejected ("uses multi-pass
% features outside the current multi-pass surface") because the driver
% hardcoded a single `%plawk_assoc_table_0`. This exercises several in-memory
% tables in one program -- each table is its own `%plawk_assoc_table_<i>`,
% created once and threaded to every pass, so a writer populates them and later
% passes read each back independently. In-memory only; durable multi-table
% storage (LMDB named sub-DBs) is a later PR in the plan.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_multitable).

% Two bare (schema-less) row tables in one program, no cache: a writer keys
% `a` by field 1 and `b` by field 2 (both storing $0), then two positional
% readers read each back. Distinct tables -> distinct contents; proves the
% driver threads more than one in-memory table.
test(two_bare_row_tables, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "pass { a[$1] = $0 ; b[$2] = $0 }\n\c
           pass rows of a as r { print r[1], r[2] }\n\c
           pass rows of b as r { print r[2], r[1] }\n",
    run_sorted(Dir, 'bare', Src, "x 10\ny 20\n", S),
    assertion(S == ["10 x", "20 y", "x 10", "y 20"]),
    !.

% A schema'd (backed) `records of` table alongside a bare positional table, in
% one program, populated together and read by name and by position. Two schema'd
% tables cannot share one store (a class-A file store is single-table; LMDB
% named sub-DBs are a later PR), so a named + a bare table is the valid mix that
% exercises the driver's N-table plumbing across reader kinds.
test(named_plus_bare_tables, [condition(clang_available)]) :-
    mdir(Dir),
    directory_file_path(Dir, 'mt.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare orders(k str, v str) }\n\c
         pass { orders[$1] = row($1, $2); b[$1] = $0 }\n\c
         pass records of orders as r { print r[\"k\"], r[\"v\"] }\n\c
         pass rows of b as r { print r[2], r[1] }\n", [Store]),
    run_sorted(Dir, 'named', Src, "a 1\nb 2\n", S),
    % orders (by name): key=$1 val=$2 ; b (positional): col2,col1 of $0
    assertion(S == ["1 a", "2 b", "a 1", "b 2"]),
    !.

% Mixed table kinds in one program: an i64 counter table `c` and a row table
% `t`, both written in pass 1, then `rows of t` and `over c` in later passes.
% The i64 and str (row) tables coexist as separate `%plawk_assoc_table_<i>`.
test(mixed_i64_and_row_tables, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "pass { c[$1]++ ; t[$1] = $0 }\n\c
           pass rows of t as r { print r[1], r[2] }\n\c
           pass over c as k { print k, c[k] }\n",
    run_sorted(Dir, 'mixed', Src, "a 10\na 20\nb 5\n", S),
    % t: last write per key wins -> a "a 20", b "b 5"; c: counts a=2 b=1
    assertion(S == ["a 2", "a 20", "b 1", "b 5"]),
    !.

% Three row tables in one program, each read by a distinct positional reader.
test(three_row_tables, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "pass { a[$1] = $0 ; b[$1] = $0 ; c[$1] = $0 }\n\c
           pass rows of a as r { print r[1] }\n\c
           pass rows of b as r { print r[2] }\n\c
           pass rows of c as r { print r[3] }\n",
    run_sorted(Dir, 'three', Src, "k p q\n", S),
    assertion(S == ["k", "p", "q"]),
    !.

:- end_tests(plawk_multitable).

% --- helpers ---------------------------------------------------------------

mdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multitable', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

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
