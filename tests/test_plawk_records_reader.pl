:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.3): the
% SAFE NAMED ROW READER. `pass records of TABLE as r { print r["col"], ... }`
% iterates TABLE's stored rows, binding each to r; columns are addressed BY
% NAME (`r["col"]`), resolved through TABLE's declared schema
% (`declare TABLE(col type, ...)`) to the column's position and extracted as
% that field of the stored row. Reordering the printed columns proves the
% mapping is by schema position, not print order. A column not in the schema
% is unsupported (the driver fails cleanly rather than emitting broken IR).
% Together with the row-capture writer (`TABLE[$k] = $0`), this is the
% "retrieve the row as an assoc array keyed by column name" shape.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_records_reader).

% `pass records of T as r { print ... }` parses to pass_records/3, with the
% body's named fields as assoc(var(r), string("col")).
test(records_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"o.db\") { declare orders(cust str, amount i64) }\n\c
         pass { orders[$1] = $0 }\n\c
         pass records of orders as r { print r[\"cust\"], r[\"amount\"] }\n",
        program_passes(_,
            [pass([rule(always, [set_row(var(orders), field(1))])]),
             pass_records(var(r), var(orders),
                 [print([assoc(var(r), string("cust")),
                         assoc(var(r), string("amount"))])])],
            [])),
    !.

% Capture rows, read them back by column NAME. Schema orders(cust, amount):
% r["cust"] = field 1, r["amount"] = field 2. Over a=10, b=5, a=20 (replace):
% a -> "a 20", b -> "b 5".
test(records_named_read, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "BEGIN cache(\"$STORE\") { declare orders(cust str, amount i64) }\n\c
           pass { orders[$1] = $0 }\n\c
           pass records of orders as r { print r[\"cust\"], r[\"amount\"] }\n",
    run_sorted(Dir, 'rec', Src, "a 10\nb 5\na 20\n", S),
    assertion(S == ["a 20", "b 5"]),
    !.

% Reordering the columns proves resolution is by SCHEMA position, not by
% print order: `print r["amount"], r["cust"]` puts the amount first.
test(records_reorder_by_schema, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "BEGIN cache(\"$STORE\") { declare orders(cust str, amount i64) }\n\c
           pass { orders[$1] = $0 }\n\c
           pass records of orders as r { print r[\"amount\"], r[\"cust\"] }\n",
    run_sorted(Dir, 'rord', Src, "a 10\nb 5\na 20\n", S),
    assertion(S == ["20 a", "5 b"]),
    !.

% A column not in the schema is outside the supported surface: the CLI
% reports it unsupported (exit 3), not a broken build.
test(records_unknown_column_unsupported, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "BEGIN cache(\"$STORE\") { declare orders(cust str, amount i64) }\n\c
           pass { orders[$1] = $0 }\n\c
           pass records of orders as r { print r[\"nope\"] }\n",
    directory_file_path(Dir, 'rbad.db', Store),
    replace_store(Src, Store, Src1),
    directory_file_path(Dir, 'rbad.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src1), close(S)),
    directory_file_path(Dir, 'rbad_bin', Bin),
    cli([build, Prog, '-o', Bin], 3),
    !.

% Arithmetic over a named column, evaluated in f64 and printed with %g:
% `r["amount"] * 2` doubles the (last) amount per key. a=10,20 -> a's row
% amount 20 -> 40; b=5 -> 10.
test(records_column_arith, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "BEGIN cache(\"$STORE\") { declare orders(cust str, amount i64) }\n\c
           pass { orders[$1] = $0 }\n\c
           pass records of orders as r { print r[\"cust\"], r[\"amount\"] * 2 }\n",
    run_sorted(Dir, 'rar', Src, "a 10\nb 5\na 20\n", S),
    assertion(S == ["a 40", "b 10"]),
    !.

% Fractional column arithmetic (the surface `/` is integer, so the print
% expression is evaluated in f64): `r["amount"] / 4`.
test(records_column_fraction, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "BEGIN cache(\"$STORE\") { declare orders(cust str, amount i64) }\n\c
           pass { orders[$1] = $0 }\n\c
           pass records of orders as r { print r[\"cust\"], r[\"amount\"] / 4 }\n",
    run_sorted(Dir, 'rfr', Src, "a 10\nb 6\n", S),
    assertion(S == ["a 2.5", "b 1.5"]),
    !.

:- end_tests(plawk_records_reader).

% --- helpers ---------------------------------------------------------------

rdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_records_reader', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% substitute $STORE with a fresh per-name store path
replace_store(Src, Store, Out) :-
    split_string(Src, "", "", [S0]),
    atomic_list_concat(Parts, "$STORE", S0),
    atomic_list_concat(Parts, Store, Out).

run_sorted(Dir, Name, Src0, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    replace_store(Src0, Store, Src),
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
