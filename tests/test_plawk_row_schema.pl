:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.1): the
% SCHEMA SURFACE. A backed-BEGIN `declare NAME(col type, ...)` carries a row
% schema -- named columns with types -- for the record readers (records of /
% rows of, later sub-phases). The schema is emitted as a separate
% cache_schema(NAME, Columns) begin action ALONGSIDE the existing
% cache_table/3, so every current cache_table consumer is untouched and a
% schema-declared table used as an ordinary i64 table behaves exactly as
% before (the schema is inert until the readers land). A bare `declare NAME`
% is unchanged.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_row_schema).

% `declare NAME(col type, ...)` yields cache_table/3 (unchanged) PLUS a
% cache_schema/2 carrying the ordered typed columns.
test(schema_declare_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"o.db\") { declare orders(cust str, amount i64) }\n\c
         { orders[$1]++ }\nEND { for (k in orders) print k, orders[k] }\n",
        program([begin([cache_table(orders, "o.db", file),
                        cache_schema(orders, [col(cust, str), col(amount, i64)])])],
                _, _)),
    !.

% A bare `declare NAME` is unchanged -- cache_table/3 only, no cache_schema.
test(bare_declare_unchanged) :-
    plawk_parse_string(
        "BEGIN cache(\"h.db\") { declare c }\n{ c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n",
        program([begin([cache_table(c, "h.db", file)])], _, _)),
    \+ plawk_parse_string(
        "BEGIN cache(\"h.db\") { declare c }\n{ c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n",
        program([begin([cache_table(_, _, _), cache_schema(_, _) | _])], _, _)),
    !.

% Mixed: a bare table and a schema'd table in one block, in order.
test(mixed_declares_parse) :-
    plawk_parse_string(
        "BEGIN cache(\"m.db\") { declare c ; declare orders(cust str, amount i64) }\n\c
         { c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
        program([begin([cache_table(c, "m.db", file),
                        cache_table(orders, "m.db", file),
                        cache_schema(orders, [col(cust, str), col(amount, i64)])])],
                _, _)),
    !.

% The schema is INERT: a schema-declared table used as an ordinary i64 table
% compiles and runs exactly as a bare-declared one would. Over a a b: a 2 / b 1.
test(schema_inert_compiles, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'sch.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare orders(cust str, amount i64) }\n\c
         { orders[$1]++ }\nEND { for (k in orders) print k, orders[k] }\n", [Store]),
    run_sorted(Dir, 'sch', Src, "a\na\nb\n", S),
    assertion(S == ["a 2", "b 1"]),
    !.

:- end_tests(plawk_row_schema).

% --- helpers ---------------------------------------------------------------

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_row_schema', Dir),
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
