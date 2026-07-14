:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk reader guards (PLAWK_MULTIPASS_CACHE.md): a WHERE-style row filter on
% the three row readers. `pass records of T as r { if (r["col"] CMP int) print
% ... }` emits only rows whose column satisfies the numeric comparison; the
% positional (`rows of T as r`, `r[N]`) and awk-native anonymous (`rows of T`,
% `$N`) readers get the same guard. The comparison value is an integer literal
% and the six operators are ==, !=, <, <=, >, >=. The guard is compiled to an
% i64 field extraction + icmp + conditional branch, so filtered rows never
% reach the print block.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_reader_guards).

% --- parse -----------------------------------------------------------------

% A records guard parses to an if/rcol_cmp wrapping the print.
test(records_guard_parses) :-
    plawk_parse_string(
        "pass records of orders as r { if (r[\"amt\"] > 100) print r[\"cust\"] }\n",
        program_passes([],
            [pass_records(var(r), var(orders),
                [if(rcol_cmp(r, "amt", gt, 100),
                    [print([assoc(var(r), string("cust"))])], [])])],
            [])),
    !.

% A positional guard parses to rpos_cmp.
test(rows_guard_parses) :-
    plawk_parse_string(
        "pass rows of t as r { if (r[2] > 5) print r[1] }\n",
        program_passes([],
            [pass_rows(var(r), var(t),
                [if(rpos_cmp(r, 2, gt, 5),
                    [print([assoc(var(r), int(1))])], [])])],
            [])),
    !.

% An anonymous `$N` guard parses to rfield_cmp.
test(anon_guard_parses) :-
    plawk_parse_string(
        "pass rows of t { if ($2 >= 10) print $1 }\n",
        program_passes([],
            [pass_rows_anon(var(t),
                [if(rfield_cmp(2, ge, 10),
                    [print([field(1)])], [])])],
            [])),
    !.

% --- records-by-name guard -------------------------------------------------

% `records of` with a column-name guard: over a schema'd store, print the cust
% column only for rows whose amt > 100.
test(records_guard_filters, [condition(clang_available)]) :-
    gdir(Dir),
    directory_file_path(Dir, 'g.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare orders(cust str, amt str) }\n\c
         pass { orders[$1] = row($1, $2) }\n\c
         pass records of orders as r { if (r[\"amt\"] > 100) print r[\"cust\"] }\n",
        [Store]),
    run_sorted(Dir, 'grec', Src, "alice 150\nbob 50\ncarol 200\n", S),
    assertion(S == ["alice", "carol"]),
    !.

% --- positional guard ------------------------------------------------------

% `rows of ... as r` with an r[N] guard: r[2] > 5.
test(rows_guard_filters, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { if (r[2] > 5) print r[1] }\n",
    run_sorted(Dir, 'grows', Src, "a 10\nb 5\nc 20\n", S),
    assertion(S == ["a", "c"]),
    !.

% --- anonymous `$N` guard --------------------------------------------------

% `rows of` (no `as`) with a $N guard: $2 >= 10.
test(anon_guard_filters, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($2 >= 10) print $1 }\n",
    run_sorted(Dir, 'ganon', Src, "a 10\nb 5\nc 20\n", S),
    assertion(S == ["a", "c"]),
    !.

% --- all six comparison operators (via the anon reader) --------------------

test(op_lt, [condition(clang_available)]) :-
    op_result("$2 < 10", S), assertion(S == ["b"]).
test(op_le, [condition(clang_available)]) :-
    op_result("$2 <= 5", S), assertion(S == ["b"]).
test(op_gt, [condition(clang_available)]) :-
    op_result("$2 > 10", S), assertion(S == ["c"]).
test(op_ge, [condition(clang_available)]) :-
    op_result("$2 >= 10", S), assertion(S == ["a", "c"]).
test(op_eq, [condition(clang_available)]) :-
    op_result("$2 == 20", S), assertion(S == ["c"]).
test(op_ne, [condition(clang_available)]) :-
    op_result("$2 != 10", S), assertion(S == ["b", "c"]).

:- end_tests(plawk_reader_guards).

% --- helpers ---------------------------------------------------------------

op_result(Cond, Sorted) :-
    gdir(Dir),
    format(atom(Src),
        "pass { t[$1] = $0 }\npass rows of t { if (~w) print $1 }\n", [Cond]),
    run_sorted(Dir, 'gop', Src, "a 10\nb 5\nc 20\n", Sorted).

gdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_reader_guards', Dir),
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
