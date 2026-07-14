:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk reader-guard extensions -- STRING equality (follow-on to the integer
% and float reader guards). A row-reader guard `if (COL == "lit")` / `!= "lit"`
% filters rows by a TEXT column: the column is extracted as a byte slice and
% compared to a string literal by length THEN memcmp -- the memcmp runs only
% when the lengths are equal, so a literal longer than the field never reads out
% of bounds. Only `==` / `!=` are meaningful (string ordering is a follow-on).
% Applies to all three readers -- named `r["col"]`, positional `r[N]`, and
% awk-native `$N`.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_guard_string).

% A string RHS parses to str(Text); ints and floats are unchanged.
test(string_guard_parses) :-
    plawk_parse_string(
        "pass records of t as r { if (r[\"cust\"] == \"alice\") print r[\"amt\"] }\n",
        program_passes([],
            [pass_records(var(r), var(t),
                [if(rcol_cmp(r, "cust", eq, str("alice")),
                    [print([assoc(var(r), string("amt"))])], [])])],
            [])),
    plawk_parse_string(
        "pass rows of t { if ($1 != \"bob\") print $1 }\n",
        program_passes([],
            [pass_rows_anon(var(t),
                [if(rfield_cmp(1, ne, str("bob")),
                    [print([field(1)])], [])])],
            [])),
    !.

% Named `==` string guard: over a schema'd store, print amt only for cust ==
% "alice".
test(records_string_eq, [condition(clang_available)]) :-
    gdir(Dir),
    directory_file_path(Dir, 'se.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(cust str, amt str) }\n\c
         pass { t[$1] = row($1, $2) }\n\c
         pass records of t as r { if (r[\"cust\"] == \"alice\") print r[\"amt\"] }\n", [Store]),
    run_sorted(Dir, 'rse', Src, "alice 100\nbob 50\ncarol 200\n", S),
    assertion(S == ["100"]),
    !.

% Named `!=` string guard: everyone but alice.
test(records_string_ne, [condition(clang_available)]) :-
    gdir(Dir),
    directory_file_path(Dir, 'sn.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(cust str, amt str) }\n\c
         pass { t[$1] = row($1, $2) }\n\c
         pass records of t as r { if (r[\"cust\"] != \"alice\") print r[\"cust\"] }\n", [Store]),
    run_sorted(Dir, 'rsn', Src, "alice 100\nbob 50\ncarol 200\n", S),
    assertion(S == ["bob", "carol"]),
    !.

% Positional `r[N]` string guard.
test(rows_string_eq, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { if (r[1] == \"bob\") print r[2] }\n",
    run_sorted(Dir, 'pse', Src, "alice 100\nbob 50\ncarol 200\n", S),
    assertion(S == ["50"]),
    !.

% Awk-native `$N` string guard, plus safety: a literal much LONGER than any
% field must not read out of bounds (length check gates the memcmp).
test(anon_string_ne_long_literal, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($1 != \"verylongliteralname\") print $1 }\n",
    run_sorted(Dir, 'ase', Src, "alice 1\nbob 2\n", S),
    assertion(S == ["alice", "bob"]),
    !.

% Equality is exact (a prefix does not match): `== "al"` matches neither
% "alice" nor "alicia".
test(string_eq_is_exact, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($1 == \"al\") print $1 }\n\c
           pass rows of t { if ($1 == \"alice\") print $1 }\n",
    run_sorted(Dir, 'exact', Src, "alice x\nalicia y\n", S),
    assertion(S == ["alice"]),
    !.

:- end_tests(plawk_guard_string).

% --- helpers ---------------------------------------------------------------

gdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_guard_string', Dir),
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
