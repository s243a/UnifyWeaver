:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk reader-guard extensions -- BOOLEAN combinations (the capstone of the
% guard extensions). A row-reader guard may combine comparisons with `&&` and
% `||`: short-circuit, `&&` binding tighter than `||`, left-associative, parens
% allowed. `if (r["amt"] > 100 && r["cust"] == "alice")`. A single comparison
% is unchanged (parses to the bare guard term); combinations parse to and(L, R)
% / or(L, R) and lower to short-circuit branches (each leaf branches to the next
% test or to print/skip). Leaves may mix kinds (int / float / string). Applies
% to all three readers.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_guard_bool).

% --- parse -----------------------------------------------------------------

% `&&` -> and(...), `||` -> or(...), a single comparison stays bare.
test(and_parses) :-
    guard_of("pass rows of t as r { if (r[2] > 3 && r[1] == \"a\") print r[1] }\n", G),
    assertion(G == and(rpos_cmp(r, 2, gt, 3), rpos_cmp(r, 1, eq, str("a")))),
    !.
test(or_parses) :-
    guard_of("pass rows of t as r { if (r[2] > 3 || r[2] < 1) print r[1] }\n", G),
    assertion(G == or(rpos_cmp(r, 2, gt, 3), rpos_cmp(r, 2, lt, 1))),
    !.
test(single_unchanged) :-
    guard_of("pass rows of t as r { if (r[2] > 3) print r[1] }\n", G),
    assertion(G == rpos_cmp(r, 2, gt, 3)),
    !.

% `&&` binds tighter than `||`: a || b && c parses as or(a, and(b, c)).
test(precedence) :-
    guard_of("pass rows of t as r { if (r[1] == \"a\" || r[2] > 3 && r[3] < 9) print r[1] }\n", G),
    assertion(G == or(rpos_cmp(r, 1, eq, str("a")),
                      and(rpos_cmp(r, 2, gt, 3), rpos_cmp(r, 3, lt, 9)))),
    !.
% Parens override: (a || b) && c parses as and(or(a, b), c).
test(parens) :-
    guard_of("pass rows of t as r { if ((r[1] == \"a\" || r[2] > 3) && r[3] < 9) print r[1] }\n", G),
    assertion(G == and(or(rpos_cmp(r, 1, eq, str("a")), rpos_cmp(r, 2, gt, 3)),
                       rpos_cmp(r, 3, lt, 9))),
    !.

% --- end to end ------------------------------------------------------------

% `&&` mixing an int and a string leaf.
test(and_int_string, [condition(clang_available)]) :-
    gdir(Dir),
    directory_file_path(Dir, 'a.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(cust str, amt str) }\n\c
         pass { t[$1] = row($1, $2) }\n\c
         pass records of t as r { if (r[\"amt\"] > 100 && r[\"cust\"] == \"alice\") print r[\"cust\"] }\n", [Store]),
    run_sorted(Dir, 'ais', Src, "alice 200\nbob 50\ncarol 300\ndan 120\n", S),
    assertion(S == ["alice"]),          % >100 AND ==alice: only alice(200)
    !.

% `||` over two comparisons.
test(or_two, [condition(clang_available)]) :-
    gdir(Dir),
    directory_file_path(Dir, 'o.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(cust str, amt str) }\n\c
         pass { t[$1] = row($1, $2) }\n\c
         pass records of t as r { if (r[\"amt\"] > 150 || r[\"cust\"] == \"bob\") print r[\"cust\"] }\n", [Store]),
    run_sorted(Dir, 'ot', Src, "alice 200\nbob 50\ncarol 300\ndan 120\n", S),
    assertion(S == ["alice", "bob", "carol"]),   % >150 (alice,carol) OR ==bob
    !.

% Three-way `&&` chain over the anon reader ($N).
test(three_way_and, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($2 > 1 && $2 < 100 && $3 == \"y\") print $1 }\n",
    run_sorted(Dir, 'tw', Src, "a 10 y\nb 200 y\nc 3 n\nd 3 y\n", S),
    assertion(S == ["a", "d"]),         % 1<v<100 AND col3==y: a(10,y), d(3,y)
    !.

% Precedence end to end: a || b && c (over the anon reader).
test(precedence_runs, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($1 == \"z\" || $2 > 5 && $3 == \"y\") print $1 }\n",
    run_sorted(Dir, 'pr', Src, "a 10 y\nb 200 y\nc 3 n\nz 0 n\nd 3 y\n", S),
    assertion(S == ["a", "b", "z"]),    % z, OR (>5 AND y): a,b ; z by name
    !.

% Parens change the meaning: (a || b) && c.
test(parens_runs, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if (($1 == \"z\" || $2 > 5) && $3 == \"y\") print $1 }\n",
    run_sorted(Dir, 'pn', Src, "a 10 y\nb 200 y\nc 3 n\nz 0 n\nd 3 y\n", S),
    assertion(S == ["a", "b"]),         % (z or >5) AND y: a,b ; z fails (col3 n)
    !.

% A float leaf inside a boolean.
test(float_in_and, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($2 > 2.5 && $3 == \"y\") print $1 }\n",
    run_sorted(Dir, 'fa', Src, "a 10 y\nc 3 n\nd 3 y\n", S),
    assertion(S == ["a", "d"]),
    !.

:- end_tests(plawk_guard_bool).

% --- helpers ---------------------------------------------------------------

guard_of(Src, Guard) :-
    plawk_parse_string(Src, program_passes(_, [pass_rows(_, _, [if(Guard, _, _)])], _)).

gdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_guard_bool', Dir),
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
