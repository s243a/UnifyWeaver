:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk query-driven reader (PLAWK_MULTIPASS_CACHE.md §3.4, phase 6;
% PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md). PR 1 landed the SURFACE: `pass
% over query(PRED(V1, ..., Vn)) { print $1, ... }` parses to
% pass_query(query(Pred, Vars), Body) -- each solution of the goal becomes a
% record, its argument variables mapped positionally to $1..$n.
%
% PR 2 lands the RUNTIME for the first supported shape: an all-query program
% whose goals are single-output (`pred(X)`) and whose bodies are `print $1`.
% Each query pass synthesises `__plawk_query_pred(L) :- findall(V, pred(V), L)`,
% runs it on the shared VM, and walks the materialised solution list into a
% table by position -- printing each solution's integer in key order (ordered,
% deterministic; the multiplicity collapses at the boundary). A query program
% OUTSIDE that surface (higher arity, richer body, END block, mixed with
% ordinary passes) is still a clean not-yet compile error.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_query_reader).

% `over query(pred(V1, V2))` parses to pass_query with the predicate name and
% its positional output-variable list; the body's `$N` address the fields.
test(query_two_args_parses) :-
    plawk_parse_string(
        "pass over query(parent(X, Y)) { print $1, $2 }\n\c
         pass { c[$1]++ }\n",
        program_passes([],
            [pass_query(query(parent, ['X', 'Y']), [print([field(1), field(2)])]),
             pass([rule(always, [inc_assoc(var(c), field(1))])])],
            [])),
    !.

% A single-argument goal.
test(query_one_arg_parses) :-
    plawk_parse_string(
        "pass over query(node(N)) { print $1 }\n\c
         pass { print $1 }\n",
        program_passes([],
            [pass_query(query(node, ['N']), [print([field(1)])]),
             pass([rule(always, [print([field(1)])])])],
            [])),
    !.

% The `(` after the predicate name distinguishes a query from a bare table:
% `pass over t as k` is still the table reader, untouched.
test(over_table_unchanged) :-
    plawk_parse_string(
        "pass over t as k { print k }\n",
        program_passes([],
            [pass_over(var(k), var(t), [print([var(k)])])],
            [])),
    !.

% A supported query program (all-query, single-output goal, body `print $1`)
% builds and RUNS: `edge/1`'s three facts materialise and print in order. No
% input file is read -- the records come from the goal, not stdin.
test(query_reader_facts_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(10).\nedge(20).\nedge(30).\n@end\n\c
           pass over query(edge(X)) { print $1 }\n",
    build_run(Dir, 'facts', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "10\n20\n30\n"),
    !.

% Non-determinism is the point: a goal with a disjunction yields every
% solution, collapsed to an ordered materialised set before the body runs.
test(query_reader_nondet_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(A) :- (A = 5 ; A = 6 ; A = 7).\n@end\n\c
           pass over query(edge(X)) { print $1 }\n",
    build_run(Dir, 'nondet', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "5\n6\n7\n"),
    !.

% A two-argument goal: each solution's arguments bind to $1, $2 from their
% per-column materialised tables and print joined by the output separator.
test(query_reader_arity2_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\nedge(3, 30).\n@end\n\c
           pass over query(edge(X, Y)) { print $1, $2 }\n",
    build_run(Dir, 'arity2', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "1 10\n2 20\n3 30\n"),
    !.

% Fields may be reordered and a non-deterministic three-arg goal still binds
% every column correctly (columns stay aligned across the per-column findall
% runs): `print $3, $1, $2` over two disjunctive solutions.
test(query_reader_arity3_reorder_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\ntri(A, B, C) :- (A = 1, B = 2, C = 3 ; A = 4, B = 5, C = 6).\n@end\n\c
           pass over query(tri(X, Y, Z)) { print $3, $1, $2 }\n",
    build_run(Dir, 'arity3', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "3 1 2\n6 4 5\n"),
    !.

% A column may be printed more than once, or a subset printed: field ordinals
% keep the SSA names unique and each `$K` reads its own column table.
test(query_reader_repeat_and_subset_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\n@end\n\c
           pass over query(edge(X, Y)) { print $1, $1, $2 }\n",
    build_run(Dir, 'repeat', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "1 1 10\n2 2 20\n"),
    !.

% A goal-body reference outside the goal's columns ($3 for a 2-arg goal) is
% outside the surface: a clean not-yet compile error (exit 2), not a
% miscompile that would read an absent column table.
test(query_reader_bad_column_not_yet) :-
    qdir(Dir),
    Src = "@prolog\npair(1, 2).\n@end\n\c
           pass over query(pair(X, Y)) { print $1, $3 }\n",
    build_status(Dir, 'badcol', Src, St),
    assertion(St == 2),
    !.

% A reader guard filters the solution set: `if ($1 > 2)` keeps only the
% matching solutions (a WHERE over the query's columns).
test(query_reader_guard_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\nedge(3, 30).\nedge(4, 40).\n@end\n\c
           pass over query(edge(X, Y)) { if ($1 > 2) print $1, $2 }\n",
    build_run(Dir, 'guard', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "3 30\n4 40\n"),
    !.

% `&&` short-circuits at the surface but lowers to pure i1 and over the
% per-column reads; the guard may read a column the body does not print.
test(query_reader_guard_and_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\nedge(3, 30).\nedge(4, 40).\n@end\n\c
           pass over query(edge(X, Y)) { if ($1 >= 2 && $2 < 40) print $1 }\n",
    build_run(Dir, 'guard_and', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "2\n3\n"),
    !.

% `||` combines two column comparisons.
test(query_reader_guard_or_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\nedge(3, 30).\n@end\n\c
           pass over query(edge(X, Y)) { if ($1 == 1 || $2 == 30) print $1, $2 }\n",
    build_run(Dir, 'guard_or', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "1 10\n3 30\n"),
    !.

% A query pass mixed with an ordinary input-reading pass: the query pass
% materialises its goal (no input), the ordinary pass scans the input file.
% Both run in one program (PR 5) -- query solutions first, then the ordinary
% pass echoes column 1 of each input line.
test(query_reader_mixed_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\n@end\n\c
           pass over query(edge(X, Y)) { print $1, $2 }\n\c
           pass { print $1 }\n",
    build_run_input(Dir, 'mixed', Src, "a b\nc d\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1 10\n2 20\na\nc\n"),
    !.

% Reverse order (ordinary pass first) with a guarded query pass: the ordinary
% pass builds a table off the input, the query pass filters its goal.
test(query_reader_mixed_reverse_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nedge(5, 50).\nedge(6, 60).\n@end\n\c
           pass { c[$1]++ }\n\c
           pass over query(edge(X, Y)) { if ($1 == 6) print $1, $2 }\n",
    build_run_input(Dir, 'mixrev', Src, "x\ny\n", Out, St),
    assertion(St == 0),
    assertion(Out == "6 60\n"),
    !.

% Determinism: the same query run in two passes yields byte-identical output
% both times -- the collapse to a materialised set is order-stable and
% repeatable (the multiplicity is fixed at the boundary; §1 intact). A
% disjunctive (non-deterministic) goal still appears in the same solution
% order in each pass.
test(query_reader_determinism_two_passes, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\ng(A) :- (A = 7 ; A = 8 ; A = 9).\n@end\n\c
           pass over query(g(X)) { print $1 }\n\c
           pass over query(g(X)) { print $1 }\n",
    build_run(Dir, 'determ', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "7\n8\n9\n7\n8\n9\n"),
    !.

% String (atom) columns: a goal binding an atom materialises via the tagged
% primitive and prints the resolved text (no build-time type needed).
test(query_reader_string_column_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\ncolor(red).\ncolor(green).\ncolor(blue).\n@end\n\c
           pass over query(color(C)) { print $1 }\n",
    build_run(Dir, 'strcol', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "red\ngreen\nblue\n"),
    !.

% A goal mixing a string column and an integer column in one tuple: each field
% branches on its own kind (atom -> text, integer -> %ld).
test(query_reader_mixed_types_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nitem(apple, 3).\nitem(pear, 5).\n@end\n\c
           pass over query(item(Name, Qty)) { print $1, $2 }\n",
    build_run(Dir, 'mixtype', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "apple 3\npear 5\n"),
    !.

% A guard on the integer column of a string+integer goal filters correctly
% (the guard reads the raw i64 value; string columns just print).
test(query_reader_string_guard_run, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "@prolog\nitem(apple, 3).\nitem(pear, 5).\nitem(plum, 2).\n@end\n\c
           pass over query(item(Name, Qty)) { if ($2 >= 3) print $1, $2 }\n",
    build_run(Dir, 'strguard', Src, [], Out, St),
    assertion(St == 0),
    assertion(Out == "apple 3\npear 5\n"),
    !.

% A program that does NOT use the query reader is unaffected (no false trigger).
test(non_query_program_builds, [condition(clang_available)]) :-
    qdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { print r[1] }\n",
    build_status(Dir, 'ok', Src, St),
    assertion(St == 0),
    !.

:- end_tests(plawk_query_reader).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

qdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_query_reader', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_status(Dir, Name, Src, Status) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

% Build a program, then run the resulting binary with Args, capturing stdout.
build_run(Dir, Name, Src, Args, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, Args,
        [stdout(pipe(RS)), stderr(std), process(RPid)]),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).

% Build a program, write Input to a file, and run the binary over that file
% (for mixed programs whose ordinary passes scan an input file).
build_run_input(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '_in.txt', InPath),
    setup_call_cleanup(open(InPath, write, IS, [encoding(utf8)]),
        write(IS, Input), close(IS)),
    build_run(Dir, Name, Src, [InPath], Out, RunStatus).
