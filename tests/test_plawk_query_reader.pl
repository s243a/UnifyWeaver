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

% A query OUTSIDE the supported surface -- here a two-argument goal -- is a
% clean not-yet compile error (exit 2), not a miscompile; the message names
% the goal as pred/arity.
test(query_reader_higher_arity_not_yet) :-
    qdir(Dir),
    Src = "@prolog\npair(1, 2).\n@end\n\c
           pass over query(pair(X, Y)) { print $1, $2 }\n",
    build_status(Dir, 'arity2', Src, St),
    assertion(St == 2),
    !.

% A query pass mixed with an ordinary pass is also outside v1 (all-query
% only) and gets the same clean not-yet error rather than the generic one.
test(query_reader_mixed_not_yet) :-
    qdir(Dir),
    Src = "@prolog\nnode(1).\n@end\n\c
           pass over query(node(N)) { print $1 }\npass { c[$1]++ }\n",
    build_status(Dir, 'mixed', Src, St),
    assertion(St == 2),
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
