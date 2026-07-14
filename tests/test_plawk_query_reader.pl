:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk query-driven reader (PLAWK_MULTIPASS_CACHE.md §3.4, phase 6;
% PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md), PR 1: the SURFACE. `pass over
% query(PRED(V1, ..., Vn)) { print $1, ... }` parses to pass_query(query(Pred,
% Vars), Body) -- each solution of the goal will become a record, its argument
% variables mapped positionally to $1..$n. The runtime materialisation (PRs
% 2-3) is not wired yet, so building such a program is a clean, specific compile
% error rather than the generic "outside the multi-pass surface".

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

% Building a query-reader program is a clean compile error (exit 2) until the
% runtime lands; the message names the goal as pred/arity.
test(query_reader_is_not_yet_error) :-
    qdir(Dir),
    Src = "pass over query(parent(X, Y)) { print $1, $2 }\npass { c[$1]++ }\n",
    build_status(Dir, 'q', Src, St),
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
