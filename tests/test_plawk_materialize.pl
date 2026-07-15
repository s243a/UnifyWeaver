:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `materialize NAME` (PLAWK_MULTIPASS_CACHE.md section 3.9), the SURFACE.
% `materialize NAME` marks a view (a `gen ... as NAME` block) or a @prolog
% relation to be materialised-and-cached: its projected/filtered rows are
% computed once into a shared table and reused by every query pass that reads
% it, instead of re-running the goal per consumer. Surface-first (mirroring the
% query-reader / generator / while arcs): the declaration parses and its
% reference is validated against the program's defined relations, but the
% materialise-and-cache runtime is a follow-on, so a program that uses it is a
% clean, specific compile error rather than the generic "outside the surface".

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_materialize).

% `materialize NAME` parses to a materialize(Name) pass-clause item, in program
% order alongside the generator that defines the view and the consuming pass.
test(materialize_parses) :-
    plawk_parse_string(
        "gen over query(edge(A, B)) as (a, b) { emit a } as srcs\n\c
         materialize srcs\n\c
         pass over query(srcs(X)) { print $1 }\n",
        program_passes([],
            [gen_block(name(srcs), over(query(edge, ['A', 'B']), [a, b]),
                 [emit(var(a))]),
             materialize(srcs),
             pass_query(query(srcs, ['X']), [print([field(1)])])],
            [])),
    !.

% A program with no `materialize` declaration is untouched.
test(no_materialize_parses) :-
    plawk_parse_string(
        "pass { print $1 }\n",
        program_passes([],
            [pass([rule(always, [print([field(1)])])])],
            [])),
    !.

% Building a program that materialises a DEFINED view (a `gen ... as NAME`
% block) is a clean not-yet compile error (exit 2) until the runtime lands.
test(materialize_defined_view_is_not_yet_error) :-
    mdir(Dir),
    Src = "@prolog\nedge(1, 10).\nedge(2, 20).\n@end\n\c
           gen over query(edge(A, B)) as (a, b) { emit a } as srcs\n\c
           materialize srcs\n\c
           pass over query(srcs(X)) { print $1 }\n",
    build_status(Dir, 'mvdef', Src, St),
    assertion(St == 2),
    !.

% Materialising a DEFINED @prolog relation is likewise a clean not-yet error.
test(materialize_prolog_relation_is_not_yet_error) :-
    mdir(Dir),
    Src = "@prolog\nnum(1).\nnum(2).\n@end\n\c
           materialize num\n\c
           pass over query(num(X)) { print $1 }\n",
    build_status(Dir, 'mvpl', Src, St),
    assertion(St == 2),
    !.

% Materialising an UNKNOWN relation (no view / @prolog defines it) is a clean
% compile error (a distinct, earlier diagnostic than the not-yet one).
test(materialize_unknown_view_is_error) :-
    mdir(Dir),
    Src = "materialize nope\npass { print $1 }\n",
    build_status(Dir, 'mvunk', Src, St),
    assertion(St == 2),
    !.

% A program with no `materialize` declaration builds normally (no false
% trigger).
test(non_materialize_program_builds, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "@prolog\nedge(1).\nedge(2).\n@end\npass over query(edge(X)) { print $1 }\n",
    build_status(Dir, 'mvok', Src, St),
    assertion(St == 0),
    !.

:- end_tests(plawk_materialize).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

mdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_materialize', Dir),
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
