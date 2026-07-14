:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `while` loops, PR 1: the SURFACE. `while (VAR CMP int) { BODY }` parses
% to while_loop(cmp(var(V), Op, int(N)), Body) -- the awk while control
% structure. The loop runtime (mutable scalar state iterated to a fixed point)
% is not wired yet, so building a program that uses one is a clean, specific
% compile error rather than the generic "outside the multi-pass surface".

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_while).

% `while (i < 3) { ... }` parses to while_loop(cmp(var(i), lt, int(3)), Body);
% the body is a general action block.
test(while_parses) :-
    plawk_parse_string(
        "{ i = 0; while (i < 3) { print i; i++ } }\n",
        program([],
            [rule(always,
                 [set(var(i), int(0)),
                  while_loop(cmp(var(i), lt, int(3)),
                      [print([var(i)]), inc(var(i))])])],
            [])),
    !.

% The comparison operators all parse.
test(while_operators_parse) :-
    forall(member(Op-Sym,
               [lt-"<", le-"<=", gt-">", ge-">=", eq-"==", ne-"!="]),
        ( format(string(Src), "{ while (n ~w 5) { n++ } }\n", [Sym]),
          plawk_parse_string(Src,
              program([],
                  [rule(always,
                       [while_loop(cmp(var(n), Op, int(5)), [inc(var(n))])])],
                  [])) )),
    !.

% Building a program with a while loop is a clean compile error (exit 2) until
% the loop runtime lands.
test(while_is_not_yet_error) :-
    wdir(Dir),
    Src = "{ i = 0; while (i < 3) { print i; i++ } }\n",
    build_status(Dir, 'w', Src, St),
    assertion(St == 2),
    !.

% A program with no while loop is unaffected (no false trigger).
test(non_while_program_builds, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ print $1 }\n",
    build_status(Dir, 'ok', Src, St),
    assertion(St == 0),
    !.

:- end_tests(plawk_while).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

wdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_while', Dir),
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
