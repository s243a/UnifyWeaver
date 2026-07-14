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

% `do { BODY } while (VAR CMP int)` parses to do_while_loop(Body, cmp(...)).
test(do_while_parses) :-
    plawk_parse_string(
        "{ i = 0; do { print i; i++ } while (i < 3) }\n",
        program([],
            [rule(always,
                 [set(var(i), int(0)),
                  do_while_loop([print([var(i)]), inc(var(i))],
                      cmp(var(i), lt, int(3)))])],
            [])),
    !.

% RUNTIME (PLAWK_CONTROL_FLOW_PLAN.md PR 2): a `while` loop iterates its
% mutable scalar state via loop-carried head phis. `{ i = 0; while (i < 3) {
% print i; i++ } }` prints 0/1/2 and stops.
test(while_runtime_counts, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 0; while (i < 3) { print i; i++ } }\n",
    build_run(Dir, 'w', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% A `while` whose condition is false at entry runs the body ZERO times.
test(while_runtime_zero_iterations, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 5; while (i < 3) { print i; i++ } }\n",
    build_run(Dir, 'wz', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == ""),
    !.

% `do-while` runs the body at least once even when the condition starts false.
test(do_while_runtime_runs_once, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 5; do { print i; i++ } while (i < 3) }\n",
    build_run(Dir, 'dw1', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "5\n"),
    !.

% `do-while` iterates when the condition holds.
test(do_while_runtime_counts, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 0; do { print i; i++ } while (i < 3) }\n",
    build_run(Dir, 'dw', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% The loop restarts per record (mutable state does not leak between records).
test(while_runtime_per_record, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 0; while (i < 2) { print i; i++ } }\n",
    build_run(Dir, 'wpr', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n0\n1\n"),
    !.

% A loop that accumulates into a scalar read from END.
test(while_runtime_accumulate_to_end, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ s = 0; i = 0; while (i < 4) { s += i; i++ } }\nEND { print s }\n",
    build_run(Dir, 'wacc', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "6\n"),
    !.

% GENERAL CONDITION (PR 3): the right side of a comparison may be another loop
% variable, not just an integer -- `while (i < n)`.
test(while_condition_var_bound, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ n = 3; i = 0; while (i < n) { print i; i++ } }\n",
    build_run(Dir, 'wcv', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% `&&` combines comparisons (both must hold): `i < 5 && i < 3` stops at 3.
test(while_condition_and, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 0; while (i < 5 && i < 3) { print i; i++ } }\n",
    build_run(Dir, 'wca', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% `||` combines comparisons (either may hold): `i < 2 || i < 3` runs while the
% weaker bound holds, stopping at 3.
test(while_condition_or, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ i = 0; while (i < 2 || i < 3) { print i; i++ } }\n",
    build_run(Dir, 'wco', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% do-while with a variable bound.
test(do_while_condition_var_bound, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "{ n = 2; i = 0; do { print i; i++ } while (i < n) }\n",
    build_run(Dir, 'dwcv', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n"),
    !.

% The general condition parses to and/or/cmp AST; a single comparison stays a
% bare cmp (the PR 2 subset is unchanged).
test(while_condition_ast_shapes) :-
    plawk_parse_string("{ while (i < n && j > 0) { i++ } }\n",
        program([], [rule(always,
            [while_loop(and(cmp(var(i), lt, var(n)), cmp(var(j), gt, int(0))),
                 [inc(var(i))])])], [])),
    plawk_parse_string("{ while (i < 3) { i++ } }\n",
        program([], [rule(always,
            [while_loop(cmp(var(i), lt, int(3)), [inc(var(i))])])], [])),
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

% Build a program, then run the binary on Input (stdin), capturing stdout.
build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
