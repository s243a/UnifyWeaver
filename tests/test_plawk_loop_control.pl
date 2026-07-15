:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk loop control -- `break` / `continue` (PLAWK_CONTROL_FLOW_PLAN.md 3b),
% the SURFACE + a guard. `continue` parses to a `continue` action. The while /
% do-while loop runtime is landed, but LOOP-LOCAL break/continue is not yet
% wired: inside a loop `break` would otherwise lower to the rule-level
% stream-break (stop reading records), a silent mis-compile. So a `break`
% inside a loop, or any `continue`, is a clean not-yet error until the runtime
% lands (it needs SSA merge phis at the loop exit + scalar `if` conditions).
% `break` OUTSIDE a loop keeps its existing stream-break meaning.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_loop_control).

% `continue` parses to a `continue` action.
test(continue_parses) :-
    plawk_parse_string("{ while (i < 3) { continue } }\n",
        program([], [rule(always,
            [while_loop(cmp(var(i), lt, int(3)), [continue])])], [])),
    !.

% `break` still parses (unchanged) as a `break` action.
test(break_parses) :-
    plawk_parse_string("{ while (i < 3) { break } }\n",
        program([], [rule(always,
            [while_loop(cmp(var(i), lt, int(3)), [break])])], [])),
    !.

% A `break` inside a loop body is a clean not-yet error (exit 2) -- guarding the
% silent stream-break mis-compile.
test(break_in_loop_is_not_yet_error) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 3) { print i; break } }\n",
    build_status(Dir, 'lb', Src, St),
    assertion(St == 2),
    !.

% A `break` inside an `if` inside a loop is likewise guarded (subtree search).
test(break_in_if_in_loop_is_not_yet_error) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 3) { if ($1 > 100) { break } i++ } }\n",
    build_status(Dir, 'lbif', Src, St),
    assertion(St == 2),
    !.

% A `continue` anywhere is a clean not-yet error (it only means loop control).
test(continue_is_not_yet_error) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 3) { continue } }\n",
    build_status(Dir, 'lc', Src, St),
    assertion(St == 2),
    !.

% A do-while body with control is also guarded.
test(break_in_do_while_is_not_yet_error) :-
    ldir(Dir),
    Src = "{ i = 0; do { break } while (i < 3) }\n",
    build_status(Dir, 'ldw', Src, St),
    assertion(St == 2),
    !.

% `break` OUTSIDE a loop (rule-level stream break) still compiles and runs --
% the guard only fires for loop bodies.
test(rule_level_break_still_runs, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "$1 == \"stop\" { break }\n{ print $1 }\n",
    build_run(Dir, 'rb', Src, "a\nstop\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\n"),
    !.

% A plain loop with no break/continue is unaffected (no false trigger).
test(plain_loop_builds, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 3) { print i; i++ } }\n",
    build_run(Dir, 'plain', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

:- end_tests(plawk_loop_control).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_loop_control', Dir),
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
