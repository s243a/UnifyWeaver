:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk loop control -- `break` / `continue` (PLAWK_CONTROL_FLOW_PLAN.md 3b).
% Loop-local break/continue is wired for `while` loops: `break` leaves the loop
% (an SSA phi at the loop's `after` merges the break value with the normal
% exit), `continue` re-tests the condition (an extra incoming to the head phi).
% `break` OUTSIDE a loop keeps its rule-level stream-break meaning. `do-while`
% break/continue is a clean not-yet error (its condition sits in the body-done
% block, needing an extra merge phi there).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_loop_control).

% `continue` / `break` parse to their actions.
test(continue_parses) :-
    plawk_parse_string("{ while (i < 3) { continue } }\n",
        program([], [rule(always,
            [while_loop(cmp(var(i), lt, int(3)), [continue])])], [])),
    !.

test(break_parses) :-
    plawk_parse_string("{ while (i < 3) { break } }\n",
        program([], [rule(always,
            [while_loop(cmp(var(i), lt, int(3)), [break])])], [])),
    !.

% --- while break ------------------------------------------------------------

% `break` leaves the loop: `while (i < 10) { if (i > 2) break; print i; i++ }`
% prints 0/1/2 and stops.
test(while_break, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 10) { if (i > 2) break; print i; i++ } }\n",
    build_run(Dir, 'wb', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% The break value flows past the loop: after breaking at i == 3, END sees i = 3
% (proves the `after` merge phi carries the break-point value).
test(while_break_state_to_end, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 10) { if (i > 2) break; i++ } }\nEND { print i }\n",
    build_run(Dir, 'wbe', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n"),
    !.

% An unconditional trailing break runs the body once.
test(while_break_trailing, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 5) { print i; i++; break } }\n",
    build_run(Dir, 'wbt', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n"),
    !.

% --- while continue ---------------------------------------------------------

% `continue` re-tests the condition, skipping the rest of the body:
% `while (i < 5) { i++; if (i > 3) continue; print i }` prints 1/2/3.
test(while_continue, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 5) { i++; if (i > 3) continue; print i } }\n",
    build_run(Dir, 'wc', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1\n2\n3\n"),
    !.

% --- boundaries -------------------------------------------------------------

% `break` OUTSIDE a loop (rule-level stream break) still compiles and runs.
test(rule_level_break_still_runs, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "$1 == \"stop\" { break }\n{ print $1 }\n",
    build_run(Dir, 'rb', Src, "a\nstop\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\n"),
    !.

% `do-while` break/continue is a clean not-yet error (runtime pending).
% --- do-while break/continue ------------------------------------------------

% `do-while` break leaves the loop (the after phi merges the post-condition exit
% with each break value).
test(do_while_break, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; do { if (i > 2) break; print i; i++ } while (i < 10) }\n",
    build_run(Dir, 'dwb', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n2\n"),
    !.

% `do-while` continue re-tests the condition (an extra merge phi in the
% body-done block feeds the condition and the back edge).
test(do_while_continue, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; do { i++; if (i > 3) continue; print i } while (i < 5) }\n",
    build_run(Dir, 'dwc', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1\n2\n3\n"),
    !.

% The do-while break value flows past the loop into END.
test(do_while_break_state_to_end, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; do { if (i > 2) break; i++ } while (i < 10) }\nEND { print i }\n",
    build_run(Dir, 'dwbe', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n"),
    !.

% --- nested loops -----------------------------------------------------------

% A while inside a while; the inner loop prints per outer iteration.
test(nested_while, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 2) { j = 0; while (j < 2) { print j; j++ } i++ } }\n",
    build_run(Dir, 'nest', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n0\n1\n"),
    !.

% An inner break breaks only the INNER loop (the loop-context stack targets the
% innermost loop).
test(nested_inner_break, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 3) { j = 0; while (j < 5) { if (j > 1) break; print j; j++ } i++ } }\n",
    build_run(Dir, 'nb', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n0\n1\n0\n1\n"),
    !.

% A while nested inside a do-while, inner break.
test(while_in_do_while, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; do { j = 0; while (j < 4) { if (j > 1) break; print j; j++ } i++ } while (i < 2) }\n",
    build_run(Dir, 'mix', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0\n1\n0\n1\n"),
    !.

% A plain loop with no break/continue is unaffected.
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
