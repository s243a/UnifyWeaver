:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `exit [n]` -- early terminate of the record loop (PLAWK_AWK_FEATURE_AUDIT
% gap 4). `exit` / `exit N` stops reading records, runs END, and returns N
% (default 0). It leaves via break_close_stream (like a rule-level `break`) but
% carries an exit code stored in @plawk_exit_code and read at the final `ret`.
% Unlike `break`, an `exit` inside a loop is NOT consumed by the loop -- it
% always ends the whole program (propagates past any enclosing loop). Slot
% values at the exit point flow into END through the break-close merge phi.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_exit).

% --- parsing ----------------------------------------------------------------

% Bare `exit` parses to `exit(int(0))` (default code 0).
test(exit_bare_parses) :-
    plawk_parse_string("{ exit }\n",
        program([], [rule(always, [exit(int(0))])], [])),
    !.

% `exit N` parses to `exit(int(N))`.
test(exit_code_parses) :-
    plawk_parse_string("{ exit 2 }\n",
        program([], [rule(always, [exit(int(2))])], [])),
    !.

% `exit` guarded by a pattern.
test(exit_guarded_parses) :-
    plawk_parse_string("$1 == \"stop\" { exit 3 }\n",
        program([],
            [rule(field_eq(1, "stop"), [exit(int(3))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `exit N` stops the record stream and returns N.
test(exit_stops_and_returns, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "$1 == \"stop\" { exit 2 }\n{ print $1 }\n",
    build_run(Dir, 'stop', Src, "a\nstop\nb\n", Out, St),
    assertion(St == 2),
    assertion(Out == "a\n"),
    !.

% Bare `exit` returns 0.
test(exit_default_zero, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ exit }\n{ print $1 }\n",
    build_run(Dir, 'def', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == ""),
    !.

% END runs on the way out of `exit`.
test(exit_runs_end, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "$1 == \"stop\" { exit 3 }\n{ print $1 }\nEND { print \"done\" }\n",
    build_run(Dir, 'end', Src, "a\nstop\nb\n", Out, St),
    assertion(St == 3),
    assertion(Out == "a\ndone\n"),
    !.

% `exit` inside an `if` branch ends the program (branch_exit path).
test(exit_in_if, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ if ($1 == \"q\") exit 5; print $1 }\n",
    build_run(Dir, 'inif', Src, "a\nq\nb\n", Out, St),
    assertion(St == 5),
    assertion(Out == "a\n"),
    !.

% `exit` in an `else` branch.
test(exit_in_else, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ if ($1 == \"go\") print \"going\"; else exit 1 }\n",
    build_run(Dir, 'inelse', Src, "go\nstop\n", Out, St),
    assertion(St == 1),
    assertion(Out == "going\n"),
    !.

% `exit` inside a loop ends the whole program (not just the loop).
test(exit_in_loop, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ i = 0; while (i < 10) { if (i > 2) exit 7; print i; i++ } }\n",
    build_run(Dir, 'inloop', Src, "x\n", Out, St),
    assertion(St == 7),
    assertion(Out == "0\n1\n2\n"),
    !.

% Scalar state at the exit point flows into END (break-close merge phi).
test(exit_state_to_end, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ x = 5; exit 4 }\nEND { print x }\n",
    build_run(Dir, 'state', Src, "a\n", Out, St),
    assertion(St == 4),
    assertion(Out == "5\n"),
    !.

% Accumulated state survives an exit into END (count records until "q").
test(exit_accum_to_end, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ x = x + 1 } $1 == \"q\" { exit 9 }\nEND { print x }\n",
    build_run(Dir, 'accum', Src, "a\nb\nq\nc\n", Out, St),
    assertion(St == 9),
    assertion(Out == "3\n"),
    !.

% Mixed program: per-record body print + a guarded exit + END.
test(exit_mixed_body_print, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ print $1 } $1 == \"z\" { exit 2 }\nEND { print \"fin\" }\n",
    build_run(Dir, 'mixed', Src, "a\nz\nb\n", Out, St),
    assertion(St == 2),
    assertion(Out == "a\nz\nfin\n"),
    !.

% A program with no exit is unaffected (returns 0).
test(no_exit_returns_zero, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ print $1 }\n",
    build_run(Dir, 'noexit', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\nb\n"),
    !.

:- end_tests(plawk_exit).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_exit', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

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
