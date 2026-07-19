:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% C-style `for (INIT; COND; UPDATE) BODY` (the awk three-part for). A parser
% normalisation pass desugars it to `INIT; while (COND) { BODY; UPDATE }`,
% reusing the while runtime -- so INIT is an assignment (`i = 0`), UPDATE is an
% increment / compound-assign (`i++`, `i += n`) or assignment, and COND is the
% general while condition. `break` exits the loop (it desugars correctly);
% loops nest.
%
% `continue` that targets the for itself is left un-desugared and cleanly
% rejected by codegen (it would skip the appended UPDATE and infinite-loop) --
% a documented follow-on. A `continue` inside a nested loop in the body is fine
% (that loop consumes it), so the for still desugars. Empty parts (`for (;;)`)
% are also a follow-on; v1 requires all three parts.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_c_for).

% Basic counting loop with i++.
test(count, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ct', "{ for (i = 0; i < 3; i++) print i }\n", "x\n", Out),
    assertion(Out == "0\n1\n2\n"), !.

% Accumulate a sum across the loop, then print it.
test(sum, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sm', "{ s = 0; for (i = 1; i <= 4; i++) s = s + i; print s }\n", "x\n", Out),
    assertion(Out == "10\n"), !.

% A compound-assign step (i += 3).
test(step_update, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'st', "{ for (i = 0; i < 10; i += 3) print i }\n", "x\n", Out),
    assertion(Out == "0\n3\n6\n9\n"), !.

% break exits the loop (the desugared while's break == the for's break).
test(break_exits, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'bk', "{ for (i = 0; i < 9; i++) { if (i == 2) break; print i } }\n", "x\n", Out),
    assertion(Out == "0\n1\n"), !.

% Loops nest.
test(nested, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ne', "{ for (i = 0; i < 2; i++) for (j = 0; j < 2; j++) print i, j }\n",
        "x\n", Out),
    assertion(Out == "0 0\n0 1\n1 0\n1 1\n"), !.

% A continue inside a nested while is consumed by that while, so the outer for
% still desugars: j==2 is skipped, the loop still increments.
test(nested_loop_continue, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'nc',
        "{ for (i = 0; i < 2; i++) { j = 0; while (j < 3) { j++; if (j == 2) continue; print i, j } } }\n",
        "x\n", Out),
    assertion(Out == "0 1\n0 3\n1 1\n1 3\n"), !.

% A condition false at entry runs the body zero times.
test(zero_iterations, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'zi', "{ for (i = 5; i < 3; i++) print i }\n", "x\n", Out),
    assertion(Out == ""), !.

:- end_tests(plawk_c_for).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_c_for', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).
