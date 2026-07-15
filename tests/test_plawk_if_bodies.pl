:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `if` bodies and regex-in-`if`, plus braceless control-flow bodies.
%
% A guarded plain body (`{ if (c) { print $1 } }`) and a regex condition
% (`if ($0 ~ /re/) { ... }`) compile and run -- the two most common awk
% conditional idioms. Braceless bodies (`if (c) print`, `while (c) x++`,
% `do stmt while (c)`) parse as a single statement, awk-style. These lock in the
% behaviour and the braceless surface.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_if_bodies).

% --- guarded plain bodies (no scalar update) --------------------------------

% `{ if ($1 > 5) { print $1 } }` -- a guarded print, the bread-and-butter awk
% conditional action.
test(if_guarded_print, [condition(clang_available)]) :-
    idir(Dir),
    build_run(Dir, 'gp', "{ if ($1 > 5) { print $1 } }\n", "3\n7\n9\n", Out, St),
    assertion(St == 0),
    assertion(Out == "7\n9\n"),
    !.

% A string-literal print in a guarded body.
test(if_guarded_literal, [condition(clang_available)]) :-
    idir(Dir),
    build_run(Dir, 'gl', "{ if ($1 > 5) { print \"big\" } }\n", "3\n7\n", Out, St),
    assertion(St == 0),
    assertion(Out == "big\n"),
    !.

% if / else with plain bodies.
test(if_else_plain_bodies, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ if ($1 > 5) { print \"big\" } else { print \"small\" } }\n",
    build_run(Dir, 'ie', Src, "3\n7\n", Out, St),
    assertion(St == 0),
    assertion(Out == "small\nbig\n"),
    !.

% --- regex conditions in `if` -----------------------------------------------

% `if ($0 ~ /re/)` -- a regex match as the condition, guarding a print.
test(if_regex_match, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ if ($0 ~ /err/) { print $1 } }\n",
    build_run(Dir, 'rx', Src, "err 1\nok 2\nerr 3\n", Out, St),
    assertion(St == 0),
    assertion(Out == "err\nerr\n"),
    !.

% `if ($0 !~ /re/)` -- the negated match.
test(if_regex_no_match, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ if ($0 !~ /err/) { print $1 } }\n",
    build_run(Dir, 'nx', Src, "err 1\nok 2\n", Out, St),
    assertion(St == 0),
    assertion(Out == "ok\n"),
    !.

% A scalar update inside a guarded `if` still works (no regression).
test(if_scalar_update_still_works, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ if ($1 > 5) { n++ } }\nEND { print n }\n",
    build_run(Dir, 'su', Src, "3\n7\n9\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2\n"),
    !.

% --- braceless bodies (awk-style single statement) --------------------------

% `if (c) print` -- a braceless then-body parses to a singleton action list.
test(braceless_if_parses) :-
    plawk_parse_string("{ if ($1 > 5) print $1 }\n",
        program([], [rule(always,
            [if(_Cond, [print([field(1)])], [])])], [])),
    !.

test(braceless_if_runs, [condition(clang_available)]) :-
    idir(Dir),
    build_run(Dir, 'bif', "{ if ($1 > 5) print $1 }\n", "3\n7\n9\n", Out, St),
    assertion(St == 0),
    assertion(Out == "7\n9\n"),
    !.

% `if (c) stmt; else stmt` -- braceless then and else across a separator.
test(braceless_if_else_runs, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ if ($1 > 5) print \"big\"; else print \"small\" }\n",
    build_run(Dir, 'bie', Src, "3\n7\n", Out, St),
    assertion(St == 0),
    assertion(Out == "small\nbig\n"),
    !.

% A braceless else-if chain.
test(braceless_else_if_chain, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ if ($1 > 8) print \"hi\"; \c
             else if ($1 > 4) print \"mid\"; \c
             else print \"lo\" }\n",
    build_run(Dir, 'bec', Src, "2\n6\n9\n", Out, St),
    assertion(St == 0),
    assertion(Out == "lo\nmid\nhi\n"),
    !.

% `while (c) stmt` -- a braceless loop body.
test(braceless_while_runs, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ i = 0; while (i < 3) i++; print i }\n",
    build_run(Dir, 'bw', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n"),
    !.

% `do stmt while (c)` -- a braceless do-while body.
test(braceless_do_while_runs, [condition(clang_available)]) :-
    idir(Dir),
    Src = "{ i = 0; do i++ while (i < 3); print i }\n",
    build_run(Dir, 'bdw', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n"),
    !.

% A braced body still parses and runs (no regression from the braceless path).
test(braced_if_still_works, [condition(clang_available)]) :-
    idir(Dir),
    build_run(Dir, 'braced', "{ if ($1 > 5) { print $1 } }\n", "3\n7\n", Out, St),
    assertion(St == 0),
    assertion(Out == "7\n"),
    !.

:- end_tests(plawk_if_bodies).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

idir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_if_bodies', Dir),
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
