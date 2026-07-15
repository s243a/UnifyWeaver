:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk scalar `if` in an END block: `END { if (COND) print ...; [else print
% ...] }`. END has no current record, so the condition is a scalar comparison
% over the final slot values (the same scalar_if(_) shape as a rule-body `if`),
% lowered in the end_print block. Each branch is a single print, using the
% prefixed print emitter so the two blocks' temporaries don't collide.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_end_if).

% `END { if (COND) print ... }` parses to end([if(scalar_if(...), [print], [])]).
test(end_if_parses) :-
    plawk_parse_string("{ n++ }\nEND { if (n > 1) print n }\n",
        program([], _Rules,
            [end([if(scalar_if(cmp(var(n), gt, int(1))), [print([var(n)])], [])])])),
    !.

test(end_if_else_parses) :-
    plawk_parse_string("{ n++ }\nEND { if (n > 1) print \"many\"; else print \"few\" }\n",
        program([], _Rules,
            [end([if(scalar_if(cmp(var(n), gt, int(1))),
                     [print([string("many")])],
                     [print([string("few")])])])])),
    !.

% --- runtime ----------------------------------------------------------------

% if/else in END, condition true.
test(end_if_else_true, [condition(clang_available)]) :-
    edir(Dir),
    Src = "{ n++ }\nEND { if (n > 1) print \"many\"; else print \"few\" }\n",
    build_run(Dir, 'ie_t', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "many\n"),
    !.

% if/else in END, condition false -> the else branch.
test(end_if_else_false, [condition(clang_available)]) :-
    edir(Dir),
    Src = "{ n++ }\nEND { if (n > 1) print \"many\"; else print \"few\" }\n",
    build_run(Dir, 'ie_f', Src, "a\n", Out, St),
    assertion(St == 0),
    assertion(Out == "few\n"),
    !.

% if with no else, condition true -> prints a scalar.
test(end_if_no_else_true, [condition(clang_available)]) :-
    edir(Dir),
    Src = "{ n++ }\nEND { if (n > 1) print n }\n",
    build_run(Dir, 'ne_t', Src, "a\nb\nc\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n"),
    !.

% if with no else, condition false -> nothing printed.
test(end_if_no_else_false, [condition(clang_available)]) :-
    edir(Dir),
    Src = "{ n++ }\nEND { if (n > 5) print n }\n",
    build_run(Dir, 'ne_f', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == ""),
    !.

% A plain `END { print ... }` still works (no regression from the new clause).
test(end_plain_print_still_works, [condition(clang_available)]) :-
    edir(Dir),
    Src = "{ n++ }\nEND { print n }\n",
    build_run(Dir, 'pp', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2\n"),
    !.

:- end_tests(plawk_end_if).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

edir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_if', Dir),
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
