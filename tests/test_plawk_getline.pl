:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% getline from a file (phase 1): `getline var < "file"`.
%   - `status = getline var < "file"` captures the 1/0/-1 return (a dual-slot
%     write: var is a string scalar holding the line, status an i64).
%   - `getline var < "file"` as a bare statement discards the status.
% The file is opened lazily and advanced one line per call; the handle is keyed
% by FILENAME in a process-wide registry, so every getline site reading the same
% file shares it (as in awk). On EOF the status is 0 and var is unchanged; an
% open/read error yields -1. The canonical loop is prime-then-re-read:
%   r = getline v < "f"; while (r > 0) { ...; r = getline v < "f" }
% (getline inside a while CONDITION is a follow-on; `getline` into $0, plain
% `getline`, and `cmd | getline` are follow-ons.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_getline).

% --- parsing ---------------------------------------------------------------

test(getline_bare_parses) :-
    plawk_parse_string("{ getline line < \"in.txt\" }\n",
        program([], [rule(always, [getline_read(line, "in.txt")])], [])),
    !.

test(getline_capture_parses) :-
    plawk_parse_string("{ r = getline line < \"in.txt\"; print r }\n",
        program([], [rule(always,
            [getline_capture(r, line, "in.txt"), print([var(r)])])], [])),
    !.

% --- runtime ---------------------------------------------------------------

% the canonical loop: read every line of a file (one shared handle across the
% two getline sites), printing each.
test(getline_loop_reads_all, [condition(clang_available)]) :-
    ldir(Dir),
    data_file(Dir, 'data.txt', "alpha\nbeta\ngamma\n", DataPath),
    format(atom(Src),
        "{ r = getline line < \"~w\"; while (r > 0) { print line; r = getline line < \"~w\" } }\n",
        [DataPath, DataPath]),
    build_run(Dir, 'gll', Src, "trigger\n", Out, St),
    assertion(St == 0), assertion(Out == "alpha\nbeta\ngamma\n"), !.

% status is 1 for each line then 0 at EOF; the var is preserved on EOF.
test(getline_status_and_eof, [condition(clang_available)]) :-
    ldir(Dir),
    data_file(Dir, 'two.txt', "one\ntwo\n", P),
    format(atom(Src),
        "{ r = getline v < \"~w\"; print r, v; r = getline v < \"~w\"; print r, v; r = getline v < \"~w\"; print r, v }\n",
        [P, P, P]),
    build_run(Dir, 'gse', Src, "t\n", Out, St),
    assertion(St == 0), assertion(Out == "1 one\n1 two\n0 two\n"), !.

% a missing file yields a -1 status.
test(getline_missing_file, [condition(clang_available)]) :-
    ldir(Dir),
    directory_file_path(Dir, 'does_not_exist.txt', Missing),
    format(atom(Src), "{ r = getline v < \"~w\"; print r }\n", [Missing]),
    build_run(Dir, 'gmf', Src, "t\n", Out, St),
    assertion(St == 0), assertion(Out == "-1\n"), !.

% the bare statement form reads into the var (status discarded).
test(getline_bare_reads, [condition(clang_available)]) :-
    ldir(Dir),
    data_file(Dir, 'bare.txt', "first\nsecond\n", P),
    format(atom(Src), "{ getline v < \"~w\"; print v }\n", [P]),
    build_run(Dir, 'gbr', Src, "t\n", Out, St),
    assertion(St == 0), assertion(Out == "first\n"), !.

:- end_tests(plawk_getline).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_getline', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

data_file(Dir, Name, Contents, Path) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
        write(S, Contents), close(S)).

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
