:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Two-argument `substr(s, m)` returns the tail of the string from position m to
% the end (awk semantics), matching gawk's `substr($0, 3)` -> everything from
% byte 3 on. It parses to `substr(Field, Start, to_end)`; the emitter maps the
% `to_end` marker to a max byte count that the runtime slice clamps to whatever
% remains after m, so no runtime change is needed. The three-argument
% `substr(s, m, n)` is unchanged.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_substr2).

% --- parsing ----------------------------------------------------------------

% `substr($0, 3)` parses to the to_end marker length.
test(two_arg_substr_parses) :-
    plawk_parse_string("{ print substr($0, 3) }\n",
        program([], [rule(always, [print([substr(field(0), 3, to_end)])])], [])),
    !.

% `substr($0, 2, 3)` still parses to an explicit length (unchanged).
test(three_arg_substr_unchanged) :-
    plawk_parse_string("{ print substr($0, 2, 3) }\n",
        program([], [rule(always, [print([substr(field(0), 2, 3)])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% Tail of the whole record from position 3.
test(two_arg_substr_record_tail, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'st', "{ print substr($0, 3) }\n", "abcdef\n", Out),
    assertion(Out == "cdef\n"), !.

% Tail of a field.
test(two_arg_substr_field_tail, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sf', "{ print substr($2, 2) }\n", "aa bbbb\n", Out),
    assertion(Out == "bbb\n"), !.

% Start at 1 returns the whole string.
test(two_arg_substr_from_one, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'so', "{ print substr($0, 1) }\n", "hello\n", Out),
    assertion(Out == "hello\n"), !.

% A start past the end yields the empty string.
test(two_arg_substr_past_end_empty, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sp', "{ print \"[\" substr($0, 10) \"]\" }\n", "abc\n", Out),
    assertion(Out == "[]\n"), !.

% Three-argument substr still works (regression).
test(three_arg_substr_runtime, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 's3', "{ print substr($0, 2, 3) }\n", "abcdef\n", Out),
    assertion(Out == "bcd\n"), !.

:- end_tests(plawk_substr2).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_substr2', Dir),
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
