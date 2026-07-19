:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Multi-char and empty OFS. OFS (the separator printed between a print's
% comma-separated fields, default a single space) is now a list of bytes rather
% than one byte: `BEGIN { OFS = ", " }` sets it to a multi-char string,
% `OFS = ""` to no separator (fields print adjacent). The separator emitters
% write one putchar per byte (plawk_ofs_sep_lines/4), so an empty OFS emits
% nothing and an N-byte OFS emits N. Single-char OFS is unchanged, and OFS
% composes with ORS (OFS between fields, ORS after the record).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_ofs).

% Default OFS is a single space (unchanged behaviour).
test(ofs_default_space, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'os', "{ print $1, $2 }\n", "a b\n", Out),
    assertion(Out == "a b\n"), !.

% A single-char OFS still works.
test(ofs_single_char, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'o1', "BEGIN { OFS = \":\" } { print $1, $2 }\n", "a b\n", Out),
    assertion(Out == "a:b\n"), !.

% A multi-char OFS: ", " between fields.
test(ofs_multichar, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'om', "BEGIN { OFS = \", \" } { print $1, $2 }\n", "a b\n", Out),
    assertion(Out == "a, b\n"), !.

% An empty OFS prints the fields adjacent.
test(ofs_empty, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'oe', "BEGIN { OFS = \"\" } { print $1, $2 }\n", "a b\n", Out),
    assertion(Out == "ab\n"), !.

% A multi-char OFS across three fields.
test(ofs_three_fields, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'o3', "BEGIN { OFS = \" | \" } { print $1, $2, $3 }\n", "a b c\n", Out),
    assertion(Out == "a | b | c\n"), !.

% OFS composes with ORS: OFS between fields, ORS terminates the record.
test(ofs_with_ors, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'oo', "BEGIN { OFS = \"-\"; ORS = \";\" } { print $1, $2 }\n",
        "a b\nc d\n", Out),
    assertion(Out == "a-b;c-d;"), !.

% A multi-char OFS in an END print.
test(ofs_end_print, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'oep', "BEGIN { OFS = \", \" } { n++ } END { print n, n }\n", "a\nb\n", Out),
    assertion(Out == "2, 2\n"), !.

:- end_tests(plawk_ofs).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ofs', Dir),
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
