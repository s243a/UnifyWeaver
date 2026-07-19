:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% RS (input record separator): the byte the runtime record reader splits input
% on, default newline. `BEGIN { RS = "X" }` sets it to a single character.
%
% The reader (@wam_stream_read_line_transient_value and its persistent sibling)
% previously compared each byte against a hardcoded newline (10); it now loads
% the @wam_rs_byte global instead, and a single-char `BEGIN { RS = "X" }` stores
% that byte into the global at startup (before the record loop), reusing the
% FS-regex startup-store seam. RS composes with FS (RS splits records, FS splits
% each record's fields). The trailing-CR strip (CRLF handling) is now gated on
% RS being newline, so a custom RS keeps a CR as ordinary data.
%
% Single-char RS only: a multi-char or empty RS makes the startup helper fail,
% so the program is cleanly rejected rather than silently reading by newline.
% getline's own record separator (the WAM getline builtin) is a separate reader
% and a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_rs).

% Default RS is a newline (unchanged behaviour).
test(rs_default, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'rd', "{ print NR, $0 }\n", "a\nb\n", Out),
    assertion(Out == "1 a\n2 b\n"), !.

% RS = ";" splits the input into records on semicolons.
test(rs_semicolon, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rs', "BEGIN { RS = \";\" } { print NR, $0 }\n", "a;b;c", Out),
    assertion(Out == "1 a\n2 b\n3 c\n"), !.

% RS = "," and the records still field-split (default FS).
test(rs_comma_fields, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rc', "BEGIN { RS = \",\" } { print $1 }\n", "x,y,z", Out),
    assertion(Out == "x\ny\nz\n"), !.

% A trailing separator does not yield an empty final record.
test(rs_trailing_separator, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rt', "BEGIN { RS = \";\" } { print $0 }\n", "a;b;", Out),
    assertion(Out == "a\nb\n"), !.

% RS and FS compose: RS splits records, FS splits each record's fields.
test(rs_fs_combo, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rf', "BEGIN { RS = \";\"; FS = \":\" } { print $2 }\n",
        "a:1;b:2;c:3", Out),
    assertion(Out == "1\n2\n3\n"), !.

% Default RS still strips a trailing CR (CRLF input): length("a") == 1.
test(rs_default_strips_crlf, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rx', "{ print length($0) }\n", "a\r\nb\r\n", Out),
    assertion(Out == "1\n1\n"), !.

% A custom RS keeps a CR as ordinary data: "a\r" has length 2.
test(rs_custom_keeps_cr, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rk', "BEGIN { RS = \";\" } { print length($0) }\n", "a\r;b", Out),
    assertion(Out == "2\n1\n"), !.

:- end_tests(plawk_rs).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_rs', Dir),
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
