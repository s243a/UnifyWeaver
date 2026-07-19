:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% RS (input record separator): the value the runtime record reader splits input
% on, default newline. A one-character value is literal; a nonempty value longer
% than one character is a POSIX ERE, matching GNU awk's regex-RS extension.
%
% Both readers (@wam_stream_read_line_transient_value and its persistent sibling)
% share this setting. RS composes with FS (RS splits records, FS splits each
% record's fields). The trailing-CR strip (CRLF handling) is gated on RS being
% exactly a one-byte newline, so a custom RS keeps a CR as ordinary data.
%
% An empty RS selects paragraph mode; its focused coverage lives in
% test_plawk_paragraph.pl. Focused ERE, RT, rejection, zero-width, and
% persistent-getline coverage lives in test_plawk_rs_regex.pl.

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

% A multi-char ERE spelling of literal "||" splits on that two-byte sequence.
test(rs_multichar, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rm', "BEGIN { RS = \"[|][|]\" } { print NR, $0 }\n", "a||b||c", Out),
    assertion(Out == "1 a\n2 b\n3 c\n"), !.

% A three-byte literal separator, expressed as an ERE with a bracketed pipe.
test(rs_multichar_three, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'r3', "BEGIN { RS = \"-[|]-\" } { print $0 }\n", "x-|-y-|-z", Out),
    assertion(Out == "x\ny\nz\n"), !.

% A multi-char RS composes with FS.
test(rs_multichar_fs, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rmf', "BEGIN { RS = \"[|][|]\"; FS = \":\" } { print $2 }\n",
        "a:1||b:2||c:3", Out),
    assertion(Out == "1\n2\n3\n"), !.

% A trailing multi-char RS does not yield an empty final record.
test(rs_multichar_trailing, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rmt', "BEGIN { RS = \"[|][|]\" } { print $0 }\n", "a||b||", Out),
    assertion(Out == "a\nb\n"), !.

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
