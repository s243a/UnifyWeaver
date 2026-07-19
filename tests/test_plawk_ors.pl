:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% ORS (output record separator): the string printed after each `print`, default
% newline. `BEGIN { ORS = "…" }` sets it to any string -- single-char,
% multi-char, or empty. It is emitted through a runtime pointer global
% (@plawk_ors_ptr, statically initialised to the ORS constant) that every print
% terminator loads and prints via `printf("%s", ptr)`, so the ORS may be any
% length and any content (a literal `%` is data, not a format directive) with no
% change to the terminator emitters. Per-record and END prints both honour it,
% and it composes with (multi-char) OFS -- OFS separates a print's fields, ORS
% terminates the record.
%
% RS (input record separator) is a separate follow-on. Multi-char OFS on the
% field-assignment join path also remains a follow-on (needs a multi-byte join).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_ors).

% Default ORS is a newline (unchanged behaviour).
test(ors_default, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'od', "{ print $1 }\n", "a\nb\n", Out),
    assertion(Out == "a\nb\n"), !.

% ORS = ";" terminates each record with a semicolon (no trailing newline).
test(ors_semicolon, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'os', "BEGIN { ORS = \";\" } { print $1 }\n", "a\nb\n", Out),
    assertion(Out == "a;b;"), !.

% A tab ORS.
test(ors_tab, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'ot', "BEGIN { ORS = \"\\t\" } { print $1 }\n", "a\nb\n", Out),
    assertion(Out == "a\tb\t"), !.

% OFS separates the fields of a print; ORS terminates the record.
test(ors_with_ofs, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'of', "BEGIN { OFS = \"-\"; ORS = \";\" } { print $1, $2 }\n",
        "a b\nc d\n", Out),
    assertion(Out == "a-b;c-d;"), !.

% ORS applies to an END print too.
test(ors_end_print, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'oe', "BEGIN { ORS = \";\" } { n++ } END { print n }\n", "a\nb\n", Out),
    assertion(Out == "2;"), !.

% A pipe ORS across three records.
test(ors_pipe, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'op', "BEGIN { ORS = \"|\" } { print $1 }\n", "x\ny\nz\n", Out),
    assertion(Out == "x|y|z|"), !.

% A multi-char ORS: each record terminated by " | ".
test(ors_multichar, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'omc', "BEGIN { ORS = \" | \" } { print $1 }\n", "a\nb\n", Out),
    assertion(Out == "a | b | "), !.

% An empty ORS concatenates records with no terminator.
test(ors_empty, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'oem', "BEGIN { ORS = \"\" } { print $1 }\n", "a\nb\n", Out),
    assertion(Out == "ab"), !.

% A literal `%` ORS is data (printed via %s), not a format directive.
test(ors_percent, [condition(clang_available)]) :-
    sdir(Dir), build_run(Dir, 'opc', "BEGIN { ORS = \"%\" } { print $1 }\n", "a\nb\n", Out),
    assertion(Out == "a%b%"), !.

% A multi-char ORS composes with a multi-char OFS.
test(ors_multichar_with_ofs, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'omo', "BEGIN { OFS = \"-\"; ORS = \" | \" } { print $1, $2 }\n",
        "a b\nc d\n", Out),
    assertion(Out == "a-b | c-d | "), !.

:- end_tests(plawk_ors).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ors', Dir),
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
