:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% ORS (output record separator): the byte printed after each `print`, default
% newline. `BEGIN { ORS = "X" }` sets it to a single character. It is sourced
% once (plawk_output_record_separator) into the `@.plawk_surface_print_newline`
% terminator global that every print emitter already references, so per-record
% and END prints both honour it with no change to the print emitters. It
% composes with OFS (OFS separates the fields of one print; ORS terminates the
% record).
%
% Single-char ORS only. Multi-char ORS, an empty ORS (a zero-length terminator),
% and a literal `%` (which would corrupt the printf terminator) are follow-ons:
% the helper fails so the program is cleanly rejected, never miscompiled. RS
% (input record separator) is a separate follow-on -- the record boundary is a
% hardcoded newline in the runtime reader.

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
