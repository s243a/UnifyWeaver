:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-char / regex field separator: `BEGIN { FS = "…" }` with a value of
% two or more characters is a POSIX ERE, and every field read splits the record
% on it. Implemented with a reserved sentinel separator byte (0) that the field
% runtime dispatches to @wam_fs_regex_field_slice_value / _count_value; the
% program stores the pattern into @wam_fs_regex_pattern_ptr at startup. Because
% the numeric/length/eq/cmp field projectors all delegate to the core slice, the
% regex FS reaches $N reads, NF, guards, and concat uniformly. A single-char FS
% is still a literal byte (awk treats a one-char FS literally). Field assignment
% with a regex FS is a documented follow-on (rejected cleanly).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_regex_fs).

% multi-char literal separator (an ERE where no char is special).
test(regex_fs_multichar_literal, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ml', "BEGIN { FS = \"::\" }\n{ print $2 }\n",
        "a::b::c\n", Out, St),
    assertion(St == 0), assertion(Out == "b\n"), !.

% a genuine regex: comma then optional spaces, joined on the default OFS (space).
test(regex_fs_comma_spaces, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'cs', "BEGIN { FS = \", *\" }\n{ print $1, $3 }\n",
        "a, b,c\n", Out, St),
    assertion(St == 0), assertion(Out == "a c\n"), !.

% a character-class regex separator.
test(regex_fs_char_class, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'cc', "BEGIN { FS = \"[0-9]+\" }\n{ print $2 }\n",
        "a12b34c\n", Out, St),
    assertion(St == 0), assertion(Out == "b\n"), !.

% NF reflects the regex split.
test(regex_fs_nf, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nf', "BEGIN { FS = \"::\" }\n{ print NF }\n",
        "a::b::c\n", Out, St),
    assertion(St == 0), assertion(Out == "3\n"), !.

% a string field-equality guard splits with the regex FS.
test(regex_fs_field_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fg', "BEGIN { FS = \"::\" }\n$1 == \"a\" { print $2 }\n",
        "a::b\nx::y\n", Out, St),
    assertion(St == 0), assertion(Out == "b\n"), !.

% numeric field arithmetic reads regex-split fields.
test(regex_fs_numeric_add, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'na', "BEGIN { FS = \"::\" }\n{ print $1 + $2 }\n",
        "3::4::5\n", Out, St),
    assertion(St == 0), assertion(Out == "7\n"), !.

% a numeric field comparison guard.
test(regex_fs_numeric_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ng', "BEGIN { FS = \"::\" }\n$2 > 5 { print $1 }\n",
        "a::9\nb::3\n", Out, St),
    assertion(St == 0), assertion(Out == "a\n"), !.

% length() of a regex-split field.
test(regex_fs_length, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ln', "BEGIN { FS = \"::\" }\n{ print length($2) }\n",
        "a::bcd\n", Out, St),
    assertion(St == 0), assertion(Out == "3\n"), !.

% concatenation of regex-split fields.
test(regex_fs_concat, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ct', "BEGIN { FS = \"::\" }\n{ x = $1 $2; print x }\n",
        "a::b\n", Out, St),
    assertion(St == 0), assertion(Out == "ab\n"), !.

% a single-char FS stays literal (a metachar is not a regex).
test(regex_fs_single_char_literal, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sc', "BEGIN { FS = \".\" }\n{ print $2 }\n",
        "a.b.c\n", Out, St),
    assertion(St == 0), assertion(Out == "b\n"), !.

% field assignment with a regex FS now works: the record splits on the FS regex
% into the field buffer, the field is set, and $0 rebuilds with OFS.
test(regex_fs_field_assign, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fa', "BEGIN { FS = \"::\"; OFS = \",\" }\n{ $2 = \"X\"; print $0 }\n",
        "a::b::c\n", Out, St),
    assertion(St == 0), assertion(Out == "a,X,c\n"), !.

:- end_tests(plawk_regex_fs).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_regex_fs', Dir),
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

build_status(Dir, Name, Src, BuildStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, BuildStatus).
