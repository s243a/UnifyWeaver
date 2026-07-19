:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Multi-char / empty OFS on the field-assignment join path. When a field is
% assigned (`$N = expr`), awk rebuilds `$0` by joining the fields with OFS. That
% rebuild used the single-byte runtime @wam_fields_join; it now uses
% @wam_fields_join_str, which takes a separator pointer + length, so the OFS may
% be multi-char or empty. The OFS is emitted as a private @.plawk_field_ofs
% constant and passed by pointer.
%
% The field-assignment driver requires an explicit non-space FS, so these use
% `BEGIN { FS = ":" }`. (A BEGIN-set OFS with field assignment is exercised here
% because the OFS constant is per-driver.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_fieldassign_ofs).

% Single-char OFS join still works (a[1] replaced, joined by ':').
test(single_char_join, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'sc', "BEGIN { FS = \":\"; OFS = \":\" } { $1 = \"x\"; print $0 }\n",
        "a:b:c\n", Out),
    assertion(Out == "x:b:c\n"), !.

% A multi-char OFS joins the rebuilt record with ", ".
test(multichar_join, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'mc', "BEGIN { FS = \":\"; OFS = \", \" } { $1 = \"x\"; print $0 }\n",
        "a:b:c\n", Out),
    assertion(Out == "x, b, c\n"), !.

% A multi-char OFS with a middle field replaced.
test(multichar_middle, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'mm', "BEGIN { FS = \":\"; OFS = \" | \" } { $2 = \"Y\"; print $0 }\n",
        "a:b:c\n", Out),
    assertion(Out == "a | Y | c\n"), !.

% An empty OFS joins the fields adjacent.
test(empty_join, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'em', "BEGIN { FS = \":\"; OFS = \"\" } { $1 = \"x\"; print $0 }\n",
        "a:b:c\n", Out),
    assertion(Out == "xbc\n"), !.

:- end_tests(plawk_fieldassign_ofs).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_fieldassign_ofs', Dir),
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
