:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk field assignment `$N = expr` -- mutate field N of the current record,
% which rebuilds `$0` (fields re-joined with OFS) and re-emits it via `print $0`.
% Backed by the @wam_record_set_field runtime primitive (split on FS, replace the
% field, join on OFS, re-intern). RHS is a string literal, an integer literal, or
% another field `$M` read from the current record; chained assignments and a
% pattern guard are supported. v1: single-char explicit FS, a single rule with no
% END, and the print-the-record idiom. Default (space) FS, in-body field reads
% after assignment, and `$0 = expr` are follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_fieldassign).

% --- parsing ----------------------------------------------------------------

test(fieldassign_string_parses) :-
    plawk_parse_string("{ $2 = \"X\"; print $0 }\n",
        program([], [rule(always, [set_field(2, string("X")), print([field(0)])])], [])),
    !.

test(fieldassign_field_rhs_parses) :-
    plawk_parse_string("{ $1 = $3; print $0 }\n",
        program([], [rule(always, [set_field(1, field(3)), print([field(0)])])], [])),
    !.

test(fieldassign_int_rhs_parses) :-
    plawk_parse_string("{ $2 = 9; print $0 }\n",
        program([], [rule(always, [set_field(2, int(9)), print([field(0)])])], [])),
    !.

% $0 = ... is not a field assignment (whole-record assign is a separate op).
test(fieldassign_rejects_zero, [fail]) :-
    plawk_parse_string("{ $0 = \"x\"; print $0 }\n",
        program([], [rule(always, [set_field(0, _) | _])], [])).

% --- runtime ----------------------------------------------------------------

% replace a middle field, keeping the separator (OFS = FS).
test(fieldassign_replace, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'rep',
        "BEGIN { FS = \",\"; OFS = \",\" }\n{ $2 = \"X\"; print $0 }\n",
        "a,b,c\nd,e,f\n", Out, St),
    assertion(St == 0), assertion(Out == "a,X,c\nd,X,f\n"), !.

% an integer RHS renders as text.
test(fieldassign_int, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'int',
        "BEGIN { FS = \",\"; OFS = \",\" }\n{ $2 = 9; print $0 }\n",
        "a,b,c\n", Out, St),
    assertion(St == 0), assertion(Out == "a,9,c\n"), !.

% a field-copy RHS reads the current record.
test(fieldassign_field_copy, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'cp',
        "BEGIN { FS = \",\"; OFS = \",\" }\n{ $1 = $3; print $0 }\n",
        "a,b,c\n", Out, St),
    assertion(St == 0), assertion(Out == "c,b,c\n"), !.

% setting a field past the end pads with empty fields.
test(fieldassign_pads, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'pad',
        "BEGIN { FS = \",\"; OFS = \",\" }\n{ $5 = \"Z\"; print $0 }\n",
        "a,b\n", Out, St),
    assertion(St == 0), assertion(Out == "a,b,,,Z\n"), !.

% the rebuild uses OFS: a comma FS with the default (space) OFS re-joins on space.
test(fieldassign_ofs_rebuild, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ofs',
        "BEGIN { FS = \",\" }\n{ $2 = \"X\"; print $0 }\n",
        "a,b,c\n", Out, St),
    assertion(St == 0), assertion(Out == "a X c\n"), !.

% chained assignments fold onto the running record.
test(fieldassign_chained, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ch',
        "BEGIN { FS = \",\"; OFS = \",\" }\n{ $1 = \"p\"; $2 = \"q\"; print $0 }\n",
        "a,b,c\n", Out, St),
    assertion(St == 0), assertion(Out == "p,q,c\n"), !.

% a pattern guard gates the assignment+print; non-matching records are dropped.
test(fieldassign_guarded, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'gd',
        "BEGIN { FS = \",\"; OFS = \",\" }\n$1 == \"err\" { $2 = \"X\"; print $0 }\n",
        "err,b,c\nok,y,z\n", Out, St),
    assertion(St == 0), assertion(Out == "err,X,c\n"), !.

% default (space) FS field assignment is out of the v1 surface: it must fail the
% build cleanly (exit 3) rather than miscompile.
test(fieldassign_default_fs_rejected, [condition(clang_available)]) :-
    ldir(Dir),
    build_status(Dir, 'df', "{ $2 = \"X\"; print $0 }\n", BuildStatus),
    assertion(BuildStatus == exit(3)), !.

:- end_tests(plawk_fieldassign).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_fieldassign', Dir),
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
