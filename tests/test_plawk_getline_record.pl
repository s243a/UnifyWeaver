:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% File getline into the current record: `getline < "file"`. A successful read
% replaces `$0` and fields are re-split lazily with the active FS; status is
% 1/0/-1, while EOF/error preserve the prior record. This form shares the
% filename registry with scalar-target getline, but v1 reads a physical newline
% record without applying RS or changing RT. It never advances NR/FNR.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_getline_record).

% --- parsing ---------------------------------------------------------------

test(record_getline_bare_parses) :-
    plawk_parse_string("{ getline < \"in.txt\" }\n",
        program([], [rule(always, [getline_file_record("in.txt")])], [])),
    !.

test(record_getline_capture_parses) :-
    plawk_parse_string("{ r = getline < \"in.txt\"; print r }\n",
        program([], [rule(always,
            [getline_file_record_capture(r, "in.txt"), print([var(r)])])], [])),
    !.

test(record_getline_while_normalises) :-
    plawk_parse_string("{ while ((getline < \"f.txt\") > 0) print $0 }\n",
        program(_, [rule(always, Actions)], _)),
    Actions = [ getline_file_record_capture('$getline_status', "f.txt"),
                while_loop(cmp(var('$getline_status'), gt, int(0)),
                    [print([field(0)]),
                     getline_file_record_capture('$getline_status', "f.txt")]) ],
    !.

% --- runtime ---------------------------------------------------------------

% A successful read updates $0 and recomputes fields/NF through a regex FS.
test(record_getline_resplits_regex_fs, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'regex_fields.txt', "left::right,tail\n", Path),
    format(atom(Src),
        "BEGIN { FS = \"[,:]+\"; OFS = \"|\" } { r = getline < \"~w\"; print r, $0, $1, $2, $3, NF }\n",
        [Path]),
    build_run(Dir, 'regex_fields', Src, "trigger\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|left::right,tail|left|right|tail|3\n"),
    !.

% EOF returns 0 and leaves $0, its fields, and NF at the last successful read.
test(record_getline_eof_preserves_record, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'one_record.txt', "one:two\n", Path),
    format(atom(Src),
        "BEGIN { FS = \":\"; OFS = \"|\" } { r = getline < \"~w\"; print r, $0, $1, $2, NF; r = getline < \"~w\"; print r, $0, $1, $2, NF }\n",
        [Path, Path]),
    build_run(Dir, 'eof_preserve', Src, "orig:main\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|one:two|one|two|2\n0|one:two|one|two|2\n"),
    !.

% An open error returns -1 and preserves the main-input record and its fields.
test(record_getline_open_error_preserves_record, [condition(clang_available)]) :-
    rdir(Dir),
    directory_file_path(Dir, 'record_getline_missing.txt', Missing),
    format(atom(Src),
        "BEGIN { FS = \":\"; OFS = \"|\" } { r = getline < \"~w\"; print r, $0, $1, $2, NF }\n",
        [Missing]),
    build_run(Dir, 'open_error', Src, "orig:main\n", Out, St),
    assertion(St == 0),
    assertion(Out == "-1|orig:main|orig|main|2\n"),
    !.

% The in-condition idiom drains the file. NR/FNR stay at the one main-input
% record throughout; redirected getline never increments either counter.
test(record_getline_while_keeps_nr_fnr, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'drain.txt', "red green\nblue gold\n", Path),
    format(atom(Src),
        "BEGIN { OFS = \"|\" } { while ((getline < \"~w\") > 0) print NR, FNR, $0, NF }\n",
        [Path]),
    build_run(Dir, 'while_nr', Src, "trigger\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|1|red green|2\n1|1|blue gold|2\n"),
    !.

% Record getline is newline-only in v1 even when the main reader has regex RS;
% it also leaves the main reader's matched RT unchanged.
test(record_getline_ignores_rs_and_preserves_rt, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'physical_line.txt', "left45right\n", Path),
    format(atom(Src),
        "BEGIN { RS = \"[0-9]+\"; OFS = \"|\" } { r = getline < \"~w\"; print r, $0, RT }\n",
        [Path]),
    build_run(Dir, 'rs_rt', Src, "trigger123", Out, St),
    assertion(St == 0),
    assertion(Out == "1|left45right|123\n"),
    !.

% Scalar-target and record-target sites use the same filename-keyed handle.
test(record_getline_shares_scalar_registry, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'shared.txt', "one\ntwo\n", Path),
    format(atom(Src),
        "{ r = getline v < \"~w\"; s = getline < \"~w\"; print r, v, s, $0 }\n",
        [Path, Path]),
    build_run(Dir, 'shared_registry', Src, "trigger\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1 one 1 two\n"),
    !.

% Growing the transient record buffer must not leave direct `$0` printers with
% the driver's stale pre-getline pointer.
test(record_getline_long_record_prints, [condition(clang_available)]) :-
    rdir(Dir),
    length(Codes, 5000),
    maplist(=(0'x), Codes),
    string_codes(Long, Codes),
    string_concat(Long, "\n", Contents),
    data_file(Dir, 'long.txt', Contents, Path),
    format(atom(Src), "{ getline < \"~w\"; print $0 }\n", [Path]),
    build_run(Dir, 'long_record', Src, "t\n", Out, St),
    assertion(St == 0),
    string_concat(Long, "\n", Expected),
    assertion(Out == Expected),
    !.

% --- explicit v1 rejection boundary ---------------------------------------

test(unsupported_getline_shapes_exit_3, [condition(clang_available)]) :-
    rdir(Dir),
    Cases = [ plain-"{ getline }\n",
              main_var-"{ getline v }\n",
              pipe-"{ cmd | getline }\n",
              dynamic_file-"{ getline < filename }\n",
              dynamic_field-"{ getline < $1 }\n",
              dynamic_paren-"{ getline < (filename) }\n",
              dynamic_scalar_target-"{ getline v < $1 }\n",
              capture_main-"{ r = getline }\n",
              capture_dynamic-"{ r = getline < $1 }\n",
              while_main-"{ while ((getline) > 0) print $0 }\n",
              while_main_var-"{ while ((getline v) > 0) print $0 }\n",
              while_dynamic-"{ while ((getline < filename) > 0) print $0 }\n",
              begin_ctx-"BEGIN { getline < \"x\" } { print $0 }\n",
              begin_capture-"BEGIN { r = getline < \"x\" } { print $0 }\n",
              end_ctx-"{ print $0 } END { getline < \"x\" }\n",
              end_capture-"{ print $0 } END { r = getline < \"x\" }\n"
            ],
    forall(member(Name-Src, Cases),
        ( build_status(Dir, Name, Src, Status),
          assertion(Status == exit(3))
        )),
    !.

:- end_tests(plawk_getline_record).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

rdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_getline_record', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

data_file(Dir, Name, Contents, Path) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
        format(S, "~s", [Contents]), close(S)).

write_prog(Dir, Name, Src, Bin-Prog) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        format(S, "~s", [Src]), close(S)),
    atom_concat(Prog0, '_bin', Bin).

build_status(Dir, Name, Src, Status) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Status).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~s", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
