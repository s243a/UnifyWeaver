:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% String builtins over the RETAINED last record in END:
% `toupper` / `tolower` / `substr` / `index` of `$0` or `$N`.
%
% ---------------------------------------------------------------------------
% WHY THESE DECLINED, AND WHAT CHANGED
%
% In awk, END sees the last record: `END { print toupper($1) }` prints the last
% line's first field upper-cased. plawk already retained the last record for END
% field reads, NF and length, and the retention gate (plawk_end_term_reads_record/1)
% already admitted these builtins -- they contain a field term. The END rewrite
% (plawk_end_lastrec_rewrite/2) turned `toupper(field(1))` into
% `toupper(end_lastrec_field(1))` through its generic compound walk, but no emitter
% had a row for that term, so every END form of these builtins declined (exit 3).
%
% The fix adds those rows: each is the in-loop row with `%line` swapped for the
% re-materialised retained record, so the runtime call doing the work is the same
% one a rule body makes. The straight-line END print, the END `if` branch and END
% printf all reach the SAME rows (plawk_end_lastrec_builtin_parts/6).
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("5 boot disk\n7 Disk io\nhello world\n").

:- begin_tests(plawk_end_record_builtins).

% --- each builtin, END-only (no rules) ------------------------------------

test(toupper_of_a_field, [condition(clang_available)]) :-
    run("END { print toupper($1) }\n", "HELLO\n"),
    !.

test(tolower_of_a_field, [condition(clang_available)]) :-
    run("END { print tolower($2) }\n", "world\n"),
    !.

test(case_of_the_whole_record, [condition(clang_available)]) :-
    run("END { print toupper($0) }\n", "HELLO WORLD\n"),
    !,
    run_with("AbC dEf\n", "END { print tolower($0) }\n", "abc def\n"),
    !.

test(substr_three_arg, [condition(clang_available)]) :-
    run("END { print substr($0, 1, 3) }\n", "hel\n"),
    !.

test(substr_two_arg_is_the_tail, [condition(clang_available)]) :-
    run("END { print substr($0, 7) }\n", "world\n"),
    !,
    run("END { print substr($2, 2) }\n", "orld\n"),
    !.

test(index_of_the_record_and_a_field, [condition(clang_available)]) :-
    run("END { print index($0, \"wor\") }\n", "7\n"),
    !,
    run("END { print index($1, \"l\") }\n", "3\n"),
    !.

test(index_absent_needle_is_zero, [condition(clang_available)]) :-
    run("END { print index($0, \"zzz\") }\n", "0\n"),
    !.

% --- with rules: the scalar driver and the assoc driver -------------------

test(with_a_counter_rule, [condition(clang_available)]) :-
    run("{ n++ } END { print toupper($1), n }\n", "HELLO 3\n"),
    !,
    run("{ n++ } END { print substr($0, 1, 3), n }\n", "hel 3\n"),
    !.

test(with_an_assoc_rule, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print toupper($1), c[\"hello\"] }\n", "HELLO 1\n"),
    !.

test(with_a_pattern_rule, [condition(clang_available)]) :-
    run("/boot/ { print } END { print toupper($1) }\n", "5 boot disk\nHELLO\n"),
    !.

% --- composition: several fields, concatenation, several prints ----------

test(several_builtins_in_one_print, [condition(clang_available)]) :-
    run("END { print toupper($1), substr($0, 1, 3), tolower($2), index($0, \"wor\") }\n",
        "HELLO hel world 7\n"),
    !.

test(inside_a_concatenation, [condition(clang_available)]) :-
    run("END { print \"p=\" index($0, \"wor\") \"!\" }\n", "p=7!\n"),
    !,
    run("END { print substr($2, 2, 2) toupper($1) }\n", "orHELLO\n"),
    !.

% Two `index` calls in two prints: each needle is a module global, and the
% multi-print renamer must keep them distinct (and the block's reference to each
% must agree with the global the globals pass defined). Different needles make a
% crossed wire visible in the output.
test(needle_globals_stay_distinct_across_prints, [condition(clang_available)]) :-
    run("END { print index($0, \"o\"); print index($0, \"w\"); print substr($0, 3) }\n",
        "5\n7\nllo world\n"),
    !,
    run("END { print index($0, \"o\") \"-\" index($0, \"d\"), index($0, \"l\") }\n",
        "5-11 3\n"),
    !.

% --- END `if` branches reach the same rows ---------------------------------

test(inside_an_end_if, [condition(clang_available)]) :-
    run("END { if (NR > 0) print toupper($1), index($0, \"d\") }\n", "HELLO 11\n"),
    !.

% --- printf: substr and index as arguments ---------------------------------

test(printf_substr_and_index, [condition(clang_available)]) :-
    run("END { printf \"%s-%d\\n\", substr($0, 1, 3), index($0, \"w\") }\n",
        "hel-7\n"),
    !,
    run("END { printf \"[%s][%d][%s]\\n\", substr($2, 2), index($1, \"ll\"), $1 }\n",
        "[orld][3][hello]\n"),
    !.

test(printf_then_print, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d %s\\n\", n, substr($0, 7, 3); print index($0, \"o\") }\n",
        "3 wor\n5\n"),
    !.

% --- edge inputs ---------------------------------------------------------

% No records: awk's END sees an empty $0, so every builtin yields empty / 0.
test(empty_input, [condition(clang_available)]) :-
    run_with("",
        "END { print toupper($1) \"|\" index($0, \"a\") \"|\" substr($0, 2) \"|\" }\n",
        "|0||\n"),
    !,
    run_with("", "END { printf \"<%s|%d>\\n\", substr($0, 1, 2), index($0, \"x\") }\n",
        "<|0>\n"),
    !.

% A field past NF is empty.
test(missing_field_is_empty, [condition(clang_available)]) :-
    run("END { print substr($5, 1, 2) \"|\" toupper($9) \"|\" index($7, \"a\") }\n",
        "||0\n"),
    !.

% A non-default FS is honoured by the retained-record read.
test(custom_field_separator, [condition(clang_available)]) :-
    run_with("a:b:c\nx:Yy:zz\n",
        "BEGIN { FS = \":\" } END { print toupper($2), substr($3, 2), index($0, \"z\") }\n",
        "YY z 6\n"),
    !.

% --- the source is the RETAINED record, not the loop's %line --------------

test(ir_reads_the_retained_record) :-
    ir_of("END { print toupper($1), index($0, \"o\"), substr($0, 1, 2) }\n", IR),
    sub_atom(IR, _, _, _, 'call i64 @plawk_lastrec_transient()'),
    sub_atom(IR, _, _, _, '@wam_atom_field_subslice_value'),
    sub_atom(IR, _, _, _, '@wam_atom_field_index_value'),
    sub_atom(IR, _, _, _, '@wam_print_ascii_upper_slice'),
    !.

% --- still declined, cleanly -------------------------------------------

% toupper/tolower lower to a PRINT-ONLY case transform, so as a printf argument they
% decline -- as they do in a record-context printf.
test(case_builtin_as_printf_arg_declines) :-
    build_status("END { printf \"%s\\n\", toupper($1) }\n", 3),
    !,
    build_status("{ printf \"%s\\n\", toupper($1) }\n", 3),
    !.

% Nearby shapes that are NOT covered here (mixed-driver END field reads, END loops,
% computed arguments, assignment of a builtin). Whatever each does, it must not be a
% clang failure: a program either builds, or declines / fails to parse.
test(nearby_shapes_do_not_miscompile) :-
    forall(member(Src,
            [ "{ print $2; c[$1]++ } END { print substr($0, 1, 2) }\n",
              "{ n++; c[$1]++ } END { print tolower($2), n }\n",
              "END { while (i < 2) { print substr($0, 1, 2); i++ } }\n",
              "END { print substr($0, NR) }\n",
              "END { print index($0, $2) }\n",
              "END { print toupper(x) }\n",
              "END { x = toupper($1); print x }\n",
              "END { if (NR > 0) printf \"%d\\n\", index($0, \"o\") }\n"
            ]),
        build_status_not_4(Src)),
    !.

:- end_tests(plawk_end_record_builtins).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_record_builtins', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'erb_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'erb', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status(Src, ExpectedStatus) :-
    build_status_of(Src, Status),
    ( Status == ExpectedStatus
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, ExpectedStatus]), fail
    ).

build_status_not_4(Src) :-
    build_status_of(Src, Status),
    ( memberchk(Status, [0, 2, 3])
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w (4 = clang miscompile)~n",
           [Src, Status]), fail
    ).

build_status_of(Src, Status) :-
    odir(Dir),
    directory_file_path(Dir, 'erb_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).

ir_of(Src, IR) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
