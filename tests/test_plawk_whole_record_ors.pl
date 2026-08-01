:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: the WHOLE-RECORD print honours ORS.
%
% `ORS` was already correct for every print that names its fields -- `print $1`,
% `print $1, $2`, `print NR`, `print "x"`, an END print -- because those go through
% plawk_print_fields_ir//4, whose base case emits the ORS terminator (load
% @plawk_ors_ptr, printf "%s"). The WHOLE-RECORD print did not:
%
%   BEGIN { ORS = "|" } { print }        emitted "a 1\n"   gawk: "a 1|"
%   BEGIN { ORS = "|" } { print $0 }     emitted "a 1\n"   gawk: "a 1|"
%
% Four separate emitters print a whole record without slicing it into fields, and
% ALL FOUR formatted it with a `"%s\n"` global (@.plawk_surface_print_line) --
% a newline baked into the format, so no ORS could reach it:
%
%   plawk_print_action_ir/4           the single-action driver     { print }
%   plawk_prefixed_print_action_ir/5  a print among statements     { n++; print }
%   the gsub driver's PrintIR         { gsub(/a/,"z"); print }
%   the field-assign driver's RecordIR  BEGIN{FS=","} { $1="z"; print }
%
% Each now prints the record with plain `%s` and then calls the SAME
% plawk_ors_terminator_ir/4 the general path uses -- one terminator emitter, four
% callers, so the ORS cannot be honoured in one spelling and dropped in another.
% This is why the fix is four small edits and no new runtime: the ORS already
% lives in a pointer global (@plawk_ors_ptr) that every driver emits, so it needs
% no threading through these emitters and no per-emitter length.
%
% The bug was most visible via the bare `print`, which desugars to `print $0` and
% so is the idiomatic spelling of exactly the broken path. tests/test_plawk_bare_print.pl
% pinned bare-vs-explicit EQUIVALENCE under ORS without blessing the wrong output;
% both sides moved together here, so that test still holds, and this suite asserts
% against gawk.
%
% Note OFS and ORS are different separators and remain so: OFS goes BETWEEN a
% print's fields, ORS AFTER the record. A whole-record print has no gaps, so OFS
% is irrelevant to it (`BEGIN{OFS="-"} {print}` does not rewrite `$0`) -- except
% in the field-assign driver, which rebuilds the record by joining fields with OFS
% and then terminates with ORS. Both are pinned.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3".
input("a 1\nb 2\nc 3\n").

% The same three records comma-separated, for the explicit-FS drivers (the
% field-assign driver requires a non-space FS).
csv_input("a,1\nb,2\nc,3\n").

:- begin_tests(plawk_whole_record_ors).

% --- the single-action driver: { print } / { print $0 } --------------------

test(bare_print_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { print }\n", "a 1|b 2|c 3|"),
    !.

test(explicit_whole_record_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { print $0 }\n", "a 1|b 2|c 3|"),
    !.

% Under a pattern, so only some records print -- the terminator rides the print,
% not the loop.
test(whole_record_under_pattern_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } /b/ { print }\n", "b 2|"),
    !.

% An EMPTY ORS: the records run together with nothing between them. This is the
% case a fixed-length format could never express.
test(empty_ors_on_whole_record, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"\" } { print }\n", "a 1b 2c 3"),
    !.

% A MULTI-BYTE ORS, including one containing a newline.
test(multi_char_ors_on_whole_record, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"--\\n\" } { print }\n", "a 1--\nb 2--\nc 3--\n"),
    !.

% The ORS is DATA, printed via `%s` -- a `%` in it is not a format directive.
test(percent_in_ors_is_data, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"%d\" } { print }\n", "a 1%db 2%dc 3%d"),
    !.

% --- the prefixed driver: a whole-record print among statements -----------

test(whole_record_after_statement_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { n++; print }\n", "a 1|b 2|c 3|"),
    !.

% Twice in one rule: each print gets its OWN terminator, under its own statement
% prefix, so the variable names cannot collide.
test(two_whole_record_prints_each_terminate, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { print; print }\n",
        "a 1|a 1|b 2|b 2|c 3|c 3|"),
    !.

test(whole_record_in_braceless_if_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { if (NR == 2) print }\n", "b 2|"),
    !.

% --- the gsub driver: print the REWRITTEN record --------------------------

test(gsub_whole_record_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { gsub(/a/, \"z\"); print }\n", "z 1|b 2|c 3|"),
    !.

test(sub_whole_record_honours_empty_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"\" } { sub(/a/, \"z\"); print $0 }\n", "z 1b 2c 3"),
    !.

test(gsub_whole_record_multi_char_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"<>\" } { gsub(/[abc]/, \"z\"); print }\n",
        "z 1<>z 2<>z 3<>"),
    !.

% --- the field-assign driver: print the REBUILT record --------------------

% This driver needs an explicit (non-space) FS, so all of these set one.

test(field_assign_whole_record_honours_ors, [condition(clang_available)]) :-
    run_csv("BEGIN { FS = \",\"; ORS = \"|\" } { $1 = \"z\"; print }\n",
        "z 1|z 2|z 3|"),
    !.

% OFS and ORS at once, and they are different things: OFS joins the rebuilt
% record's fields, ORS terminates it.
test(field_assign_ofs_and_ors_compose, [condition(clang_available)]) :-
    run_csv("BEGIN { FS = \",\"; OFS = \"-\"; ORS = \"|\" } { $1 = \"z\"; print $0 }\n",
        "z-1|z-2|z-3|"),
    !.

test(field_assign_empty_ors, [condition(clang_available)]) :-
    run_csv("BEGIN { FS = \",\"; ORS = \"\" } { $1 = \"z\"; print }\n",
        "z 1z 2z 3"),
    !.

% --- OFS does NOT leak into a whole-record print -------------------------

% `print` emits `$0` verbatim; awk only rebuilds `$0` with OFS when a field is
% ASSIGNED. So a bare OFS setting must not change the record's interior.
test(ofs_does_not_rewrite_the_record, [condition(clang_available)]) :-
    run("BEGIN { OFS = \"-\"; ORS = \"|\" } { print }\n", "a 1|b 2|c 3|"),
    !.

% --- regressions: the prints that were already correct -------------------

test(default_ors_whole_record_unchanged, [condition(clang_available)]) :-
    run("{ print }\n", "a 1\nb 2\nc 3\n"),
    !.

test(default_ors_explicit_whole_record_unchanged, [condition(clang_available)]) :-
    run("{ print $0 }\n", "a 1\nb 2\nc 3\n"),
    !.

test(default_ors_gsub_whole_record_unchanged, [condition(clang_available)]) :-
    run("{ gsub(/a/, \"z\"); print }\n", "z 1\nb 2\nc 3\n"),
    !.

test(default_ors_field_assign_unchanged, [condition(clang_available)]) :-
    run_csv("BEGIN { FS = \",\" } { $1 = \"z\"; print }\n", "z 1\nz 2\nz 3\n"),
    !.

test(single_field_ors_unchanged, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { print $1 }\n", "a|b|c|"),
    !.

test(field_list_ofs_and_ors_unchanged, [condition(clang_available)]) :-
    run("BEGIN { OFS = \"-\"; ORS = \"|\" } { print $1, $2 }\n", "a-1|b-2|c-3|"),
    !.

test(nr_print_ors_unchanged, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { print NR }\n", "1|2|3|"),
    !.

test(end_print_ors_unchanged, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" } { n++ } END { print n }\n", "3|"),
    !.

% --- equivalence: whole-record and single-field terminate alike -----------

% The defect was an ASYMMETRY -- the two spellings disagreed on the terminator.
% Asserted directly: for the same ORS, the bytes after each record are the same
% whether the print names `$0` or `$1`.
test(whole_record_and_field_agree_on_the_terminator,
        [condition(clang_available)]) :-
    forall(member(Ors, ["|", "", "<>"]),
        ( format(string(WholeSrc),
              "BEGIN { ORS = \"~w\" } { print }\n", [Ors]),
          format(string(FieldSrc),
              "BEGIN { ORS = \"~w\" } { print $1 }\n", [Ors]),
          % `$0` is "a 1"/"b 2"/"c 3", `$1` is "a"/"b"/"c"; the ORS after each
          % record is the same string in both.
          atomics_to_string(["a 1", Ors, "b 2", Ors, "c 3", Ors], WholeExpected),
          atomics_to_string(["a", Ors, "b", Ors, "c", Ors], FieldExpected),
          run_out(WholeSrc, WholeOut),
          run_out(FieldSrc, FieldOut),
          assertion(WholeOut == WholeExpected),
          assertion(FieldOut == FieldExpected)
        )),
    !.

% --- IR shape ------------------------------------------------------------

% The whole-record print now LOADS the ORS pointer, exactly as a field print does.
test(whole_record_ir_loads_the_ors_pointer) :-
    plawk_parse_string("{ print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'load i8*, i8** @plawk_ors_ptr'))),
    !.

% ...and no longer USES the newline-baked format global. The global is still
% DEFINED in the runtime-globals block (other emitters reference it), so the test
% pins the getelementptr USE (`[4 x i8]* @.plawk_surface_print_line`) rather than
% the definition (`[4 x i8] c"..."`).
test(whole_record_ir_does_not_use_the_newline_format) :-
    plawk_parse_string("{ print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '[4 x i8]* @.plawk_surface_print_line')),
    !.

% Same for a whole-record print among statements (the prefixed emitter).
test(prefixed_whole_record_ir_loads_the_ors_pointer) :-
    plawk_parse_string("{ n++; print }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'load i8*, i8** @plawk_ors_ptr'))),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '[4 x i8]* @.plawk_surface_print_line')),
    !.

% Same for the gsub driver.
test(gsub_whole_record_ir_loads_the_ors_pointer) :-
    plawk_parse_string("{ gsub(/a/, \"z\"); print }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'load i8*, i8** @plawk_ors_ptr'))),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '[4 x i8]* @.plawk_surface_print_line')),
    !.

% Same for the field-assign driver.
test(field_assign_ir_loads_the_ors_pointer) :-
    plawk_parse_string("BEGIN { FS = \",\" } { $1 = \"z\"; print }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'load i8*, i8** @plawk_ors_ptr'))),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '[4 x i8]* @.plawk_surface_print_line')),
    !.

% The terminator is ONE emitter. Its three lines are exactly what every caller
% appends, so a caller cannot invent its own spelling.
test(one_shared_terminator_emitter) :-
    plawk_native_codegen:plawk_ors_terminator_ir(f, p, r, Lines),
    % destructured OUTSIDE the assertion: assertion/1 does not propagate bindings
    length(Lines, 3),
    Lines = [FmtLine, LoadLine, PrintLine],
    assertion(sub_atom(FmtLine, _, _, _, '@.plawk_surface_print_string')),
    assertion(sub_atom(LoadLine, _, _, _, '@plawk_ors_ptr')),
    assertion(sub_atom(PrintLine, _, _, _, '@printf')),
    !.

% Every whole-record emitter appends those three lines. Checked by counting the
% ORS loads: one per print, so two prints in a rule give two.
test(each_whole_record_print_gets_its_own_terminator) :-
    plawk_parse_string("{ print; print }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    findall(B, sub_atom(DriverIR, B, _, _, 'load i8*, i8** @plawk_ors_ptr'),
        Loads),
    length(Loads, N),
    assertion(N >= 2),
    !.

:- end_tests(plawk_whole_record_ors).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_whole_record_ors', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_input(Src, Input, Out),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

run_csv(Src, Expected) :-
    csv_input(Input),
    run_input(Src, Input, Out),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

run_out(Src, Out) :-
    input(Input),
    run_input(Src, Input, Out).

run_input(Src, Input, Out) :-
    odir(Dir),
    directory_file_path(Dir, 'wro_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'wro', Prog0),
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
    process_wait(Pid, exit(0)).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
