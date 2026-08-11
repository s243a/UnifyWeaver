:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A bare scalar variable INSIDE an END concatenation -- `END { print "s=" s }`,
% `END { print s "!" }` -- for every slot kind.
%
% ---------------------------------------------------------------------------
% A THIRD COPY, AND WHAT MADE IT INVISIBLE
%
% This closes WRONG OUTPUT, not a decline. Before the fix:
%
%     { s = $2 } END { print s }              -> disk      correct
%     { s = $2 } END { print "s=" s }         -> s=25      WRONG   (gawk: s=disk)
%     { s = $2 } END { print s "!" }          -> 25!       WRONG   (gawk: disk!)
%     { if (0) s = $2 } END { print "s=" s }  -> s=0       WRONG   (gawk: s=)
%
% 25 is the interned ATOM ID of the text, printed as an i64. Standalone
% `END { print s }` went through plawk_end_scalar_var_print_lines/4, which dispatches
% on SLOT KIND: string / strnum slots resolve the id to text (id 0 rendering EMPTY),
% counters and doubles go through plawk_end_numeric_print_lines/6. The concat part
% emitter, plawk_end_field_print_lines(var(Name), ...), had resolved the slot itself
% and then called the NUMERIC render unconditionally -- one arm of a three-arm
% dispatch. It now calls the shared emitter, so the copy is gone.
%
% This is the SECOND time this same emitter was found duplicated. The first was the
% mixed walker's copy (tests/test_plawk_mixed_scalar_print.pl), and the failure mode
% was identical: a copy that did not merely lag behind on new behaviour, but silently
% un-did SHIPPED behaviour for a subset of programs. Two things are new and worth
% recording, because both are about why nobody noticed:
%
%  1. A COMMENT ASSERTING AGREEMENT IS NOT EVIDENCE OF IT. The copy carried
%     "the SAME numeric render the standalone END print uses, so a counter or double
%     in a concatenation cannot disagree with one printed on its own about whether an
%     unset value is empty". Every word of that was TRUE -- and it named two of the
%     three arms. The claim reads as a finished audit of the emitter while covering
%     the two cases that worked. A note about sharing should name what is NOT shared.
%
%  2. TWO OF THREE ARMS COVERED READS AS COVERED. Two suites already exercised a
%     scalar in an END concat, and both picked a working arm:
%     tests/test_plawk_concat.pl's `concat_in_end` uses `s += $1` (a counter), and
%     tests/test_plawk_unset_scalar.pl pins the unset render for a counter and for a
%     double. The construct looked tested from every angle except the one that was
%     broken. So this suite walks the slot kinds DELIBERATELY, as a row, rather than
%     testing "a scalar in a concat" and picking a representative -- the choice of
%     representative is exactly what hid this for as long as it was hidden.
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

% Three records; the LAST is "7 disk". $2 of it is "disk" -- text that is NOT a
% number, so an atom id printed as an i64 cannot coincidentally look right. $1 is 7
% and c["5"] is 2.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_end_concat_scalar).

% --- the slot kinds, as a row -------------------------------------------
%
% string / strnum are the arm that was broken; counter and double are the arm that
% worked. All four are here so the row moves together if it moves again.

test(strnum_slot_from_a_field, [condition(clang_available)]) :-
    run("{ s = $2 } END { print \"s=\" s }\n", "s=disk\n"),
    !.

test(string_slot_from_a_literal, [condition(clang_available)]) :-
    run("{ s = \"lit\" } END { print \"s=\" s }\n", "s=lit\n"),
    !.

test(counter_slot, [condition(clang_available)]) :-
    run("{ n++ } END { print \"n=\" n }\n", "n=3\n"),
    !.

test(double_slot, [condition(clang_available)]) :-
    run("{ x = $1 + 0.5 } END { print \"x=\" x }\n", "x=7.5\n"),
    !.

% A strnum slot holding text that LOOKS numeric still prints as its text, not as the
% id: `$1` of the last record is 7, and the id happens to be a small integer too, so
% this is the case where a wrong render is hardest to spot by eye.
test(strnum_slot_whose_text_is_numeric, [condition(clang_available)]) :-
    run("{ s = $1 } END { print \"s=\" s }\n", "s=7\n"),
    !.

% --- unset renders empty, for the same row -----------------------------
%
% Unset was the second wrong output: id 0 printed as the number 0. A string slot's
% unset renders EMPTY, exactly as the standalone print already did.

test(unset_strnum_slot_is_empty, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") s = $2 } END { print \"s=\" s }\n", "s=\n"),
    !.

test(unset_string_slot_is_empty, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") s = \"q\" } END { print \"s=\" s }\n", "s=\n"),
    !.

test(unset_counter_is_empty, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") n++ } END { print \"n=\" n }\n", "n=\n"),
    !.

test(unset_double_is_empty, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") x += 1.5 } END { print \"x=\" x }\n", "x=\n"),
    !.

% --- the scalar's POSITION in the concat --------------------------------
%
% The part emitter is indexed per part, so a scalar must render the same whether it
% leads, trails, or sits between literals -- and twice in one concat must not collide.

test(scalar_trailing, [condition(clang_available)]) :-
    run("{ s = $2 } END { print s \"!\" }\n", "disk!\n"),
    !.

test(scalar_between_literals, [condition(clang_available)]) :-
    run("{ s = $2 } END { print \"[\" s \"]\" }\n", "[disk]\n"),
    !.

test(the_same_scalar_twice_in_one_concat, [condition(clang_available)]) :-
    run("{ s = $2 } END { print \"a\" s \"b\" s \"c\" }\n", "adiskbdiskc\n"),
    !.

% Two concats in one print: each gets its own separator, and the per-part indices of
% one must not collide with the other's.
test(two_concats_in_one_print, [condition(clang_available)]) :-
    run("{ s = $2; n++ } END { print \"s=\" s, \"n=\" n }\n", "s=disk n=3\n"),
    !.

% --- every END route that reaches the shared concat emitter -------------
%
% plawk_end_field_print_lines/4 is reached from the scalar-only walker, the mixed
% walker and the assoc walker, so the fix lands in all three at once. Each route is
% pinned, because "fixed in the route I probed" is how the first copy survived.

test(mixed_route_with_an_assoc_read, [condition(clang_available)]) :-
    run("{ s = $2; c[$1]++ } END { print \"s=\" s, c[\"5\"] }\n", "s=disk 2\n"),
    !.

test(statement_list_route, [condition(clang_available)]) :-
    run("{ s = $2 } END { print \"s=\" s; print \"again=\" s }\n",
        "s=disk\nagain=disk\n"),
    !.

% A concat beside a retained-record read: the concat part and the field projection
% share the print, and neither disturbs the other's naming.
test(beside_a_retained_record_read, [condition(clang_available)]) :-
    run("{ s = $2 } END { print \"s=\" s, \"f=\" $1 }\n", "s=disk f=7\n"),
    !.

% --- one emitter, both positions ---------------------------------------
%
% The property the fix buys, checked on the emitted instructions rather than only on
% output: a scalar printed on its own and the same scalar inside a concat must lower
% through the SAME render. Compared on the string arm, which is the one that drifted.
%
% The concat is `print s "!"` rather than `print "x" s` so the SCALAR IS PART 0. The
% part emitter is indexed CombinedIndex = PrintIndex*1000 + PartIndex, so a leading
% scalar gets index 0 -- the same index the standalone print's only field gets -- and
% the two line sets compare byte-for-byte with no name normalising. (With the scalar
% in second position the indices are 0 and 1 by construction, and any normaliser
% written to hide that is one more thing that can quietly stop matching. Choosing
% comparable inputs beats normalising incomparable ones.)
test(standalone_and_concat_emit_the_same_string_render) :-
    str_lines("{ s = $2 } END { print s }\n", Standalone),
    str_lines("{ s = $2 } END { print s \"!\" }\n", InConcat),
    assertion(Standalone \== []),
    assertion(Standalone == InConcat),
    !.

% ...and the numeric arm is unchanged by the fix, which is why the 29 pre-existing
% golden-corpus programs stayed byte-identical.
%
% Checked on the CONSTRUCT'S OWN names: the absence of `end_str_` in the emitted
% print, not the absence of `wam_atom_to_string`. The first draft asserted the latter
% and failed, because that runtime function is declared module-wide whatever the
% program does -- a driver-wide mnemonic, which is never a valid pin for one
% construct's behaviour.
test(a_counter_in_a_concat_still_uses_the_numeric_render) :-
    build_ll("{ n++ } END { print \"n=\" n }\n", LL),
    assertion(sub_string(LL, _, _, _, "printed_end_i64_")),
    assertion(\+ sub_string(LL, _, _, _, "end_str_")),
    !.

% The string arm's three instructions must all be present -- resolve, the id-0 test,
% and the select that turns unset into the empty C string. A resolve without the
% select would print whatever atom 0 is.
test(the_string_arm_emits_resolve_test_and_select) :-
    build_ll("{ s = $2 } END { print \"s=\" s }\n", LL),
    assertion(sub_string(LL, _, _, _, "call i8* @wam_atom_to_string")),
    assertion(sub_string(LL, _, _, _, "_empty_")),
    assertion(sub_string(LL, _, _, _, "select i1 %end_str_empty_")),
    !.

% --- boundaries, pinned with their kind --------------------------------

% A for-in loop VARIABLE in a concat still declines. Not this defect and not a slot
% kind: `k` is the iteration key, not a scalar slot, so it is a missing PART KIND in
% the shared concat emitter -- the same kind of gap as the assoc read pinned in
% tests/test_plawk_assoc_end_record.pl, and a follow-on rather than a narrowing.
% Paired with the working plain form so the difference is visible.
test(a_for_in_loop_variable_in_a_concat_declines) :-
    build_status("{ c[$1]++ } END { for (k in c) print \"k=\" k }\n", 3),
    !.

test(the_same_loop_variable_printed_plainly_works, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k }\n", "5\n7\n"),
    !.

% printf `%s` of a scalar was never wrong: plawk_end_printf_arg/7 carries its own
% slot-kind dispatch WITH the string arm. It is a third implementation of the same
% decision, but it produces printf call-arguments rather than print instructions, so
% it is not the copy this change deleted. Pinned so a future merge of the two has a
% statement of what printf already does.
test(printf_of_a_string_scalar_in_end_was_already_right,
        [condition(clang_available)]) :-
    run("{ s = $2 } END { printf \"s=%s\\n\", s }\n", "s=disk\n"),
    !,
    run("{ if ($1 == \"ZZZ\") s = $2 } END { printf \"s=%s\\n\", s }\n", "s=\n"),
    !.

:- end_tests(plawk_end_concat_scalar).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_concat_scalar', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    run_raw(Input, Src, Out),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
            [Src, Out, Expected]), fail
    ).

% for-in iteration order is unspecified in awk, so compare sorted lines.
run_sorted(Src, Expected) :-
    input(Input),
    run_raw(Input, Src, Out),
    sorted_lines(Out, Got),
    sorted_lines(Expected, Want),
    (   Got == Want
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
            [Src, Got, Want]), fail
    ).

sorted_lines(Text, Sorted) :-
    split_string(Text, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, Sorted).

run_raw(Input, Src, Out) :-
    odir(Dir),
    directory_file_path(Dir, 'ec_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ec', Prog0),
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

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ec_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

% The string-render instructions an END scalar print contributes, by their own
% generated names (`%end_str_*` / `%printed_end_str_*`, which plawk_end_scalar_var_
% print_lines/4 names). Raw lines: the callers choose inputs whose print indices
% agree, so there is nothing to normalise.
str_lines(Src, Lines) :-
    build_ll(Src, LL),
    split_string(LL, "\n", " ", All),
    include(str_line, All, Lines).

str_line(Line) :-
    sub_string(Line, _, _, _, "end_str_"),
    !.

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
