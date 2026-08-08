:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `END { print NAME }` for a bare scalar, in the MIXED route -- a program whose rules also
% touch an associative table.
%
% ---------------------------------------------------------------------------
% TWO GAPS THAT WERE NOT MISSING FEATURES
%
% The bare-scalar END print was implemented TWICE: once in the scalar-only END walker and
% once in the mixed (scalar + assoc) walker. Only the scalar copy was ever taught anything
% after the split, so the same surface construct behaved differently depending on whether
% the program happened to touch an assoc table:
%
%                                                       plawk      gawk
%   { if ($1=="ZZZ") n++ } END { print n }              (empty)    (empty)   ok
%   { if ($1=="ZZZ") n++; c[$1]++ } END { print n, . }   0          (empty)   WRONG
%   { s = $1 } END { print s }                          the text   the text  ok
%   { s = $1; c[$1]++ } END { print s, . }              DECLINE    the text  gap
%
% Neither was a missing feature. Both behaviours existed ten lines away in the other copy:
% unset-renders-empty was designed and landed deliberately, and a string/strnum slot print
% was already there. The mixed copy simply never received either, and it could not be seen
% from inside either suite, because each suite exercises the route it owns.
%
% The fix was to DELETE the second copy: both walkers now call
% plawk_end_scalar_var_print_lines/4, and both divergences closed without either being
% implemented. That is the case for generalising rather than patching -- a duplicate does
% not merely fail to gain new behaviour, it silently un-does shipped behaviour for a subset
% of programs. One emitter cannot drift from itself.
%
% This suite pins the MIXED route specifically. It deliberately duplicates cases that
% tests/test_plawk_unset_scalar.pl already pins for the scalar route: the whole defect was
% two routes diverging, so the coverage has to be per-route and stated twice. A shared
% harness parameterised over "the route" would re-introduce exactly the assumption that
% hid the bug.
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

% $1 is 5, 5, 7 -- so c["5"] is 2 and c["7"] is 1; no record has $1 == "ZZZ".
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_mixed_scalar_print).

% --- an UNSET counter renders empty in the mixed route too ---------------

% WAS: `0`. The mixed copy printed the slot as a plain i64, so a never-assigned counter
% showed its zero-initialised register instead of awk's empty string.
test(an_unset_counter_prints_empty_beside_an_assoc_read,
        [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") n++; c[$1]++ } END { print n, c[\"5\"] }\n", " 2\n"),
    !.

% The counter first, so the missing value is not merely a trailing empty field.
test(an_unset_counter_prints_empty_before_another_field,
        [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") n++; c[$1]++ } END { print c[\"5\"], n }\n", "2 \n"),
    !.

% An ASSIGNED counter still prints its value -- presence, not value, is what renders
% empty, so a real zero must still print `0`.
test(an_assigned_counter_prints_its_value, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print n, c[\"5\"] }\n", "3 2\n"),
    !.

test(a_counter_assigned_zero_still_prints_zero, [condition(clang_available)]) :-
    run("{ n = 0; c[$1]++ } END { print n, c[\"5\"] }\n", "0 2\n"),
    !.

% --- a STRING scalar prints its text in the mixed route -----------------

% WAS: a clean decline (exit 3). The mixed copy demanded a `scalar_counter` slot, so a
% field-assigned (strnum) scalar had no clause at all.
test(a_field_assigned_scalar_prints_its_text, [condition(clang_available)]) :-
    run("{ s = $1; c[$1]++ } END { print s, c[\"5\"] }\n", "7 2\n"),
    !.

test(a_literal_assigned_scalar_prints_its_text, [condition(clang_available)]) :-
    run("{ s = \"tag\"; c[$1]++ } END { print s, c[\"5\"] }\n", "tag 2\n"),
    !.

% An UNSET string scalar is empty, not the atom-id-0 text.
test(an_unset_string_scalar_prints_empty, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") s = $2; c[$1]++ } END { print s, c[\"5\"] }\n", " 2\n"),
    !.

% --- a DOUBLE scalar in the mixed route ---------------------------------

test(a_double_scalar_prints_in_the_mixed_route, [condition(clang_available)]) :-
    run("{ x = $1 / 2; c[$1]++ } END { print x, c[\"5\"] }\n", "3.5 2\n"),
    !.

% --- the scalar-only route is unchanged ---------------------------------
%
% Stated here as well as in its own suite, deliberately: these are the DONOR behaviours,
% and the point of the change is that both routes now produce them from one emitter. If a
% future edit breaks the donor, this suite says so too.

test(scalar_only_unset_counter_still_empty, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") n++ } END { print n }\n", "\n"),
    !.

test(scalar_only_string_scalar_still_prints_text, [condition(clang_available)]) :-
    run("{ s = $1 } END { print s }\n", "7\n"),
    !.

test(scalar_only_assigned_counter_still_prints_value, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "3\n"),
    !.

% --- the two routes emit the SAME instructions for the same print -------
%
% The structural pin. A bare-scalar END print must lower identically whether or not the
% program touches an assoc table -- that equality is what the shared emitter buys, and its
% absence is what the two gaps above were. Compared on the instruction lines the scalar
% print emits, since the surrounding programs necessarily differ.
test(both_routes_emit_the_same_bare_scalar_print) :-
    scalar_print_lines("{ s = $1 } END { print s }\n", ScalarOnly),
    scalar_print_lines("{ s = $1; c[$1]++ } END { print s, c[\"5\"] }\n", Mixed),
    assertion(ScalarOnly \== []),
    assertion(ScalarOnly == Mixed),
    !.

test(both_routes_emit_the_same_unset_counter_print) :-
    scalar_print_lines("{ if ($1 == \"ZZZ\") n++ } END { print n }\n", ScalarOnly),
    scalar_print_lines("{ if ($1 == \"ZZZ\") n++; c[$1]++ } END { print n, c[\"5\"] }\n",
        Mixed),
    assertion(ScalarOnly \== []),
    assertion(ScalarOnly == Mixed),
    !.

:- end_tests(plawk_mixed_scalar_print).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_scalar_print', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ms_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ms', Prog0),
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

% The instruction lines a bare-scalar END print contributes, pulled out of the emitted IR
% by their own generated variable names (`%end_str_*` / `%end_i64_*` / `%end_f64_*` /
% `%printed_end_*`) -- never a driver-wide mnemonic, which would match in every program.
scalar_print_lines(Src, Lines) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, Text),
    split_string(Text, "\n", " ", All),
    include(scalar_print_line, All, Lines).

scalar_print_line(Line) :-
    % The OFS separator and the ORS terminator also carry `printed_end_` names, and a
    % two-field print has a separator a one-field print does not -- so they are excluded
    % explicitly. Without this the comparison fails on a difference that is not the
    % scalar print's.
    \+ sub_string(Line, _, _, _, "printed_end_separator"),
    \+ sub_string(Line, _, _, _, "printed_end_newline"),
    (   sub_string(Line, _, _, _, "%end_str_")
    ;   sub_string(Line, _, _, _, "%end_i64_")
    ;   sub_string(Line, _, _, _, "%end_f64_")
    ;   sub_string(Line, _, _, _, "%printed_end_")
    ),
    !.

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
