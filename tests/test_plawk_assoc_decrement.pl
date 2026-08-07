:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `arr[k]--` and `arr[k] -= D` -- decrementing an associative element.
%
%   { c[$1]++; c[$1]-- }        PARSE ERROR
%   { c[$1] -= 1 }              PARSE ERROR
%   { c[$1] += -1 }             PARSE ERROR
%
% ---------------------------------------------------------------------------
% THE FOLLOW-ON WAS SIZED WRONG, AND THAT IS THE INTERESTING PART
%
% This was recorded as needing "a row in each of four `inc_assoc` walkers": the
% table-name walker, the update-action gate, the body-action spec and the increment
% planner. The reasoning was that `arr[k]++` is the action family `inc_assoc`, so
% `arr[k]--` had to become a sibling family with rows in all four places.
%
% It does not. `arr[k] += N` ALREADY parses to `add_assoc/3` -- a different existing
% family with its own complete set of rows -- so `arr[k]--` desugars to
% `add_assoc(var(C), Key, int(-1))` and NO walker is touched. Adding a `dec_assoc`
% would have created a THIRD representation of "change this element by N", when
% `inc_assoc` and `add_assoc` are already two, which is the very defect the original
% note was written to avoid.
%
% The shape of the mistake: the follow-on was sized by looking at the nearest
% relative (`inc_assoc`, which shares the `++` spelling) instead of asking which
% existing family the surface form belongs to (`add_assoc`, which shares the
% semantics). Worth checking on the next "this needs a row in every walker" estimate.
%
% ---------------------------------------------------------------------------
% ONE CLAUSE, OPERATOR CHOSEN AFTER THE KEY
%
% The assoc `+=` clause cuts as soon as it has parsed `TABLE[key]`. A separate `-=`
% clause placed after it is therefore unreachable -- and was: `c[$1] -= 1` stayed a
% parse error until the two were merged into one clause that parses the key once and
% then dispatches on the operator. A `!` that commits before the distinguishing token
% has been read makes any later alternative dead.
%
% ---------------------------------------------------------------------------
% KEY COVERAGE, AND THE ASYMMETRY THAT REMAINS
%
% `arr[k]--` is `arr[k] += -1`, so it supports exactly the keys `+=` supports:
%
%              field key $1   literal key "x"   scalar-var key k
%   c[K]++          yes            no                 yes
%   c[K] += N       yes            no                 yes
%   c[K]--          yes            no                 yes
%
% The scalar-var column read `no / no` for the add_assoc rows when this suite was
% written -- `c[k]++` compiled and `c[k]--` declined -- and that was NOT this change's
% asymmetry but the pre-existing `inc_assoc` / `add_assoc` key-coverage gap, inherited
% because the decrement rides `add_assoc`. That gap is now closed; see
% tests/test_plawk_assoc_key_coverage.pl. The literal-key column is still `no` for
% BOTH families and is pinned below in a pair, so it stays attributed to the family
% that owns it.
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

% Four records; $1 of the first is INFO, so c["INFO"] counts 1.
input("INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n").

:- begin_tests(plawk_assoc_decrement).

% --- the decrement -------------------------------------------------------

test(assoc_decrement_cancels_an_increment, [condition(clang_available)]) :-
    run("{ c[$1]++; c[$1]-- } END { print c[\"INFO\"] }\n", "0\n"),
    !.

test(assoc_decrement_alone_goes_negative, [condition(clang_available)]) :-
    run("{ c[$1]-- } END { print c[\"DEBUG\"] }\n", "-2\n"),
    !.

% --- `-=` on an element --------------------------------------------------

test(assoc_subtract_assign, [condition(clang_available)]) :-
    run("{ c[$1]++; c[$1] -= 1 } END { print c[\"INFO\"] }\n", "0\n"),
    !.

test(assoc_subtract_assign_delta_not_one, [condition(clang_available)]) :-
    run("{ c[$1] += 3; c[$1] -= 2 } END { print c[\"INFO\"] }\n", "1\n"),
    !.

test(assoc_subtract_assign_negative_delta, [condition(clang_available)]) :-
    run("{ c[$1] -= -1 } END { print c[\"INFO\"] }\n", "1\n"),
    !.

% --- negative `+=` delta -------------------------------------------------

test(assoc_negative_add_delta, [condition(clang_available)]) :-
    run("{ c[$1]++; c[$1] += -1 } END { print c[\"INFO\"] }\n", "0\n"),
    !.

% --- all three spellings are one term ------------------------------------

% The claim that no walker was touched, made checkable: every spelling parses to the
% same add_assoc/3 term, so nothing downstream can tell them apart.
test(every_spelling_parses_to_the_same_add_assoc_term) :-
    plawk_parse_string("{ c[$1]-- } END { print c[\"x\"] }\n", Dec),
    plawk_parse_string("{ c[$1] -= 1 } END { print c[\"x\"] }\n", SubAssign),
    plawk_parse_string("{ c[$1] += -1 } END { print c[\"x\"] }\n", NegAdd),
    assertion(Dec == SubAssign),
    assertion(Dec == NegAdd),
    assertion(Dec = program([],
        [rule(always, [add_assoc(var(c), field(1), int(-1))])], [_])),
    !.

% ...and therefore emit identical IR.
test(every_spelling_emits_identical_ir) :-
    plawk_parse_string("{ c[$1]-- } END { print c[\"x\"] }\n", Dec),
    plawk_parse_string("{ c[$1] += -1 } END { print c[\"x\"] }\n", NegAdd),
    plawk_program_native_driver_ir(Dec, 'input.txt', DecIR),
    plawk_program_native_driver_ir(NegAdd, 'input.txt', NegAddIR),
    assertion(DecIR == NegAddIR),
    !.

% The decrement is NOT a new action family: no `dec_assoc` term exists anywhere.
test(no_dec_assoc_family_was_introduced) :-
    plawk_parse_string("{ c[$1]-- } END { print c[\"x\"] }\n", Program),
    assertion(\+ ( sub_term(Sub, Program), nonvar(Sub),
                   functor(Sub, dec_assoc, _) )),
    !.

% --- the scalar decrement is untouched ----------------------------------

test(scalar_decrement_unchanged, [condition(clang_available)]) :-
    run("{ n++; n-- } END { print n }\n", "0\n"),
    !.

% `n--` must not be captured by the assoc clause: the `[` is what distinguishes
% them, and the assoc clause is tried first.
test(scalar_decrement_parses_as_scalar) :-
    plawk_parse_string("{ n-- } END { print n }\n",
        program([], [rule(always, [dec(var(n))])], [_])),
    !.

% --- regressions: the increment forms ------------------------------------

test(assoc_increment_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"INFO\"] }\n", "1\n"),
    !.

test(assoc_add_assign_unchanged, [condition(clang_available)]) :-
    run("{ c[$1] += 1 } END { print c[\"INFO\"] }\n", "1\n"),
    !.

test(assoc_field_delta_unchanged, [condition(clang_available)]) :-
    run("{ c[$1] += $2 } END { print c[\"INFO\"] }\n", "0\n"),
    !.

% A scalar-var key still works for `++`, which is the half of the asymmetry below
% that compiles.
test(assoc_increment_with_a_scalar_key_unchanged, [condition(clang_available)]) :-
    run("{ k = $1; c[k]++ } END { print c[\"INFO\"] }\n", "1\n"),
    !.

% --- the inherited key-coverage gap, pinned in pairs --------------------
%
% Each pair is (the `+=` form, the `--` form) for one key kind. They decline
% TOGETHER because the decrement rides add_assoc, so the restriction belongs to that
% family and not to this change. If a pair ever splits, the desugaring has been
% bypassed.

% WAS: both declined (status 3), pinned as the inherited gap. The scalar-var key is
% now covered for the whole add_assoc family -- see
% tests/test_plawk_assoc_key_coverage.pl for the full matrix. Kept here, inverted, so
% this suite still records which half of the gap moved.
test(a_scalar_var_key_now_works_for_add_assign_and_decrement_alike,
        [condition(clang_available)]) :-
    run("{ k = $1; c[k] += 1 } END { print c[\"INFO\"] }\n", "1\n"),
    run("{ k = $1; c[k]-- } END { print c[\"INFO\"] }\n", "-1\n"),
    !.

test(a_literal_key_declines_for_add_assign_and_decrement_alike) :-
    build_status("{ c[\"x\"] += 1 } END { print c[\"x\"] }\n", 3),
    build_status("{ c[\"x\"]-- } END { print c[\"x\"] }\n", 3),
    !.

% A NON-LITERAL `-=` delta is refused rather than mis-negated -- the negation is a
% parse-time literal negation, exactly as for the scalar `-=`. Note `+= $2` works, so
% this is the same deliberate asymmetry recorded in
% tests/test_plawk_negative_delta.pl.
test(non_literal_assoc_subtract_assign_is_refused) :-
    build_status("{ c[$1] -= $2 } END { print c[\"INFO\"] }\n", 2),
    !.

:- end_tests(plawk_assoc_decrement).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_decrement', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ad_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ad', Prog0),
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

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ad_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
