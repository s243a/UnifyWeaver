:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Numeric SPECIALS as operands of an `if` / `while` / `do-while` condition --
% `if (length > 3)`, `if (n < NF)`, `if (n < NR)`, `if (n < ARGC)`.
%
% ---------------------------------------------------------------------------
% A ROW OF SILENT WRONG OUTPUTS, AND A CONTRACT ONLY ONE SIDE HONOURED
%
% Everything here printed the WRONG THING before, and none of it declined:
%
%     { if (length > 3) print $1 }        printed nothing;  gawk prints 3 records
%     { n = 3; if (n < length) print $1 } printed nothing;  gawk prints 3 records
%     { n = 1; if (n < NF) print $1 }     printed nothing;  gawk prints 3 records
%     { n = 1; if (n < NR) print $1 }     printed nothing;  gawk prints 2 records
%     { n = 0; if (n < ARGC) print $1 }   printed nothing;  gawk prints 3 records
%     END { if (length == 6) … }          took the else branch for a 6-byte record
%
% while the BARE-PATTERN spelling of the same comparison was correct throughout:
%
%     length > 3 { print $1 }             correct
%     NF > 1 { print $1 }                 correct
%
% Two causes, and the second is the more instructive.
%
% 1. TWO LISTS OF ONE SET. The condition grammar's special-operand list
%    (match_special_name//1: NF, NR, FNR, RSTART, RLENGTH, ARGC) omits `length`, while
%    the bare-pattern list (special_cmp_operand//1: NF, NR, FNR, length) includes it.
%    Neither is a superset of the other. So which spelling of `length > 3` worked
%    depended on which grammar read it.
%
% 2. A CONTRACT WITH ONLY ONE SIDE IMPLEMENTED -- the right-hand operand. The codegen
%    was ready and waiting: plawk_while_cond_build/8 carries a
%    `cmp(Lhs, Op, special('NF'))` row for the reversed operand order, and
%    plawk_while_cond_operand/8 resolves RSTART / RLENGTH / ARGC / NR on EITHER side
%    (it takes a Side argument). The PARSER never produced a special on the right at
%    all. The row sat unreachable while `identifier//1` captured the name as an
%    ordinary variable.
%
%    This is the sharpest variant of the campaign's recurring class so far. It is not
%    two implementations of one property disagreeing about a value -- it is one side of
%    a contract never emitting what the other side already handles correctly. Nothing
%    in the codegen looked wrong; nothing in the parser looked incomplete, because a
%    fallback covered the case. Reading either file alone showed no defect.
%
% WHAT TURNED THE DISAGREEMENT INTO WRONG OUTPUT rather than a decline: the identifier
% fallbacks carried no `scalar_cmp_reserved_name/1` guard, though every sibling
% production in the bare-pattern grammar did. Without it, any special this grammar has
% not been taught is silently captured as a variable, resolves to an unassigned slot
% worth 0, and the comparison is quietly false. The guard is the durable half of the
% fix: it makes the NEXT omission from either list a decline. It also makes
% `{ if (int > 2) … }` a parse error, which is what gawk does -- `int` is a builtin, and
% plawk had been comparing a phantom.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Records "5 boot" (6 bytes, NF 2), "5 trace" (7, NF 2), "7 disk" (6, NF 2).
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_cond_specials).

% --- `length` as a condition operand -----------------------------------

test(length_on_the_left, [condition(clang_available)]) :-
    run("{ if (length > 3) print $1 }\n", "5\n5\n7\n"),
    !.

% == picks out only the 6-byte records, so this cannot pass by accident the way a
% `> 3` that matches everything could.
test(length_equality_selects_a_subset, [condition(clang_available)]) :-
    run("{ if (length == 6) print $1 }\n", "5\n7\n"),
    !.

test(length_on_the_right, [condition(clang_available)]) :-
    run("{ n = 3; if (n < length) print $1 }\n", "5\n5\n7\n"),
    !.

% The parenthesised spelling is the same special.
test(length_of_field_zero_spelling, [condition(clang_available)]) :-
    run("{ if (length($0) > 3) print $1 }\n", "5\n5\n7\n"),
    !.

% --- every special on the RIGHT, which is the row that was wrong --------

test(nf_on_the_right, [condition(clang_available)]) :-
    run("{ n = 1; if (n < NF) print $1 }\n", "5\n5\n7\n"),
    !,
    run("{ n = 2; if (n == NF) print $1 }\n", "5\n5\n7\n"),
    !.

test(nr_on_the_right, [condition(clang_available)]) :-
    run("{ n = 1; if (n < NR) print $1 }\n", "5\n7\n"),
    !.

test(fnr_on_the_right_is_the_nr_alias, [condition(clang_available)]) :-
    run("{ n = 1; if (n < FNR) print $1 }\n", "5\n7\n"),
    !.

test(argc_on_the_right, [condition(clang_available)]) :-
    run("{ n = 0; if (n < ARGC) print $1 }\n", "5\n5\n7\n"),
    !.

% --- the left side, which already worked, so these are regressions ------

test(nf_on_the_left_unchanged, [condition(clang_available)]) :-
    run("{ if (NF > 1) print $1 }\n", "5\n5\n7\n"),
    !.

test(nr_on_the_left_unchanged, [condition(clang_available)]) :-
    run("{ if (NR > 1) print $1 }\n", "5\n7\n"),
    !.

test(argc_on_the_left_unchanged, [condition(clang_available)]) :-
    run("{ if (ARGC > 0) print $1 }\n", "5\n5\n7\n"),
    !.

% --- the same grammar drives while / do-while --------------------------
%
% if / while / do-while share while_condition//1, so the fix reaches all three. A
% loop is the case where a wrong condition is worst: `while (n < NF)` reading a
% phantom 0 exits immediately.

test(nf_on_the_right_of_a_while, [condition(clang_available)]) :-
    run("{ n = 0; while (n < NF) { n++ }; print n }\n", "2\n2\n2\n"),
    !.

test(nf_on_the_right_of_a_do_while, [condition(clang_available)]) :-
    run("{ n = 0; do { n++ } while (n < NF); print n }\n", "2\n2\n2\n"),
    !.

test(length_on_the_right_of_a_while, [condition(clang_available)]) :-
    run("{ n = 4; while (n < length) { n++ }; print n }\n", "6\n7\n6\n"),
    !.

% --- combinators -------------------------------------------------------

test(specials_compose_with_and_or, [condition(clang_available)]) :-
    run("{ if (NF > 1 && NR > 1) print $1 }\n", "5\n7\n"),
    !,
    run("{ n = 1; if (n < NF || n > 9) print $1 }\n", "5\n5\n7\n"),
    !,
    run("{ if (length > 3 && NF == 2) print $1 }\n", "5\n5\n7\n"),
    !.

% --- the bare-pattern spelling, which was right all along --------------
%
% Kept as regressions because the fix touched the list the bare patterns read from
% (special_cmp_operand//1 is now called by the condition grammar too, with its
% argument bound, rather than copied).

test(bare_pattern_spellings_unchanged, [condition(clang_available)]) :-
    run("length > 3 { print $1 }\n", "5\n5\n7\n"),
    !,
    run("length == 6 { print $1 }\n", "5\n7\n"),
    !,
    run("NF > 1 { print $1 }\n", "5\n5\n7\n"),
    !,
    run("NR > 1 { print $1 }\n", "5\n7\n"),
    !.

% --- ordinary variables still work, in both positions ------------------
%
% The reserved-name guard must not reject names that merely resemble specials, and
% `identifier_boundary//0` is what keeps `NRX` an identifier.

test(ordinary_variables_unaffected, [condition(clang_available)]) :-
    run("{ n = 1; m = 2; if (n < m) print $1 }\n", "5\n5\n7\n"),
    !,
    run("{ n++; if (n > 1) print $1 }\n", "5\n7\n"),
    !,
    run("{ s = \"x\"; if (s == \"x\") print $1 }\n", "5\n5\n7\n"),
    !.

test(a_name_that_merely_starts_like_a_special, [condition(clang_available)]) :-
    run("{ NRX = 2; if (NRX == 2) print $1 }\n", "5\n5\n7\n"),
    !,
    run("{ lengthy = 2; if (lengthy == 2) print $1 }\n", "5\n5\n7\n"),
    !.

% --- the parse level, where the defect lived ---------------------------
%
% Pinned as TERMS, not only as output, because clause order and a missing guard carry
% the semantics invisibly -- the recorded prescription for this defect variant is to
% assert the parse, so the next ordering change fails with the reason on the label.

test(bare_length_in_a_condition_is_the_special_not_a_variable) :-
    plawk_parse_string("{ if (length > 3) print $1 }\n", Program),
    Program = program(_, [rule(always, [if(scalar_if(Cond), _, _)])], _),
    % THE defect: this used to be cmp(var(length), gt, int(3)).
    assertion(Cond == cmp(special(length), gt, int(3))),
    !.

test(the_parenthesised_length_parses_to_the_same_term) :-
    plawk_parse_string("{ if (length($0) > 3) print $1 }\n", A),
    plawk_parse_string("{ if (length > 3) print $1 }\n", B),
    assertion(A == B),
    !.

test(a_special_on_the_right_parses_as_a_special) :-
    plawk_parse_string("{ n = 1; if (n < NF) print $1 }\n", Program),
    Program = program(_, [rule(always, [_, if(scalar_if(Cond), _, _)])], _),
    % THE defect: this used to be cmp(var(n), lt, var('NF')).
    assertion(Cond == cmp(var(n), lt, special('NF'))),
    !.

test(fnr_normalises_to_nr_on_the_right_too) :-
    plawk_parse_string("{ n = 1; if (n < FNR) print $1 }\n", A),
    plawk_parse_string("{ n = 1; if (n < NR) print $1 }\n", B),
    assertion(A == B),
    !.

% --- the reserved-name guard -------------------------------------------

% `int` is a builtin, not a variable. gawk rejects this as a syntax error; plawk used
% to accept it and compare a phantom slot worth 0. It is now a PARSE ERROR (status 2),
% which is the same answer gawk gives.
test(a_builtin_name_is_not_a_condition_variable) :-
    build_status("{ if (int > 2) print \"x\" }\n", 2),
    !.

% --- END conditions: declines, and now consistently -------------------
%
% There is no current record in END -- `%line` holds the `end_of_file` sentinel -- and
% no condition emitter knows about the RETAINED record (the END *print* path does;
% conditions never go through plawk_end_lastrec_rewrite/2). So a record-reading END
% condition must decline.
%
% `NF` already did, for want of an emitter row. `length` did NOT: it had a row, so it
% compiled and measured the 11-byte sentinel, which is how
% `END { if (length == 6) print "six"; else print "no" }` printed `no` for a 6-byte
% last record. The two now agree, and this pins them TOGETHER so the next change to
% either moves both or fails here.
test(record_reading_end_conditions_decline_alike) :-
    build_status("{ n++ } END { if (length == 6) print \"six\"; else print \"no\" }\n", 3),
    build_status("{ n++ } END { if (length > 3) print \"long\"; else print \"short\" }\n", 3),
    build_status("{ n++ } END { if (NF == 2) print \"two\"; else print \"no\" }\n", 3),
    !.

% ...while a scalar END condition is unaffected, and a record read in the BRANCH still
% works -- it is the CONDITION that cannot reach the retained record, not the branch.
test(scalar_end_conditions_and_record_reads_in_branches_unaffected,
        [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print \"three\"; else print \"no\" }\n", "three\n"),
    !,
    run("{ n++ } END { if (n == 3) print length }\n", "6\n"),
    !,
    run("{ n++ } END { if (n == 3) print NF }\n", "2\n"),
    !.

% --- a follow-on, pinned with its reason -------------------------------

% `length($N)` for N > 0 is a PARSE ERROR in a condition -- and equally in a bare
% pattern, which is the point: special_cmp_operand//1 accepts only bare `length` and
% `length($0)`, so both spellings refuse it identically. A consistent gap rather than a
% new asymmetry, and admitting it means carrying a field index on the special term.
% Pinned as a pair so the two cannot drift apart the way `length` itself did.
test(length_of_a_positive_field_is_not_yet_a_comparison_operand) :-
    build_status("{ if (length($1) > 0) print $1 }\n", 2),
    build_status("length($1) > 0 { print $1 }\n", 2),
    !.

:- end_tests(plawk_cond_specials).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_cond_specials', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'cs_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'cs', Prog0),
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
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
            [Src, Out, Expected]), fail
    ).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'cs_reject', Prog0),
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
