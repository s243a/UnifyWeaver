:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `n--` -- post-decrement.
%
% `n--` was a PARSE ERROR everywhere while `n++` worked; `n = n - 1` was the only
% spelling of a decrement. It is the natural update for a countdown loop --
%
%   while (n > 0) { print n; n-- }
%
% -- which is what made this worth doing before loops in END: that idiom needs
% this to be writable at all.
%
% ---------------------------------------------------------------------------
% WHY IT IS TWO PARSER PRODUCTIONS AND ONE CODEGEN ROW
%
% The inverse of this campaign's usual finding. `n--` reports the SAME
% `add(const(_))` scalar update `n++` already reports, with a delta of -1, so
% every downstream consumer of that operation -- the strnum-safety check, the
% double-typing fixpoint, the slot emitters -- already covers it. No new emitter,
% no walker to teach.
%
% That was CHECKED before writing, not assumed: scalar `inc(var(_))` appears in
% exactly two codegen places, and every other `inc` mention is `inc_assoc`, the
% array form. `same_operation_shape_as_increment` below pins the equivalence that
% the argument rests on.
%
% ---------------------------------------------------------------------------
% WHAT IS DELIBERATELY LEFT OUT
%
%   arr[k]--     PARSE ERROR, on purpose. The assoc increment is a whole action
%                family (`inc_assoc`, with rows in the table-name walker, the
%                update-action gate, the body-action spec and the increment
%                planner), so a decrement there needs a row in EVERY one of them.
%                A missing row is exactly how this codebase's recurring defect
%                gets in, so a clean parse error plus this pin beats four-fifths
%                of a feature.
%
%   n += -1      PARSE ERROR too (SINCE FIXED, see the pin below) -- note WHERE:
%                the parser's compound-assign
%                delta does not accept a negative literal, so this is rejected
%                before codegen, not by the `add(var(N), int(V))` clause's
%                `V >= 0` guard. Worth pinning precisely because the result is now
%                ODD: `n--` compiles and its longhand equivalent does not.
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

:- begin_tests(plawk_decrement).

% --- the decrement itself -------------------------------------------------

test(single_decrement, [condition(clang_available)]) :-
    run("{ n = 2; n--; print n }\n", "1\n1\n1\n"),
    !.

test(two_decrements, [condition(clang_available)]) :-
    run("{ n = 5; n--; n--; print n }\n", "3\n3\n3\n"),
    !.

% Down to zero, and PAST zero -- the delta is a signed add, not a saturating
% "decrement if positive".
test(decrement_to_zero, [condition(clang_available)]) :-
    run("{ n = 1; n--; if (n == 0) print \"zero\" }\n",
        "zero\nzero\nzero\n"),
    !.

test(decrement_below_zero, [condition(clang_available)]) :-
    run("{ n = 0; n--; print n }\n", "-1\n-1\n-1\n"),
    !.

% --- THE motivating case: countdown loops --------------------------------

% Braced body. This is the shape that could not be written at all before.
test(countdown_while_braced, [condition(clang_available)]) :-
    run("{ n = 3; while (n > 0) { print n; n-- } }\n",
        "3\n2\n1\n3\n2\n1\n3\n2\n1\n"),
    !.

% Braceless body: the decrement IS the loop body.
test(countdown_while_braceless, [condition(clang_available)]) :-
    run("{ n = 3; while (n > 0) n-- ; print n }\n", "0\n0\n0\n"),
    !.

test(countdown_do_while, [condition(clang_available)]) :-
    run("{ n = 2; do { print n; n-- } while (n > 0) }\n",
        "2\n1\n2\n1\n2\n1\n"),
    !.

% A decrementing C-for UPDATE -- this is why decrement_action//1 is registered in
% for_c_simple//1 as well as in action//1.
test(countdown_c_for, [condition(clang_available)]) :-
    run("{ for (i = 3; i > 0; i--) print i }\n",
        "3\n2\n1\n3\n2\n1\n3\n2\n1\n"),
    !.

% --- a decrement of a RULE-ACCUMULATED counter ---------------------------

% The slot is carried across records, so this exercises the decrement against a
% loop-head phi value rather than a freshly assigned constant.
test(decrement_of_an_accumulated_counter, [condition(clang_available)]) :-
    run("{ n++; if (NR == 3) { n--; print n } }\n", "2\n"),
    !.

% --- regressions: the increment side ------------------------------------

test(increment_unchanged, [condition(clang_available)]) :-
    run("{ n++; print n }\n", "1\n2\n3\n"),
    !.

test(incrementing_c_for_unchanged, [condition(clang_available)]) :-
    run("{ for (i = 0; i < 2; i++) print i }\n", "0\n1\n0\n1\n0\n1\n"),
    !.

test(compound_add_unchanged, [condition(clang_available)]) :-
    run("{ n = 0; n += 2; print n }\n", "2\n2\n2\n"),
    !.

test(counter_into_end_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "3\n"),
    !.

% --- pinned parse errors -------------------------------------------------

% `arr[k]--` was a PARSE ERROR, pinned here with its increment sibling to show the
% refusal was about the assoc action family rather than about `--` being
% unparseable. It now PARSES -- desugared to `arr[k] += -1`, the existing add_assoc
% family, so no walker was touched. tests/test_plawk_assoc_decrement.pl owns it.
%
% The form pinned here uses a LITERAL key. That was a parse error, then a DECLINE, and it
% now COMPILES -- the literal key landed as the arity-1 case of the multi-dimensional key
% builder, and the rule-body literal read landed once the two array key spaces
% (interned-text vs raw position) were resolved at plan time. Kept as a PAIR with the `+=`
% form throughout all three states, which is the point: the decrement rides add_assoc, so
% whatever `c["a"] += 1` does, `c["a"]--` does. See tests/test_plawk_literal_assoc_key.pl
% and tests/test_plawk_posarray_keyspace.pl.
test(assoc_decrement_works_where_add_assign_works, [condition(clang_available)]) :-
    run("{ c[\"a\"] += 1; print c[\"a\"] }\n", "1\n2\n3\n"),
    run("{ c[\"a\"]--; print c[\"a\"] }\n", "-1\n-2\n-3\n"),
    !.

test(assoc_decrement_with_a_field_key_compiles, [condition(clang_available)]) :-
    run("{ c[$1]++; c[$1]-- } END { print c[\"a\"] }\n", "0\n"),
    !.

test(assoc_increment_still_works, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"] }\n", "1\n"),
    !.

% `n += -1` was rejected by the PARSER's compound-assign delta (a negative literal),
% and then by a `Value >= 0` guard in the codegen -- two layers, which is why fixing
% the parser alone only turned the parse error into a decline. Pinned here because
% `n--` compiled while its longhand equivalent did not.
%
% Both restrictions are gone: the longhand now compiles and agrees with `n--`, which
% is what this row always implied should be true. tests/test_plawk_negative_delta.pl
% owns the behaviour; the pin is inverted rather than deleted so the asymmetry it
% recorded stays visible.
test(negative_compound_add_agrees_with_decrement, [condition(clang_available)]) :-
    run("{ n = 2; n += -1; print n }\n", "1\n1\n1\n"),
    run("{ n = 2; n--; print n }\n", "1\n1\n1\n"),
    !.

% --- structure -----------------------------------------------------------

% `n--` parses to its own action term, mirroring `inc`.
test(decrement_parses_to_its_own_term) :-
    plawk_parse_string("{ n-- }\n",
        program([], [rule(always, [Action])], [])),
    assertion(Action == dec(var(n))),
    !.

test(increment_still_parses_to_inc) :-
    plawk_parse_string("{ n++ }\n",
        program([], [rule(always, [Action])], [])),
    assertion(Action == inc(var(n))),
    !.

% THE test the whole "one row, no new emitter" argument rests on: `n--` and `n++`
% report the SAME scalar-update operation shape, differing only in the delta. If
% this ever diverges, the decrement has grown its own lowering and every consumer
% of add(const(_)) needs re-checking for it.
test(same_operation_shape_as_increment) :-
    plawk_native_codegen:plawk_scalar_action_update(inc(var(n)), IncName, IncOp),
    plawk_native_codegen:plawk_scalar_action_update(dec(var(n)), DecName, DecOp),
    assertion(IncName == n),
    assertion(DecName == n),
    assertion(IncOp == add(const(1))),
    assertion(DecOp == add(const(-1))),
    !.

% A decrementing C-for desugars to the same while-loop shape an incrementing one
% does, with the decrement as the appended UPDATE.
test(c_for_desugars_with_the_decrement_as_update) :-
    plawk_parse_string("{ for (i = 3; i > 0; i--) print i }\n",
        program([], [rule(always, Actions)], [])),
    assertion(Actions = [set(var(i), int(3)), while_loop(_, Body)]),
    assertion(last(Body, dec(var(i)))),
    !.

:- end_tests(plawk_decrement).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_decrement', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'dec_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'dec', Prog0),
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
    directory_file_path(Dir, 'dec_reject', Prog0),
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
