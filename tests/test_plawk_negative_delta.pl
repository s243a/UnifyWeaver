:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% NEGATIVE compound-assign deltas, and the `-=` operator.
%
%   { n += -1 }      PARSE ERROR      while `n--` compiled and means the same thing
%   { n -= 1 }       PARSE ERROR      the operator did not exist at all
%   { n = -1 }       PARSE ERROR
%   { x += -1.5 }    PARSE ERROR
%
% Two restrictions stacked, which is why this looked like one bug and was two:
%
%   1. the parser's `scalar_delta_expr//1` required a NON-NEGATIVE literal, so a
%      leading `-` had no production at all -- a parse error, not a decline;
%   2. `plawk_scalar_action_update/3` then required `Value >= 0` for both
%      `add(var(N), int(V))` and `set(var(N), int(V))`.
%
% Fixing only the parser turned the parse errors into DECLINES, which is how the
% second guard surfaced. Worth remembering: when a surface form is refused, check
% whether more than one layer refuses it before declaring it fixed.
%
% ---------------------------------------------------------------------------
% THE ASYMMETRY THAT SHOULD HAVE GIVEN IT AWAY
%
% `n--` has compiled since decrement landed, and it lowers to exactly
% add(const(-1)) -- so the `add` path has been emitting negative constants all
% along. The codegen comment beside that row even recorded the asymmetry
% ("`n--` compiles, `n += -1` does not") and pinned it rather than asking why. A
% surface restriction that the emitter demonstrably does not need is a restriction
% with no owner.
%
% ---------------------------------------------------------------------------
% `-=` IS SUGAR, NOT A FEATURE
%
% `n -= D` parses to the SAME add/2 term `n += -D` produces, so no codegen path
% learns anything new and the emitted IR is identical. A test asserts the parse
% trees are equal, which is what makes that claim checkable rather than asserted.
%
% Only a LITERAL delta can be negated at parse time. `n -= $2` needs a real
% subtraction in the update emitter (which knows `add` and `set` only), so it stays
% a clean refusal instead of being silently mis-negated -- pinned below.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Four records, second field 1/2/3/4 -- so `n += $2` sums to 10.
input("a 1\nb 2\nc 3\nd 4\n").

:- begin_tests(plawk_negative_delta).

% --- negative integer deltas ---------------------------------------------

test(negative_integer_delta, [condition(clang_available)]) :-
    run("{ n += -1 } END { print n }\n", "-4\n"),
    !.

test(negative_integer_delta_not_one, [condition(clang_available)]) :-
    run("{ n += -2 } END { print n }\n", "-8\n"),
    !.

test(negative_integer_assignment, [condition(clang_available)]) :-
    run("{ n = -1 } END { print n }\n", "-1\n"),
    !.

% --- the `-=` operator ---------------------------------------------------

test(subtract_assign, [condition(clang_available)]) :-
    run("{ n -= 1 } END { print n }\n", "-4\n"),
    !.

% Negating a negative delta: `-= -1` is `+= 1`.
test(subtract_assign_negative_delta, [condition(clang_available)]) :-
    run("{ n -= -1 } END { print n }\n", "4\n"),
    !.

test(increment_then_subtract_assign, [condition(clang_available)]) :-
    run("{ n++; n -= 1 } END { print n }\n", "0\n"),
    !.

% --- doubles take the same path ------------------------------------------

test(negative_float_delta, [condition(clang_available)]) :-
    run("{ x += -1.5 } END { print x }\n", "-6\n"),
    !.

test(subtract_assign_float, [condition(clang_available)]) :-
    run("{ x -= 1.5 } END { print x }\n", "-6\n"),
    !.

test(subtract_assign_negative_float, [condition(clang_available)]) :-
    run("{ x -= -1.5 } END { print x }\n", "6\n"),
    !.

test(negative_float_assignment, [condition(clang_available)]) :-
    run("{ x = -1.5 } END { print x }\n", "-1.5\n"),
    !.

% --- `-=` is a desugaring, not a feature ---------------------------------

% The claim that no codegen path learns anything new, made checkable: `n -= 1`
% parses to the term `n += -1` parses to.
test(subtract_assign_parses_as_negated_add) :-
    plawk_parse_string("{ n -= 1 } END { print n }\n", Sugar),
    plawk_parse_string("{ n += -1 } END { print n }\n", Explicit),
    assertion(Sugar == Explicit),
    assertion(Sugar = program([], [rule(always, [add(var(n), int(-1))])], [_])),
    !.

test(subtract_assign_float_parses_as_negated_add) :-
    plawk_parse_string("{ x -= 1.5 } END { print x }\n", Sugar),
    plawk_parse_string("{ x += -1.5 } END { print x }\n", Explicit),
    assertion(Sugar == Explicit),
    !.

% Both spellings therefore emit the same IR -- the stronger form of the same claim.
test(subtract_assign_emits_identical_ir) :-
    plawk_parse_string("{ n -= 1 } END { print n }\n", Sugar),
    plawk_parse_string("{ n += -1 } END { print n }\n", Explicit),
    plawk_program_native_driver_ir(Sugar, 'input.txt', SugarIR),
    plawk_program_native_driver_ir(Explicit, 'input.txt', ExplicitIR),
    assertion(SugarIR == ExplicitIR),
    !.

% ...and `n--`, which is where the negative constant was already proven safe.
test(decrement_and_negative_delta_agree, [condition(clang_available)]) :-
    run("{ n-- } END { print n }\n", "-4\n"),
    run("{ n += -1 } END { print n }\n", "-4\n"),
    !.

% --- the float/int ordering hazard --------------------------------------

% "-1.5" must not stop at the integer prefix "-1" and leave ".5" to choke the
% statement -- the same ordering trap the unsigned clauses have, which is why the
% negative float clause is registered before the negative integer one. If they were
% swapped this would be a parse error.
test(negative_float_is_not_split_at_the_integer_prefix) :-
    plawk_parse_string("{ x += -1.5 } END { print x }\n",
        program([], [rule(always, [add(var(x), float_const(-15, 10))])], [_])),
    !.

test(negative_integer_delta_stays_an_integer) :-
    plawk_parse_string("{ n += -1 } END { print n }\n",
        program([], [rule(always, [add(var(n), int(-1))])], [_])),
    !.

% --- regressions: the positive forms are untouched ----------------------

test(positive_integer_delta_unchanged, [condition(clang_available)]) :-
    run("{ n += 1 } END { print n }\n", "4\n"),
    !.

test(field_delta_unchanged, [condition(clang_available)]) :-
    run("{ n += $2 } END { print n }\n", "10\n"),
    !.

test(positive_float_delta_unchanged, [condition(clang_available)]) :-
    run("{ x += 1.5 } END { print x }\n", "6\n"),
    !.

test(increment_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "4\n"),
    !.

% --- clean refusals, pinned ---------------------------------------------

% --- non-literal deltas ---------------------------------------------------
%
% `n -= $2` was refused, on the reasoning that a real subtraction was needed in the
% update emitter (which knows `add` and `set`). It is not: `add` of a NEGATION is the
% same arithmetic, and `0 - E` is the `sub_i64/2` term the parser already builds for
% the surface subtraction in `n += 0 - $2`. So `n -= E` desugars to exactly the term
% `n += 0 - E` produces and needs no new operation.
%
% A literal is still negated in place, so `n -= 1` stays the single-instruction
% `add i64 %slot, -1` it has always been rather than growing a subtract.

test(non_literal_subtract_assign, [condition(clang_available)]) :-
    run("{ n -= $2 } END { print n }\n", "-10\n"),
    !.

test(subtract_assign_a_special, [condition(clang_available)]) :-
    run("{ n -= NR } END { print n }\n", "-10\n"),
    !.

test(subtract_assign_a_builtin, [condition(clang_available)]) :-
    run("{ n -= length($1) } END { print n }\n", "-4\n"),
    !.

% Nested arithmetic on the right: `n -= $1 - $2` is n -= ($1 - $2).
test(subtract_assign_an_arithmetic_expression, [condition(clang_available)]) :-
    run("{ n -= $2 - 1 } END { print n }\n", "-6\n"),
    !.

% THE EQUIVALENCE, which is what makes the desugaring safe to state generally:
% whatever `+= 0 - E` does -- including declining -- `-= E` does identically, because
% it is the same term. Pinned as an equivalence rather than by enumerating which
% expressions work.
test(non_literal_subtract_assign_is_the_explicit_negation) :-
    plawk_parse_string("{ n -= $2 } END { print n }\n", Sugar),
    plawk_parse_string("{ n += 0 - $2 } END { print n }\n", Explicit),
    assertion(Sugar == Explicit),
    assertion(Sugar = program([],
        [rule(always, [add(var(n), sub_i64(int(0), field(2)))])], [_])),
    !.

test(non_literal_subtract_assign_emits_identical_ir) :-
    plawk_parse_string("{ n -= $2 } END { print n }\n", Sugar),
    plawk_parse_string("{ n += 0 - $2 } END { print n }\n", Explicit),
    plawk_program_native_driver_ir(Sugar, 'input.txt', SugarIR),
    plawk_program_native_driver_ir(Explicit, 'input.txt', ExplicitIR),
    assertion(SugarIR == ExplicitIR),
    !.

% A LITERAL still takes the in-place negation, not the subtract -- so the common form
% does not regress into an extra instruction.
test(a_literal_delta_stays_an_in_place_negation) :-
    plawk_parse_string("{ n -= 1 } END { print n }\n",
        program([], [rule(always, [add(var(n), int(-1))])], [_])),
    assertion(\+ plawk_parser:plawk_negate_literal_delta(field(2), _)),
    assertion(plawk_parser:plawk_negate_literal_delta(int(3), int(-3))),
    assertion(plawk_parser:plawk_negate_literal_delta(float_const(15, 10),
        float_const(-15, 10))),
    !.

% The fallback is total: a non-literal always becomes a negation rather than failing.
test(the_scalar_negation_helper_is_total) :-
    assertion(plawk_parser:plawk_negate_scalar_delta(int(3), int(-3))),
    assertion(plawk_parser:plawk_negate_scalar_delta(field(2),
        sub_i64(int(0), field(2)))),
    assertion(plawk_parser:plawk_negate_scalar_delta(special('NR'),
        sub_i64(int(0), special('NR')))),
    !.

% --- the assoc half is NOT fixed here, and is attributed ------------------

% `c[$1] -= $2` still declines -- because the assoc delta accepts only a bare field
% or an integer literal, so `c[$1] += 0 - $2` declines too. Pinned as a PAIR so the
% restriction stays attributed to the assoc delta production rather than to `-=`.
test(a_non_literal_assoc_delta_is_refused_for_both_spellings) :-
    build_status("{ c[$1] += 0 - $2 } END { print c[\"a\"] }\n", 2),
    build_status("{ c[$1] -= $2 } END { print c[\"a\"] }\n", 2),
    !.

% A compound assign as an END STATEMENT declines -- for `+=`, `-=` and `++` alike,
% so it is the END shape and not this change. Pinned as a trio so the distinction
% stays visible.
test(compound_assign_in_end_declines_for_every_spelling) :-
    build_status("{ n++ } END { n += 1; print n }\n", 3),
    build_status("{ n++ } END { n -= 1; print n }\n", 3),
    build_status("{ n++ } END { n++; print n }\n", 3),
    !.

:- end_tests(plawk_negative_delta).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_negative_delta', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'nd_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'nd', Prog0),
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
    directory_file_path(Dir, 'nd_reject', Prog0),
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
