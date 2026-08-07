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
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

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

% A NON-LITERAL `-=` delta. `n += $2` works, so this is an asymmetry -- a deliberate
% one: negating at parse time only works for a literal, and the update emitter knows
% `add` and `set` only, so a real subtraction is its own change. Refused rather than
% mis-negated.
test(non_literal_subtract_assign_is_refused) :-
    build_status("{ n -= $2 } END { print n }\n", 2),
    !.

test(the_negation_helper_refuses_non_literals) :-
    assertion(plawk_native_codegen:true),   % module loaded
    assertion(\+ plawk_parser:plawk_negate_literal_delta(field(2), _)),
    assertion(plawk_parser:plawk_negate_literal_delta(int(3), int(-3))),
    assertion(plawk_parser:plawk_negate_literal_delta(float_const(15, 10),
        float_const(-15, 10))),
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
