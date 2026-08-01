:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `$0` against a string literal as a TERNARY CONDITION.
%
% This closes the loop on a miscompile from #4035, and the history is the point.
%
% In #4035 the whole-record condition was ADMITTED into the ternary and printed
% 0/0/0 where gawk gives 0/1/0. The fix then was to require `Index >= 1` in both
% the gate and the emitter -- correct at the time, because the only comparator
% available projected a FIELD SLICE out of the record and answers false for index
% 0. Excluding `$0` turned a miscompile into a clean decline.
%
% The facts changed when `$0 OP "str"` landed as a rule PATTERN, bringing a
% whole-record strcmp -- a comparison that IS correct at index 0. The exclusion
% then stopped protecting anything and started being a gap, so the ternary
% condition now reuses plawk_record_str_cmp_guard_ir/5: the very emitter that
% pattern uses. One emitter, two surfaces -- a whole-record comparison means the
% same thing written as a rule pattern or as a ternary condition.
%
% The gate (plawk_ternary_cond_ok/3) and the emitter (plawk_ternary_cond_ir/8)
% gained their matching clauses in ONE commit, because their DIVERGENCE -- a gate
% admitting what the emitter could not lower correctly -- is precisely what made
% the original attempt a miscompile instead of a decline.
%
% Because the condition emitter is shared across branch types, this works with
% i64 branches and string branches alike, in assignment, print and printf.
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

:- begin_tests(plawk_ternary_record_cond).

% --- THE regression from #4035 --------------------------------------------

% The exact program that printed 0/0/0 against gawk's 0/1/0. First and foremost.
test(the_4035_miscompile_now_matches_gawk, [condition(clang_available)]) :-
    run("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

test(the_4035_shape_in_a_print, [condition(clang_available)]) :-
    run("{ print $0 == \"b 2\" ? 1 : 0 }\n", "0\n1\n0\n"),
    !.

% --- the operators --------------------------------------------------------

test(record_inequality_condition, [condition(clang_available)]) :-
    run("{ x = $0 != \"b 2\" ? 1 : 0; print x }\n", "1\n0\n1\n"),
    !.

test(record_less_than_condition, [condition(clang_available)]) :-
    run("{ x = $0 < \"b\" ? 1 : 0; print x }\n", "1\n0\n0\n"),
    !.

test(record_greater_equal_condition, [condition(clang_available)]) :-
    run("{ x = $0 >= \"b 2\" ? 1 : 0; print x }\n", "0\n1\n1\n"),
    !.

% The REVERSED order, which needed only that normalise_ternary_cmp/4 stop
% requiring a positive index -- it mirrors through the same swap_cmp_op/2 as
% every other reversed form.
test(record_condition_reversed, [condition(clang_available)]) :-
    run("{ x = \"b 2\" == $0 ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

% --- composes with everything the condition emitter already served --------

% STRING branches: the condition emitter is shared across branch types, so this
% needed no extra work -- which is the property worth pinning.
test(record_condition_with_string_branches, [condition(clang_available)]) :-
    run("{ x = $0 == \"b 2\" ? \"hit\" : \"miss\"; print x }\n",
        "miss\nhit\nmiss\n"),
    !.

test(record_condition_in_printf, [condition(clang_available)]) :-
    run("{ printf \"%d\\n\", $0 == \"b 2\" ? 1 : 0 }\n", "0\n1\n0\n"),
    !.

test(record_condition_parenthesised, [condition(clang_available)]) :-
    run("{ x = ($0 == \"b 2\" ? 1 : 0); print x }\n", "0\n1\n0\n"),
    !.

% --- regressions: positive fields still take the FIELD comparator ---------

test(positive_field_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = $1 == \"b\" ? 1 : 0; print x }\n", "0\n1\n0\n"),
    !.

test(positive_field_ordering_unchanged, [condition(clang_available)]) :-
    run("{ x = $1 <= \"b\" ? 1 : 2; print x }\n", "1\n1\n2\n"),
    !.

test(numeric_condition_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

% --- structure: the gate and the emitter agree ----------------------------

% The gate now admits `$0` with a string literal, for all six operators.
test(gate_admits_the_whole_record) :-
    forall(member(Op, [eq, ne, lt, le, gt, ge]),
        assertion(plawk_native_codegen:plawk_ternary_cond_ok(field(0), Op,
            string("b 2")))),
    !.

% ...and so does the emitter. This PAIR is the test that matters: a gate that
% admits what the emitter cannot lower correctly is exactly how #4035 became a
% miscompile rather than a decline, so the two are asserted together.
test(emitter_matches_the_gate) :-
    forall(member(Op, [eq, ne, lt, le, gt, ge]),
        ( assertion(plawk_native_codegen:plawk_ternary_cond_ok(field(0), Op,
              string("b 2"))),
          assertion(plawk_native_codegen:plawk_ternary_cond_ir(
              cmp(field(0), Op, string("b 2")), 32, tbase, tglobal,
              _CondIR, _Globals, _OperandSetup, _CondLines))
        )),
    !.

% Both operand orders normalise to the same condition term.
test(both_orders_normalise_alike) :-
    plawk_parse_string("{ x = $0 == \"b 2\" ? 1 : 0 }\n",
        program([], [rule(always, [set(var(x), Forward)])], [])),
    plawk_parse_string("{ x = \"b 2\" == $0 ? 1 : 0 }\n",
        program([], [rule(always, [set(var(x), Reversed)])], [])),
    assertion(Forward == Reversed),
    assertion(Forward = ternary(cmp(field(0), eq, string("b 2")),
        int(1), int(0))),
    !.

% --- IR shape -------------------------------------------------------------

% The condition is the whole-record strcmp, NOT the field-slice comparator that
% answers false at index 0. This is the assertion that would have caught #4035.
test(record_condition_ir_uses_the_record_strcmp) :-
    plawk_parse_string("{ x = $0 == \"b 2\" ? 1 : 0; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_reccmp = call i32 @strcmp'))),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '@wam_atom_field_str_cmp_value')),
    assertion(once(sub_atom(DriverIR, _, _, _, 'select i1'))),
    !.

% A positive field still uses the field comparator -- the two lowerings stay
% distinct, so this change cannot have quietly rerouted ordinary field
% conditions.
test(positive_field_condition_ir_unchanged) :-
    plawk_parse_string("{ x = $1 == \"b\" ? 1 : 0; print x }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@wam_atom_field_str_cmp_value'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_reccmp = call i32 @strcmp')),
    !.

:- end_tests(plawk_ternary_record_cond).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_record_cond', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'trc_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'trc', Prog0),
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

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
