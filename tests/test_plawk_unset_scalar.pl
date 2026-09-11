:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% An UNSET SCALAR prints nothing, not 0.
%
%   { n++ } END { print n }        on EMPTY input:  printed 0, gawk prints ""
%   $1 == "ZZZ" { n++ } END { print n }           : printed 0, gawk prints ""
%
% awk's uninitialized value is a DUAL: the empty string in string context, 0 in
% numeric context. `print` is string context, so an unset variable contributes no
% bytes. plawk printed 0 because a counter lives in an i64 register initialised to
% 0 and the END render read that register numerically -- the STORAGE decided the
% type, which is exactly what awk's model says must not happen.
%
% Note the canonical program is affected: `{ n++ } END { print n }` on empty input
% never executes the increment. So this is not an exotic corner.
%
% ---------------------------------------------------------------------------
% PRESENCE, NOT VALUE
%
% The test is whether the scalar was ASSIGNED, never whether its value is 0:
% `{ n = 0 } END { print n }` prints 0, because it was assigned. That is the same
% rule @wam_assoc_i64_print already applies to absent array elements (it probes the
% occupied bit, not the stored i64) -- absent elements were fixed that way, and
% string scalars already rendered an unset atom id 0 as empty. Counters were the
% last slot kind still deciding by storage type.
%
% ---------------------------------------------------------------------------
% A MONOTONIC MARK, SO NO PHI THREADING
%
% @plawk_slot_assigned is a bit table; an update stores `true` into its slot's
% entry. The mark only ever goes false->true, so it needs no SSA threading through
% the record loop's phis -- a plain store where the update is emitted runs exactly
% when the assignment runs, including inside `if` branches and loop bodies. The
% alternative (a companion i1 slot threaded through every phi) would have meant an
% arity change across ~34 clauses of the sequence emitter.
%
% The mark is written by ONE wrapper over every clause of the update emitter
% (plawk_scalar_update_operation_ir/9), not per clause: an update that forgot to
% mark would render an assigned counter as unset, which is wrong output.
%
% ---------------------------------------------------------------------------
% WHY IT CANNOT PRODUCE WRONG OUTPUT (the trackability rule)
%
% A slot renders empty-when-unset only if EVERY reachable assignment to its name is
% update-shaped -- i.e. goes through that one marking emitter.
% plawk_unset_tracked_slots/3 enforces this. A name also written by a getline
% capture, a `gsub` count or a dynrec bind (each of which writes a slot value
% through its own emitter, with no mark) is NOT tracked and keeps printing 0.
%
% So "tracked" means "every write is marked", by construction rather than by
% diligence -- the failure mode of a missed writer is an unfixed divergence, never
% a wrongly-empty number.
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

input("INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n").

:- begin_tests(plawk_unset_scalar).

% --- never assigned at all ------------------------------------------------

% The canonical counter program, on EMPTY input: the increment never runs.
test(counter_on_empty_input_prints_nothing, [condition(clang_available)]) :-
    run_input("{ n++ } END { print n }\n", "", "\n"),
    !.

test(accumulator_on_empty_input_prints_nothing, [condition(clang_available)]) :-
    run_input("{ s += 1 } END { print s }\n", "", "\n"),
    !.

% Dead code after `next`: the increment is unreachable, so the scalar can only ever
% be unset. This is the shape that had 12 stale expectations in the prefix-print
% suite before absent ARRAY elements were fixed; the scalar half is fixed here.
test(unreachable_increment_prints_nothing, [condition(clang_available)]) :-
    run("$1 == \"DEBUG\" { next; skipped++ } { total++ } END { print total, skipped }\n",
        "2 \n"),
    !.

% --- assigned only under a condition that never fires ---------------------
%
% These are the cases no static analysis could settle: the assignment is reachable
% and simply never executes. They are why the mark is a runtime bit.

test(guarded_rule_that_never_matches_prints_nothing, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { n++ } END { print n }\n", "\n"),
    !.

test(if_in_a_rule_body_that_never_fires_prints_nothing, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") n++ } END { print n }\n", "\n"),
    !.

% Nested `if`: the mark is stored where the update is emitted, so depth is
% irrelevant -- unlike a walker that has to learn each nesting level.
test(nested_if_that_never_fires_prints_nothing, [condition(clang_available)]) :-
    run("{ if ($1 == \"DEBUG\") { if ($2 == \"zzz\") n++ } } END { print n }\n",
        "\n"),
    !.

% One branch fires and the other does not, in the same program.
test(if_else_marks_only_the_branch_that_ran, [condition(clang_available)]) :-
    run("{ if ($1 == \"ZZZ\") a++; else b++ } END { print a, b }\n", " 4\n"),
    !.

% Two counters, one assigned and one not.
test(one_counter_assigned_one_not, [condition(clang_available)]) :-
    run("$1 == \"DEBUG\" { d++ } $1 == \"ZZZ\" { z++ } END { print d, z }\n",
        "2 \n"),
    !.

% --- assigned: unchanged, including an assigned ZERO ---------------------

test(assigned_counter_prints_its_value, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "4\n"),
    !.

% PRESENCE, NOT VALUE: an explicitly assigned 0 prints 0. Testing the value
% instead of the mark would print nothing here, which is the bug this design
% avoids -- and the same trap the str-valued assoc tables document (atom id 0 is a
% legitimate value, so the value cannot be the unset test).
test(explicitly_assigned_zero_prints_zero, [condition(clang_available)]) :-
    run("{ n = 0 } END { print n }\n", "0\n"),
    !.

test(guarded_rule_that_does_match_prints_its_value, [condition(clang_available)]) :-
    run("$1 == \"DEBUG\" { n++ } END { print n }\n", "2\n"),
    !.

test(if_in_a_rule_body_that_fires_prints_its_value, [condition(clang_available)]) :-
    run("{ if ($1 == \"DEBUG\") n++ } END { print n }\n", "2\n"),
    !.

% Assigned inside a rule-body LOOP.
test(counter_assigned_in_a_loop_prints_its_value, [condition(clang_available)]) :-
    run("{ i = 0; while (i < 2) { n++; i++ } } END { print n }\n", "8\n"),
    !.

% Assigned in the END block itself, then printed there.
test(counter_assigned_in_end_prints_its_value, [condition(clang_available)]) :-
    run("{ n++ } END { m = 0; while (m < 2) { m++ }; print m }\n", "2\n"),
    !.

% --- context matters: numeric contexts still read 0 ---------------------
%
% The dual value: the mark changes only the STRING-context render.

test(unset_counter_in_arithmetic_is_zero, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { n++ } END { print n + 0 }\n", "0\n"),
    !.

test(unset_counter_in_printf_d_is_zero, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { n++ } END { printf \"%d\\n\", n }\n", "0\n"),
    !.

% --- composition ---------------------------------------------------------

% A concatenation goes through the same counter render as a standalone print, so
% the two cannot disagree about an unset value.
test(unset_counter_in_a_concatenation, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { n++ } END { print \"n=\" n }\n", "n=\n"),
    !.

test(assigned_counter_in_a_concatenation, [condition(clang_available)]) :-
    run("$1 == \"DEBUG\" { n++ } END { print \"n=\" n }\n", "n=2\n"),
    !.

% An unset counter beside other fields still emits its OFS separators.
test(unset_counter_keeps_its_separators, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { n++ } END { print \"a\", n, \"b\" }\n", "a  b\n"),
    !.

% A statement list: each print is renamed by statement index, so two counter
% renders cannot collide.
test(unset_counter_in_a_statement_list, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { n++ } END { print n; print \"x\" }\n", "\nx\n"),
    !.

% --- the trackability rule ----------------------------------------------

% A simple counter is tracked; a name with no assignment at all is tracked (it can
% only be unset).
test(a_simple_counter_is_tracked) :-
    plawk_parse_string("$1 == \"ZZZ\" { n++ } END { print n }\n",
        program(_B, Rules, _E)),
    plawk_native_codegen:plawk_scalar_state_plan(Rules, [var(n)], StatePlan),
    plawk_native_codegen:plawk_state_plan_tracked(StatePlan, Tracked),
    plawk_native_codegen:plawk_state_plan_slots(StatePlan, Slots),
    nth0(Index, Slots, scalar_counter(n)),
    assertion(memberchk(Index, Tracked)),
    !.

% A name written by a NON-update emitter is refused. `gsub_count` writes its count
% slot through its own emitter with no mark, so tracking it would render an
% assigned count as unset -- wrong output. This is the assertion that makes the
% safety rule explicit rather than incidental.
test(a_gsub_count_name_is_not_tracked) :-
    Rules = [rule(always, [gsub_count(c, global, "o", "0", field(0))])],
    plawk_native_codegen:plawk_scalar_state_plan(Rules, [var(c)], StatePlan),
    plawk_native_codegen:plawk_state_plan_tracked(StatePlan, Tracked),
    plawk_native_codegen:plawk_state_plan_slots(StatePlan, Slots),
    ( nth0(Index, Slots, scalar_counter(c))
    -> assertion(\+ memberchk(Index, Tracked))
    ;  true          % no counter slot for it at all is equally safe
    ),
    !.

% The container-vs-leaf distinction: an `if` reports itself as assigning whatever
% its branches assign, so the walker must yield the LEAF. Yielding the container
% too made every conditionally-assigned counter untracked.
test(the_walker_yields_leaves_not_containers) :-
    Action = if(scalar_if(cmp(var(x), eq, int(1))), [inc(var(n))], []),
    findall(Leaf,
        plawk_native_codegen:plawk_action_subaction(Action, Leaf),
        Leaves),
    assertion(Leaves == [inc(var(n))]),
    !.

% The mark table is fixed-width, and the tracker refuses any slot at or beyond it,
% so a mark can never index out of bounds.
test(the_mark_table_width_bounds_tracking) :-
    plawk_native_codegen:plawk_slot_assigned_width(Width),
    assertion(integer(Width)),
    assertion(Width > 0),
    % a store is emitted only below the width
    plawk_native_codegen:plawk_scalar_assigned_store_ir(scalar_counter(n), 0,
        rule_0, 1, InIR),
    assertion(InIR \== ''),
    plawk_native_codegen:plawk_scalar_assigned_store_ir(scalar_counter(n), Width,
        rule_0, 1, OutIR),
    assertion(OutIR == ''),
    !.

% --- IR shape -----------------------------------------------------------

% The mark store lands in the record loop, and the END render selects the format
% rather than branching. Pinned on this construct's OWN names.
test(a_tracked_counter_emits_a_mark_and_a_format_select) :-
    plawk_parse_string("$1 == \"ZZZ\" { n++ } END { print n }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@plawk_slot_assigned'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'store i1 true'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%end_asg_0 = load i1'))),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@.plawk_surface_print_unset'))),
    !.

% A string scalar's END print is untouched -- it already tested its own unset
% sentinel, and must not grow a second mechanism for the same property.
test(a_string_scalar_print_is_unchanged) :-
    plawk_parse_string("{ s = $1 } END { print s }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%end_str_empty_0'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%end_asg_0')),
    !.

% --- doubles, on the same footing as counters ----------------------------
%
% An unset double printed 0 for a release after counters were fixed: they were two
% slot kinds with two renders and two lists of "which kinds get a mark". They now
% share ONE render (plawk_end_numeric_print_lines/6) and ONE table
% (plawk_numeric_slot_print/5) that the render, the mark and the trackability check
% all key off -- so a numeric kind is marked exactly when it can be rendered, rather
% than by three lists kept in step by hand.

test(unset_double_prints_nothing, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { x += 1.5 } END { print x }\n", "\n"),
    !.

test(unset_double_on_empty_input_prints_nothing, [condition(clang_available)]) :-
    run_input("{ x += 1.5 } END { print x }\n", "", "\n"),
    !.

test(assigned_double_prints_its_value, [condition(clang_available)]) :-
    run("$1 == \"DEBUG\" { x += 1.5 } END { print x }\n", "3\n"),
    !.

% PRESENCE, NOT VALUE, for doubles too.
test(explicitly_assigned_zero_double_prints_zero, [condition(clang_available)]) :-
    run("{ x = 0.0 } END { print x }\n", "0\n"),
    !.

% A division result is a double slot, so it travels the same path.
test(unset_division_result_prints_nothing, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { x = 7 / 2 } END { print x }\n", "\n"),
    !.

test(unset_double_in_a_concatenation, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { x += 1.5 } END { print \"x=\" x }\n", "x=\n"),
    !.

% The dual value holds for doubles: numeric context still reads 0.
test(unset_double_in_arithmetic_is_zero, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { x += 1.5 } END { print x + 0 }\n", "0\n"),
    !.

% A counter and a double in one program, both unset.
test(unset_counter_and_double_together, [condition(clang_available)]) :-
    run("$1 == \"ZZZ\" { x += 1.5; n++ } END { print n, x }\n", " \n"),
    !.

% --- one table drives mark, render and trackability ---------------------

% Both numeric kinds are marked; string/strnum are not (their unset atom id 0
% already renders empty) -- and the mark keys off the SAME table as the render, so
% the two cannot disagree about which kinds have an unset path.
%
% This REPLACES an `only_counter_slots_are_marked` pin that asserted doubles get no
% mark. That pin was correct when only counters had an unset render, and it is the
% test that would have flagged the doubles gap if anyone had asked why a slot kind
% was excluded rather than merely asserting that it was. Stated as "the marked kinds
% are exactly the renderable kinds", it now says the property instead of the
% then-current membership.
test(both_numeric_kinds_are_marked_and_no_others) :-
    forall(member(Slot, [scalar_counter(n), scalar_double(x)]),
        ( plawk_native_codegen:plawk_scalar_assigned_store_ir(Slot, 0, rule_0, 1, IR),
          assertion(IR \== '') )),
    forall(member(Slot, [scalar_string(s), scalar_strnum(t), scalar_record_number]),
        ( plawk_native_codegen:plawk_scalar_assigned_store_ir(Slot, 0, rule_0, 1, IR),
          assertion(IR == '') )),
    !.

% The table itself: exactly the kinds with an unset render.
test(the_numeric_print_table_lists_both_kinds) :-
    assertion(plawk_native_codegen:plawk_numeric_slot_print(scalar_counter(n),
        _K1, _G1, _B1, i64)),
    assertion(plawk_native_codegen:plawk_numeric_slot_print(scalar_double(x),
        _K2, _G2, _B2, double)),
    assertion(\+ plawk_native_codegen:plawk_numeric_slot_print(scalar_string(s),
        _K3, _G3, _B3, _T3)),
    !.

% A double slot is tracked on the same terms as a counter -- and refused on the same
% terms when a non-update emitter writes it.
test(a_double_slot_is_tracked_like_a_counter) :-
    plawk_parse_string("$1 == \"ZZZ\" { x += 1.5 } END { print x }\n",
        program(_B, Rules, _E)),
    plawk_native_codegen:plawk_scalar_state_plan(Rules, [var(x)], StatePlan),
    plawk_native_codegen:plawk_state_plan_tracked(StatePlan, Tracked),
    plawk_native_codegen:plawk_state_plan_slots(StatePlan, Slots),
    nth0(Index, Slots, scalar_double(x)),
    assertion(memberchk(Index, Tracked)),
    !.

:- end_tests(plawk_unset_scalar).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_unset_scalar', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_input(Src, Input, Expected).

run_input(Src, Input, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'us_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'us', Prog0),
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
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == 0
    -> true
    ;  format(user_error, "~n~w~n  got      ~q rc=~w~n  expected ~q~n",
           [Src, Out, RC, Expected]), fail
    ).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
