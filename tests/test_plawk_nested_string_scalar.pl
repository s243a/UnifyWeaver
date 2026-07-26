:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: a string scalar assigned inside an `if` / loop body.
%
% `{ if (NR > 1) s = $1 } END { print s }` compiled and printed `0` instead of
% the field text -- WRONG OUTPUT, not a decline. Two collectors asked
% overlapping questions about the same rules and walked different shapes:
%
%   * plawk_scalar_update_name_expr/3 descends into `if` branches and
%     foreach / while / do-while bodies, so the name DID get a slot;
%   * plawk_scalar_string_names/2 and the strnum collector's action enumerator
%     looked at TOP-LEVEL rule-body actions only, so the name's TYPE was decided
%     by a view that never saw the assignment.
%
% The slot therefore existed as a plain i64 counter and the END print rendered
% the default 0. Both collectors now walk the shared plawk_scalar_nested_action/2
% helper, so the slot and its type are decided over the same actions.
%
% Fixing the reach exposed a second, PRE-EXISTING bug (present at the merge base
% with no nesting involved): a name written both as a field copy and as a string
% (`{ s = "x"; s = $1 }`) types as a string slot, because a literal write
% disqualifies strnum -- but the field-copy STORE only had a strnum clause, so it
% fell through to the generic numeric store, put the field's numeric value (0 for
% non-numeric text) in the slot, and printing resolved atom id 0 to empty. String
% and strnum slots both hold an interned atom id, so both write paths are now
% behind plawk_slot_holds_text/1.
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

% Three records: "a 1" / "b 2" / "c 3". The last record's $1 is "c", so a guard
% of NR > 1 leaves "c" in the slot at END.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_nested_string_scalar).

% --- the reported bug ------------------------------------------------------

% A bare field copy inside an `if` body. Printed `0` before.
test(if_body_field_copy, [condition(clang_available)]) :-
    run("{ if (NR > 1) s = $1 } END { print s }\n", "c\n"),
    !.

% A string literal inside an `if` body.
test(if_body_string_literal, [condition(clang_available)]) :-
    run("{ if (NR > 1) s = \"hit\" } END { print s }\n", "hit\n"),
    !.

% A concatenation inside an `if` body.
test(if_body_concat, [condition(clang_available)]) :-
    run("{ if (NR > 1) s = $1 $2 } END { print s }\n", "c3\n"),
    !.

% Arbitrary nesting depth, not just one level.
test(nested_if_in_if, [condition(clang_available)]) :-
    run("{ if (NR > 1) { if (NR > 2) s = $1 } } END { print s }\n", "c\n"),
    !.

% A while body, not just an if body -- the shared walker covers foreach /
% while / do-while too.
test(while_body_field_copy, [condition(clang_available)]) :-
    run("{ i = 0; while (i < 1) { t = $1; i++ } } END { print t }\n", "c\n"),
    !.

% The value reaches an END printf argument as text, not as a number.
test(if_body_string_reaches_end_printf, [condition(clang_available)]) :-
    run("{ if (NR > 1) s = $1 } END { printf \"[%s]\\n\", s }\n", "[c]\n"),
    !.

% --- the mixed field-copy + string case -----------------------------------

% A name written BOTH ways inside branches. A literal write disqualifies strnum,
% so the slot is string-typed; the field-copy write must still intern the text.
test(if_else_field_copy_and_literal, [condition(clang_available)]) :-
    run("{ if (NR > 1) s = $1; else s = \"first\" } END { print s }\n", "c\n"),
    !.

% The same mixed shape with NO nesting at all -- this was broken at the merge
% base too, and is the second bug the reach fix exposed. Field copy last wins.
test(top_level_mixed_literal_then_field, [condition(clang_available)]) :-
    run("{ s = \"x\"; s = $1 } END { print s }\n", "c\n"),
    !.

% Literal last wins, the order that already worked.
test(top_level_mixed_field_then_literal, [condition(clang_available)]) :-
    run("{ s = $1; s = \"x\" } END { print s }\n", "x\n"),
    !.

% --- regressions: shapes that already worked ------------------------------

test(top_level_field_copy, [condition(clang_available)]) :-
    run("{ s = $1 } END { print s }\n", "c\n"),
    !.

test(top_level_string_literal, [condition(clang_available)]) :-
    run("{ s = \"x\" } END { print s }\n", "x\n"),
    !.

test(string_accumulation, [condition(clang_available)]) :-
    run("{ s = s $1 } END { print s }\n", "abc\n"),
    !.

% A numeric counter incremented inside an `if` body must stay an i64 counter --
% widening the STRING collector's reach must not make numeric names string-typed.
test(if_body_counter_stays_numeric, [condition(clang_available)]) :-
    run("{ if (NR > 1) n++ } END { print n }\n", "2\n"),
    !.

% A double slot assigned in an `if` body stays double.
test(if_body_double_stays_double, [condition(clang_available)]) :-
    run("{ if (NR > 1) d = $2 / 2 } END { print d }\n", "1.5\n"),
    !.

% --- slot typing, directly -------------------------------------------------

% The string-name collector sees a nested assignment. This is the specific
% disagreement that caused the bug: the slot collector always saw it.
test(nested_string_assignment_is_collected) :-
    Rules = [rule(always, [if(cmp(special('NR'), gt, int(1)),
                              [set(var(s), string("x"))], [])])],
    plawk_native_codegen:plawk_scalar_string_names(Rules, Strings),
    assertion(Strings == [s]),
    !.

% A nested field copy seeds strnum-ness, so `s` types as a text-holding slot
% rather than an i64 counter.
test(nested_field_copy_types_as_text) :-
    Rules = [rule(always, [if(cmp(special('NR'), gt, int(1)),
                              [set(var(s), field(1))], [])])],
    plawk_native_codegen:plawk_scalar_typed_slots(Rules, [s], Slots),
    assertion(Slots = [Slot]),
    Slots = [Slot],
    assertion(plawk_native_codegen:plawk_slot_holds_text(Slot)),
    !.

% A nested increment still types as a plain counter (no over-widening).
test(nested_increment_types_as_counter) :-
    Rules = [rule(always, [if(cmp(special('NR'), gt, int(1)),
                              [inc(var(n))], [])])],
    plawk_native_codegen:plawk_scalar_typed_slots(Rules, [n], Slots),
    assertion(Slots == [scalar_counter(n)]),
    !.

% Both text-holding slot kinds are behind one guard, so a write emitter cannot
% support one and silently mis-store the other as a number.
test(both_text_slot_kinds_recognised) :-
    assertion(plawk_native_codegen:plawk_slot_holds_text(scalar_string(x))),
    assertion(plawk_native_codegen:plawk_slot_holds_text(scalar_strnum(x))),
    assertion(\+ plawk_native_codegen:plawk_slot_holds_text(scalar_counter(x))),
    assertion(\+ plawk_native_codegen:plawk_slot_holds_text(scalar_double(x))),
    !.

% The nested walker yields the container itself plus every action at any depth,
% for each container kind the scalar collectors care about.
test(nested_action_walker_reaches_every_depth) :-
    Inner = set(var(s), field(1)),
    forall(member(Container,
               [ if(cmp(special('NR'), gt, int(1)), [Inner], [])
               , if(cmp(special('NR'), gt, int(1)), [], [Inner])
               , while_loop(cmp(var(i), lt, int(1)), [Inner])
               , do_while_loop([Inner], cmp(var(i), lt, int(1)))
               , foreach_loop(layout, [Inner])
               , if(cmp(special('NR'), gt, int(1)),
                    [if(cmp(special('NR'), gt, int(2)), [Inner], [])], [])
               ]),
        ( plawk_native_codegen:plawk_scalar_nested_action(Container, Inner)
        -> true
        ;  format(user_error, "~nnot reached in: ~q~n", [Container]), fail
        )),
    !.

% --- IR shape --------------------------------------------------------------

% A nested field copy interns the field's bytes rather than storing its numeric
% value: the store goes through @wam_intern_atom, not the generic `add i64 0`.
test(nested_field_copy_ir_interns) :-
    plawk_parse_string("{ if (NR > 1) s = $1 } END { print s }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_intern_atom'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value'))),
    !.

:- end_tests(plawk_nested_string_scalar).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_nested_string_scalar', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build Src, run it over the shared input, require Expected byte-for-byte.
run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ns_bin', StaleBin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(StaleBin) -> delete_file(StaleBin) ; true ),
    directory_file_path(Dir, 'ns', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', StaleBin], 0),
    process_create(StaleBin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    assertion(Out == Expected).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
