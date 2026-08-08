:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `arr[k] += N` / `arr[k] -= N` / `arr[k]--` at a SCALAR-VARIABLE key -- closing the
% `inc_assoc` / `add_assoc` key-coverage gap that tests/test_plawk_assoc_decrement.pl
% recorded and attributed:
%
%              field key $1   literal key "x"   scalar-var key k
%   c[K]++          yes            no                yes
%   c[K] += N       yes            no                yes  <- was no
%   c[K]--          yes            no                yes  <- was no
%
% ---------------------------------------------------------------------------
% THREE LISTS OF THE SAME SET, AND ONE OF THEM MISCOMPILED
%
% `@wam_assoc_i64_inc(table, key, delta)` takes the delta as an ARGUMENT, so
% `arr[k]++` and `arr[k] += N` were always the same call with a different constant.
% What made one compile and the other decline was not a missing operation but FOUR
% independent lists of which svar-keyed action shapes exist, each written at a
% different site, each naming `inc_assoc` and none naming `add_assoc`:
%
%   1. the mixed route's admission gate      (plawk_assoc_update_action/1)
%   2. the table-registration walker         (plawk_assoc_increment_spec_in_action/2)
%   3. the assoc-only route's body spec      (plawk_assoc_body_action_spec/2)
%   4. the strnum read-use gate              (plawk_strnum_action_unsafe_read/3)
%
% Three of the four turn a missing row into a DECLINE. The fourth does not, and that
% is the part worth keeping: gate 4 decides the key scalar's REPRESENTATION. Reading
% `k` as an assoc key is a supported strnum read, so `k` keeps its interned-atom-id
% slot; an unrecognised read deactivates strnum and `k` becomes a plain i64 counter,
% whose key is the DECIMAL of the field parsed as a number -- "0" for "INFO". So while
% gates 1-3 were still refusing `c[k] += N` on its own, a program that mixed the two
% spellings (`{ k = $1; c[k]++; c[k] += 2 }`) passed gate 1 on the `++`, registered
% its table on the `++`, and BUILT -- keying the whole table on "0" and printing an
% empty line where gawk prints 3. A build that runs and prints the wrong thing, which
% is worse than the decline the other three gates were giving.
%
% The lesson is not "add the fourth row". It is that a gate's failure mode decides how
% much a missing row costs, and a representation-selecting gate fails SILENTLY. All
% four now defer to plawk_assoc_scalar_key_update/4 -- one predicate naming the set --
% so the next key shape added is added once. Pinned below at the IR level (the key
% must be interned from the field slice, not from a decimal) so a re-split of the
% lists is caught as a wrong key rather than as a wrong count.
%
% ---------------------------------------------------------------------------
% WHAT IS STILL REFUSED
%
% Both remaining columns of the matrix decline cleanly, pinned:
%   - a LITERAL key (`c["x"] += 1`, `c["x"]++`) -- refused for BOTH families, so it
%     belongs to neither this change nor the decrement.
%   - a NON-LITERAL delta at an svar key (`c[k] += $2`) -- the delta reaches the
%     emitter as an integer constant. `c[$1] += $2` (a FIELD key) is unaffected.
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

% Four records; $1 is INFO once, DEBUG twice, ERROR once.
input("INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n").

:- begin_tests(plawk_assoc_key_coverage).

% --- the mixed route (END is a plain print) --------------------------------

test(svar_key_add_assign_one, [condition(clang_available)]) :-
    run("{ k = $1; c[k] += 1 } END { print c[\"INFO\"] }\n", "1\n"),
    !.

test(svar_key_add_assign_delta_not_one, [condition(clang_available)]) :-
    run("{ k = $1; c[k] += 2 } END { print c[\"DEBUG\"] }\n", "4\n"),
    !.

test(svar_key_decrement, [condition(clang_available)]) :-
    run("{ k = $1; c[k]-- } END { print c[\"DEBUG\"] }\n", "-2\n"),
    !.

test(svar_key_subtract_assign, [condition(clang_available)]) :-
    run("{ k = $1; c[k] -= 3 } END { print c[\"INFO\"] }\n", "-3\n"),
    !.

test(svar_key_negative_add_delta, [condition(clang_available)]) :-
    run("{ k = $1; c[k] += -1 } END { print c[\"INFO\"] }\n", "-1\n"),
    !.

test(svar_key_subtract_a_negative, [condition(clang_available)]) :-
    run("{ k = $1; c[k] -= -1 } END { print c[\"INFO\"] }\n", "1\n"),
    !.

% Mixing the spellings in one rule body -- the shape that BUILT and printed the wrong
% answer while the gates disagreed (see the header). Both updates must land on the
% same, text-derived key.
test(svar_key_increment_then_add_assign, [condition(clang_available)]) :-
    run("{ k = $1; c[k]++; c[k] += 2 } END { print c[\"INFO\"] }\n", "3\n"),
    !.

test(svar_key_increment_then_decrement_cancels, [condition(clang_available)]) :-
    run("{ k = $1; c[k]++; c[k]-- } END { print c[\"INFO\"] }\n", "0\n"),
    !.

% Two keys read back, so a collapsed key (everything landing on one entry) shows up
% as a wrong pair rather than a plausible single number.
test(svar_key_two_keys_stay_distinct, [condition(clang_available)]) :-
    run("{ k = $1; c[k] += 2 } END { print c[\"INFO\"], c[\"DEBUG\"] }\n", "2 4\n"),
    !.

% The update in an if/else branch, both arms.
test(svar_key_add_assign_in_a_branch, [condition(clang_available)]) :-
    run("{ k = $1; if ($2 == \"trace\") c[k]--; else c[k] += 5 } END { print c[\"DEBUG\"], c[\"INFO\"] }\n",
        "-2 5\n"),
    !.

% A NUMERIC (counter) key still works: the counter is interned via its decimal
% spelling, so the delta rides the same path as the strnum key.
test(counter_key_add_assign, [condition(clang_available)]) :-
    run("{ n = NR % 2; c[n] += 2 } END { print c[\"0\"], c[\"1\"] }\n", "4 4\n"),
    !.

% --- the assoc-only route (END is a for-in) --------------------------------
%
% A different driver with its own body-action spec, planner and emitter -- gate 3.
% Sorted, because awk's for-in order is unspecified.

test(svar_key_add_assign_forin_end, [condition(clang_available)]) :-
    run_sorted("{ k = $1; c[k] += 2 } END { for (kk in c) print kk, c[kk] }\n",
        "DEBUG 4\nERROR 2\nINFO 2\n"),
    !.

test(svar_key_decrement_forin_end, [condition(clang_available)]) :-
    run_sorted("{ k = $1; c[k]-- } END { for (kk in c) print kk, c[kk] }\n",
        "DEBUG -2\nERROR -1\nINFO -1\n"),
    !.

test(svar_key_mixed_spellings_forin_end, [condition(clang_available)]) :-
    run_sorted("{ k = $1; c[k]++; c[k] -= 3 } END { for (kk in c) print kk, c[kk] }\n",
        "DEBUG -4\nERROR -2\nINFO -2\n"),
    !.

% --- the key's representation, at the IR level -----------------------------
%
% The silent-failure pin. When the strnum read-use gate did not recognise the
% add-assign, `k` deactivated to an i64 counter and the key became
% @wam_intern_i64_decimal(<field parsed as a number>) -- "0" for "INFO". The key must
% come from the field's BYTES.
test(svar_key_add_assign_keys_on_the_field_text_not_a_decimal) :-
    build_ll("{ k = $1; c[k] += 2 } END { print c[\"INFO\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@wam_atom_field_slice_value")),
    assertion(\+ sub_string(LL, _, _, _, "@wam_intern_i64_decimal")),
    !.

% ...and the same program compiled with `++` instead of `+=` keys the same way, so the
% pin above is a property of the key and not of the spelling.
test(svar_key_increment_keys_on_the_field_text_too) :-
    build_ll("{ k = $1; c[k]++ } END { print c[\"INFO\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@wam_atom_field_slice_value")),
    assertion(\+ sub_string(LL, _, _, _, "@wam_intern_i64_decimal")),
    !.

% The delta reaches the runtime call as its own constant -- the hardcoded `i64 1` is
% what made `c[k] += 2` unrepresentable even once the gates admitted it.
test(the_delta_reaches_the_runtime_call) :-
    build_ll("{ k = $1; c[k] += 7 } END { print c[\"INFO\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_0, i64 %rule_0_body_slot_0_op_0_snum_id, i64 7)")),
    !.

test(a_negative_delta_reaches_the_runtime_call) :-
    build_ll("{ k = $1; c[k]-- } END { print c[\"INFO\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "i64 -1)")),
    !.

% --- one operation, two spellings ----------------------------------------
%
% `c[k]++` and `c[k] += 1` are the same update, so they must emit the same program --
% byte for byte, including the key. This is the check that the four gates still agree:
% each of them recognising only one spelling shows up here as a diff (or, for the
% strnum gate, as two different key expressions) rather than as a count that happens
% to look plausible.
test(the_increment_and_the_add_assign_emit_identical_ir) :-
    build_ll("{ k = $1; c[k]++ } END { print c[\"INFO\"] }\n", IncIR),
    build_ll("{ k = $1; c[k] += 1 } END { print c[\"INFO\"] }\n", AddIR),
    assertion(IncIR == AddIR),
    !.

% ...and the same for the assoc-only route, which is a different driver end to end.
test(the_two_spellings_emit_identical_ir_in_the_forin_route) :-
    build_ll("{ k = $1; c[k]++ } END { for (kk in c) print kk, c[kk] }\n", IncIR),
    build_ll("{ k = $1; c[k] += 1 } END { for (kk in c) print kk, c[kk] }\n", AddIR),
    assertion(IncIR == AddIR),
    !.

% `c[k]--` and `c[k] -= 1` and `c[k] += -1` likewise -- all three are one add_assoc
% term, so nothing downstream can tell them apart.
test(every_decrement_spelling_emits_identical_ir) :-
    build_ll("{ k = $1; c[k]-- } END { print c[\"INFO\"] }\n", DecIR),
    build_ll("{ k = $1; c[k] -= 1 } END { print c[\"INFO\"] }\n", SubIR),
    build_ll("{ k = $1; c[k] += -1 } END { print c[\"INFO\"] }\n", NegIR),
    assertion(DecIR == SubIR),
    assertion(DecIR == NegIR),
    !.

% --- what is still refused, pinned ---------------------------------------

% WAS: both declined (status 3), pinned here as unrelated to the scalar-var key kind
% this suite covers. Correctly attributed -- and it turned out to be unrelated in the
% implementation too: a STRING-literal key is the arity-1 case of the multi-dimensional
% key builder, needing no part of the machinery this suite exercises. Kept, inverted;
% the full matrix is in tests/test_plawk_literal_assoc_key.pl.
test(a_string_literal_key_now_works_for_both_spellings, [condition(clang_available)]) :-
    run("{ c[\"x\"] += 1 } END { print c[\"x\"] }\n", "4\n"),
    run("{ c[\"x\"]++ } END { print c[\"x\"] }\n", "4\n"),
    !.

% WAS a decline, and the reason recorded here was correct at the time: `arr[N]` was
% claimed by the raw-integer key space that positional (split) tables read, so admitting
% the update made `{ c[5]++; print c[5] }` store and load different keys. That cause is
% gone -- the reads resolve the key space from the table's kind, and a positional table
% declines the UPDATE separately, as unrepresentable rather than unresolved. So the
% integer key works, and it is the SAME key as the string spelling, which is what awk
% means by a subscript. tests/test_plawk_int_subscript.pl owns the matrix.
test(an_integer_literal_key_now_works_and_is_the_same_key,
        [condition(clang_available)]) :-
    run("{ c[5] += 1 } END { print c[\"5\"] }\n", "4\n"),
    run("{ c[5]++ } END { print c[\"5\"] }\n", "4\n"),
    !.

% A non-literal delta at an svar key declines cleanly (`+=`) or is a parse error
% (`-=`, whose negation is a parse-time literal negation). A FIELD key with the same
% delta works, so this is a key-kind restriction, not a delta one.
test(a_non_literal_delta_at_an_svar_key_is_refused) :-
    build_status("{ k = $1; c[k] += $2 } END { print c[\"INFO\"] }\n", 3),
    build_status("{ k = $1; c[k] -= $2 } END { print c[\"INFO\"] }\n", 2),
    !.

test(a_non_literal_delta_at_a_field_key_still_works, [condition(clang_available)]) :-
    run("{ c[$1] += $2 } END { print c[\"INFO\"] }\n", "0\n"),
    !.

% A DOUBLE key is still a clean not-yet for the add-assign, as it is for the
% increment (tests/test_plawk_assoc_varkey.pl): no float->string key path.
test(a_double_key_still_declines) :-
    build_status("{ d = NR / 2; c[d] += 2 } END { print c[\"1\"] }\n", 3),
    !.

% --- regressions: the field-key family is untouched ----------------------

test(field_key_add_assign_unchanged, [condition(clang_available)]) :-
    run("{ c[$1] += 2 } END { print c[\"DEBUG\"] }\n", "4\n"),
    !.

test(field_key_decrement_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]-- } END { print c[\"DEBUG\"] }\n", "-2\n"),
    !.

test(scalar_add_assign_unchanged, [condition(clang_available)]) :-
    run("{ n += 2 } END { print n }\n", "8\n"),
    !.

:- end_tests(plawk_assoc_key_coverage).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_key_coverage', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    run_(Src, Expected, plain).

run_sorted(Src, Expected) :-
    run_(Src, Expected, sorted).

run_(Src, Expected, Mode) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'kc_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'kc', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out0),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Mode == sorted -> sort_lines(Out0, Out) ; Out = Out0 ),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

% awk's for-in order is unspecified, so for-in output is compared as a multiset of
% lines. The trailing newline is preserved.
sort_lines(In, Out) :-
    split_string(In, "\n", "", Parts0),
    ( append(Parts, [""], Parts0) -> true ; Parts = Parts0 ),
    msort(Parts, Sorted),
    atomic_list_concat(Sorted, '\n', Joined),
    ( Sorted == [] -> Out = "" ; format(string(Out), "~w\n", [Joined]) ).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'kc_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

% The emitted LLVM IR, straight from the code generator -- no clang needed, so these
% pins run everywhere.
build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
