:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: a MIXED (scalar slots + assoc tables) END statement list --
% `{ c[$1]++; n++ } END { print n; print c["a"] }`.
%
% The FOURTH and last END chain to take a statement list, completing the set:
%
%   scalar   tests/test_plawk_multi_print_end.pl, test_plawk_end_exit.pl
%   for-in   tests/test_plawk_end_chain.pl        (blocks, interleaved)
%   assoc    tests/test_plawk_assoc_end_list.pl   (straight-line)
%   mixed    THIS FILE                            (straight-line)
%
% Generalizing the fourth one turned the previous PR's assoc-specific walkers into
% shared ones parameterized by chain kind, so the two loop-free chains now differ
% in exactly two places: which print emitter resolves a field (the mixed chain
% reads scalar slots as well as tables), and which printf arguments are reachable.
% One walker, two configurations -- the same anti-drift discipline as the rest of
% this campaign.
%
% The mixed chain's printf accepts a `var` argument, because its slots exist,
% unlike the pure-assoc chain's literal-only gate. `NR` stays excluded in both:
% this driver's record counter is not the one the END printf emitter references,
% so admitting it would emit invalid IR rather than a decline.
%
% INHERITED BOUNDARY, not introduced here: the mixed driver requires the END to
% reference a table. `{ c[$1]++; n++ } END { print n }` -- a single print with no
% assoc reference -- declines at the merge base too, so the statement list declines
% for the same reason. Pinned below so a future fix to that admission updates both.
%
% gawk 5.2 is the oracle for every expectation here, exit STATUS included.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3". So `n` is 3 and c["a"], c["b"] are 1.
% Every program reads the table by LITERAL key, so output order is deterministic.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_mixed_end_list).

% --- scalar and assoc statements in one END -------------------------------

test(scalar_then_assoc_print, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n; print c[\"a\"] }\n", "3\n1\n", 0),
    !.

test(assoc_then_scalar_print, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; print n }\n", "1\n3\n", 0),
    !.

test(three_statements, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n; print c[\"a\"]; print c[\"b\"] }\n",
        "3\n1\n1\n", 0),
    !.

% A statement can still print a scalar and a table element together, as the
% single-print driver does.
test(combined_field_print_then_scalar, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n, c[\"a\"]; print \"done\" }\n",
        "3 1\ndone\n", 0),
    !.

test(literal_between_reads, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; print \"--\"; print n }\n",
        "1\n--\n3\n", 0),
    !.

% Two literal-key lookups in separate statements: each needs its own key global.
test(two_assoc_lookups, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; print c[\"b\"]; print n }\n",
        "1\n1\n3\n", 0),
    !.

% --- printf ---------------------------------------------------------------

% Unlike the pure-assoc chain, a `var` argument works here: the mixed chain has
% scalar slots, so it reads `n`'s final slot.
test(printf_scalar_var_argument, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; printf \"n=%d\\n\", n }\n",
        "1\nn=3\n", 0),
    !.

test(printf_literal_argument, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; printf \"%d\\n\", 7 }\n", "1\n7\n", 0),
    !.

test(printf_constant_arithmetic, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; printf \"%d\\n\", 2 * 3 }\n",
        "1\n6\n", 0),
    !.

% printf appends no ORS, so the next print closes the line.
test(printf_appends_no_newline, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { printf \"v=\"; print c[\"a\"] }\n", "v=1\n", 0),
    !.

% --- exit -----------------------------------------------------------------

test(assoc_read_then_exit, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; exit 3 }\n", "1\n", 3),
    !.

test(both_prints_then_exit, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; print n; exit 3 }\n", "1\n3\n", 3),
    !.

test(bare_exit_is_zero, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; exit }\n", "1\n", 0),
    !.

test(exit_truncates_following_statements, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; exit 1; print n }\n", "1\n", 1),
    !.

test(exit_first_suppresses_all_output, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { exit 5; print c[\"a\"] }\n", "", 5),
    !.

% --- regressions: the other three chains and the single print -------------

test(mixed_single_print_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n, c[\"a\"] }\n", "3 1\n", 0),
    !.

test(assoc_list_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; print \"done\" }\n", "1\ndone\n", 0),
    !.

test(scalar_list_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n; print \"x\" }\n", "3\nx\n", 0),
    !.

test(forin_chain_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print \"done\" }\n",
        ["a", "b", "c", "done"], 0),
    !.

% --- clean declines -------------------------------------------------------

% `NR` as a printf argument would reference a counter this driver does not define
% -- it must DECLINE, not emit invalid IR (which surfaces as a clang failure,
% status 4).
test(printf_nr_argument_declines) :-
    build_status("{ c[$1]++; n++ } END { print c[\"a\"]; printf \"%d\\n\", NR }\n", 3),
    !.

% WAS a decline. The statement-list driver threads the EndRecord token now, so a
% `$N` printf argument projects from the retained last record. The NR pin above did
% not move -- that exclusion is about this driver's counter naming, not the record.
test(printf_field_argument_now_works, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print c[\"a\"]; printf \"%s\\n\", $1 }\n",
        "1\nc\n", 0),
    !.

% INHERITED: the mixed driver requires the END to reference a table. A single
% print with no assoc reference declines at the merge base too, so a statement
% list of the same shape declines for the same reason -- not a regression, and
% pinned so a future fix to that admission updates both.
test(no_assoc_reference_single_print_declines) :-
    build_status("{ c[$1]++; n++ } END { print n }\n", 3),
    !.

test(no_assoc_reference_statement_list_declines) :-
    build_status("{ c[$1]++; n++ } END { print n; exit 3 }\n", 3),
    !.

% --- structure ------------------------------------------------------------

% The two loop-free chains share one classifier, differing only in the printf
% argument gate. Everything else about the statement vocabulary is identical.
test(shared_classifier_differs_only_on_printf_args) :-
    Statements = [print([string("a")]), exit(int(1))],
    % Both kinds accept it, and classify it identically (assertion/1 does not
    % propagate bindings, so call directly to compare the results).
    plawk_native_codegen:plawk_end_list_statements(assoc, Statements, ItemsAssoc),
    plawk_native_codegen:plawk_end_list_statements(mixed, Statements, ItemsMixed),
    assertion(ItemsAssoc == ItemsMixed),
    assertion(ItemsAssoc == [print([string("a")]), exit(int(1))]),
    % printf: the mixed chain admits a scalar var, the assoc chain does not
    Printf = [print([string("a")]), printf(string("%d\n"), [var(n)])],
    assertion(plawk_native_codegen:plawk_end_list_statements(mixed, Printf, _)),
    assertion(\+ plawk_native_codegen:plawk_end_list_statements(assoc, Printf, _)),
    !.

% Neither loop-free chain claims a lone plain print (the single-print drivers keep
% those shapes byte-identically), nor a for-in (the block chain's job).
test(loop_free_chains_exclude_lone_print_and_forin) :-
    forall(member(Kind, [assoc, mixed]),
        ( assertion(\+ plawk_native_codegen:plawk_end_list_statements(Kind,
              [print([string("a")])], _)),
          assertion(\+ plawk_native_codegen:plawk_end_list_statements(Kind,
              [for_in(var(k), var(c), [print([var(k)])]), print([string("x")])],
              _))
        )),
    !.

% Both END print emitters append the table frees in their base case, so both need
% the non-freeing variant a statement list uses.
test(both_print_emitters_have_a_non_freeing_variant) :-
    Plan = assoc_plan([c], []),
    % The assoc emitter takes an EndRecord capability token now (it grew field/NF-in-END
    % support). `no_end_record` here: this test is about the table frees, and a string literal
    % reads no record, so the token cannot affect what is asserted.
    plawk_native_codegen:plawk_assoc_end_print_ir([string("x")], Plan, 32, [32],
        no_end_record, AssocWith),
    plawk_native_codegen:plawk_assoc_end_print_body_ir([string("x")], Plan, 32,
        [32], no_end_record, AssocWithout),
    assertion(once(sub_atom(AssocWith, _, _, _, '@wam_assoc_i64_free'))),
    assertion(\+ sub_atom(AssocWithout, _, _, _, '@wam_assoc_i64_free')),
    % The mixed emitter takes an EndRecord capability token now (it grew NF-in-END support,
    % which needs the retained last record). `no_end_record` here: this test is about the
    % table frees, and a string literal reads no record -- so the token's value is
    % irrelevant to what is asserted, and passing the no-op one keeps it that way.
    plawk_native_codegen:plawk_mixed_end_print_ir([string("x")],
        state_plan([]), Plan, [32], no_end_record, MixedWith),
    plawk_native_codegen:plawk_mixed_end_print_body_ir([string("x")],
        state_plan([]), Plan, [32], no_end_record, MixedWithout),
    assertion(once(sub_atom(MixedWith, _, _, _, '@wam_assoc_i64_free'))),
    assertion(\+ sub_atom(MixedWithout, _, _, _, '@wam_assoc_i64_free')),
    !.

% --- IR shape -------------------------------------------------------------

% The tables are freed exactly ONCE for the whole list.
test(ir_frees_each_table_once) :-
    plawk_parse_string(
        "{ c[$1]++; n++ } END { print c[\"a\"]; print c[\"b\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    findall(B, sub_atom(DriverIR, B, _, _,
        '@wam_assoc_i64_free(%WamAssocI64Table* %plawk_assoc_table_0)'), Frees),
    assertion(Frees = [_]),
    !.

% Two literal-key lookups emit two distinct key globals.
test(ir_two_key_globals_are_distinct) :-
    plawk_parse_string(
        "{ c[$1]++; n++ } END { print c[\"a\"]; print c[\"b\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_assoc_print_key_0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_assoc_print_key1_0'))),
    !.

% A scalar statement reads its FINAL slot, established by the break-close phi.
test(ir_scalar_statement_reads_final_slot) :-
    plawk_parse_string("{ c[$1]++; n++ } END { print c[\"a\"]; print n }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%final_slot_0'))),
    !.

test(ir_exit_stores_code) :-
    plawk_parse_string("{ c[$1]++; n++ } END { print c[\"a\"]; exit 3 }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'store i32 3, i32* @plawk_exit_code'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'ret i32 %plawk_exit_ec'))),
    !.

:- end_tests(plawk_mixed_end_list).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_end_list', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected, ExpectedStatus) :-
    run_raw(Src, Out, Status),
    assertion(Out == Expected),
    assertion(Status == ExpectedStatus).

% Lines as a SORTED set, for the for-in regression whose order is hash-dependent.
run_sorted(Src, ExpectedSortedLines, ExpectedStatus) :-
    run_raw(Src, Out, Status),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, SortedLines),
    msort(ExpectedSortedLines, ExpectedSorted),
    assertion(SortedLines == ExpectedSorted),
    assertion(Status == ExpectedStatus).

run_raw(Src, Out, Status) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'mel_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'mel', Prog0),
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
    process_wait(Pid, exit(Status)).

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline, distinct from 2 = parse error and 4 = clang failure).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'mel_decline', Prog0),
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
