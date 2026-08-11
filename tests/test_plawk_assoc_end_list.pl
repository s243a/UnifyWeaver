:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: a LOOP-FREE assoc END statement list -- `END { print c["a"]; print "done" }`.
%
% The pure-assoc END driver took a SINGLE `print`, so any second statement
% declined. This is the sibling of the mixed for-in chain
% (tests/test_plawk_end_chain.pl) and completes that work: the END statement list
% is now uniform across all three chains -- scalar, for-in, and assoc.
%
% Unlike the for-in chain there is no loop, so the whole END is STRAIGHT-LINE: no
% basic blocks, no predecessor labels, no per-statement tail. Each statement's IR
% is emitted in order, renamed by statement index, and the tables are freed once
% at the end. It reuses the chain's per-statement rename, which covers BOTH global
% families that need uniquifying: string literals (`end_`-prefixed) and literal
% assoc keys (`plawk_assoc_print_key_N`, which the string rename does not reach --
% so two literal-key lookups would otherwise share one key global).
%
% `printf` IS supported here, but only with LITERAL arguments. The pure-assoc
% chain has no scalar state plan, so a `var` argument has no slot to read; and
% `NR` would emit a reference to a counter this driver does not define, which
% would be invalid IR rather than a decline. The gate restricts arguments to
% literals and constant arithmetic, keeping the useful case (a fixed trailer line)
% without either hazard -- the declines below pin that boundary.
%
% gawk 5.2 is the oracle for every expectation here, exit STATUS included.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3", so c["a"] and c["b"] are both 1. Every
% program here reads the table by LITERAL key, so output order is deterministic
% and can be compared byte-for-byte (no for-in, hence no hash-order dependence).
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_assoc_end_list).

% --- several statements ---------------------------------------------------

test(assoc_read_then_plain_print, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; print \"done\" }\n", "1\ndone\n", 0),
    !.

test(plain_print_then_assoc_read, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print \"start\"; print c[\"a\"] }\n", "start\n1\n", 0),
    !.

% TWO literal-key lookups in separate statements: each must resolve its own key
% global. Sharing `..._key_0` was the failure the per-statement rename prevents.
test(two_assoc_reads, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; print c[\"b\"] }\n", "1\n1\n", 0),
    !.

test(three_statements, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print \"h\"; print c[\"a\"]; print c[\"b\"] }\n",
        "h\n1\n1\n", 0),
    !.

% An absent key is an uninitialised element: empty in string context.
test(absent_key_then_print, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"zz\"]; print \"after\" }\n", "\nafter\n", 0),
    !.

% --- printf with literal arguments ---------------------------------------

test(assoc_read_then_printf, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; printf \"x\\n\" }\n", "1\nx\n", 0),
    !.

test(printf_then_assoc_read, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { printf \"n=%d\\n\", 7; print c[\"a\"] }\n", "n=7\n1\n", 0),
    !.

% Constant arithmetic folds, as in every other printf context.
test(printf_constant_arithmetic, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; printf \"%d\\n\", 2 * 3 }\n", "1\n6\n", 0),
    !.

% printf appends no ORS, so a following print closes the line.
test(printf_appends_no_newline, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { printf \"v=\"; print c[\"a\"] }\n", "v=1\n", 0),
    !.

% --- exit ----------------------------------------------------------------

test(assoc_read_then_exit, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; exit 3 }\n", "1\n", 3),
    !.

test(exit_only_after_assoc_program, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { exit 4 }\n", "", 4),
    !.

test(bare_exit_is_zero, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; exit }\n", "1\n", 0),
    !.

% Statements after the exit are dead, as in every other chain.
test(exit_truncates_following_statements, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; exit 1; print \"never\" }\n", "1\n", 1),
    !.

test(exit_first_suppresses_all_output, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { exit 5; print c[\"a\"] }\n", "", 5),
    !.

% An END exit still overrides a rule-level one (its store happens later).
test(end_exit_after_print_and_printf, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; printf \"z\\n\"; exit 6 }\n", "1\nz\n", 6),
    !.

% --- regressions ---------------------------------------------------------

% A lone plain print stays with the single-print driver (byte-identical IR).
test(single_assoc_print_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"] }\n", "1\n", 0),
    !.

% The for-in END driver is untouched.
test(forin_end_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k }\n", ["a", "b", "c"], 0),
    !.

% So is the mixed for-in chain from the previous PR.
test(mixed_forin_chain_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print \"done\" }\n",
        ["a", "b", "c", "done"], 0),
    !.

% --- clean declines: the printf argument boundary ------------------------

% A `var` argument has no slot in the pure-assoc chain.
test(printf_var_arg_declines) :-
    build_status("{ c[$1]++ } END { print c[\"a\"]; printf \"%d\\n\", n }\n", 3),
    !.

% `NR` would reference a counter this driver does not define -- it must DECLINE,
% not emit invalid IR (which would surface as a clang failure, status 4).
test(printf_nr_arg_declines) :-
    build_status("{ c[$1]++ } END { print c[\"a\"]; printf \"%d\\n\", NR }\n", 3),
    !.

% WAS a decline -- "END has no current record". It has a RETAINED one now: the
% statement-list driver threads the EndRecord token, and the printf arg gate admits
% `$N`/`NF`, which project from the retained last record rather than reading a slot.
% The `var` and `NR` pins above did NOT move -- those exclusions are about this
% driver's missing scalar plan and counter, not about the record, and the boundary
% is per-capability. `$1` of the last record ("c 3") is "c".
test(printf_field_arg_now_works, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; printf \"%s\\n\", $1 }\n",
        "1\nc\n", 0),
    !.

% WAS a decline -- "not wired to the record counter here either". It is now, and the reason it
% came for free is worth recording: NR needs no capability, only something to ask.
% plawk_end_nr_value/2 already falls back to `%plawk_nr`, the counter this driver emits when a
% print mentions NR, so the new clause passes `state_plan([], [])` -- the honest statement that
% this route has no scalar slots, not a stub.
%
% Note this works in the STATEMENT-LIST form, which still declines for a field read and NF.
% That is not an inconsistency: those two need the retained-record token that
% plawk_end_list_bodies/8 does not carry, and NR does not. The boundary is per-capability, not
% per-route.
test(nr_plain_print_now_works, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"a\"]; print NR }\n", "1\n3\n", 0),
    !.

% A MIXED scalar+assoc program with a statement list is the fourth chain, landed
% as the sibling to this one; behaviour is covered in
% tests/test_plawk_mixed_end_list.pl. Asserted here only as no-longer-declining,
% since this suite's driver deliberately does not claim it.
test(mixed_scalar_assoc_statement_list_compiles, [condition(clang_available)]) :-
    build_status("{ c[$1]++; n++ } END { print n; print c[\"a\"] }\n", 0),
    !.

% --- structure -----------------------------------------------------------

% The driver claims a statement list but NOT a lone plain print, so the
% single-print clause keeps its shape.
test(statement_list_excludes_a_lone_print) :-
    assertion(plawk_native_codegen:plawk_end_list_statements(assoc,
        [print([string("a")]), print([string("b")])], _)),
    assertion(plawk_native_codegen:plawk_end_list_statements(assoc,
        [print([string("a")]), exit(int(1))], _)),
    assertion(plawk_native_codegen:plawk_end_list_statements(assoc,
        [exit(int(1))], _)),
    assertion(\+ plawk_native_codegen:plawk_end_list_statements(assoc,
        [print([string("a")])], _)),
    % a for-in is not a statement of a loop-free list -- that is the chain's job
    assertion(\+ plawk_native_codegen:plawk_end_list_statements(assoc,
        [for_in(var(k), var(c), [print([var(k)])]), print([string("x")])], _)),
    !.

% The printf argument gate: literals and constant arithmetic in, everything that
% needs runtime state out.
test(printf_arg_gate) :-
    assertion(plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc,
        string("x"))),
    assertion(plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc, int(1))),
    assertion(plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc,
        add_i64(int(1), int(2)))),
    assertion(\+ plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc, var(n))),
    assertion(\+ plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc,
        special('NR'))),
    % WAS negative: a field argument is admitted now -- it reads the RETAINED last
    % record via the threaded EndRecord token, not runtime scalar state, so the
    % gate's dividing line ("everything that needs runtime state out") is unchanged
    % even though this row flipped sides of it.
    assertion(plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc,
        field(1))),
    assertion(plawk_native_codegen:plawk_end_list_printf_arg_ok(assoc,
        special('NF'))),
    % the MIXED chain admits a `var` (its slots exist) but still not NR
    assertion(plawk_native_codegen:plawk_end_list_printf_arg_ok(mixed, var(n))),
    assertion(\+ plawk_native_codegen:plawk_end_list_printf_arg_ok(mixed,
        special('NR'))),
    !.

% printf arguments reference no table, so they contribute no print fields to the
% table plan -- only the prints do.
test(statement_fields_come_from_prints_only) :-
    plawk_native_codegen:plawk_end_list_statement_fields(
        [print([assoc(var(c), string("a"))]),
         printf(string("%d\n"), [int(7)]),
         print([string("x")])],
        Fields),
    assertion(Fields == [assoc(var(c), string("a")), string("x")]),
    !.

% --- IR shape ------------------------------------------------------------

% The tables are freed exactly ONCE for the whole statement list. A per-statement
% free is what double-freed in the for-in chain, so the non-freeing print body is
% used here too.
test(ir_frees_each_table_once) :-
    plawk_parse_string("{ c[$1]++ } END { print c[\"a\"]; print c[\"b\"] }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    findall(B, sub_atom(DriverIR, B, _, _,
        '@wam_assoc_i64_free(%WamAssocI64Table* %plawk_assoc_table_0)'), Frees),
    assertion(Frees = [_]),
    !.

% Two literal-key lookups emit two distinct key globals, each defined and
% referenced consistently.
test(ir_two_key_globals_are_distinct) :-
    plawk_parse_string("{ c[$1]++ } END { print c[\"a\"]; print c[\"b\"] }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_assoc_print_key_0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_assoc_print_key1_0'))),
    !.

test(ir_exit_stores_code) :-
    plawk_parse_string("{ c[$1]++ } END { print c[\"a\"]; exit 3 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'store i32 3, i32* @plawk_exit_code'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'ret i32 %plawk_exit_ec'))),
    !.

:- end_tests(plawk_assoc_end_list).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_end_list', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build + run, requiring Expected output byte-for-byte and ExpectedStatus.
run(Src, Expected, ExpectedStatus) :-
    run_raw(Src, Out, Status),
    assertion(Out == Expected),
    assertion(Status == ExpectedStatus).

% As run/3 but comparing lines as a SORTED set, for the for-in regressions whose
% iteration order is hash-dependent.
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
    directory_file_path(Dir, 'ael_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ael', Prog0),
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
    directory_file_path(Dir, 'ael_decline', Prog0),
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
