:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: NR in an END-`if` CONDITION must emit %plawk_nr.
%
% `{ n++ } END { if (NR == 3) print "yes"; else print "no" }` previously reached
% clang and failed with exit status 4: `use of undefined value '%plawk_nr'`.
% The END-if condition operand already resolved NR to %plawk_nr
% (plawk_while_cond_operand/8), but counter discovery only walked print/update
% fields -- not the condition -- so the phi was never emitted.
%
% Fix: plawk_end_if_print_fields/4 contributes special('NR') when
% plawk_cond_expr_uses_nr/1 sees NR (the same walker ternary/combinators use).
%
% Explicit OUT-OF-SCOPE declines (exit 3), pinned with their owning boundary:
%   * reversed `if (3 == NR)` -- not in END-if condition vocabulary
%   * END-only (no rule) -- separate driver gap
%   * assoc rules + scalar END-if -- no driver admits that condition/route pairing
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

% Three records: "5 boot" / "5 trace" / "7 disk" -- NR == 3 after the loop.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_end_if_nr).

% --- must compile and match gawk -----------------------------------------

test(end_if_nr_eq_else, [condition(clang_available)]) :-
    run("{ n++ } END { if (NR == 3) print \"yes\"; else print \"no\" }\n",
        "yes\n"),
    !.

test(end_if_nr_eq_no_else, [condition(clang_available)]) :-
    run("{ n++ } END { if (NR == 3) print \"yes\" }\n", "yes\n"),
    !.

test(end_if_nr_and_scalar, [condition(clang_available)]) :-
    run("{ n++ } END { if (NR > 1 && n == 3) print \"yes\" }\n", "yes\n"),
    !.

test(end_if_nr_or_scalar, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3 || NR == 0) print \"yes\" }\n", "yes\n"),
    !.

% Empty input: NR == 0 is true; the counter phi must still exist.
test(end_if_nr_eq_zero_empty_input, [condition(clang_available)]) :-
    run_input("{ n++ } END { if (NR == 0) print \"empty\" }\n", "", "empty\n"),
    !.

% Coexistence: NR in the condition + retained-record field/NF in branches.
test(end_if_nr_cond_with_field_and_nf_branches, [condition(clang_available)]) :-
    run("{ n++ } END { if (NR == 3) print $1; else print NF }\n", "7\n"),
    !.

% Non-NR twin of the coexistence case (already worked; regression pin).
test(end_if_scalar_cond_with_field_and_nf_branches, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print $1; else print NF }\n", "7\n"),
    !.

% --- IR contract ---------------------------------------------------------

test(end_if_nr_condition_defines_plawk_nr) :-
    plawk_parse_string(
        "{ n++ } END { if (NR == 3) print \"yes\"; else print \"no\" }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '%plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%current_nr = add i64 %plawk_nr, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'icmp eq i64 %plawk_nr, 3'))),
    !.

test(end_if_and_or_nr_condition_defines_plawk_nr) :-
    plawk_parse_string(
        "{ n++ } END { if (NR > 1 && n == 3) print \"yes\" }\n", ProgramAnd),
    plawk_program_native_driver_ir(ProgramAnd, 'input.txt', IRAnd),
    assertion(once(sub_atom(IRAnd, _, _, _, '%plawk_nr = phi i64'))),
    plawk_parse_string(
        "{ n++ } END { if (n == 3 || NR == 0) print \"yes\" }\n", ProgramOr),
    plawk_program_native_driver_ir(ProgramOr, 'input.txt', IROr),
    assertion(once(sub_atom(IROr, _, _, _, '%plawk_nr = phi i64'))),
    !.

% END condition WITHOUT NR must not acquire a record counter.
test(end_if_without_nr_emits_no_plawk_nr) :-
    plawk_parse_string(
        "{ n++ } END { if (n == 3) print \"yes\"; else print \"no\" }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%plawk_nr')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%current_nr')),
    !.

% Record-independent END condition: no retained-record globals (pay-per-use).
test(end_if_nr_condition_emits_no_retained_record_globals) :-
    plawk_parse_string(
        "{ n++ } END { if (NR == 3) print \"yes\"; else print \"no\" }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_lastrec_buf')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_lastrec_store')),
    !.

% --- out-of-scope declines (exit 3), owning boundary pinned ---------------

% Reversed operand order is not in the END-if condition vocabulary.
test(reversed_nr_end_if_condition_declines) :-
    build_status("{ n++ } END { if (3 == NR) print \"yes\" }\n", 3),
    !.

% END-only programs (no rule) decline entirely -- separate driver gap.
test(end_only_nr_if_declines) :-
    build_status("END { if (NR == 3) print \"yes\" }\n", 3),
    !.

% Assoc rules plus this scalar END-if select no driver: the assoc END-if route admits
% membership conditions, while the scalar END-if route does not admit the assoc rules.
test(assoc_end_if_with_nr_condition_declines) :-
    build_status(
        "{ c[$1]++ } END { if (NR == 3) print \"yes\"; else print \"no\" }\n",
        3),
    !.

:- end_tests(plawk_end_if_nr).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_if_nr', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_input_status(Src, Input, Expected, 0).

run_input(Src, Input, Expected) :-
    run_input_status(Src, Input, Expected, 0).

run_input_status(Src, Input, Expected, ExpectedRC) :-
    odir(Dir),
    directory_file_path(Dir, 'ein_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ein', Prog0),
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
    gawk_check(Src, Input, Expected, ExpectedRC),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q rc=~w~n  expected ~q rc=~w~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).

gawk_check(Src, Input, Expected, ExpectedRC) :-
    odir(Dir),
    directory_file_path(Dir, 'ein_gawk.plawk', GawkProg),
    setup_call_cleanup(open(GawkProg, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    process_create(path(gawk), ['-f', GawkProg],
        [stdin(pipe(In)), stdout(pipe(OS)), stderr(null), process(Pid)]),
    format(In, '~w', [Input]),
    close(In),
    read_string(OS, _, Out),
    close(OS),
    process_wait(Pid, exit(RC)),
    assertion(Out == Expected),
    assertion(RC == ExpectedRC).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ein_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
