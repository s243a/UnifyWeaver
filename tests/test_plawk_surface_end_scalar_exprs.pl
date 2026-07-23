:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- dynamic user:plawk_endexpr_marker/0.

user:plawk_endexpr_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_end_scalar_exprs, [condition(clang_available)]).

test(parses_end_scalar_division_by_nr) :-
    plawk_parse_string("{ sum += $2 } END { print \"avg\", sum / NR }\n", Program),
    assertion(Program == program([], [rule(always, [add(var(sum), field(2))])],
        [end([print([string("avg"), div_i64(var(sum), special('NR'))])])])).

test(parses_scalar_read_in_update) :-
    plawk_parse_string("{ avg = $2 / 2; total += avg } END { print total }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(avg), div_i64(field(2), int(2))),
         add(var(total), var(avg))])],
        [end([print([var(total)])])])).

test(parses_scalar_self_read) :-
    plawk_parse_string("{ x = x + 2 } END { print x }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(x), add_i64(var(x), int(2)))])],
        [end([print([var(x)])])])).

test(parses_bare_nr_in_end_print) :-
    plawk_parse_string("{ n++ } END { print \"records\", NR }\n", Program),
    assertion(Program == program([], [rule(always, [inc(var(n))])],
        [end([print([string("records"), special('NR')])])])).

test(end_expr_ir_uses_final_slots_and_loop_nr_phi) :-
    plawk_parse_string("{ sum += $2 } END { print sum / NR }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]'))),
    % `/` is floating-point: the final slot promotes via sitofp and divides with fdiv.
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_end_expr_0_lhs = sitofp i64 %final_slot_0 to double'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_end_expr_0 = fdiv double'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(scalar_read_ir_uses_current_slot_value) :-
    plawk_parse_string("{ avg = $2 / 2; total += avg } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    % total's add reads avg's freshly assigned op value (%rule_0_body_
    % slot_0_op_0), not avg's stale loop-phi input. avg = $2/2 is float division,
    % so avg (and, by propagation, total) is a double slot: the add is fadd double.
    assertion(once(sub_atom(DriverIR, _, _, _, '= fadd double %slot_1, %rule_0_body_slot_0_op_0'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_end_average_report) :-
    run_endexpr_print_smoke("{ sum += $2 } END { print \"avg\", sum / NR }\n",
        "a 10\nb 20\nc 30\n",
        "avg 20\n").

test(surface_scalar_read_accumulates_intermediate) :-
    run_endexpr_print_smoke("{ avg = $2 / 2; total += avg } END { print total }\n",
        "a 10\nb 20\n",
        "15\n").

test(surface_scalar_self_read_updates) :-
    run_endexpr_print_smoke("{ x = x + 2 } END { print x }\n",
        "a\nb\nc\n",
        "6\n").

test(surface_bare_nr_in_end) :-
    run_endexpr_print_smoke("{ n++ } END { print \"records\", NR }\n",
        "a\nb\n",
        "records 2\n").

test(surface_cross_rule_scalar_read) :-
    run_endexpr_print_smoke("$1 == \"A\" { base = $2 } $1 == \"B\" { total += base } END { print total }\n",
        "A 5\nB x\nA 7\nB y\nB z\n",
        "19\n").

test(surface_mixed_end_expr_with_assoc_lookup) :-
    run_endexpr_print_smoke("{ total++; counts[$1]++ } END { print total * 2, counts[\"ERROR\"] }\n",
        "ERROR a\nWARN b\nERROR c\n",
        "6 2\n").

% `/` is floating-point, so an empty-input average is IEEE 0.0/0.0 = nan (printed
% "-nan"), not the old guarded integer 0. gawk fatals on division by zero here;
% plawk keeps its lenient float policy (x/0.0 = inf, 0.0/0.0 = nan) rather than
% aborting, consistent with its other f64 division.
test(surface_empty_input_average_is_nan) :-
    run_endexpr_print_smoke("{ sum += $2 } END { print sum / NR }\n",
        "",
        "-nan\n").

test(surface_assignments_apply_in_source_order) :-
    run_endexpr_print_smoke("{ a = NR; b = a + 1; a = 100 } END { print a, b }\n",
        "one row\n",
        "100 2\n").

test(surface_read_before_any_write_is_zero) :-
    run_endexpr_print_smoke("{ first = seen + 1; seen = 5 } END { print first, seen }\n",
        "only row\n",
        "1 5\n").

test(surface_end_expr_precedence_and_parens) :-
    run_endexpr_print_smoke("{ hits++; total += 3 } END { print (hits + total) * 2, hits + total * 2 }\n",
        "a\nb\n",
        "16 14\n").

run_endexpr_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_end_scalar_exprs', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    directory_file_path(Dir, 'plawk_surface_end_scalar_exprs.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_endexpr_marker/0 ],
        [module_name('plawk_surface_end_scalar_exprs')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_end_scalar_exprs_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface end scalar exprs output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_end_scalar_exprs_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_end_scalar_exprs).
