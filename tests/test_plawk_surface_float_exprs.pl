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

:- dynamic user:plawk_float_marker/0.

user:plawk_float_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_float_exprs, [condition(clang_available)]).

test(parses_float_literal_as_exact_ratio) :-
    plawk_parse_string("{ print $2 * 1.5 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([mul_i64(field(2), float_const(15, 10))])])], [])).

test(parses_float_field_coercion) :-
    plawk_parse_string("{ print float($2) }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([float_field(2)])])], [])).

test(parses_float_in_printf_arg) :-
    plawk_parse_string("{ printf \"%.2f;\", $2 / 3.0 }\n", Program),
    assertion(Program == program([], [rule(always,
        [printf(string("%.2f;"),
            [div_i64(field(2), float_const(30, 10))])])], [])).

test(integer_division_stays_integer) :-
    plawk_parse_string("{ print $2 / 2.0, $2 / 2 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([div_i64(field(2), float_const(20, 10)),
                div_i64(field(2), int(2))])])], [])).

test(float_expr_ir_uses_fdiv_ratio_and_promotion) :-
    plawk_parse_string("{ print ($2 + 1) * 0.5 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '= fdiv double 5.0, 10.0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= sitofp i64 '))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= fmul double '))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(float_field_ir_uses_strtod_helper) :-
    plawk_parse_string("{ print float($2) }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_f64_value(%Value %line, i64 2, i8 32)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_float_multiply_prints_g_format) :-
    run_float_print_smoke("{ print $2 * 1.5 }\n",
        "a 10\nb 3\n",
        "15\n4.5\n").

% `/` is ALWAYS floating-point in awk, so `$2 / 2` and `$2 / 2.0` are identical
% (both 3.5) -- there is no integer-division form of `/`.
test(surface_division_is_always_float) :-
    run_float_print_smoke("{ print $2 / 2.0, $2 / 2 }\n",
        "a 7\n",
        "3.5 3.5\n").

test(surface_float_field_uses_strtod_semantics) :-
    run_float_print_smoke("{ print float($2) }\n",
        "a 3.14\nb 2.5rest\nc abc\n",
        "3.14\n2.5\n0\n").

test(surface_printf_fixed_precision) :-
    run_float_print_smoke("{ printf \"%.2f;\", $2 / 3.0 }\n",
        "a 10\nb 1\n",
        "3.33;0.33;").

test(surface_parenthesized_int_subtree_promotes) :-
    run_float_print_smoke("{ print ($2 + 1) * 0.5 }\n",
        "a 9\n",
        "5\n").

test(surface_nr_promotes_to_double) :-
    run_float_print_smoke("{ print NR * 0.5 }\n",
        "a\nb\n",
        "0.5\n1\n").

test(surface_decimal_literals_round_correctly) :-
    run_float_print_smoke("{ print 0.1 + 0.2 }\n",
        "x\n",
        "0.3\n").

test(surface_integer_valued_double_prints_without_point) :-
    run_float_print_smoke("{ print 2.0 * $2 }\n",
        "a 10\n",
        "20\n").

test(surface_printf_g_and_e_formats) :-
    run_float_print_smoke("{ printf \"%g|%e\\n\", float($2) * 2.0, float($2) }\n",
        "a 1.25\n",
        "2.5|1.250000e+00\n").

test(surface_float_field_plus_int_field) :-
    run_float_print_smoke("{ print float($2) + $3 }\n",
        "a 0.5 2\n",
        "2.5\n").

test(surface_float_composes_with_guards) :-
    run_float_print_smoke("$1 == \"ERROR\" { print $3 * 1.5 }\n",
        "ERROR disk 10\nWARN cpu 4\nERROR net 3\n",
        "15\n4.5\n").

% A double-valued scalar prints with %g in a rule body (previously only END
% print handled a double scalar; a rule-body `print x` was rejected).
test(surface_rule_body_double_scalar_print) :-
    run_float_print_smoke("{ x = 1.5; print x }\n",
        "a\nb\n",
        "1.5\n1.5\n").

% ... in a comma list beside a field.
test(surface_rule_body_double_scalar_comma) :-
    run_float_print_smoke("{ x = 1.5; print x, $1 }\n",
        "7\n",
        "1.5 7\n").

% ... and glued by juxtaposition-concat.
test(surface_rule_body_double_scalar_concat) :-
    run_float_print_smoke("{ x = 1.5; print x $1 }\n",
        "7\n",
        "1.57\n").

% ... and after a string-literal prefix.
test(surface_rule_body_double_scalar_prefix) :-
    run_float_print_smoke("{ x = 2.5; print \"v=\" x }\n",
        "a\n",
        "v=2.5\n").

% An integer-valued double still prints without a decimal point (%g).
test(surface_rule_body_double_scalar_integral) :-
    run_float_print_smoke("{ x = 2.0 * $1; print x }\n",
        "5\n",
        "10\n").

run_float_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_float_exprs', Dir),
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
    directory_file_path(Dir, 'plawk_surface_float_exprs.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_float_marker/0 ],
        [module_name('plawk_surface_float_exprs')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_float_exprs_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface float exprs output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_float_exprs_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_float_exprs).
