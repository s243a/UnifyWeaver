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

:- dynamic user:plawk_arith_marker/0.

user:plawk_arith_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_arith_exprs, [condition(clang_available)]).

test(parses_field_plus_field) :-
    plawk_parse_string("{ print $2 + $3 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([add_i64(field(2), field(3))])])], [])).

test(parses_multiplicative_precedence) :-
    plawk_parse_string("{ print $2 + $3 * $4 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([add_i64(field(2), mul_i64(field(3), field(4)))])])], [])).

test(parses_parenthesized_grouping) :-
    plawk_parse_string("{ print ($2 + $3) * $4 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([mul_i64(add_i64(field(2), field(3)), field(4))])])], [])).

test(parses_left_associative_subtraction) :-
    plawk_parse_string("{ print $2 - $3 - $4 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([sub_i64(sub_i64(field(2), field(3)), field(4))])])], [])).

test(parses_div_and_mod) :-
    plawk_parse_string("{ print $2 / $3, $2 % $3 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([div_i64(field(2), field(3)),
                mod_i64(field(2), field(3))])])], [])).

test(parses_mixed_primary_arithmetic) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR * 10 + NF, length($2) * 2 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([add_i64(mul_i64(special('NR'), int(10)), special('NF')),
                mul_i64(length(field(2)), int(2))])])], [])).

test(parses_scalar_binary_update) :-
    plawk_parse_string("{ sum += $2 * $3; last = $2 + $3 } END { print sum, last }\n", Program),
    assertion(Program == program([], [rule(always,
        [add(var(sum), mul_i64(field(2), field(3))),
         set(var(last), add_i64(field(2), field(3)))])],
        [end([print([var(sum), var(last)])])])).

test(parses_printf_binary_arg) :-
    plawk_parse_string("{ printf \"%d\\n\", $2 + $3 * 2 }\n", Program),
    assertion(Program == program([], [rule(always,
        [printf(string("%d\n"),
            [add_i64(field(2), mul_i64(field(3), int(2)))])])], [])).

test(bare_fields_still_parse_as_slices) :-
    plawk_parse_string("{ print $2, $3 }\n", Program),
    assertion(Program == program([], [rule(always,
        [print([field(2), field(3)])])], [])).

test(legacy_primary_minus_constant_ast_unchanged) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR - 1, length($0) - 3 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([sub_i64(special('NR'), int(1)),
                sub_i64(length(field(0)), int(3))])])], [])).

% `/` is floating-point (fdiv, no integer guard); `%` stays integer and keeps its
% zero-divisor / INT64_MIN-overflow guard (srem).
test(arith_ir_division_is_float_modulo_stays_guarded_integer) :-
    plawk_parse_string("{ print $2 / $3, $2 % $3 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'fdiv double'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'sdiv i64')),
    assertion(once(sub_atom(DriverIR, _, _, _, '_den_zero = icmp eq i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_lhs_min = icmp eq i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_safe_den = select i1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_raw = srem i64'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(arith_ir_nested_operands_get_unique_names) :-
    plawk_parse_string("{ print index($2, \"a\") + index($3, \"b\") }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_lhs = call i64 @wam_atom_field_index_value(%Value %line, i64 2'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_rhs = call i64 @wam_atom_field_index_value(%Value %line, i64 3'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_prints_basic_binary_ops) :-
    run_arith_print_smoke("{ print $2 + $3, $2 - $3, $2 * $3 }\n",
        "x 7 2\ny 10 5\n",
        "9 5 14\n15 5 50\n").

test(surface_precedence_and_parens) :-
    run_arith_print_smoke("{ print ($2 + $3) * $4, $2 + $3 * $4 }\n",
        "x 2 3 4\n",
        "20 14\n").

% `/` is floating-point (7/2 = 3.5); `%` stays integer.
test(surface_division_and_modulo) :-
    run_arith_print_smoke("{ print $2 / $3, $2 % $3 }\n",
        "a 7 2\nb -7 2\n",
        "3.5 1\n-3.5 -1\n").

% float `/` by zero is IEEE inf (gawk fatals; plawk stays lenient); `%` by zero
% keeps its guard and yields 0.
test(surface_float_division_by_zero_is_inf) :-
    run_arith_print_smoke("{ print $2 / $3, $2 % $3 }\n",
        "a 7 0\n",
        "inf 0\n").

% INT64_MIN / -1 has no integer-overflow trap under float `/` -- it is just the
% double 2^63 (printed by %g); `%` keeps the integer overflow guard (-> 0).
test(surface_int64_min_division_is_float) :-
    run_arith_print_smoke("{ print $2 / $3, $2 % $3 }\n",
        "a -9223372036854775808 -1\n",
        "9.22337e+18 0\n").

test(surface_nonnumeric_fields_coerce_to_zero) :-
    run_arith_print_smoke("{ print $2 + $3 }\n",
        "a nope 5\nb 3 4\n",
        "5\n7\n").

test(surface_scalar_accumulates_binary_expr) :-
    run_arith_print_smoke("{ sum += $2 * $3 } END { print sum }\n",
        "a 2 3\nb 4 5\n",
        "26\n").

test(surface_guarded_rule_prints_nr_nf_arithmetic) :-
    run_arith_print_smoke("$1 == \"ERROR\" { print NR * 10 + NF }\n",
        "INFO a\nERROR b c\n",
        "23\n").

test(surface_printf_binary_arg) :-
    run_arith_print_smoke("{ printf \"%d;\", $2 + $3 * 2 }\n",
        "x 1 2\ny 3 4\n",
        "5;11;").

% `100 / NF` is float (100/3 = 33.3333); `NF % 2` stays integer.
test(surface_constant_operands_with_nf) :-
    run_arith_print_smoke("{ print 100 / NF, NF % 2 }\n",
        "a b c d\nx y z\n",
        "25 0\n33.3333 1\n").

run_arith_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_arith_exprs', Dir),
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
    directory_file_path(Dir, 'plawk_surface_arith_exprs.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_arith_marker/0 ],
        [module_name('plawk_surface_arith_exprs')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_arith_exprs_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface arith exprs output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_arith_exprs_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_arith_exprs).
