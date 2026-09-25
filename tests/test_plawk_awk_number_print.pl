:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% How a double is rendered as text: awk's OFMT/CONVFMT rule, not C's `%g`.
%
% ---------------------------------------------------------------------------
% awk prints an INTEGRAL number as an integer and every other number through
% "%.6g". plawk printed every double with raw "%g", so an integral value of a million
% or more came out in exponent form:
%
%     { print $2 * 1.0 }   on 1250000     gawk: 1250000     plawk: 1.25e+06
%
% Silent wrong output, and a blocker for typing field arithmetic as double (phase C
% of the numeric-fields fix), which would turn every large integer sum into this.
%
% One runtime rule, @wam_awk_num_fmt: integral -> "%.0f" (full digits, as gawk prints
% 1e30), else "%.6g". Integral: inside +-2^63 the value survives an i64 round trip;
% beyond, every finite double is. It serves `print` (@wam_print_awk_number, which
% keeps the unset-aware render: an empty format prints nothing) and the CONVFMT text
% a strnum-vs-number STRING comparison is made against (@wam_strnum_cmp_double).
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_awk_number_print).

test(integral_double_prints_as_integer, [condition(clang_available)]) :-
    run("a 1250000\nb 1234567\n", "{ print $2 * 1.0 }\n", "1250000\n1234567\n"),
    !,
    run("a 1250000\n", "{ x = $2 * 3.0; print x }\n", "3750000\n"),
    !.

test(non_integral_uses_ofmt, [condition(clang_available)]) :-
    run("a 1250000\nb 1\n", "{ print $2 / 3 }\n", "416667\n0.333333\n"),
    !,
    run("a 2500001\n", "{ print $2 / 2 }\n", "1.25e+06\n"),
    !.

test(double_scalar_in_end, [condition(clang_available)]) :-
    run("a 1000000\nb 2000000\n", "{ s += $2 * 1.0 } END { print s }\n", "3000000\n"),
    !,
    run("a 2\nb 3\n", "{ s += $2 * 0.25 } END { print s }\n", "1.25\n"),
    !.

test(literal_arithmetic, [condition(clang_available)]) :-
    run("x\n", "{ print 1234567.0 * 1 }\n", "1234567\n"),
    !.

% The IR goes through the awk renderer, not a raw %g printf.
test(print_goes_through_the_awk_renderer) :-
    ir_of("{ print $2 * 1.0 }\n", IR),
    sub_atom(IR, _, _, _, 'call i32 @wam_print_awk_number(i8*'),
    !.

:- end_tests(plawk_awk_number_print).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_awk_number_print', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'anp_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'anp', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(BO)), stderr(null), process(BPid)]),
    read_string(BO, _, _), close(BO),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

ir_of(Src, IR) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR).
