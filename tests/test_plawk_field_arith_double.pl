:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Field arithmetic is DOUBLE, as in awk -- phase C of the numeric-fields fix.
%
% ---------------------------------------------------------------------------
% awk's numbers are doubles, and a field's numeric value is its strtod reading.
% plawk typed field arithmetic as i64 and read the field with a strict integer
% parse defaulting to 0, so any non-integer data silently vanished:
%
%     { s += $2 } END { print s }   on 30.25, 5, 3abc    gawk 38.25   plawk 5
%     { print $2 + 1 }                                   gawk 31.25   plawk 1
%     $2 + 0 > 10                                        gawk true    plawk false
%
% The typing rule (one place): a field read as an OPERAND of arithmetic makes the
% tree double (plawk_arith_field_operand/1 in plawk_expr_is_double/1), and
% `s += $N` accumulates float_field(N). The existing double fixpoint carries it to
% slots; doubles read fields via strtod (phase A) and print the awk way, integral
% values as integers (the awk-number-print fix), so INTEGER DATA PRINTS AS BEFORE.
% Arithmetic comparison patterns (`$I ARITH K CMP V`, `$I ARITH $J CMP V`) and
% ternary comparisons of double trees compare with fcmp. A double under printf's
% `%d` is truncated (@wam_awk_f64_to_i64), so `printf "%d", s` keeps working.
%
% NOT double: `print $2` and `x = $2` (text / strnum copies), counters (`n++`),
% NR, NF, length, int(). Assoc tables stay i64 (`c[$1] += $2`) -- a follow-on.
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

input("alice 30.25 NY 7\nbob 5 LA -2.9\ncarol 3abc SF x\n").

:- begin_tests(plawk_field_arith_double).

test(accumulator, [condition(clang_available)]) :-
    run("{ s += $2 } END { print s }\n", "38.25\n"),
    !,
    run("{ n++; s += $2 } END { print n, s, s / n }\n", "3 38.25 12.75\n"),
    !,
    run("{ s += $2 * 2 } END { print s }\n", "76.5\n"),
    !.

test(expressions, [condition(clang_available)]) :-
    run("{ print $2 + 1 }\n", "31.25\n6\n4\n"),
    !,
    run("{ print $2 * 2 }\n", "60.5\n10\n6\n"),
    !,
    run("{ print $2 % 3 }\n", "0.25\n2\n0\n"),
    !,
    run("{ print ($2 + 1) * 1.5 }\n", "46.875\n9\n6\n"),
    !,
    run("{ x = $2 + 0; print x }\n", "30.25\n5\n3\n"),
    !.

% Integer data prints exactly as it did: integral doubles print as integers.
test(integer_data_unchanged, [condition(clang_available)]) :-
    run_with("a 1000000\nb 2500000\nc 7\n", "{ s += $2 } END { print s }\n",
        "3500007\n"),
    !,
    run_with("a 1000000\nb 2500000\nc 7\n", "{ print $2 + 1 }\n",
        "1000001\n2500001\n8\n"),
    !.

% A double under %d is truncated; %f takes it as is.
test(printf_of_double_accumulators, [condition(clang_available)]) :-
    run("{ s += $2 } END { printf \"%d %.2f\\n\", s, s }\n", "38 38.25\n"),
    !,
    run_with("a 1000000\nb 7\n", "{ printf \"%5d|\\n\", $2 * 2 }\n",
        "2000000|\n   14|\n"),
    !.

test(arithmetic_comparison_patterns, [condition(clang_available)]) :-
    run_with("4 30.25\n7 5\n9 3abc\n2 x\n", "$2 + 0 > 10\n", "4 30.25\n"),
    !,
    run_with("4 30.25\n7 5\n9 3abc\n2 x\n", "$1 % 2 == 0\n", "4 30.25\n2 x\n"),
    !,
    run_with("4 30.25\n7 5\n9 3abc\n2 x\n", "$1 + $2 > 10\n",
        "4 30.25\n7 5\n9 3abc\n"),
    !,
    run_with("4 30.25\n7 5\n9 3abc\n2 x\n", "$2 - 1 >= 4\n", "4 30.25\n7 5\n"),
    !.

test(arithmetic_comparison_in_if_and_ternary, [condition(clang_available)]) :-
    run_with("4 30.25\n7 5\n9 3abc\n2 x\n", "{ if ($2 * 2 > 10) print $1 }\n", "4\n"),
    !,
    run_with("4 30.25\n7 5\n9 3abc\n2 x\n", "{ print ($2 + 1 > 10) ? 1 : 0 }\n",
        "1\n0\n0\n0\n"),
    !.

% Copies stay text; counters stay integers.
test(copies_and_counters_are_not_double, [condition(clang_available)]) :-
    run("{ x = $2; print x }\n", "30.25\n5\n3abc\n"),
    !,
    ir_of("{ n++ } END { print n }\n", IR),
    \+ sub_atom(IR, _, _, _, 'phi double'),
    !.

test(ir_accumulator_is_a_double_slot_read_by_strtod) :-
    ir_of("{ s += $2 } END { print s }\n", IR),
    sub_atom(IR, _, _, _, 'phi double'),
    sub_atom(IR, _, _, _, '@wam_atom_field_f64_value(%Value %line, i64 2'),
    !.

:- end_tests(plawk_field_arith_double).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_field_arith_double', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'fad_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'fad', Prog0),
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
