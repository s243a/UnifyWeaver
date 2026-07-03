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

:- dynamic user:plawk_f64slot_marker/0.

user:plawk_f64slot_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_float_slots, [condition(clang_available)]).

test(parses_bare_float_delta) :-
    plawk_parse_string("{ sum += 0.5 } END { print sum }\n", Program),
    assertion(Program == program([],
        [rule(always, [add(var(sum), float_const(5, 10))])],
        [end([print([var(sum)])])])).

test(parses_float_field_delta) :-
    plawk_parse_string("{ sum += float($2) } END { print sum }\n", Program),
    assertion(Program == program([],
        [rule(always, [add(var(sum), float_field(2))])],
        [end([print([var(sum)])])])).

test(double_slot_ir_uses_double_phis_and_fadd) :-
    plawk_parse_string("{ sum += $2 * 1.5 } END { print sum }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '%slot_0 = phi double [0.0, %check_handle_value], [%next_slot_0, %continue_loop]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi double '))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%final_slot_0 = phi double '))),
    assertion(once(sub_atom(DriverIR, _, _, _, ' = fadd double %slot_0, '))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@printf(i8* %end_f64_fmt_0, double %final_slot_0)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(i64_slots_stay_i64) :-
    plawk_parse_string("{ n++ ; total += $2 } END { print n, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'phi double')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'fadd')),
    !.

test(fixpoint_promotes_transitive_reads) :-
    % b reads a; a is double, so b must become double too.
    plawk_parse_string("{ a = 1.5 ; b = a + 1 } END { print b }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@printf(i8* %end_f64_fmt_0, double %final_slot_'))),
    !.

test(end_arith_on_double_slot_promotes_to_f64) :-
    % END arithmetic reading a double slot promotes the whole expression
    % to double and prints %g (was rejected before the f64 END slice).
    plawk_parse_string("{ sum += 0.5 } END { print sum + 1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, ' = fadd double '))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'printed_end_expr_f64_0'))),
    !.

test(surface_end_f64_average) :-
    % The classic average: double slot divided by NR, IEEE fdiv, %g print.
    run_f64_smoke("{ sum += float($2) ; n++ } END { print sum / NR, n * 2, sum + 0.5 }\n",
        "a 2.5\nb 0.5\nc 1.5\nd 3.5\n",
        "2 8 8.5\n").

test(surface_end_float_literal_expr) :-
    run_f64_smoke("{ n++ } END { print n * 1.5 }\n",
        "a\nb\nc\n",
        "4.5\n").

test(surface_double_accumulator_text) :-
    run_f64_smoke("{ sum += float($2) * 1.5 ; n++ } END { print n, sum }\n",
        "a 2.5\nb 0.5\nc 1.0\n",
        "3 6\n").

test(surface_double_if_else_and_next) :-
    run_f64_smoke("$1 == \"skip\" { next } { if ($2 > 2) { big += 0.5 } else { small += 0.25 } } END { print big, small }\n",
        "x 5\nskip 9\nx 1\nx 3\n",
        "1 0.25\n").

test(surface_double_break) :-
    run_f64_smoke("$1 == \"stop\" { break } { sum += 0.5 } END { print sum }\n",
        "a 1\nb 2\nstop 0\nc 3\n",
        "1\n").

test(surface_double_set_overwrites) :-
    run_f64_smoke("{ last = float($2) } END { print last }\n",
        "a 2.5\nb 0.125\n",
        "0.125\n").

test(surface_binary_double_accumulator) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_float_slots', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } $1 > 10 { sum += float($2) } END { print sum }\n", Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_stream_read_record'))),
    emit_probe(Dir, DriverIR, BinPath),
    write_i64_f64_records(InputPath, [5-bits(0x4023800000000000),      % 9.75
                                      20-bits(0x4004000000000000),     % 2.5
                                      30-bits(0x3FD0000000000000)]),   % 0.25
    process_create(BinPath, [], [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == "2.75\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_i64_f64_records(Path, Pairs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(A-bits(Bits), Pairs),
            ( write_i64_le(Out, A), write_i64_le(Out, Bits) )),
        close(Out)).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_f64slot_marker/0 ],
        [module_name('plawk_float_slots')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, BuildOut),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk float slots build output]~n~w~n", [BuildOut]),
       throw(plawk_float_slots_build_failed(Status))
    ).

run_f64_smoke(Source, InputText, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_float_slots', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, Out, [encoding(utf8)]),
        write(Out, InputText),
        close(Out)),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath),
    process_create(BinPath, [], [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk float slots run output]~n~w~n", [OutStr]),
       throw(plawk_float_slots_run_failed(Status))
    ),
    !.

:- end_tests(plawk_float_slots).
