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

:- dynamic user:plawk_binrec_marker/0.

user:plawk_binrec_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_binary_records, [condition(clang_available)]).

test(parses_binfmt_begin_assignment) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } $1 > 100 { sum += $2 } END { print sum }\n", Program),
    assertion(Program == program([begin([set(var('BINFMT'), string("i64 i64"))])],
        [rule(field_cmp(1, gt, 100), [add(var(sum), field(2))])],
        [end([print([var(sum)])])])).

test(binary_ir_loads_fields_without_parsing) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } $1 > 100 { sum += $2 } END { print sum }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_stream_read_record(%Value %handle, i64 16, i8* %rec)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_fp = getelementptr i8, i8* %rec, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_fp = getelementptr i8, i8* %rec, i64 8'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_cmp_value')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'read_line')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(binary_ir_f64_field_uses_double_load) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } { print $1, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'bitcast i8* %plawk_binfield_1_fp to double*'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@printf(i8* %binfield_fmt_1, double %plawk_binfield_1)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_f64_value')),
    !.

test(binary_mode_rejects_text_shaped_programs) :-
    Rejects = [
        "BEGIN { BINFMT = \"i64 i64\" } { print $0 }\n",
        "BEGIN { BINFMT = \"i64 i64\" } /^ERR/ { c++ } END { print c }\n",
        "BEGIN { BINFMT = \"i64 i64\" } $1 == \"x\" { c++ } END { print c }\n",
        "BEGIN { BINFMT = \"i64 f64\" } { sum += $2 } END { print sum }\n",
        "BEGIN { BINFMT = \"i64 i64\" } { c += $3 } END { print c }\n",
        "BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { print counts[\"x\"] }\n",
        "BEGIN { BINFMT = \"i64 i64\" } { print length($1) }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_binary_guard_and_sum) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 i64\" } $1 > 100 { hits++; sum += $2 } END { print hits, sum }\n",
        i64_pairs([50-1, 200-2, 300-3]),
        "2 5\n").

test(surface_binary_prints_i64_and_f64_fields) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 f64\" } { print $1, $2 }\n",
        i64_f64_pairs([1-2.5, 2-0.5]),
        "1 2.5\n2 0.5\n").

test(surface_binary_f64_arithmetic) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 f64\" } { print float($2) * 2.0 }\n",
        i64_f64_pairs([1-2.5, 2-0.5]),
        "5\n1\n").

test(surface_binary_negative_values_and_nf) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 i64\" } { print NR, NF, $1 * $2 + 1 }\n",
        i64_pairs([(-7)-4]),
        "1 2 -27\n").

test(surface_binary_combined_guard) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 i64\" } $1 > 100 && $2 < 3 { c++ } END { print c }\n",
        i64_pairs([50-1, 200-2, 300-3]),
        "1\n").

test(surface_binary_empty_input_runs_end) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 i64\" } { total += $2 } END { print total }\n",
        i64_pairs([]),
        "0\n").

test(surface_binary_printf) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 i64\" } { printf \"%d;%d\\n\", $1, $2 }\n",
        i64_pairs([50-1, 200-2]),
        "50;1\n200;2\n").

test(surface_binary_if_else_and_next) :-
    run_binary_smoke("BEGIN { BINFMT = \"i64 i64\" } $2 == 9 { skipped++; next } { if ($1 > 100) { big++ } else { small++ } } END { print big, small, skipped }\n",
        i64_pairs([50-1, 200-2, 5-9, 300-3]),
        "2 1 1\n").

test(surface_binary_trailing_partial_record_fails) :-
    build_binary_probe("BEGIN { BINFMT = \"i64 i64\" } { total += $2 } END { print total }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    setup_call_cleanup(
        open(InputPath, write, Out, [type(binary)]),
        ( write_i64_le(Out, 1), write_i64_le(Out, 2),
          write_i64_le(Out, 3) ),   % half a record trails
        close(Out)),
    process_create(BinPath, [],
                   [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Status),
    assertion(Status == exit(11)),
    !.

test(surface_binary_reads_stdin) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binary_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } $1 > 100 { sum += $2 } END { print sum }\n", Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv, DriverIR),
    emit_probe(Dir, DriverIR, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_records(InputPath, i64_pairs([50-1, 200-2, 300-3])),
    format(atom(Cmd), '~w < ~w', [BinPath, InputPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == "5\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% IEEE-754 bit patterns for the dyadic doubles used in these tests.
double_bits(2.5, 0x4004000000000000).
double_bits(0.5, 0x3FE0000000000000).

write_f64_le(Out, V) :-
    double_bits(V, Bits),
    write_i64_le(Out, Bits).

write_records(Path, i64_pairs(Pairs)) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(A-B, Pairs),
            ( write_i64_le(Out, A), write_i64_le(Out, B) )),
        close(Out)).
write_records(Path, i64_f64_pairs(Pairs)) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(A-B, Pairs),
            ( write_i64_le(Out, A), write_f64_le(Out, B) )),
        close(Out)).

build_binary_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binary_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_binrec_marker/0 ],
        [module_name('plawk_binary_records')], LLPath),
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
    ;  format(user_error, "~n[plawk binary records build output]~n~w~n", [BuildOut]),
       throw(plawk_binary_records_build_failed(Status))
    ).

run_binary_smoke(Source, Records, ExpectedOutput) :-
    build_binary_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_records(InputPath, Records),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk binary records run output]~n~w~n", [OutStr]),
       throw(plawk_binary_records_run_failed(Status))
    ),
    !.

:- end_tests(plawk_binary_records).
