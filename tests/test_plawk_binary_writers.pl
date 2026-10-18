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

:- dynamic user:plawk_binwr_marker/0.

user:plawk_binwr_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_binary_writers, [condition(clang_available)]).

test(parses_writebin_action) :-
    plawk_parse_string("BEGIN { OUTFMT = \"i64 i64\" } { writebin $1, $2 }\n", Program),
    assertion(Program == program([begin([set(var('OUTFMT'), string("i64 i64"))])],
        [rule(always, [writebin([field(1), field(2)])])],
        [])).

test(writebin_ir_stores_and_fwrites) :-
    plawk_parse_string("BEGIN { OUTFMT = \"i64 f64\" } { writebin $1, float($2) }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_wbuf = alloca [16 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'store i64 '))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'store double '))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@fwrite('))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@stdout = external global i8*'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(writebin_rejections) :-
    Rejects = [
        % no OUTFMT declared
        "{ writebin $1, $2 }\n",
        % argument count does not match the layout
        "BEGIN { OUTFMT = \"i64 i64\" } { writebin $1 }\n",
        % double expression into an i64 slot
        "BEGIN { OUTFMT = \"i64\" } { writebin 1.5 }\n",
        % string field into an i64 slot in binary input mode
        "BEGIN { BINFMT = \"s8 i64\" ; OUTFMT = \"i64\" } { writebin $1 }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.txt', _))
        ;  true
        )).

test(surface_text_to_binary_converter) :-
    % Single-action driver: convert text lines to binary records.
    build_writer_probe("BEGIN { OUTFMT = \"i64 f64\" } { writebin $1, float($2) }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.txt', InputPath),
    write_text(InputPath, "5 2.5\n20 4.5\n30 1.25\n"),
    run_capture_raw(BinPath, [], Bytes),
    decode_i64_f64(Bytes, Records),
    assertion(Records == [5-2.5, 20-4.5, 30-1.25]),
    !.

test(surface_binary_transform_with_guard_and_scalars) :-
    % Scalar driver: binary in, guarded binary out, END count as trailer.
    build_writer_probe("BEGIN { BINFMT = \"i64 f64\" ; OUTFMT = \"i64 f64\" } $1 > 10 { n++ ; writebin $1 * 10, float($2) * 0.5 } END { print n }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    encode_i64_f64(InputPath, [5-2.5, 20-4.5, 30-1.25]),
    run_capture_raw(BinPath, [], Bytes),
    string_length(Bytes, Len),
    RecBytes is Len - 2,
    sub_string(Bytes, 0, RecBytes, _, RecStr),
    sub_string(Bytes, RecBytes, 2, 0, Trailer),
    assertion(Trailer == "2\n"),
    decode_i64_f64(RecStr, Records),
    assertion(Records == [200-2.25, 300-0.625]),
    !.

test(surface_writebin_nr_and_expr_args) :-
    build_writer_probe("BEGIN { OUTFMT = \"i64 i64\" } { writebin NR, $1 + 1 }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.txt', InputPath),
    write_text(InputPath, "10\n20\n"),
    run_capture_raw(BinPath, [], Bytes),
    decode_i64_i64(Bytes, Records),
    assertion(Records == [1-11, 2-21]),
    !.

test(surface_plawk_to_plawk_pipeline) :-
    % Program A converts text to binary; program B consumes A's binary
    % output and aggregates natively - no text in between.
    build_writer_probe("BEGIN { OUTFMT = \"i64 f64\" } { writebin $1, float($2) }\n",
        DirA, BinA),
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binary_writers_b', DirB),
    clean_dir(DirB),
    make_directory_path(DirB),
    directory_file_path(DirB, 'stage.bin', StagePath),
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } $1 > 10 { sum += float($2) } END { print sum }\n", ProgramB),
    plawk_program_native_driver_ir(ProgramB, StagePath, DriverB),
    emit_probe(DirB, DriverB, BinB),
    directory_file_path(DirA, 'input.txt', InputPath),
    write_text(InputPath, "5 2.5\n20 4.5\n30 1.25\n"),
    format(atom(Cmd), '~w > ~w && ~w', [BinA, StagePath, BinB]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == "5.75\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_text(Path, Text) :-
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

i64_bytes(V, Bytes) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    findall(B, ( between(0, 7, I), B is (V64 >> (8 * I)) /\ 0xFF ), Bytes).

% dyadic doubles used in these tests, as IEEE-754 bit patterns
double_bits(2.5,   0x4004000000000000).
double_bits(4.5,   0x4012000000000000).
double_bits(1.25,  0x3FF4000000000000).
double_bits(2.25,  0x4002000000000000).
double_bits(0.625, 0x3FE4000000000000).

encode_i64_f64(Path, Pairs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(A-F, Pairs),
            ( i64_bytes(A, ABytes), forall(member(B, ABytes), put_byte(Out, B)),
              double_bits(F, Bits),
              i64_bytes(Bits, FBytes), forall(member(B, FBytes), put_byte(Out, B)) )),
        close(Out)).

decode_i64_f64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_i64_f64_codes(Codes, Records).

decode_i64_f64_codes([], []).
decode_i64_f64_codes(Codes, [A-F | Records]) :-
    length(ABytes, 8), length(FBytes, 8),
    append(ABytes, Rest0, Codes), append(FBytes, Rest, Rest0),
    le_i64(ABytes, A),
    le_i64(FBytes, Bits),
    double_bits(F, Bits),
    decode_i64_f64_codes(Rest, Records).

decode_i64_i64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_i64_i64_codes(Codes, Records).

decode_i64_i64_codes([], []).
decode_i64_i64_codes(Codes, [A-B | Records]) :-
    length(ABytes, 8), length(BBytes, 8),
    append(ABytes, Rest0, Codes), append(BBytes, Rest, Rest0),
    le_i64(ABytes, A), le_i64(BBytes, B),
    decode_i64_i64_codes(Rest, Records).

le_i64(Bytes, Value) :-
    foldl([B, I0-V0, I-V]>>( V is V0 + (B << (8 * I0)), I is I0 + 1 ),
        Bytes, 0-0, _-Unsigned),
    ( Unsigned >= 0x8000000000000000
    -> Value is Unsigned - 0x10000000000000000
    ;  Value = Unsigned
    ).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_binwr_marker/0 ],
        [module_name('plawk_binary_writers')], LLPath),
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
    ;  format(user_error, "~n[plawk binary writers build output]~n~w~n", [BuildOut]),
       throw(plawk_binary_writers_build_failed(Status))
    ).

build_writer_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binary_writers', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    plawk_parse_string(Source, Program),
    ( sub_string(Source, _, _, _, "BINFMT")
    -> directory_file_path(Dir, 'input.bin', InputPath)
    ;  directory_file_path(Dir, 'input.txt', InputPath)
    ),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath).

run_capture_raw(BinPath, Args, Bytes) :-
    process_create(BinPath, Args,
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).

:- end_tests(plawk_binary_writers).
