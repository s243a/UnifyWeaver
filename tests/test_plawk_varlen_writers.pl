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

:- dynamic user:plawk_vlw_marker/0.

user:plawk_vlw_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_varlen_writers, [condition(clang_available)]).

test(varlen_outfmt_ir_uses_per_slot_fwrites) :-
    plawk_parse_string("BEGIN { OUTFMT = \"lps16 i64\" } { writebin $1, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    % lps slot: one fwrite for the length, one for the payload; numeric
    % slot: one staged fwrite. No single whole-record fwrite.
    assertion(once(sub_atom(DriverIR, _, _, _, '_lwr = call i64 @fwrite('))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_pwr = call i64 @fwrite('))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_wr = call i64 @fwrite('))),
    % text-mode source: slice + null guard + cap clamp
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_over = icmp ugt i64 '))),
    !.

test(fixed_outfmt_keeps_single_record_fwrite) :-
    plawk_parse_string("BEGIN { OUTFMT = \"s16 i64\" } { writebin $1, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 24, i64 1, i8*'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_lwr = ')),
    !.

test(varlen_writer_rejections) :-
    Rejects = [
        % literal longer than the cap
        "BEGIN { OUTFMT = \"lps4\" } { writebin \"toolong\" }\n",
        % source cap wider than the output cap
        "BEGIN { BINFMT = \"lps8 i64\" ; OUTFMT = \"lps4\" } { writebin $1 }\n",
        % numeric input field into an lps slot
        "BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"lps8\" } { writebin $1 }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.txt', _))
        ;  true
        )).

test(surface_text_to_varlen_exact_lengths) :-
    % Wire lengths are exact: no padding for short values, no clamp
    % needed at exactly the cap.
    build_vlw_probe("BEGIN { OUTFMT = \"lps16 i64\" } { writebin $1, $2 }\n",
        text("alpha 5\nverylongname1234 7\n"), _Dir, BinPath),
    run_capture_raw(BinPath, Bytes, exit(0)),
    decode_lps_i64(Bytes, Records),
    assertion(Records == ["alpha"-5, "verylongname1234"-7]),
    !.

test(surface_text_clamps_to_cap) :-
    build_vlw_probe("BEGIN { OUTFMT = \"lps4 i64\" } { writebin $1, NR }\n",
        text("longvalue x\nab y\n"), _Dir, BinPath),
    run_capture_raw(BinPath, Bytes, exit(0)),
    decode_lps_i64(Bytes, Records),
    assertion(Records == ["long"-1, "ab"-2]),
    !.

test(surface_varlen_transform_with_literal_and_empty) :-
    build_vlw_probe("BEGIN { BINFMT = \"i64 lps8\" ; OUTFMT = \"lps4 lps8 i64\" } $1 > 10 { writebin \"tag\", $2, $1 * 2 }\n",
        lps_recs([rec(5, "aa"), rec(20, "bee"), rec(40, "")]), _Dir, BinPath),
    run_capture_raw(BinPath, Bytes, exit(0)),
    decode_lps_lps_i64(Bytes, Records),
    assertion(Records == [t("tag", "bee", 40), t("tag", "", 80)]),
    !.

test(surface_varlen_round_trip) :-
    % Writer output is byte-compatible with the varlen READER: stage 1
    % converts text to lps16 records, stage 2 reads them back with
    % BINFMT and aggregates.
    build_vlw_probe("BEGIN { OUTFMT = \"lps16 i64\" } { writebin $1, $2 }\n",
        text("alpha 5\nbeta 7\nalpha 2\n"), Dir1, Bin1),
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_varlen_writers_b', Dir2),
    clean_dir(Dir2),
    make_directory_path(Dir2),
    directory_file_path(Dir2, 'stage.bin', StagePath),
    plawk_parse_string("BEGIN { BINFMT = \"lps16 i64\" } $1 == \"alpha\" { sum += $2 } END { print sum }\n", Program2),
    plawk_program_native_driver_ir(Program2, StagePath, Driver2),
    emit_probe(Dir2, Driver2, Bin2),
    format(atom(Cmd), '~w > ~w && ~w', [Bin1, StagePath, Bin2]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == "7\n"),
    assertion(Dir1 \== ''),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

le_i64(Bytes, Value) :-
    foldl([B, I0-V0, I-V]>>( V is V0 + (B << (8 * I0)), I is I0 + 1 ),
        Bytes, 0-0, _-Unsigned),
    ( Unsigned >= 0x8000000000000000
    -> Value is Unsigned - 0x10000000000000000
    ;  Value = Unsigned
    ).

take_i64(Codes, V, Rest) :-
    length(Bytes, 8),
    append(Bytes, Rest, Codes),
    le_i64(Bytes, V).

take_lps(Codes, S, Rest) :-
    take_i64(Codes, Len, Rest0),
    length(PayloadCodes, Len),
    append(PayloadCodes, Rest, Rest0),
    string_codes(S, PayloadCodes).

decode_lps_i64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_lps_i64_codes(Codes, Records).

decode_lps_i64_codes([], []).
decode_lps_i64_codes(Codes, [S-V | Records]) :-
    take_lps(Codes, S, Rest0),
    take_i64(Rest0, V, Rest),
    decode_lps_i64_codes(Rest, Records).

decode_lps_lps_i64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_lps_lps_i64_codes(Codes, Records).

decode_lps_lps_i64_codes([], []).
decode_lps_lps_i64_codes(Codes, [t(A, B, V) | Records]) :-
    take_lps(Codes, A, Rest0),
    take_lps(Rest0, B, Rest1),
    take_i64(Rest1, V, Rest),
    decode_lps_lps_i64_codes(Rest, Records).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_vlw_marker/0 ],
        [module_name('plawk_varlen_writers')], LLPath),
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
    ;  format(user_error, "~n[plawk varlen writers build output]~n~w~n", [BuildOut]),
       throw(plawk_varlen_writers_build_failed(Status))
    ).

build_vlw_probe(Source, Input, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_varlen_writers', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    ( Input = text(Text)
    ->  directory_file_path(Dir, 'input.txt', InputPath),
        setup_call_cleanup(
            open(InputPath, write, Out, [encoding(utf8)]),
            write(Out, Text),
            close(Out))
    ;   Input = lps_recs(Recs),
        directory_file_path(Dir, 'input.bin', InputPath),
        setup_call_cleanup(
            open(InputPath, write, Out, [type(binary)]),
            forall(member(rec(V, S), Recs),
                ( write_i64_le(Out, V),
                  string_codes(S, Codes),
                  length(Codes, Len),
                  write_i64_le(Out, Len),
                  forall(member(C, Codes), put_byte(Out, C)) )),
            close(Out))
    ),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath).

run_capture_raw(BinPath, Bytes, ExpectedStatus) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == ExpectedStatus).

:- end_tests(plawk_varlen_writers).
