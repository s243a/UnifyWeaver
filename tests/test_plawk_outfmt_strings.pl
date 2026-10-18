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

:- dynamic user:plawk_ostr_marker/0.

user:plawk_ostr_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_outfmt_strings, [condition(clang_available)]).

test(sN_output_ir_uses_memcpy_memset) :-
    plawk_parse_string("BEGIN { OUTFMT = \"s8 i64\" } { writebin $1, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    % s8 + i64 = 16-byte records
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_wbuf = alloca [16 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@llvm.memset.p0i8.i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@llvm.memcpy.p0i8.p0i8.i64'))),
    % text-mode source: slice + clamp guard
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_cap = select i1 '))),
    !.

test(sN_literal_source_ir) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"s4 i64\" } { writebin \"tag\", $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'c"tag\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 3, i1 false)'))),
    !.

test(sN_output_rejections) :-
    Rejects = [
        % literal longer than the slot
        "BEGIN { OUTFMT = \"s4\" } { writebin \"toolong\" }\n",
        % source field wider than the output slot
        "BEGIN { BINFMT = \"s8 i64\" ; OUTFMT = \"s4\" } { writebin $1 }\n",
        % numeric input field into a string slot
        "BEGIN { BINFMT = \"s8 i64\" ; OUTFMT = \"s8\" } { writebin $2 }\n",
        % string literal into an i64 slot
        "BEGIN { OUTFMT = \"i64\" } { writebin \"x\" }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.txt', _))
        ;  true
        )).

test(surface_text_to_binary_string_slot) :-
    % Short values NUL-pad; a value longer than the slot clamps to width.
    build_ostr_probe("BEGIN { OUTFMT = \"s8 i64\" } { writebin $1, $2 }\n",
        text("alpha 5\nbee 7\nlongname12 9\n"), Dir, BinPath),
    run_capture_raw(BinPath, Bytes),
    decode_s8_i64(Bytes, Records),
    assertion(Records == ["alpha"-5, "bee"-7, "longname"-9]),
    assertion(Dir \== ''),
    !.

test(surface_binary_string_passthrough_with_tag) :-
    build_ostr_probe("BEGIN { BINFMT = \"s8 i64\" ; OUTFMT = \"s4 s8 i64\" } $2 > 1 { writebin \"tag\", $1, $2 * 2 }\n",
        s8_i64([rec("alpha", 5), rec("bee", 7)]), _Dir, BinPath),
    run_capture_raw(BinPath, Bytes),
    decode_s4_s8_i64(Bytes, Records),
    assertion(Records == [t("tag", "alpha", 10), t("tag", "bee", 14)]),
    !.

test(surface_empty_field_writes_zero_slot) :-
    % A record with fewer fields than referenced: the slot is all zeros.
    build_ostr_probe("BEGIN { OUTFMT = \"s8 i64\" } { writebin $2, NR }\n",
        text("one\nalpha beta\n"), _Dir, BinPath),
    run_capture_raw(BinPath, Bytes),
    decode_s8_i64(Bytes, Records),
    assertion(Records == [""-1, "beta"-2]),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_sfield(Out, String, Width) :-
    string_codes(String, Codes),
    length(Codes, Len),
    Len =< Width,
    forall(member(C, Codes), put_byte(Out, C)),
    Pad is Width - Len,
    forall(between(1, Pad, _), put_byte(Out, 0)).

le_i64(Bytes, Value) :-
    foldl([B, I0-V0, I-V]>>( V is V0 + (B << (8 * I0)), I is I0 + 1 ),
        Bytes, 0-0, _-Unsigned),
    ( Unsigned >= 0x8000000000000000
    -> Value is Unsigned - 0x10000000000000000
    ;  Value = Unsigned
    ).

sfield_string(Codes, String) :-
    append(Prefix, Suffix, Codes),
    ( Suffix = [0 | _] ; Suffix = [] ),
    \+ member(0, Prefix),
    !,
    string_codes(String, Prefix).

decode_s8_i64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_s8_i64_codes(Codes, Records).

decode_s8_i64_codes([], []).
decode_s8_i64_codes(Codes, [S-V | Records]) :-
    length(SBytes, 8), length(VBytes, 8),
    append(SBytes, Rest0, Codes), append(VBytes, Rest, Rest0),
    sfield_string(SBytes, S),
    le_i64(VBytes, V),
    decode_s8_i64_codes(Rest, Records).

decode_s4_s8_i64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_s4_s8_i64_codes(Codes, Records).

decode_s4_s8_i64_codes([], []).
decode_s4_s8_i64_codes(Codes, [t(A, B, V) | Records]) :-
    length(ABytes, 4), length(BBytes, 8), length(VBytes, 8),
    append(ABytes, Rest0, Codes),
    append(BBytes, Rest1, Rest0),
    append(VBytes, Rest, Rest1),
    sfield_string(ABytes, A),
    sfield_string(BBytes, B),
    le_i64(VBytes, V),
    decode_s4_s8_i64_codes(Rest, Records).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_ostr_marker/0 ],
        [module_name('plawk_outfmt_strings')], LLPath),
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
    ;  format(user_error, "~n[plawk outfmt strings build output]~n~w~n", [BuildOut]),
       throw(plawk_outfmt_strings_build_failed(Status))
    ).

build_ostr_probe(Source, Input, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_outfmt_strings', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    ( Input = text(Text)
    ->  directory_file_path(Dir, 'input.txt', InputPath),
        setup_call_cleanup(
            open(InputPath, write, Out, [encoding(utf8)]),
            write(Out, Text),
            close(Out))
    ;   Input = s8_i64(Recs),
        directory_file_path(Dir, 'input.bin', InputPath),
        setup_call_cleanup(
            open(InputPath, write, Out, [type(binary)]),
            forall(member(rec(S, V), Recs),
                ( write_sfield(Out, S, 8), write_i64_le(Out, V) )),
            close(Out))
    ),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath).

run_capture_raw(BinPath, Bytes) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).

:- end_tests(plawk_outfmt_strings).
