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

:- dynamic user:plawk_uwb_marker/0.

user:plawk_uwb_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_union_writebin, [condition(clang_available)]).

test(union_writebin_ir_shape) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" ; OUTFMT = \"i64 i64\" } case 0 { { writebin $1, NR } } case 1 { { writebin $2, NR } }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % shared output buffer in the entry block, one write per rule
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_wbuf = alloca [16 x i8]'))),
    % arm 1's source is its i64 at offset 16 (after the lps16 slot)
    assertion(once(sub_atom(DriverIR, _, _, _, 'i8* %rec, i64 16'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(union_writebin_rejections) :-
    Rejects = [
        % writebin still demands OUTFMT
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } case 0 { { writebin $1 } }\n",
        % arm 1's $1 is a string; the output slot is i64
        "BEGIN { BINFMT = \"case(i64 | lps8)\" ; OUTFMT = \"i64\" } case 1 { { writebin $1 } }\n",
        % argument count must match the output layout
        "BEGIN { BINFMT = \"case(i64 | lps8)\" ; OUTFMT = \"i64 i64\" } case 0 { { writebin $1 } }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_union_normalizer) :-
    % A pure normalizer: two record kinds in, one fixed layout out
    % (value, NR). No END, no scalars -- just per-arm writebin.
    run_uwb_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" ; OUTFMT = \"i64 i64\" } case 0 { { writebin $1, NR } } case 1 { { writebin $2, NR } }\n",
        [m(50, 1.5), e("boom", 7), m(200, 2.5)],
        Bytes),
    decode_i64_pairs(Bytes, Records),
    assertion(Records == [50-1, 7-2, 200-3]),
    !.

test(surface_union_lps_passthrough_with_tag_guards) :-
    % Tag-guard spelling; arm 1's lps16 flows into an lps16 output
    % slot; arm-0 records have no rule and are read + skipped.
    run_uwb_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" ; OUTFMT = \"lps16 i64\" } TAG == 1 && $2 > 5 { writebin $1, $2 }\n",
        [e("boom", 7), m(50, 1.5), e("skip", 2), e("full16bytes4sure", 9)],
        Bytes),
    decode_lps_i64(Bytes, Records),
    assertion(Records == ["boom"-7, "full16bytes4sure"-9]),
    !.

:- end_tests(plawk_union_writebin).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5, 0x3FF8000000000000).
double_bits(2.5, 0x4004000000000000).

% m(V, F): tag 0, i64, f64.  e(S, C): tag 1, lps16, i64.
write_uwb_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Rec, Recs), write_uwb_record(Out, Rec)),
        close(Out)).

write_uwb_record(Out, m(V, F)) :-
    write_i64_le(Out, 0),
    write_i64_le(Out, V),
    double_bits(F, Bits),
    write_i64_le(Out, Bits).
write_uwb_record(Out, e(S, C)) :-
    write_i64_le(Out, 1),
    string_codes(S, Codes),
    length(Codes, Len),
    write_i64_le(Out, Len),
    forall(member(Code, Codes), put_byte(Out, Code)),
    write_i64_le(Out, C).

le_i64(Bytes, Value) :-
    foldl([B, I0-V0, I-V]>>( V is V0 + (B << (8 * I0)), I is I0 + 1 ),
        Bytes, 0-0, _-Unsigned),
    ( Unsigned >= 0x8000000000000000
    -> Value is Unsigned - 0x10000000000000000
    ;  Value = Unsigned
    ).

decode_i64_pairs(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_i64_pairs_codes(Codes, Records).

decode_i64_pairs_codes([], []).
decode_i64_pairs_codes(Codes, [A-B | Records]) :-
    length(ABytes, 8), length(BBytes, 8),
    append(ABytes, Rest0, Codes), append(BBytes, Rest, Rest0),
    le_i64(ABytes, A),
    le_i64(BBytes, B),
    decode_i64_pairs_codes(Rest, Records).

decode_lps_i64(Bytes, Records) :-
    string_codes(Bytes, Codes),
    decode_lps_i64_codes(Codes, Records).

decode_lps_i64_codes([], []).
decode_lps_i64_codes(Codes, [S-V | Records]) :-
    length(LenBytes, 8),
    append(LenBytes, Rest0, Codes),
    le_i64(LenBytes, Len),
    length(SBytes, Len),
    append(SBytes, Rest1, Rest0),
    string_codes(S, SBytes),
    length(VBytes, 8),
    append(VBytes, Rest, Rest1),
    le_i64(VBytes, V),
    decode_lps_i64_codes(Rest, Records).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_uwb_marker/0 ],
        [module_name('plawk_union_writebin')], LLPath),
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
    ;  format(user_error, "~n[plawk union writebin build output]~n~w~n", [BuildOut]),
       throw(plawk_union_writebin_build_failed(Status))
    ).

run_uwb_smoke(Source, Recs, Bytes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_union_writebin', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath),
    write_uwb_records(InputPath, Recs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    !.
