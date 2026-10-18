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

:- dynamic user:plawk_vlr_marker/0.

user:plawk_vlr_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_varlen_records, [condition(clang_available)]).

test(varlen_ir_uses_field_by_field_reads) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 lps16\" } $1 > 10 { c++ } END { print c }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % access layout: 8 + 16 = 24-byte buffer
    assertion(once(sub_atom(DriverIR, _, _, _, 'malloc(i64 24)'))),
    % shared length scratch + per-field reads instead of one fixed read
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_len_scratch = alloca i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 8, i8* %vr_f0_dst'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 8, i8* %vr_len_i8'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_f1_fits = icmp ule i64 %vr_f1_n, 16'))),
    % only the record's first read may hit clean EOF
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_f0_eof'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%vr_f1_eof')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(fixed_layouts_keep_single_read_skeleton) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 s16\" } $1 > 10 { c++ } END { print c }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 24, i8* %rec'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'vr_len_scratch')),
    !.

test(surface_varlen_guard_print_equality) :-
    run_vlr_smoke("BEGIN { BINFMT = \"i64 lps16\" } $1 > 10 { hits++ ; print $2, $1 } $2 == \"skipme\" { skips++ } END { print hits, skips }\n",
        [rec(5, "small"), rec(20, "hello"), rec(30, ""), rec(40, "exactly16bytes!!"), rec(7, "skipme")],
        "hello 20\n 30\nexactly16bytes!! 40\n3 1\n").

test(surface_varlen_groupby) :-
    run_vlr_sorted_smoke("BEGIN { BINFMT = \"i64 lps8\" } { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        [rec(5, "a"), rec(9, "bb"), rec(5, "ccc"), rec(5, "")],
        ["5 3", "9 1"]).

test(surface_varlen_to_fixed_binary_writer) :-
    % lps input accesses as sN, so it flows into a fixed sN output slot.
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_varlen_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string("BEGIN { BINFMT = \"i64 lps8\" ; OUTFMT = \"s8 i64\" } $1 > 10 { writebin $2, $1 }\n", Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath),
    write_vlr_records(InputPath, [rec(5, "aa"), rec(20, "bee"), rec(40, "longnam8")]),
    run_capture_raw(BinPath, Bytes, exit(0)),
    decode_s8_i64(Bytes, Records),
    assertion(Records == ["bee"-20, "longnam8"-40]),
    !.

test(surface_varlen_error_paths) :-
    build_vlr_probe("BEGIN { BINFMT = \"i64 lps16\" } { c++ } END { print c }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    % oversized length prefix (17 > 16)
    write_raw(InputPath, [i64(20), i64(17), bytes("xxxxxxxxxxxxxxxxx")]),
    run_capture_raw(BinPath, _, exit(11)),
    % truncated payload (claims 10, has 4)
    write_raw(InputPath, [i64(20), i64(10), bytes("abcd")]),
    run_capture_raw(BinPath, _, exit(11)),
    % EOF mid-record (numeric field then nothing)
    write_raw(InputPath, [i64(20)]),
    run_capture_raw(BinPath, _, exit(11)),
    % empty input is a clean EOF: END runs
    write_raw(InputPath, []),
    run_capture_raw(BinPath, Out, exit(0)),
    assertion(Out == "0\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_raw(Path, Items) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Item, Items),
            ( Item = i64(V) -> write_i64_le(Out, V)
            ; Item = bytes(S) -> ( string_codes(S, Cs),
                                   forall(member(C, Cs), put_byte(Out, C)) )
            )),
        close(Out)).

% one record: 8-byte i64, 8-byte length, payload
write_vlr_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(V, S), Recs),
            ( write_i64_le(Out, V),
              string_codes(S, Codes),
              length(Codes, Len),
              write_i64_le(Out, Len),
              forall(member(C, Codes), put_byte(Out, C)) )),
        close(Out)).

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

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_vlr_marker/0 ],
        [module_name('plawk_varlen_records')], LLPath),
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
    ;  format(user_error, "~n[plawk varlen records build output]~n~w~n", [BuildOut]),
       throw(plawk_varlen_records_build_failed(Status))
    ).

build_vlr_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_varlen_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
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

run_vlr_smoke(Source, Recs, ExpectedOutput) :-
    build_vlr_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_vlr_records(InputPath, Recs),
    run_capture_raw(BinPath, OutStr, exit(0)),
    assertion(OutStr == ExpectedOutput),
    !.

run_vlr_sorted_smoke(Source, Recs, ExpectedLines) :-
    build_vlr_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_vlr_records(InputPath, Recs),
    run_capture_raw(BinPath, OutStr, exit(0)),
    split_string(OutStr, "\n", "", Split0),
    exclude(==(""), Split0, Lines0),
    msort(Lines0, SortedLines),
    msort(ExpectedLines, SortedExpected),
    assertion(SortedLines == SortedExpected),
    !.

:- end_tests(plawk_varlen_records).
