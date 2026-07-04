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

:- dynamic user:plawk_uas_marker/0.

user:plawk_uas_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_union_assoc, [condition(clang_available)]).

test(union_assoc_ir_keys_are_arm_relative_loads) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 | lps8 i64)\" } case 0 { { counts[$1]++ } } case 1 { { counts[$2]++ } } END { for (k in counts) print k, counts[k] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % the union reader's tag switch feeds one shared table; arm 0's key
    % loads at offset 0, arm 1's i64 key after its 8-byte string slot
    assertion(once(sub_atom(DriverIR, _, _, _, 'switch i64 %vr_tag'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_new'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%assoc_rule_0_action_0_key_fp = getelementptr i8, i8* %rec, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%assoc_rule_1_action_0_key_fp = getelementptr i8, i8* %rec, i64 8'))),
    % raw i64 keys: no interning anywhere in the update path
    assertion(\+ sub_atom(DriverIR, _, _, _, 'assoc_rule_0_action_0_key_id')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(union_assoc_rejections) :-
    Rejects = [
        % arm 1's $1 is a string: not a raw-i64 key
        "BEGIN { BINFMT = \"case(i64 | lps8 i64)\" } case 1 { { counts[$1]++ } } END { print counts[5] }\n",
        % string lookups stay text-mode-only
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } case 0 { { counts[$1]++ } } END { print counts[\"x\"] }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_union_groupby_forin) :-
    % One shared keyspace across arms: metric ids and event values
    % count into the same table.
    run_uas_sorted_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 0 { { counts[$1]++ } } case 1 { { counts[$2]++ } } END { for (k in counts) print k, counts[k] }\n",
        [m(5, 1.5), e("x", 9), m(5, 2.5), e("y", 5), m(9, 1.5)],
        ["5 3", "9 2"]).

test(surface_union_groupby_tag_guards_and_lookup) :-
    % Tag-guard spelling with a residual guard, END int lookup.
    run_uas_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } TAG == 0 && $1 > 10 { counts[$1]++ } TAG == 1 { counts[$2]++ } END { print counts[20], counts[5] }\n",
        [m(20, 1.5), e("a", 20), m(5, 2.5), m(20, 1.5), e("b", 5)],
        "3 1\n").

test(surface_union_groupby_forin_writebin) :-
    % Group-by to BINARY output over a union stream: one (key, count)
    % record per group.
    build_uas_probe("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" ; OUTFMT = \"i64 i64\" } case 0 { { counts[$1]++ } } case 1 { { counts[$2]++ } } END { for (k in counts) writebin k, counts[k] }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_uas_records(InputPath, [m(5, 1.5), e("x", 9), m(5, 2.5), e("y", 5), m(9, 1.5)]),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    decode_i64_pairs(Bytes, Records0),
    msort(Records0, Records),
    assertion(Records == [5-3, 9-2]),
    !.

:- end_tests(plawk_union_assoc).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5, 0x3FF8000000000000).
double_bits(2.5, 0x4004000000000000).

% m(V, F): tag 0, i64, f64.  e(S, C): tag 1, lps16, i64.
write_uas_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Rec, Recs), write_uas_record(Out, Rec)),
        close(Out)).

write_uas_record(Out, m(V, F)) :-
    write_i64_le(Out, 0),
    write_i64_le(Out, V),
    double_bits(F, Bits),
    write_i64_le(Out, Bits).
write_uas_record(Out, e(S, C)) :-
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

build_uas_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_union_assoc', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_uas_marker/0 ],
        [module_name('plawk_union_assoc')], LLPath),
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
    ;  format(user_error, "~n[plawk union assoc build output]~n~w~n", [BuildOut]),
       throw(plawk_union_assoc_build_failed(Status))
    ).

run_capture(BinPath, OutStr) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).

run_uas_smoke(Source, Recs, ExpectedOutput) :-
    build_uas_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_uas_records(InputPath, Recs),
    run_capture(BinPath, OutStr),
    assertion(OutStr == ExpectedOutput),
    !.

run_uas_sorted_smoke(Source, Recs, ExpectedLines) :-
    build_uas_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_uas_records(InputPath, Recs),
    run_capture(BinPath, OutStr),
    split_string(OutStr, "\n", "", Split0),
    exclude(==(""), Split0, Lines0),
    msort(Lines0, SortedLines),
    msort(ExpectedLines, SortedExpected),
    assertion(SortedLines == SortedExpected),
    !.
