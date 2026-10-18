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

:- dynamic user:plawk_uot_marker/0.

user:plawk_uot_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_union_out, [condition(clang_available)]).

test(union_outfmt_writes_tag_then_arm_slots) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 f64)\" ; OUTFMT = \"case(i64 | i64 lps8)\" } TAG == 0 && $1 > 100 { writebin case 0, $1 ; next } TAG == 0 { writebin case 1, $1, \"low\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % shared buffer sized to the widest arm (i64 + lps8 slot = 16),
    % element pointer resolved once in entry
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_wbuf = alloca [16 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_wbuf_p = getelementptr inbounds [16 x i8]'))),
    % each site stores its constant tag and writes 8 bytes first
    assertion(once(sub_atom(DriverIR, _, _, _, 'store i64 0, i64* %'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'store i64 1, i64* %'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(union_outfmt_rejections) :-
    Rejects = [
        % arm index beyond the declared arms
        "BEGIN { BINFMT = \"i64 f64\" ; OUTFMT = \"case(i64 | i64)\" } { writebin case 2, $1 }\n",
        % a plain writebin cannot pick an arm
        "BEGIN { BINFMT = \"i64 f64\" ; OUTFMT = \"case(i64 | i64)\" } { writebin $1 }\n",
        % an arm-targeted writebin is meaningless against a flat layout
        "BEGIN { BINFMT = \"i64 f64\" ; OUTFMT = \"i64\" } { writebin case 0, $1 }\n",
        % argument count must match the arm
        "BEGIN { BINFMT = \"i64 f64\" ; OUTFMT = \"case(i64 i64)\" } { writebin case 0, $1 }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_split_stream_into_tagged_output) :-
    % Records split into a tagged stream: big ids to arm 0, everything
    % else to arm 1 with a label. A pure retagger: no END, no scalars.
    run_uot_writer("BEGIN { BINFMT = \"case(i64 f64)\" ; OUTFMT = \"case(i64 | i64 lps8)\" } TAG == 0 && $1 > 100 { writebin case 0, $1 ; next } TAG == 0 { writebin case 1, $1, \"low\" }\n",
        [rec(200, 1.5), rec(5, 2.5), rec(300, 0.25)],
        OutBytes),
    expected_bytes([i64(0), i64(200),
                    i64(1), i64(5), i64(3), bytes("low"),
                    i64(0), i64(300)],
        Expected),
    assertion(OutBytes == Expected),
    !.

test(surface_tagged_output_roundtrips_through_union_reader) :-
    % The capstone: the tagged writer's output is byte-compatible with
    % the union READER, so a second plawk program consumes it directly.
    run_uot_writer("BEGIN { BINFMT = \"case(i64 f64)\" ; OUTFMT = \"case(i64 | i64 lps8)\" } TAG == 0 && $1 > 100 { writebin case 0, $1 ; next } TAG == 0 { writebin case 1, $1, \"low\" }\n",
        [rec(200, 1.5), rec(5, 2.5), rec(300, 0.25)],
        OutBytes),
    run_uot_reader("BEGIN { BINFMT = \"case(i64 | i64 lps8)\" } TAG == 0 { bigsum += $1 } TAG == 1 && $2 == \"low\" { lows++ } END { print bigsum, lows }\n",
        OutBytes, ReaderOut),
    assertion(ReaderOut == "500 1\n"),
    !.

:- end_tests(plawk_union_out).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5,  0x3FF8000000000000).
double_bits(2.5,  0x4004000000000000).
double_bits(0.25, 0x3FD0000000000000).

write_items(Path, Items) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Item, Items), write_item(Out, Item)),
        close(Out)).

write_item(Out, i64(V)) :-
    write_i64_le(Out, V).
write_item(Out, f64(F)) :-
    double_bits(F, Bits),
    write_i64_le(Out, Bits).
write_item(Out, bytes(S)) :-
    string_codes(S, Cs),
    forall(member(C, Cs), put_byte(Out, C)).

expected_bytes(Items, Bytes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_union_out_expect.bin', Path),
    write_items(Path, Items),
    setup_call_cleanup(
        open(Path, read, In, [type(binary)]),
        read_string(In, _, Bytes),
        close(In)).

build_uot_probe(Source, SubDir, ModuleName, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, SubDir, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_uot_marker/0 ],
        [module_name(ModuleName)], LLPath),
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
    ;  format(user_error, "~n[plawk union out build output]~n~w~n", [BuildOut]),
       throw(plawk_union_out_build_failed(Status))
    ).

run_capture_binary(BinPath, Bytes) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).

% rec(I64, F64) input records for the single-arm-union writer programs
% (each on the wire as tag 0, i64, f64)
run_uot_writer(Source, Recs, OutBytes) :-
    build_uot_probe(Source, 'uw_plawk_union_out_w', 'plawk_union_out_w',
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    findall(Item,
        ( member(rec(I, F), Recs),
          member(Item, [i64(0), i64(I), f64(F)])
        ),
        Items),
    write_items(InputPath, Items),
    run_capture_binary(BinPath, OutBytes).

run_uot_reader(Source, InputBytes, OutStr) :-
    build_uot_probe(Source, 'uw_plawk_union_out_r', 'plawk_union_out_r',
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    setup_call_cleanup(
        open(InputPath, write, Out, [type(binary)]),
        ( string_codes(InputBytes, Codes),
          forall(member(C, Codes), put_byte(Out, C)) ),
        close(Out)),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).
