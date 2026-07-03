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

:- dynamic user:plawk_urp_marker/0.

user:plawk_urp_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_union_rep, [condition(clang_available)]).

test(rep_arm_sizes_buffer_and_reads_in_arm) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 rep4(i64 f64) | lps16 i64)\" } case 0 { { foreach { n++ } } } END { print n }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % widest arm: 8 id + 8 count + 4*16 elems + 16 staging = 96
    assertion(once(sub_atom(DriverIR, _, _, _, 'malloc(i64 96)'))),
    % the rep reads inside arm 0's field sequence (bulk read: fixed
    % elements), and foreach is the runtime loop
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_a0_f1_bytes = mul i64 %vr_a0_f1_n, 16'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_j = phi i64 [1, %'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(rep_arm_rejections) :-
    Rejects = [
        % foreach needs a rep in ITS arm (arm 1 has none)
        "BEGIN { BINFMT = \"case(i64 rep4(i64) | lps16 i64)\" } case 1 { { foreach { n++ } } } END { print n }\n",
        % element arity still checked per arm ($3 in a 2-field element)
        "BEGIN { BINFMT = \"case(i64 rep4(i64 f64) | i64)\" } case 0 { { foreach { s += $3 } } } END { print s }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_rep_arm_foreach_dispatch) :-
    % Arm 0 records carry element lists that foreach aggregates; arm 1
    % events are guarded per record -- one loop, shared END.
    run_urp_smoke("BEGIN { BINFMT = \"case(i64 rep4(i64 f64) | lps16 i64)\" } case 0 { $1 > 0 { foreach { n++ ; wsum += float($2) } } } case 1 { $1 == \"boom\" { events++ } } END { print n, wsum, events }\n",
        [ [i64(0), i64(1), i64(2), i64(5), f64(1.5), i64(20), f64(2.5)],
          [i64(1), i64(4), bytes("boom"), i64(3)],
          [i64(0), i64(2), i64(0)],
          [i64(1), i64(1), bytes("x"), i64(7)],
          [i64(0), i64(3), i64(1), i64(30), f64(0.25)] ],
        "3 4.25 1\n").

test(surface_rep_lps_arm_with_tag_guards) :-
    % All three slices composed: tag-guard sugar, a rep arm, and lps
    % strings inside the rep's elements.
    run_urp_smoke("BEGIN { BINFMT = \"case(rep4(lps8 i64) | i64)\" } TAG == 0 { foreach { if ($1 == \"hot\") { hits++ }; total += $2 } } TAG == 1 { other++ } END { print hits, total, other }\n",
        [ [i64(0), i64(2), i64(3), bytes("hot"), i64(5), i64(4), bytes("cool"), i64(7)],
          [i64(1), i64(99)],
          [i64(0), i64(1), i64(3), bytes("hot"), i64(2)] ],
        "2 14 1\n").

test(surface_rep_arm_error_paths) :-
    build_urp_probe("BEGIN { BINFMT = \"case(i64 rep4(i64 f64) | lps16 i64)\" } case 0 { { foreach { n++ } } } END { print n }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    % count above the rep cap inside arm 0
    write_items(InputPath, [i64(0), i64(1), i64(5)]),
    run_capture_raw(BinPath, _, exit(11)),
    % truncated element region (count 2, one element present)
    write_items(InputPath, [i64(0), i64(1), i64(2), i64(5), f64(1.5)]),
    run_capture_raw(BinPath, _, exit(11)),
    % unknown tag
    write_items(InputPath, [i64(9)]),
    run_capture_raw(BinPath, _, exit(11)),
    % clean EOF at a record boundary runs END
    write_items(InputPath, []),
    run_capture_raw(BinPath, Out, exit(0)),
    assertion(Out == "0\n"),
    !.

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

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_urp_marker/0 ],
        [module_name('plawk_union_rep')], LLPath),
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
    ;  format(user_error, "~n[plawk union rep build output]~n~w~n", [BuildOut]),
       throw(plawk_union_rep_build_failed(Status))
    ).

build_urp_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_union_rep', Dir),
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

% each record is spelled as a flat item list (tag first)
run_urp_smoke(Source, Records, ExpectedOutput) :-
    build_urp_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    append(Records, Items),
    write_items(InputPath, Items),
    run_capture_raw(BinPath, OutStr, exit(0)),
    assertion(OutStr == ExpectedOutput),
    !.

:- end_tests(plawk_union_rep).
