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

:- dynamic user:plawk_fwb_marker/0.

user:plawk_fwb_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_forin_writebin, [condition(clang_available)]).

test(parses_forin_writebin_body) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n", Program),
    Program = program(_, _, [end([for_in(var(k), var(counts), Body)])]),
    assertion(Body == [writebin([var(k), assoc(var(counts), var(k))])]).

test(forin_writebin_ir_shape) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_wbuf = alloca [16 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_iter_next'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_value_at'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@fwrite('))),
    % the group loop emits records, not text
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_to_string')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(forin_writebin_rejections) :-
    Rejects = [
        % text-mode keys are atom ids: binary input only
        "BEGIN { OUTFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        % arity mismatch against OUTFMT
        "BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        % no OUTFMT at all
        "BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        % sN output slice not supported
        "BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"s8 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_groupby_to_binary_records) :-
    run_fwb_smoke("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        [5-0, (-3)-0, 5-0, 9-0, (-3)-0, 5-0],
        [(-3)-2, 5-3, 9-1]).

test(surface_guarded_groupby_to_binary) :-
    run_fwb_smoke("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64 i64\" } $2 > 100 { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        [5-200, 5-50, 9-300, 5-150],
        [5-2, 9-1]).

test(surface_groupby_empty_input_writes_nothing) :-
    run_fwb_smoke("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        [],
        []).

test(surface_groupby_pipeline_two_stages) :-
    % Stage 1 groups raw events into (key, count) binary records; stage 2
    % consumes those records and sums the counts -- binary end to end.
    build_fwb_probe("BEGIN { BINFMT = \"i64 i64\" ; OUTFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) writebin k, counts[k] }\n",
        Dir1, Bin1),
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_forin_writebin_b', Dir2),
    clean_dir(Dir2),
    make_directory_path(Dir2),
    directory_file_path(Dir2, 'stage.bin', StagePath),
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } { total += $2 } END { print total }\n", Program2),
    plawk_program_native_driver_ir(Program2, StagePath, Driver2),
    emit_probe(Dir2, Driver2, Bin2),
    directory_file_path(Dir1, 'input.bin', InputPath),
    write_pairs(InputPath, [5-0, (-3)-0, 5-0, 9-0, (-3)-0, 5-0]),
    format(atom(Cmd), '~w > ~w && ~w', [Bin1, StagePath, Bin2]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == "6\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_pairs(Path, Pairs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(A-B, Pairs),
            ( write_i64_le(Out, A), write_i64_le(Out, B) )),
        close(Out)).

le_i64(Bytes, Value) :-
    foldl([B, I0-V0, I-V]>>( V is V0 + (B << (8 * I0)), I is I0 + 1 ),
        Bytes, 0-0, _-Unsigned),
    ( Unsigned >= 0x8000000000000000
    -> Value is Unsigned - 0x10000000000000000
    ;  Value = Unsigned
    ).

decode_pairs(Bytes, Pairs) :-
    string_codes(Bytes, Codes),
    decode_pair_codes(Codes, Pairs).

decode_pair_codes([], []).
decode_pair_codes(Codes, [A-B | Pairs]) :-
    length(ABytes, 8), length(BBytes, 8),
    append(ABytes, Rest0, Codes), append(BBytes, Rest, Rest0),
    le_i64(ABytes, A), le_i64(BBytes, B),
    decode_pair_codes(Rest, Pairs).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_fwb_marker/0 ],
        [module_name('plawk_forin_writebin')], LLPath),
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
    ;  format(user_error, "~n[plawk forin writebin build output]~n~w~n", [BuildOut]),
       throw(plawk_forin_writebin_build_failed(Status))
    ).

build_fwb_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_forin_writebin', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath).

run_fwb_smoke(Source, InputPairs, ExpectedSortedPairs) :-
    build_fwb_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_pairs(InputPath, InputPairs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, Bytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    decode_pairs(Bytes, Pairs),
    msort(Pairs, SortedPairs),
    msort(ExpectedSortedPairs, SortedExpected),
    assertion(SortedPairs == SortedExpected),
    !.

:- end_tests(plawk_forin_writebin).
