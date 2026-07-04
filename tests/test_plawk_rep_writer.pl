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

:- dynamic user:plawk_rpw_marker/0.

user:plawk_rpw_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_rep_writer, [condition(clang_available)]).

test(rep_outfmt_writes_count_then_bulk_elements) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" ; OUTFMT = \"i64 rep4(i64 f64)\" } $1 > 0 { writebin $1, $2 ; kept++ } END { print kept }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % the rep slot: live count staged + written (8 bytes), then one
    % bulk fwrite of count*16 element bytes straight from %rec
    assertion(once(sub_atom(DriverIR, _, _, _, '_cwr = call i64 @fwrite(i8* %'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_ewr = call i64 @fwrite(i8* %'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(rep_outfmt_rejections) :-
    Rejects = [
        % caps must match (a bigger output cap cannot invent elements,
        % a smaller one could not hold the count)
        "BEGIN { BINFMT = \"i64 rep4(i64 f64)\" ; OUTFMT = \"i64 rep8(i64 f64)\" } { writebin $1, $2 }\n",
        % element layouts must match exactly
        "BEGIN { BINFMT = \"i64 rep4(i64 f64)\" ; OUTFMT = \"i64 rep4(i64)\" } { writebin $1, $2 }\n",
        % the rep argument must be the input rep's count field
        "BEGIN { BINFMT = \"i64 rep4(i64 f64)\" ; OUTFMT = \"i64 rep4(i64 f64)\" } { writebin $1, $1 }\n",
        % element layouts must match even in loop mode (input s8 vs
        % output lps8 differ on the wire)
        "BEGIN { BINFMT = \"i64 rep4(s8 i64)\" ; OUTFMT = \"i64 rep4(lps8 i64)\" } { writebin $1, $2 }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_rep_filter_is_byte_exact) :-
    % A stream filter: guarded records pass through byte-identical
    % (count + live elements only -- the writer never pads).
    Keep = [rec(1, [e(5, 1.5), e(20, 2.5)]), rec(3, [])],
    Drop = [rec(-2, [e(9, 1.5)])],
    Input = [rec(1, [e(5, 1.5), e(20, 2.5)]), rec(-2, [e(9, 1.5)]), rec(3, [])],
    run_rpw_smoke("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" ; OUTFMT = \"i64 rep4(i64 f64)\" } $1 > 0 { writebin $1, $2 }\n",
        Input, OutBytes),
    records_bytes(Keep, ExpectedBytes),
    records_bytes(Drop, DropBytes),
    assertion(OutBytes == ExpectedBytes),
    assertion(OutBytes \== DropBytes).

test(rep_lps_elements_write_in_a_loop) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep4(lps8 i64)\" ; OUTFMT = \"i64 rep4(lps8 i64)\" } $1 > 0 { writebin $1, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % a writer-side loop: head phi, live length via strnlen, per-field
    % fwrites -- and no bulk element write
    assertion(once(sub_atom(DriverIR, _, _, _, '_j = phi i64 [ 1, %'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@strnlen(i8* %'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_pwr = call i64 @fwrite'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_ewr = call i64 @fwrite')),
    !.

test(surface_rep_lps_filter_is_byte_exact) :-
    % Same byte-exact filter contract as the fixed-element case, with
    % variable-size elements: name lengths 3, 0, and the full cap 8.
    Keep = [rec2(1, [f("hot", 5), f("", 7)]), rec2(4, [f("full8chr", 2)])],
    Input = [rec2(1, [f("hot", 5), f("", 7)]), rec2(-9, [f("drop", 1)]),
             rec2(4, [f("full8chr", 2)])],
    run_rpw2_smoke("BEGIN { BINFMT = \"i64 rep4(lps8 i64)\" ; OUTFMT = \"i64 rep4(lps8 i64)\" } $1 > 0 { writebin $1, $2 }\n",
        Input, OutBytes),
    records2_bytes(Keep, ExpectedBytes),
    assertion(OutBytes == ExpectedBytes).

test(surface_rep_passthrough_in_union_arm) :-
    % Composes with case blocks: arm 0 carries the rep, its rule
    % passes it through with a rewritten leading field.
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 rep2(i64) | i64)\" ; OUTFMT = \"i64 rep2(i64)\" } TAG == 0 { writebin NR, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_ewr = call i64 @fwrite(i8* %'))),
    !.

:- end_tests(plawk_rep_writer).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5, 0x3FF8000000000000).
double_bits(2.5, 0x4004000000000000).

% rec(Id, Elems): i64 id, i64 count, then per element i64 + f64
write_rpw_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        write_rpw_stream(Out, Recs),
        close(Out)).

write_rpw_stream(Out, Recs) :-
    forall(member(rec(Id, Elems), Recs),
        ( write_i64_le(Out, Id),
          length(Elems, Count),
          write_i64_le(Out, Count),
          forall(member(e(V, F), Elems),
              ( write_i64_le(Out, V),
                double_bits(F, Bits),
                write_i64_le(Out, Bits) )) )).

% rec2(Id, Elems): i64 id, i64 count, then per element lps8 + i64
write_rpw2_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec2(Id, Elems), Recs),
            ( write_i64_le(Out, Id),
              length(Elems, Count),
              write_i64_le(Out, Count),
              forall(member(f(S, V), Elems),
                  ( string_codes(S, Codes),
                    length(Codes, Len),
                    write_i64_le(Out, Len),
                    forall(member(C, Codes), put_byte(Out, C)),
                    write_i64_le(Out, V) )) )),
        close(Out)).

records2_bytes(Recs, Bytes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_rep_writer_expect2.bin', Path),
    write_rpw2_records(Path, Recs),
    setup_call_cleanup(
        open(Path, read, In, [type(binary)]),
        read_string(In, _, Bytes),
        close(In)).

run_rpw2_smoke(Source, Recs, OutBytes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_rep_writer2', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_rpw_marker/0 ],
        [module_name('plawk_rep_writer2')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(BuildOutS)), stderr(std), process(BuildPid)]),
    read_string(BuildOutS, _, BuildOut),
    close(BuildOutS),
    process_wait(BuildPid, BuildStatus),
    ( BuildStatus == exit(0)
    -> true
    ;  format(user_error, "~n[plawk rep writer2 build output]~n~w~n", [BuildOut]),
       throw(plawk_rep_writer2_build_failed(BuildStatus))
    ),
    write_rpw2_records(InputPath, Recs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, OutBytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    !.

% the byte string a record list occupies on the wire
records_bytes(Recs, Bytes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_rep_writer_expect.bin', Path),
    write_rpw_records(Path, Recs),
    setup_call_cleanup(
        open(Path, read, In, [type(binary)]),
        read_string(In, _, Bytes),
        close(In)).

run_rpw_smoke(Source, Recs, OutBytes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_rep_writer', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_rpw_marker/0 ],
        [module_name('plawk_rep_writer')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(BuildOutS)), stderr(std), process(BuildPid)]),
    read_string(BuildOutS, _, BuildOut),
    close(BuildOutS),
    process_wait(BuildPid, BuildStatus),
    ( BuildStatus == exit(0)
    -> true
    ;  format(user_error, "~n[plawk rep writer build output]~n~w~n", [BuildOut]),
       throw(plawk_rep_writer_build_failed(BuildStatus))
    ),
    write_rpw_records(InputPath, Recs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    set_stream(Stdout, type(binary)),
    read_string(Stdout, _, OutBytes),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    !.
