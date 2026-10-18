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

:- dynamic user:plawk_rls_marker/0.

user:plawk_rls_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_rep_strings, [condition(clang_available)]).

test(rep_with_lps_elements_reads_element_by_element) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep4(lps8 i64)\" } { foreach { total += $2 } } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % 8 id + 8 count + 4*16 elements + 16 staging
    assertion(once(sub_atom(DriverIR, _, _, _, 'malloc(i64 96)'))),
    % a per-element read loop, not one bulk region read
    assertion(once(sub_atom(DriverIR, _, _, _, 'vr_f1_lh:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_f1_j = phi i64 [ 1, %vr_f1_read ]'))),
    % the element's lps string parses through the shared length scratch
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_f1_e_f0_lstatus'))),
    % and its i64 sibling reads relative to the current element base
    assertion(once(sub_atom(DriverIR, _, _, _, 'i8* %vr_f1_eb, i64 8'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_bytes = mul')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(rep_with_fixed_elements_keeps_bulk_read) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep4(s8 i64)\" } { foreach { total += $2 } } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_bytes = mul'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'vr_f1_lh:')),
    !.

test(toplevel_s_field_reads_its_exact_width) :-
    plawk_parse_string("BEGIN { BINFMT = \"s4 lps8\" } $1 == \"abcd\" { c++ } END { print c }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % 4 wire bytes for s4, not a hardcoded 8
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 4, i8* %vr_f0_dst'))),
    !.

test(surface_foreach_string_guard_aggregates) :-
    run_rls_smoke("BEGIN { BINFMT = \"i64 rep4(lps8 i64)\" } $1 > 0 { recs++ } { foreach { if ($1 == \"hot\") { hits++ }; total += $2 } } END { print recs, hits, total }\n",
        [rec(1, [e("hot", 5), e("cold", 7)]),
         rec(2, []),
         rec(3, [e("warm", 1), e("hot", 2), e("hotx", 3), e("full8chr", 4)])],
        "3 2 22\n").

test(surface_foreach_prints_element_strings) :-
    run_rls_smoke("BEGIN { BINFMT = \"i64 rep4(lps8 i64)\" } { foreach { print $1, $2 } } END { print done }\n",
        [rec(1, [e("hot", 5), e("", 7), e("full8chr", 9)])],
        "hot 5\n 7\nfull8chr 9\n0\n").

test(surface_toplevel_s_field_equality) :-
    build_rls_probe("BEGIN { BINFMT = \"s4 lps8\" } $1 == \"abcd\" { c++ } END { print c }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_raw(InputPath, [bytes("abcd"), i64(1), bytes("x"),
                          bytes("efgh"), i64(2), bytes("yy")]),
    run_capture_raw(BinPath, Out, exit(0)),
    assertion(Out == "1\n"),
    !.

test(surface_rep_lps_error_paths) :-
    build_rls_probe("BEGIN { BINFMT = \"i64 rep4(lps8 i64)\" } { foreach { total += $2 } } END { print total }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    % element string longer than its cap (9 > 8)
    write_raw(InputPath, [i64(1), i64(1), i64(9), bytes("ninechars"), i64(5)]),
    run_capture_raw(BinPath, _, exit(11)),
    % count above the rep cap (5 > 4)
    write_raw(InputPath, [i64(1), i64(5)]),
    run_capture_raw(BinPath, _, exit(11)),
    % truncated element payload (claims 6, has 2)
    write_raw(InputPath, [i64(1), i64(1), i64(6), bytes("ab")]),
    run_capture_raw(BinPath, _, exit(11)),
    % EOF between elements (count 2, one element present)
    write_raw(InputPath, [i64(1), i64(2), i64(3), bytes("abc"), i64(5)]),
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

% one record: i64 id, i64 count, then per element an lps8 (i64 length +
% payload) and an i64 value
write_rls_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(Id, Elems), Recs),
            ( write_i64_le(Out, Id),
              length(Elems, Count),
              write_i64_le(Out, Count),
              forall(member(e(S, V), Elems),
                  ( string_codes(S, Codes),
                    length(Codes, Len),
                    write_i64_le(Out, Len),
                    forall(member(C, Codes), put_byte(Out, C)),
                    write_i64_le(Out, V) )) )),
        close(Out)).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_rls_marker/0 ],
        [module_name('plawk_rep_strings')], LLPath),
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
    ;  format(user_error, "~n[plawk rep strings build output]~n~w~n", [BuildOut]),
       throw(plawk_rep_strings_build_failed(Status))
    ).

build_rls_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_rep_strings', Dir),
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

run_rls_smoke(Source, Recs, ExpectedOutput) :-
    build_rls_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_rls_records(InputPath, Recs),
    run_capture_raw(BinPath, OutStr, exit(0)),
    assertion(OutStr == ExpectedOutput),
    !.

:- end_tests(plawk_rep_strings).
