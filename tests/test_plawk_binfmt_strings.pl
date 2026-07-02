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

:- dynamic user:plawk_binstr_marker/0.

user:plawk_binstr_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_binfmt_strings, [condition(clang_available)]).

test(string_field_layout_offsets) :-
    plawk_parse_string("BEGIN { BINFMT = \"s8 f64 i64\" } { print $2, $3 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % record size = 8 + 8 + 8 = 24; f64 at 8, i64 at 16.
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 24, i8* %rec'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_fp = getelementptr i8, i8* %rec, i64 8'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_fp = getelementptr i8, i8* %rec, i64 16'))),
    !.

test(string_field_print_uses_strnlen_slice) :-
    plawk_parse_string("BEGIN { BINFMT = \"s8 i64\" } { print $1, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@strnlen('))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 8)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'read_line')),
    !.

test(string_eq_guard_uses_memcmp_and_nul_check) :-
    plawk_parse_string("BEGIN { BINFMT = \"s8 i64\" } $1 == \"ERR\" { c++ } END { print c }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@memcmp('))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_nul_ok = icmp eq i8 '))),
    !.

test(full_width_key_skips_nul_check) :-
    plawk_parse_string("BEGIN { BINFMT = \"s8 i64\" } $1 == \"ERRINGXX\" { c++ } END { print c }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@memcmp('))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_nul_ok')),
    !.

test(oversized_key_is_statically_false) :-
    plawk_parse_string("BEGIN { BINFMT = \"s4 i64\" } $1 == \"TOOLONG\" { c++ } END { print c }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '= icmp eq i1 true, false'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@memcmp(')),
    !.

test(binfmt_string_rejects_text_shaped_programs) :-
    Rejects = [
        % string field in arithmetic
        "BEGIN { BINFMT = \"s8 i64\" } { c += $1 } END { print c }\n",
        % numeric compare on a string field
        "BEGIN { BINFMT = \"s8 i64\" } $1 > 5 { c++ } END { print c }\n",
        % string equality on an i64 field
        "BEGIN { BINFMT = \"s8 i64\" } $2 == \"x\" { c++ } END { print c }\n",
        % string field as an i64 assoc key
        "BEGIN { BINFMT = \"s8 i64\" } { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        % float() coercion of a string field
        "BEGIN { BINFMT = \"s8 f64\" } { sum += float($1) } END { print sum }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_string_field_print_and_guard) :-
    run_binstr_smoke("BEGIN { BINFMT = \"s8 i64\" } $1 == \"ERR\" { c += $2 } { print $1, $2 } END { print c }\n",
        [rec("ERR", 5), rec("OK", 1), rec("ERR", 7), rec("ERRINGXX", 9)],
        "ERR 5\nOK 1\nERR 7\nERRINGXX 9\n12\n").

test(surface_full_width_key_and_mixed_layout) :-
    % s8 f64 i64: full-width key match, guard combinator, f64 accumulation.
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binfmt_strings', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string("BEGIN { BINFMT = \"s8 f64 i64\" } $1 == \"ERRINGXX\" && $3 > 1 { hits++ ; sum += float($2) } END { print hits, sum }\n", Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath),
    setup_call_cleanup(
        open(InputPath, write, Out, [type(binary)]),
        ( write_sfield(Out, "ERRINGXX", 8), write_f64_bits(Out, 0x4004000000000000), write_i64_le(Out, 3),   % 2.5
          write_sfield(Out, "ERRINGXX", 8), write_f64_bits(Out, 0x3FF0000000000000), write_i64_le(Out, 0),   % 1.0
          write_sfield(Out, "ERR", 8),      write_f64_bits(Out, 0x4022000000000000), write_i64_le(Out, 9),   % 9.0
          write_sfield(Out, "ERRINGXX", 8), write_f64_bits(Out, 0x3FE0000000000000), write_i64_le(Out, 2) ), % 0.5
        close(Out)),
    run_probe(BinPath, OutStr),
    assertion(OutStr == "2 3\n"),
    !.

test(surface_empty_string_field_prints_empty) :-
    run_binstr_smoke("BEGIN { BINFMT = \"s4 i64\" } { print $1, $2 }\n",
        [rec("", 7), rec("ab", 9)],
        " 7\nab 9\n").

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_f64_bits(Out, Bits) :-
    write_i64_le(Out, Bits).

write_sfield(Out, String, Width) :-
    string_codes(String, Codes),
    length(Codes, Len),
    Len =< Width,
    forall(member(C, Codes), put_byte(Out, C)),
    Pad is Width - Len,
    forall(between(1, Pad, _), put_byte(Out, 0)).

write_recs(Path, Width, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(S, V), Recs),
            ( write_sfield(Out, S, Width), write_i64_le(Out, V) )),
        close(Out)).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_binstr_marker/0 ],
        [module_name('plawk_binfmt_strings')], LLPath),
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
    ;  format(user_error, "~n[plawk binfmt strings build output]~n~w~n", [BuildOut]),
       throw(plawk_binfmt_strings_build_failed(Status))
    ).

run_probe(BinPath, OutStr) :-
    process_create(BinPath, [], [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk binfmt strings run output]~n~w~n", [OutStr]),
       throw(plawk_binfmt_strings_run_failed(Status))
    ).

run_binstr_smoke(Source, Recs, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binfmt_strings', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath),
    % width comes from the first sN in the source
    ( sub_string(Source, _, _, _, "s8") -> Width = 8 ; Width = 4 ),
    write_recs(InputPath, Width, Recs),
    run_probe(BinPath, OutStr),
    assertion(OutStr == ExpectedOutput),
    !.

:- end_tests(plawk_binfmt_strings).
