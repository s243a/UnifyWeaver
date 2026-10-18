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

:- dynamic user:plawk_transient_marker/0.
:- dynamic user:plawk_whole_line_error/1.

user:plawk_transient_marker.

% Matches by atom identity against the whole record, so it only works
% if $0 foreign-call arguments intern the transient line to a real atom.
user:plawk_whole_line_error('ERROR disk full').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_transient_line_records, [condition(clang_available)]).

test(driver_reads_records_transiently) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_stream_read_line_transient_value(%Value %handle)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_stream_read_line_value(%Value %handle)')),
    !.

test(foreign_whole_record_arg_interns_line) :-
    plawk_parse_string("plawk_whole_line_error($0) { hits++ } END { print hits }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', [wam_vm(100, 20)], DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_len = call i64 @strlen(i8* %line_s)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_id = call i64 @wam_intern_atom(i8* %line_s, i64 %'))),
    !.

% Long line followed by short lines: a stale NUL terminator or reused
% buffer residue would corrupt the later prints.
test(surface_line_buffer_reuse_prints_each_record_exactly) :-
    run_transient_print_smoke("{ print $0 }\n",
        "first-very-long-line-with-plenty-of-payload-bytes-here\nab\nlonger again\n",
        "first-very-long-line-with-plenty-of-payload-bytes-here\nab\nlonger again\n").

% Field-slice assoc keys intern real atoms, so counts keyed off the
% mutating transient buffer must still deduplicate across records.
test(surface_assoc_keys_survive_buffer_reuse) :-
    run_transient_print_smoke("{ counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "ERROR disk\nWARN cpu\nERROR net\n",
        "2 1\n").

test(surface_foreign_dollar0_guard_matches_by_atom_identity) :-
    run_transient_foreign_smoke("plawk_whole_line_error($0) { hits++ } END { print hits }\n",
        "ERROR disk full\nWARN cpu hot\nERROR disk full\n",
        "2\n").

% 500k unique ~100-byte lines (~50 MB of line text) under a 60 MB
% address-space cap: interning every line would blow the limit, the
% transient buffer keeps memory constant.
test(surface_unbounded_unique_stream_runs_in_constant_memory) :-
    build_transient_binary("{ total += $3 } END { print total }\n", Dir, BinPath),
    directory_file_path(Dir, 'unique_wide.txt', InputPath),
    Count = 500000,
    write_wide_lines(InputPath, Count),
    format(atom(Cmd), 'ulimit -v 61440; exec ~w ~w', [BinPath, InputPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    Expected is Count * (Count - 1) // 2,
    format(atom(ExpectedLine), '~w~n', [Expected]),
    atom_string(ExpectedLine, ExpectedStr),
    assertion(OutStr == ExpectedStr),
    !.

build_transient_binary(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_transient_line_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_transient_marker/0 ],
        [module_name('plawk_transient_line_records')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    compile_probe(LLPath, BinPath).

compile_probe(LLPath, BinPath) :-
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, BuildOut),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[transient line records build output]~n~w~n", [BuildOut]),
       throw(plawk_transient_line_records_build_failed(Status))
    ).

write_wide_lines(Path, Count) :-
    Limit is Count - 1,
    length(PadCodes, 80),
    maplist(=(0'x), PadCodes),
    atom_codes(Pad, PadCodes),
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(between(0, Limit, I),
            format(Out, 'k~w ~w ~w~n', [I, Pad, I])),
        close(Out)).

run_transient_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_transient_line_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_transient_marker/0 ],
        [module_name('plawk_transient_line_records')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    compile_probe(LLPath, BinPath),
    run_and_compare(BinPath, ExpectedOutput).

run_transient_foreign_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_transient_line_records', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    plawk_parse_string(Source, Program),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_whole_line_error/1,
          user:plawk_transient_marker/0
        ],
        [module_name('plawk_transient_line_records')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_program_native_driver_ir(Program, InputPath,
        [wam_vm(InstrCount, LabelCount)], DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    compile_probe(LLPath, BinPath),
    run_and_compare(BinPath, ExpectedOutput).

run_and_compare(BinPath, ExpectedOutput) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[transient line records run output]~n~w~n", [OutStr]),
       throw(plawk_transient_line_records_run_failed(Status))
    ),
    !.

:- end_tests(plawk_transient_line_records).
