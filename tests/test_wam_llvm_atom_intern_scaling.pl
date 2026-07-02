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

:- dynamic user:plawk_intern_marker/0.

user:plawk_intern_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_atom_intern_scaling, [condition(clang_available)]).

% Regression guard for the O(n^2) intern scan: read_line interns every
% record, so before the FNV hash index a 200k-unique-line stream took
% over two minutes; with it, well under a second. The 60-second bound
% is deliberately loose for slow CI while still catching a quadratic
% relapse (which lands at minutes, not seconds).
test(unique_line_stream_interning_is_not_quadratic) :-
    build_sum_binary(Dir, BinPath),
    directory_file_path(Dir, 'unique.txt', InputPath),
    write_numbered_lines(InputPath, 200000),
    get_time(T0),
    run_binary(BinPath, InputPath, OutStr),
    get_time(T1),
    Elapsed is T1 - T0,
    % sum of 0..199999
    assertion(OutStr == "19999900000\n"),
    ( Elapsed < 60
    -> true
    ;  throw(atom_intern_scaling_regressed(seconds(Elapsed)))
    ),
    !.

% Interning must still deduplicate: repeated lines reuse one dynamic
% atom, and keys equal to statically interned atoms resolve to the
% static ids (counted correctly through the assoc table).
test(repeated_and_static_atoms_still_deduplicate) :-
    build_count_binary(Dir, BinPath),
    directory_file_path(Dir, 'mixed.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, Out, [type(binary)]),
        forall(between(1, 500, _),
            format(Out, 'ERROR disk 1~nWARN cpu 2~n', [])),
        close(Out)),
    run_binary(BinPath, InputPath, OutStr),
    assertion(OutStr == "500 500\n"),
    !.

build_sum_binary(Dir, BinPath) :-
    build_probe_binary("{ total += $3 } END { print total }\n",
        'intern_sum', Dir, BinPath).

build_count_binary(Dir, BinPath) :-
    build_probe_binary("{ counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n",
        'intern_count', Dir, BinPath).

build_probe_binary(Source, Name, Dir, BinPath) :-
    tmp_root(Root),
    atom_concat('uw_wam_llvm_atom_intern_', Name, DirName),
    directory_file_path(Root, DirName, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_intern_marker/0 ],
        [module_name(Name)], LLPath),
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
    ;  format(user_error, "~n[atom intern scaling build output]~n~w~n", [BuildOut]),
       throw(atom_intern_scaling_build_failed(Status))
    ).

write_numbered_lines(Path, Count) :-
    Limit is Count - 1,
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(between(0, Limit, I),
            format(Out, 'k~w comp ~w~n', [I, I])),
        close(Out)).

run_binary(BinPath, InputPath, OutStr) :-
    format(atom(Cmd), '~w ~w', [BinPath, InputPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  throw(atom_intern_scaling_run_failed(Status))
    ).

:- end_tests(wam_llvm_atom_intern_scaling).
