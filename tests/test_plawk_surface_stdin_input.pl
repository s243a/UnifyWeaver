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

:- dynamic user:plawk_stdin_marker/0.

user:plawk_stdin_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_stdin_input, [condition(clang_available)]).

test(stdin_driver_emits_argv_main) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv, DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'define i32 @main(i32 %argc, i8** %argv)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_stream_open_fd_value(i64 0)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.wam_stream_stdin_dash = private constant [2 x i8] c"-\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@.wam_stream_input_path')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(stdin_driver_keeps_check_handle_value_phi_predecessor) :-
    % Scalar loop phis name %check_handle_value as the loop entry
    % predecessor; the argv/stdin main must preserve that block label.
    plawk_parse_string("$1 == \"ERROR\" { count++ } END { print count }\n", Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv, DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%slot_0 = phi i64 [0, %check_handle_value], [%next_slot_0, %continue_loop]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_stream_open_value(%Value %path)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_stream_open_fd_value(i64 0)'))),
    !.

test(concrete_path_driver_unchanged) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'define i32 @main() {'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.wam_stream_input_path'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_stream_open_fd_value')),
    !.

test(stdin_print_rule_reads_all_input_modes) :-
    run_stdin_smoke("$1 == \"ERROR\" { print $0 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "ERROR disk full\nERROR net down\n").

test(stdin_scalar_end_reads_all_input_modes) :-
    run_stdin_smoke("$1 == \"ERROR\" { count++ } END { print count }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2\n").

test(stdin_assoc_end_reads_all_input_modes) :-
    run_stdin_smoke("{ counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 1\n").

test(stdin_forin_end_reads_all_input_modes) :-
    run_stdin_smoke_sorted("{ counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        "INFO boot ok\nERROR disk full\nERROR net down\n",
        ["ERROR 2", "INFO 1"]).

test(stdin_missing_file_exits_with_open_failure) :-
    build_stdin_binary("$1 == \"ERROR\" { print $0 }\n", _Dir, BinPath),
    format(atom(Cmd), '~w /nonexistent_plawk_input_file', [BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Status),
    assertion(Status == exit(10)),
    !.

build_stdin_binary(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_stdin_input', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, stdin_or_argv, DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    directory_file_path(Dir, 'plawk_surface_stdin_input.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_stdin_marker/0 ],
        [module_name('plawk_surface_stdin_input')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_stdin_input_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, BuildOut),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk stdin input build output]~n~w~n", [BuildOut]),
       throw(plawk_surface_stdin_input_build_failed(Status))
    ).

% One binary, four awk-style invocations: an argv path, stdin
% redirection, the "-" stdin alias, and a shell pipe.
run_stdin_smoke(Source, Input, ExpectedOutput) :-
    build_stdin_binary(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    forall(member(Mode, [argv, redirect, dash, pipe]),
        ( run_stdin_mode(Mode, BinPath, InputPath, OutStr),
          ( OutStr == ExpectedOutput
          -> true
          ;  format(user_error,
                 "~n[plawk stdin input ~w mode output]~nexpected: ~q~ngot: ~q~n",
                 [Mode, ExpectedOutput, OutStr]),
             throw(plawk_surface_stdin_input_mode_failed(Mode))
          )
        )),
    !.

run_stdin_smoke_sorted(Source, Input, ExpectedSortedLines) :-
    build_stdin_binary(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    maplist(atom_string, ExpectedSortedLines, ExpectedStrings0),
    msort(ExpectedStrings0, ExpectedStrings),
    forall(member(Mode, [argv, redirect, dash, pipe]),
        ( run_stdin_mode(Mode, BinPath, InputPath, OutStr),
          split_string(OutStr, "\n", "", Lines0),
          exclude(==(""), Lines0, Lines),
          msort(Lines, SortedLines),
          ( SortedLines == ExpectedStrings
          -> true
          ;  format(user_error,
                 "~n[plawk stdin input ~w mode output]~nexpected: ~q~ngot: ~q~n",
                 [Mode, ExpectedStrings, SortedLines]),
             throw(plawk_surface_stdin_input_mode_failed(Mode))
          )
        )),
    !.

run_stdin_mode(Mode, BinPath, InputPath, OutStr) :-
    stdin_mode_command(Mode, BinPath, InputPath, Cmd),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk stdin input ~w mode output]~n~w~n",
              [Mode, OutStr]),
       throw(plawk_surface_stdin_input_run_failed(Mode, Status))
    ).

stdin_mode_command(argv, BinPath, InputPath, Cmd) :-
    format(atom(Cmd), '~w ~w', [BinPath, InputPath]).
stdin_mode_command(redirect, BinPath, InputPath, Cmd) :-
    format(atom(Cmd), '~w < ~w', [BinPath, InputPath]).
stdin_mode_command(dash, BinPath, InputPath, Cmd) :-
    format(atom(Cmd), '~w - < ~w', [BinPath, InputPath]).
stdin_mode_command(pipe, BinPath, InputPath, Cmd) :-
    format(atom(Cmd), 'cat ~w | ~w', [InputPath, BinPath]).

:- end_tests(plawk_surface_stdin_input).
