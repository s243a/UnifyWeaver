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

:- dynamic user:plawk_surface_marker/0.

user:plawk_surface_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_prefix_print, [condition(clang_available)]).

test(parses_prefix_print_rule) :-
    plawk_parse_string("/^ERROR/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(prefix("ERROR"), [print(field(0))])], [])).

test(parses_field_eq_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"), [print(field(0))])], [])).

test(surface_prefix_prints_matching_records) :-
    run_surface_print_smoke("/^ERROR/ { print $0 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "ERROR disk full\nERROR net down\n").

test(surface_field_eq_prints_matching_records) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print $0 }\n",
        "INFO boot ok\nNOTERROR misleading\nWARN cpu hot\nERROR net down\n",
        "ERROR net down\n").

run_surface_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_prefix_print', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    directory_file_path(Dir, 'plawk_surface_prefix_print.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_surface_marker/0 ],
        [module_name('plawk_surface_prefix_print')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_prefix_print_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface prefix print output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_prefix_print_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_prefix_print).
