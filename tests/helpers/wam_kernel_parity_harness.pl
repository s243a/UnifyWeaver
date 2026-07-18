:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% wam_kernel_parity_harness.pl — shared helpers for recursive-kernel
% contract parity suites (TC2/TD3/TPD4/TSPD5/WSP3/ASTAR4).
%
% Reuses smoke_paths for writable tmp roots. Does not implement any
% graph algorithm; parity suites keep independent Prolog oracles and
% generated-runtime drivers locally.

:- module(wam_kernel_parity_harness,
          [ dotnet_available/0,
            gcc_available/0,
            cargo_available/0,
            elixir_available/0,
            go_available/0,
            scala_available/0,
            rscript_available/0,
            ghc_available/0,
            llvm_tools_available/0,
            executable_available/1,
            tmp_dir/2,
            read_file_string/2,
            write_file_string/2,
            run_dotnet_build/3,
            run_dotnet_run/3,
            dotnet_status_exit/2
          ]).

:- use_module(library(filesex),
              [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('smoke_paths', [tmp_root/1, clean_dir/1]).

toolchain_available(Cmd, Args) :-
    catch(
        ( process_create(path(Cmd), Args,
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

dotnet_available :- toolchain_available(dotnet, ['--version']).
gcc_available     :- toolchain_available(gcc, ['--version']).
cargo_available   :- toolchain_available(cargo, ['--version']).
elixir_available  :- toolchain_available(elixir, ['--version']).
go_available      :- toolchain_available(go, [version]).
scala_available   :- toolchain_available(scalac, ['-version']).
rscript_available :- toolchain_available('Rscript', ['--version']).
ghc_available     :- toolchain_available(ghc, ['--version']).

executable_available(Name) :-
    toolchain_available(Name, ['--version']).

llvm_tools_available :-
    executable_available(llc),
    executable_available(clang).

%! tmp_dir(+Tag, -Dir) is det.
%  Unique writable directory under smoke_paths:tmp_root/1.
tmp_dir(Tag, Dir) :-
    tmp_root(Root),
    get_time(T),
    Stamp is round(T * 1000000),
    format(atom(Leaf), 'uw_wam_parity_~w_~w', [Tag, Stamp]),
    directory_file_path(Root, Leaf, Dir),
    clean_dir(Dir),
    make_directory_path(Dir).

read_file_string(Path, String) :-
    read_file_to_string(Path, String, []).

write_file_string(Path, String) :-
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, String),
        close(Out)).

run_dotnet_build(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['build', '--nologo', '-v', 'q', '-c', 'Release'],
            [cwd(Dir),
             environment([
                 'DOTNET_NOLOGO'='1',
                 'DOTNET_ROLL_FORWARD'='Major'
             ]),
             stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, Status),
          dotnet_status_exit(Status, Exit),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

run_dotnet_run(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['run', '--no-build', '-c', 'Release', '--no-launch-profile', '--'],
            [cwd(Dir),
             environment([
                 'DOTNET_NOLOGO'='1',
                 'DOTNET_ROLL_FORWARD'='Major'
             ]),
             stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, Status),
          dotnet_status_exit(Status, Exit),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

dotnet_status_exit(exit(Code), Code).
dotnet_status_exit(killed(Signal), Code) :-
    Code is 128 + Signal.
