% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% streaming_glue.pl — glue for streaming data pipelines.
%
% Implements Phase S1.2 of the streaming-pipelines design:
%   - Resolves declare_target(Pred, rust, [leaf(true), native_crate(X)])
%     declarations: locate the Cargo crate, build it if needed, register
%     the resulting binary.
%   - Resolves declare_target(Pred, python, [leaf(true), script_path(X)])
%     declarations: locate the script.
%   - Composes a producer+consumer pair into a shell pipeline with
%     TSV-over-pipe transport (Phase S1 of the design).
%   - Runs the pipeline with caller-supplied environment variables
%     driving the consumer's filter/key/value configuration.
%
% The consumer-side filter/key/value logic lives in environment
% variables rather than being transpiled from Prolog. Transpiling
% `forall/2` composition into a consumer script is a future milestone
% (declarative-specialization direction in 01-philosophy.md).
%
% See docs/design/cross-target-glue/streaming-pipelines/ for the
% full design.

:- module(streaming_glue, [
    % Build / registration
    ensure_streaming_binary/2,      % +Pred/Arity, -BinaryPath
    ensure_streaming_script/2,      % +Pred/Arity, -ScriptPath

    % Producer + consumer resolution
    resolve_producer/2,             % +Pred/Arity, -Invocation
    resolve_consumer/2,             % +Pred/Arity, -Invocation

    % Python runtime resolution
    resolve_python_exec/2,          % +Pred/Arity, -PythonExec
    detect_python_exec/3,           % +MinVersion, +PipPackages, -Exec

    % Pipeline generation + execution
    generate_streaming_pipeline/4,  % +ProducerPred, +ConsumerPred,
                                    % +PipelineOpts, -ScriptText
    run_streaming_pipeline/4        % +ProducerPred, +ConsumerPred,
                                    % +PipelineOpts, -ExitCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/target_mapping', [
    predicate_target_options/3
]).
:- use_module('native_glue', [
    compile_if_needed/4,
    register_binary/4,
    compiled_binary/3
]).

%% ============================================
%% Path resolution
%% ============================================

%% repo_root(-Dir)
%  Resolve the repository root relative to this file.
%  streaming_glue.pl lives at src/unifyweaver/glue/, so the root is
%  three directories up.
repo_root(Root) :-
    source_file(repo_root(_), ThisFile),
    file_directory_name(ThisFile, GlueDir),       % .../src/unifyweaver/glue
    file_directory_name(GlueDir, UwDir),          % .../src/unifyweaver
    file_directory_name(UwDir, SrcDir),           % .../src
    file_directory_name(SrcDir, Root).            % repo root

%% resolve_crate_dir(+CrateName, -Dir)
%  native_crate(Name) resolves to src/unifyweaver/runtime/rust/<Name>/.
resolve_crate_dir(CrateName, Dir) :-
    repo_root(Root),
    format(atom(Dir), '~w/src/unifyweaver/runtime/rust/~w', [Root, CrateName]).

%% resolve_script_path(+RelPath, -AbsPath)
%  script_path is either absolute or relative to the repo root.
resolve_script_path(RelPath, AbsPath) :-
    (   is_absolute_file_name(RelPath)
    ->  AbsPath = RelPath
    ;   repo_root(Root),
        format(atom(AbsPath), '~w/~w', [Root, RelPath])
    ).

%% ============================================
%% Producer resolution — target-agnostic
%% ============================================
%
%  resolve_producer/2 returns a term describing how to invoke the
%  producer at the shell layer. The shape is:
%
%    producer(Kind, Path, Options)
%
%  where Kind is one of:
%    rust_binary  — Path is a compiled native binary, invoked directly
%    awk_script   — Path is a .awk file, invoked via gawk -f
%
%  Options carry target-specific extras like input_filter(zcat) for
%  AWK (which can't read .gz natively).
%
%  Adding a new target (haskell, C, etc.) means adding another clause
%  here; the pipeline generator branches on Kind.

%% resolve_producer(+Pred/Arity, -Invocation)
resolve_producer(Pred/Arity, producer(rust_binary, BinaryPath, Options)) :-
    predicate_target_options(Pred/Arity, rust, Options),
    option(leaf(true), Options),
    option(native_crate(CrateName), Options),
    !,
    resolve_crate_dir(CrateName, CrateDir),
    format(atom(CargoToml), '~w/Cargo.toml', [CrateDir]),
    compile_if_needed(Pred/Arity, rust, CargoToml, BinaryPath0),
    % compile_if_needed guesses a binary path based on Cargo.toml's
    % basename, which doesn't match the crate's actual [[bin]] name.
    % Fix up to the real path.
    format(atom(BinaryPath), '~w/target/release/~w', [CrateDir, CrateName]),
    (   BinaryPath0 == BinaryPath
    ->  true
    ;   register_binary(Pred/Arity, rust, BinaryPath, [])
    ).

resolve_producer(Pred/Arity, producer(awk_script, ScriptPath, Options)) :-
    predicate_target_options(Pred/Arity, awk, Options),
    option(leaf(true), Options),
    option(script_path(RelPath), Options),
    !,
    resolve_script_path(RelPath, ScriptPath).

%% ensure_streaming_binary(+Pred/Arity, -BinaryPath)
%  Legacy wrapper kept for backward compatibility.  Prefer resolve_producer/2.
ensure_streaming_binary(Pred/Arity, BinaryPath) :-
    resolve_producer(Pred/Arity, producer(rust_binary, BinaryPath, _)).

%% ============================================
%% Consumer resolution — target-agnostic
%% ============================================
%
%  resolve_consumer/2 parallels resolve_producer/2.  Returns:
%
%    consumer(Kind, Path, Options)
%
%  Kinds:
%    python_script — Path is a .py file, invoked via a detected
%                    Python interpreter (see resolve_python_exec/2)
%    dotnet_dll    — Path is a compiled .NET DLL, invoked via
%                    `dotnet path.dll`.  Project is built with
%                    `dotnet build -c Release` if needed.
%
%  Adding another consumer target (JVM, Ruby, ...) means adding
%  another clause here plus one in format_consumer_cmd/4.

%% resolve_consumer(+Pred/Arity, -Invocation)
resolve_consumer(Pred/Arity, consumer(python_script, ScriptPath, Options)) :-
    predicate_target_options(Pred/Arity, python, Options),
    option(leaf(true), Options),
    option(script_path(RelPath), Options),
    !,
    resolve_script_path(RelPath, ScriptPath).

resolve_consumer(Pred/Arity, consumer(dotnet_dll, DllPath, Options)) :-
    predicate_target_options(Pred/Arity, csharp, Options),
    option(leaf(true), Options),
    option(project_dir(RelDir), Options),
    !,
    resolve_script_path(RelDir, AbsDir),
    ensure_dotnet_built(AbsDir, DllPath).

%% ensure_streaming_script(+Pred/Arity, -ScriptPath)
%  Legacy wrapper: python-only.  Prefer resolve_consumer/2.
ensure_streaming_script(Pred/Arity, ScriptPath) :-
    resolve_consumer(Pred/Arity, consumer(python_script, ScriptPath, _)).

%% ensure_dotnet_built(+ProjectDir, -DllPath)
%  Build `dotnet build -c Release` if the DLL is missing or older
%  than the .csproj file.  Returns the path to the compiled DLL.
%  Expects exactly one .csproj in the directory.
ensure_dotnet_built(ProjectDir, DllPath) :-
    find_csproj(ProjectDir, CsprojPath, BaseName),
    find_dll_path(ProjectDir, BaseName, DllPath),
    (   exists_file(DllPath),
        exists_file(CsprojPath),
        time_file(DllPath, BinTime),
        time_file(CsprojPath, SrcTime),
        BinTime >= SrcTime
    ->  true
    ;   format(user_error,
               '[streaming_glue] dotnet build -c Release (~w)~n', [ProjectDir]),
        format(atom(Cmd), 'cd "~w" && dotnet build -c Release', [ProjectDir]),
        shell(Cmd, 0)
    ).

find_csproj(ProjectDir, CsprojPath, BaseName) :-
    directory_files(ProjectDir, Files),
    member(F, Files),
    file_name_extension(BaseName, csproj, F),
    !,
    format(atom(CsprojPath), '~w/~w', [ProjectDir, F]).

%% find_dll_path: locate bin/Release/netX.Y/<Base>.dll.
%  The framework subdir (net8.0, net9.0, ...) is discovered by listing.
find_dll_path(ProjectDir, BaseName, DllPath) :-
    format(atom(ReleaseDir), '~w/bin/Release', [ProjectDir]),
    (   exists_directory(ReleaseDir),
        directory_files(ReleaseDir, Entries),
        member(Entry, Entries),
        sub_atom(Entry, 0, _, _, 'net'),
        format(atom(TargetDir), '~w/~w', [ReleaseDir, Entry]),
        exists_directory(TargetDir)
    ->  format(atom(DllPath), '~w/~w.dll', [TargetDir, BaseName])
    ;   % Not yet built — guess a default path; the build step will
        % populate it.  We assume net9.0 as a sensible modern default.
        format(atom(DllPath),
               '~w/bin/Release/net9.0/~w.dll', [ProjectDir, BaseName])
    ).

%% ============================================
%% Python runtime resolution
%% ============================================
%
%  Declarations in the predicate's target options:
%    python_exec(Exec)               — explicit interpreter (overrides all)
%    python_min_version(Version)     — e.g. '3.9'
%    python_max_version(Version)     — optional upper bound
%    pip_packages(List)              — e.g. [lmdb, numpy] — must be importable
%
%  The glue probes common interpreters (python3.13..python3.9, python3)
%  and picks the first that satisfies every declared requirement.

%% resolve_python_exec(+Pred/Arity, -Exec)
%  Pick a Python interpreter that satisfies the predicate's declared
%  requirements. If python_exec/1 is in the declaration, use that
%  verbatim.
resolve_python_exec(Pred/Arity, Exec) :-
    predicate_target_options(Pred/Arity, python, Options),
    (   option(python_exec(Exec0), Options)
    ->  Exec = Exec0
    ;   option(python_min_version(MinV), Options, '3.0'),
        option(pip_packages(Pkgs), Options, []),
        detect_python_exec(MinV, Pkgs, Exec)
    ).

%% detect_python_exec(+MinVersion, +PipPackages, -Exec)
%  Probe common Python interpreters and return the first one that:
%    - has version >= MinVersion
%    - can import every package in PipPackages
%  Fails if none match.
detect_python_exec(MinVersion, PipPackages, Exec) :-
    % Note: python3.X parses as dict syntax in SWI — must quote as atoms.
    Candidates = ['python3.13', 'python3.12', 'python3.11',
                  'python3.10', 'python3.9', python3],
    member(Exec, Candidates),
    python_exec_satisfies(Exec, MinVersion, PipPackages),
    !.

%% python_exec_satisfies(+Exec, +MinVersion, +PipPackages)
python_exec_satisfies(Exec, MinVersion, PipPackages) :-
    python_exec_version(Exec, Version),
    version_at_least(Version, MinVersion),
    forall(member(Pkg, PipPackages),
           python_can_import(Exec, Pkg)).

%% python_exec_version(+Exec, -Version)
%  Returns e.g. '3.9' for python3.9.  Fails if interpreter not found.
python_exec_version(Exec, Version) :-
    format(atom(Cmd),
           '~w -c "import sys; print(\'%d.%d\' % sys.version_info[:2])" 2>/dev/null',
           [Exec]),
    catch(
        setup_call_cleanup(
            open(pipe(Cmd), read, S),
            read_string(S, _, Raw),
            close(S)),
        _, fail
    ),
    split_string(Raw, "", "\n \t", [VerStr|_]),
    VerStr \= "",
    atom_string(Version, VerStr).

%% python_can_import(+Exec, +Package)
python_can_import(Exec, Package) :-
    format(atom(Cmd), '~w -c "import ~w" 2>/dev/null', [Exec, Package]),
    shell(Cmd, 0).

%% version_at_least(+Version, +Required)
%  Both in "MAJOR.MINOR" form. Lexicographic on integer components.
version_at_least(Version, Required) :-
    parse_version(Version, VMaj, VMin),
    parse_version(Required, RMaj, RMin),
    (   VMaj > RMaj
    ;   VMaj =:= RMaj, VMin >= RMin
    ), !.

parse_version(V, Maj, Min) :-
    atom_string(V, S),
    split_string(S, ".", "", Parts),
    Parts = [MajStr, MinStr | _],
    number_string(Maj, MajStr),
    number_string(Min, MinStr).

%% ============================================
%% Pipeline composition
%% ============================================

%% generate_streaming_pipeline(+ProducerPred, +ConsumerPred,
%%                             +PipelineOpts, -ScriptText)
%
%  Produce a bash script that pipes the producer binary's stdout into
%  the consumer script's stdin.
%
%  PipelineOpts keys:
%    producer_args(Args)    List of CLI arg strings for the producer
%    consumer_args(Args)    List of CLI arg strings for the consumer
%    env(Pairs)             List of Name=Value pairs for the consumer env
%    python_exec(Exec)      Python interpreter, defaults to python3
%
generate_streaming_pipeline(Producer, Consumer, Opts, Script) :-
    resolve_producer(Producer, ProducerInv),
    resolve_consumer(Consumer, ConsumerInv),

    option(producer_args(ProdArgs), Opts, []),
    option(consumer_args(ConsArgs), Opts, []),
    option(env(EnvPairs), Opts, []),

    % Python interpreter resolution (used only for python_script consumers):
    %   1. explicit pipeline-level override: python_exec(X) in Opts
    %   2. consumer declaration's python_min_version + pip_packages
    %      satisfied by a detected interpreter
    %   3. fall back to python3
    (   option(python_exec(Override), Opts)
    ->  RealPython = Override
    ;   catch(resolve_python_exec(Consumer, RealPython), _, fail)
    ->  true
    ;   RealPython = python3
    ),

    format_producer_cmd(ProducerInv, ProdArgs, ProdCmd),
    format_consumer_cmd(ConsumerInv, ConsArgs, RealPython, ConsCmd),
    format_env(EnvPairs, EnvStr),

    format(string(Script),
'#!/bin/bash
# Generated by UnifyWeaver streaming_glue
set -euo pipefail
~w~w | ~w
', [EnvStr, ProdCmd, ConsCmd]).

%% format_producer_cmd(+producer(Kind, Path, Opts), +Args, -CmdStr)
%  Render the shell command for a producer based on its Kind. AWK
%  producers get wrapped with a `zcat | gawk -f script` incantation
%  when input_filter(zcat) is declared; the Args become zcat's path
%  rather than script args.
format_producer_cmd(producer(rust_binary, Path, _), Args, Cmd) :-
    format_args(Args, ArgStr),
    format(string(Cmd), '"~w"~w', [Path, ArgStr]).
format_producer_cmd(producer(awk_script, Path, Opts), Args, Cmd) :-
    option(awk_exec(AwkExec), Opts, gawk),
    (   option(input_filter(zcat), Opts),
        Args = [InputPath|_]
    ->  format(string(Cmd), 'zcat "~w" | ~w -f "~w"',
               [InputPath, AwkExec, Path])
    ;   % Plain: awk -f script <args>
        format_args(Args, ArgStr),
        format(string(Cmd), '~w -f "~w"~w', [AwkExec, Path, ArgStr])
    ).

% Note on env format: format_env/2 emits `export NAME="value"; ...`
% lines so the vars are visible to both ends of the pipe. A bare
% `NAME=val cmd1 | cmd2` would only apply to cmd1.

%% format_consumer_cmd(+consumer(Kind, Path, Opts), +Args, +PythonExec, -CmdStr)
%  Render the shell command for a consumer based on its Kind.
%  PythonExec is pre-resolved for python_script consumers and ignored
%  otherwise.
format_consumer_cmd(consumer(python_script, Path, _), Args, PythonExec, Cmd) :-
    format_args(Args, ArgStr),
    format(string(Cmd), '~w "~w"~w', [PythonExec, Path, ArgStr]).
format_consumer_cmd(consumer(dotnet_dll, Path, Options), Args, _, Cmd) :-
    option(dotnet_exec(Exec), Options, dotnet),
    format_args(Args, ArgStr),
    format(string(Cmd), '~w "~w"~w', [Exec, Path, ArgStr]).

format_args([], "").
format_args([A|Rest], Str) :-
    format(string(Head), ' "~w"', [A]),
    format_args(Rest, Tail),
    string_concat(Head, Tail, Str).

format_env([], "").
format_env([Name=Value|Rest], Str) :-
    format(string(Head), 'export ~w="~w"\n', [Name, Value]),
    format_env(Rest, Tail),
    string_concat(Head, Tail, Str).

%% run_streaming_pipeline(+ProducerPred, +ConsumerPred, +PipelineOpts, -ExitCode)
%
%  Generate the pipeline script and execute it via bash -c.  Returns
%  the shell exit code.
run_streaming_pipeline(Producer, Consumer, Opts, ExitCode) :-
    generate_streaming_pipeline(Producer, Consumer, Opts, Script),
    setup_call_cleanup(
        tmp_file_stream(text, TmpFile, Stream),
        (   write(Stream, Script),
            close(Stream),
            format(atom(Cmd), 'bash "~w"', [TmpFile]),
            shell(Cmd, ExitCode)
        ),
        catch(delete_file(TmpFile), _, true)
    ).
