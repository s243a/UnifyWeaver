:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% backend_loader.pl - Parallel backend auto-loader
% Discovers and registers all parallel execution backends
%
% Usage:
%   ?- use_module(unifyweaver(core/backend_loader)).
%   ?- load_all_backends.
%   ?- list_backends(Backends).

:- module(backend_loader, [
    load_all_backends/0,
    load_backend/1,
    available_backends/1
]).

:- use_module(library(lists)).

% Import the parallel_backend module for registration
:- use_module(unifyweaver(core/parallel_backend)).

%% ============================================
%% BACKEND DEFINITIONS
%% ============================================

%% backend_definition(+Name, +Module, +Description, +Requirements)
%  Define available backends with their modules and requirements
%
%  Requirements:
%  - tool(ToolName) - External tool must be available
%  - module(ModuleName) - Python/language module must be installed
%  - env(VarName) - Environment variable must be set
%  - any(ReqList) - Any of the requirements must be met
backend_definition(bash_fork, bash_fork_backend,
    'Pure bash fork-based parallelization (no external dependencies)',
    [tool(bash)]).

backend_definition(gnu_parallel, gnu_parallel_backend,
    'GNU Parallel for batch processing',
    [tool(parallel)]).

backend_definition(dask_distributed, dask_distributed_backend,
    'Dask distributed computing (Python)',
    [tool(python3), module(dask)]).

backend_definition(hadoop_streaming, hadoop_streaming_backend,
    'Hadoop Streaming for MapReduce (stdin/stdout)',
    [any([tool(hadoop), env('HADOOP_HOME')])]).

backend_definition(hadoop_native, hadoop_native_backend,
    'Hadoop Native API (Java/Scala/Kotlin in-process)',
    [tool(java), any([tool(hadoop), env('HADOOP_HOME')])]).

backend_definition(spark, spark_backend,
    'Apache Spark (PySpark or Scala)',
    [any([tool('spark-submit'), env('SPARK_HOME')])]).

%% ============================================
%% LOADER PREDICATES
%% ============================================

%% load_all_backends
%  Load and register all available backends
%  Backends with missing requirements are skipped with a warning
load_all_backends :-
    format('[BackendLoader] Discovering backends...~n', []),
    findall(Name, backend_definition(Name, _, _, _), AllBackends),
    length(AllBackends, Total),
    format('[BackendLoader] Found ~w backend definitions~n', [Total]),

    % Try to load each backend
    maplist(try_load_backend, AllBackends, Results),

    % Count successes
    include(=(success), Results, Successes),
    length(Successes, Loaded),
    format('[BackendLoader] Successfully loaded ~w/~w backends~n', [Loaded, Total]),

    % List registered backends
    list_backends(Backends),
    (   Backends = []
    ->  format('[BackendLoader] WARNING: No backends available~n', [])
    ;   format('[BackendLoader] Available backends: ~w~n', [Backends])
    ).

%% try_load_backend(+Name, -Result)
%  Attempt to load a backend, returning success or skipped
try_load_backend(Name, Result) :-
    backend_definition(Name, Module, Description, Requirements),
    (   check_requirements(Requirements)
    ->  catch(
            (   load_backend_module(Name, Module),
                register_backend(Name, Module),
                format('[BackendLoader] Loaded: ~w - ~w~n', [Name, Description]),
                Result = success
            ),
            Error,
            (   format('[BackendLoader] ERROR loading ~w: ~w~n', [Name, Error]),
                Result = error
            )
        )
    ;   format('[BackendLoader] Skipped ~w: requirements not met~n', [Name]),
        Result = skipped
    ).

%% load_backend(+Name)
%  Load a specific backend by name
load_backend(Name) :-
    (   backend_definition(Name, Module, Description, Requirements)
    ->  (   check_requirements(Requirements)
        ->  load_backend_module(Name, Module),
            register_backend(Name, Module),
            format('[BackendLoader] Loaded: ~w - ~w~n', [Name, Description])
        ;   format('[BackendLoader] Cannot load ~w: requirements not met~n', [Name]),
            fail
        )
    ;   format('[BackendLoader] Unknown backend: ~w~n', [Name]),
        fail
    ).

%% available_backends(-Backends)
%  Get list of backends that can be loaded (requirements met)
available_backends(Backends) :-
    findall(backend(Name, Description),
            (   backend_definition(Name, _, Description, Requirements),
                check_requirements(Requirements)
            ),
            Backends).

%% ============================================
%% REQUIREMENT CHECKING
%% ============================================

%% check_requirements(+Requirements)
%  Check if all requirements are met
check_requirements([]) :- !.
check_requirements([Req|Rest]) :-
    check_requirement(Req),
    !,
    check_requirements(Rest).

%% check_requirement(+Requirement)
%  Check a single requirement
check_requirement(tool(Tool)) :-
    !,
    tool_available(Tool).

check_requirement(module(Module)) :-
    !,
    python_module_available(Module).

check_requirement(env(Var)) :-
    !,
    getenv(Var, _).

check_requirement(any(Requirements)) :-
    !,
    member(Req, Requirements),
    check_requirement(Req),
    !.

check_requirement(all(Requirements)) :-
    !,
    check_requirements(Requirements).

%% tool_available(+Tool)
%  Check if a command-line tool is available
tool_available(Tool) :-
    catch(
        (   process_create(path(which), [Tool],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(ExitCode)),
            ExitCode =:= 0
        ),
        _,
        fail
    ).

%% python_module_available(+Module)
%  Check if a Python module is available
python_module_available(Module) :-
    atom_string(Module, ModuleStr),
    format(atom(CheckCmd), 'import ~w', [ModuleStr]),
    catch(
        (   process_create(path(python3), ['-c', CheckCmd],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(ExitCode)),
            ExitCode =:= 0
        ),
        _,
        fail
    ).

%% ============================================
%% MODULE LOADING
%% ============================================

%% load_backend_module(+Name, +Module)
%  Load the Prolog module for a backend
load_backend_module(Name, Module) :-
    % Build module path
    format(atom(ModulePath), 'unifyweaver(core/backends/~w)', [Name]),

    % Try to load the module
    catch(
        use_module(ModulePath),
        Error,
        (   format('[BackendLoader] Failed to load module ~w: ~w~n', [Module, Error]),
            throw(Error)
        )
    ).

%% ============================================
%% AUTO-INITIALIZATION
%% ============================================

%% On module load, register built-in backends that don't require checking
:- initialization(register_builtin_backends, now).

register_builtin_backends :-
    % Always register bash_fork as it has minimal requirements
    catch(
        (   use_module(unifyweaver(core/backends/bash_fork)),
            register_backend(bash_fork, bash_fork_backend)
        ),
        _,
        true
    ).
