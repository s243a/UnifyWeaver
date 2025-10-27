:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% parallel_backend.pl - Parallel execution backend system
% Provides plugin-based backends for parallel batch processing

:- module(parallel_backend, [
    % Plugin interface
    backend_init/2,              % +Config, -Handle
    backend_execute/4,           % +Handle, +Partitions, +ScriptPath, -Results
    backend_cleanup/1,           % +Handle

    % Backend registration
    register_backend/2,          % +BackendName, +Module
    list_backends/1,             % -Backends

    % Configuration
    set_default_backend/1,       % +Backend
    get_default_backend/1        % -Backend
]).

:- use_module(library(lists)).

%% ============================================
%% PLUGIN REGISTRY
%% ============================================

%% register_backend(+BackendName, +Module)
%  Register a parallel execution backend plugin
%
%  @arg BackendName Atom identifying the backend (e.g., gnu_parallel, bash_fork)
%  @arg Module Module implementing the backend interface
%
%  Backend modules must implement:
%  - backend_init_impl(+Config, -State)
%  - backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  - backend_cleanup_impl(+State)
:- dynamic registered_backend/2.

register_backend(BackendName, Module) :-
    (   atom(BackendName)
    ->  true
    ;   throw(error(type_error(atom, BackendName),
                    context(register_backend/2, 'Backend name must be atom')))
    ),
    (   atom(Module)
    ->  true
    ;   throw(error(type_error(atom, Module),
                    context(register_backend/2, 'Module name must be atom')))
    ),
    % Remove existing registration if present
    retractall(registered_backend(BackendName, _)),
    % Add new registration
    assertz(registered_backend(BackendName, Module)),
    format('[ParallelBackend] Registered backend: ~w (module: ~w)~n', [BackendName, Module]).

%% list_backends(-Backends)
%  Get list of all registered parallel backends
list_backends(Backends) :-
    findall(backend(Name, Module),
            registered_backend(Name, Module),
            Backends).

%% ============================================
%% DEFAULT BACKEND CONFIGURATION
%% ============================================

:- dynamic default_parallel_backend/1.

%% set_default_backend(+Backend)
%  Set the default parallel backend
%
%  @arg Backend Backend term (e.g., gnu_parallel(workers(4)))
set_default_backend(Backend) :-
    retractall(default_parallel_backend(_)),
    assertz(default_parallel_backend(Backend)).

%% get_default_backend(-Backend)
%  Get the default parallel backend
get_default_backend(Backend) :-
    (   default_parallel_backend(Backend)
    ->  true
    ;   % Default: GNU Parallel with 4 workers
        Backend = gnu_parallel(workers(4))
    ).

%% ============================================
%% BACKEND LIFECYCLE
%% ============================================

%% backend_init(+Config, -Handle)
%  Initialize a parallel backend with given configuration
%
%  @arg Config Backend configuration term (e.g., gnu_parallel(workers(4)))
%  @arg Handle Opaque handle for subsequent operations
%
%  @example Initialize GNU Parallel backend
%    ?- backend_init(gnu_parallel(workers(4)), Handle).
%    Handle = handle(gnu_parallel, gnu_parallel_backend, state(...)).
backend_init(Config, Handle) :-
    % Extract backend name from config term
    (   compound(Config)
    ->  functor(Config, BackendName, _),
        Config =.. [BackendName|Args]
    ;   atom(Config)
    ->  BackendName = Config,
        Args = []
    ;   throw(error(type_error(backend_config, Config),
                    context(backend_init/2, 'Invalid backend config term')))
    ),

    % Look up registered module for this backend
    (   registered_backend(BackendName, Module)
    ->  true
    ;   throw(error(existence_error(parallel_backend, BackendName),
                    context(backend_init/2,
                           'Backend not registered. Use register_backend/2')))
    ),

    % Build backend-specific configuration
    BackendConfig = [backend_args(Args)],

    % Initialize backend module
    call(Module:backend_init_impl(BackendConfig, BackendState)),

    % Create handle
    Handle = handle(BackendName, Module, BackendState).

%% backend_execute(+Handle, +Partitions, +ScriptPath, -Results)
%  Execute script on partitions in parallel
%
%  @arg Handle Backend handle from backend_init/2
%  @arg Partitions List of partition(ID, Data) terms
%  @arg ScriptPath Path to bash script to execute
%  @arg Results List of result terms from parallel execution
%
%  @example Execute script on partitions
%    ?- backend_execute(Handle, Partitions, 'script.sh', Results).
%    Results = [result(0, [1,2,3]), result(1, [4,5,6]), ...].
backend_execute(handle(BackendName, Module, State), Partitions, ScriptPath, Results) :-
    length(Partitions, NumPartitions),
    format('[ParallelBackend] ~w: Executing on ~w partitions~n',
           [BackendName, NumPartitions]),
    call(Module:backend_execute_impl(State, Partitions, ScriptPath, Results)),
    format('[ParallelBackend] ~w: Execution complete~n', [BackendName]).

%% backend_cleanup(+Handle)
%  Clean up backend resources
%
%  @arg Handle Backend handle from backend_init/2
backend_cleanup(handle(BackendName, Module, State)) :-
    call(Module:backend_cleanup_impl(State)),
    format('[ParallelBackend] ~w: Cleaned up~n', [BackendName]).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% validate_handle(+Handle)
%  Validate backend handle structure
validate_handle(Handle) :-
    (   Handle = handle(BackendName, Module, _State)
    ->  (   atom(BackendName)
        ->  true
        ;   throw(error(type_error(atom, BackendName),
                        context(validate_handle/1, 'Invalid backend name')))
        ),
        (   atom(Module)
        ->  true
        ;   throw(error(type_error(atom, Module),
                        context(validate_handle/1, 'Invalid module name')))
        )
    ;   throw(error(type_error(backend_handle, Handle),
                    context(validate_handle/1, 'Invalid handle structure')))
    ).
