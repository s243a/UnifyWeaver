:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% partitioner.pl - Core partitioning system for batch processing
% Provides plugin-based partitioning strategies for splitting data streams

:- module(partitioner, [
    % Plugin interface
    partitioner_init/3,           % +Strategy, +Config, -Handle
    partitioner_partition/3,      % +Handle, +DataStream, -Partitions
    partitioner_assign/3,         % +Handle, +Item, -PartitionID
    partitioner_cleanup/1,        % +Handle

    % Strategy registration
    register_partitioner/2,       % +StrategyName, +Module
    list_partitioners/1,          % -Strategies

    % Configuration
    set_default_strategy/1,       % +Strategy
    get_default_strategy/1,       % -Strategy

    % Utilities
    count_partitions/2,           % +Partitions, -Count
    get_partition_sizes/2         % +Partitions, -Sizes
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% ============================================
%% PLUGIN REGISTRY
%% ============================================

%% register_partitioner(+StrategyName, +Module)
%  Register a partitioning strategy plugin
%
%  @arg StrategyName Atom identifying the strategy (e.g., fixed_size, hash_based)
%  @arg Module Module implementing the strategy interface
%
%  Strategy modules must implement:
%  - strategy_init(+Config, -Handle)
%  - strategy_partition(+Handle, +DataStream, -Partitions)
%  - strategy_assign(+Handle, +Item, -PartitionID)
%  - strategy_cleanup(+Handle)
:- dynamic registered_partitioner/2.

register_partitioner(StrategyName, Module) :-
    (   atom(StrategyName)
    ->  true
    ;   throw(error(type_error(atom, StrategyName),
                    context(register_partitioner/2, 'Strategy name must be atom')))
    ),
    (   atom(Module)
    ->  true
    ;   throw(error(type_error(atom, Module),
                    context(register_partitioner/2, 'Module name must be atom')))
    ),
    % Remove existing registration if present
    retractall(registered_partitioner(StrategyName, _)),
    % Add new registration
    assertz(registered_partitioner(StrategyName, Module)),
    format('[Partitioner] Registered strategy: ~w (module: ~w)~n', [StrategyName, Module]).

%% list_partitioners(-Strategies)
%  Get list of all registered partitioning strategies
list_partitioners(Strategies) :-
    findall(strategy(Name, Module),
            registered_partitioner(Name, Module),
            Strategies).

%% ============================================
%% DEFAULT STRATEGY CONFIGURATION
%% ============================================

:- dynamic default_partition_strategy/1.

%% set_default_strategy(+Strategy)
%  Set the default partitioning strategy
%
%  @arg Strategy Strategy term (e.g., fixed_size(rows(1000)))
set_default_strategy(Strategy) :-
    retractall(default_partition_strategy(_)),
    assertz(default_partition_strategy(Strategy)).

%% get_default_strategy(-Strategy)
%  Get the default partitioning strategy
get_default_strategy(Strategy) :-
    (   default_partition_strategy(Strategy)
    ->  true
    ;   % Default: fixed size with 1000 rows per partition
        Strategy = fixed_size(rows(1000))
    ).

%% ============================================
%% PARTITIONER LIFECYCLE
%% ============================================

%% partitioner_init(+Strategy, +Config, -Handle)
%  Initialize a partitioner with given strategy and configuration
%
%  @arg Strategy Strategy term (e.g., fixed_size(rows(1000)), hash_based(...))
%  @arg Config Additional configuration options (list of Key=Value)
%  @arg Handle Opaque handle for subsequent operations
%
%  @example Initialize fixed-size partitioner
%    ?- partitioner_init(fixed_size(rows(1000)), [], Handle).
%    Handle = handle(fixed_size, fixed_size_partitioner, state(...)).
partitioner_init(Strategy, Config, Handle) :-
    % Extract strategy name from strategy term
    (   compound(Strategy)
    ->  functor(Strategy, StrategyName, _),
        Strategy =.. [StrategyName|Args]
    ;   atom(Strategy)
    ->  StrategyName = Strategy,
        Args = []
    ;   throw(error(type_error(strategy, Strategy),
                    context(partitioner_init/3, 'Invalid strategy term')))
    ),

    % Look up registered module for this strategy
    (   registered_partitioner(StrategyName, Module)
    ->  true
    ;   throw(error(existence_error(partitioner_strategy, StrategyName),
                    context(partitioner_init/3,
                           'Strategy not registered. Use register_partitioner/2')))
    ),

    % Build strategy-specific configuration
    StrategyConfig = [strategy_args(Args)|Config],

    % Initialize strategy module
    call(Module:strategy_init(StrategyConfig, StrategyState)),

    % Create handle
    Handle = handle(StrategyName, Module, StrategyState).

%% partitioner_partition(+Handle, +DataStream, -Partitions)
%  Partition a data stream into chunks
%
%  @arg Handle Partitioner handle from partitioner_init/3
%  @arg DataStream List of data items to partition
%  @arg Partitions List of partition(ID, Data) terms
%
%  @example Partition data stream
%    ?- partitioner_partition(Handle, [1,2,3,4,5,6,7,8,9,10], Partitions).
%    Partitions = [partition(0, [1,2,3]), partition(1, [4,5,6]), ...].
partitioner_partition(handle(StrategyName, Module, State), DataStream, Partitions) :-
    call(Module:strategy_partition(State, DataStream, Partitions)),
    length(Partitions, NumPartitions),
    format('[Partitioner] ~w: Created ~w partitions from ~w items~n',
           [StrategyName, NumPartitions, _]).

%% partitioner_assign(+Handle, +Item, -PartitionID)
%  Assign a single item to a partition (for streaming)
%
%  @arg Handle Partitioner handle from partitioner_init/3
%  @arg Item Single data item
%  @arg PartitionID Partition identifier (integer)
%
%  @example Assign item to partition
%    ?- partitioner_assign(Handle, row(1, "alice", 25), PartitionID).
%    PartitionID = 3.
partitioner_assign(handle(_StrategyName, Module, State), Item, PartitionID) :-
    call(Module:strategy_assign(State, Item, PartitionID)).

%% partitioner_cleanup(+Handle)
%  Clean up partitioner resources
%
%  @arg Handle Partitioner handle from partitioner_init/3
partitioner_cleanup(handle(StrategyName, Module, State)) :-
    call(Module:strategy_cleanup(State)),
    format('[Partitioner] ~w: Cleaned up~n', [StrategyName]).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% count_partitions(+Partitions, -Count)
%  Count number of partitions
count_partitions(Partitions, Count) :-
    length(Partitions, Count).

%% get_partition_sizes(+Partitions, -Sizes)
%  Get sizes of all partitions
%
%  @arg Partitions List of partition(ID, Data) terms
%  @arg Sizes List of size(ID, Count) terms
get_partition_sizes(Partitions, Sizes) :-
    maplist(partition_size, Partitions, Sizes).

partition_size(partition(ID, Data), size(ID, Count)) :-
    length(Data, Count).

%% ============================================
%% PARTITIONER HANDLE VALIDATION
%% ============================================

%% validate_handle(+Handle)
%  Validate partitioner handle structure
validate_handle(Handle) :-
    (   Handle = handle(StrategyName, Module, _State)
    ->  (   atom(StrategyName)
        ->  true
        ;   throw(error(type_error(atom, StrategyName),
                        context(validate_handle/1, 'Invalid strategy name')))
        ),
        (   atom(Module)
        ->  true
        ;   throw(error(type_error(atom, Module),
                        context(validate_handle/1, 'Invalid module name')))
        )
    ;   throw(error(type_error(partitioner_handle, Handle),
                    context(validate_handle/1, 'Invalid handle structure')))
    ).
