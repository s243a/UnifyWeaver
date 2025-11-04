:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% fixed_size.pl - Fixed-size partitioning strategy
% Splits data into partitions of fixed row count or byte size

:- module(fixed_size_partitioner, [
    strategy_init/2,
    strategy_partition/3,
    strategy_assign/3,
    strategy_cleanup/1
]).

:- use_module(library(lists)).

%% ============================================
%% STRATEGY IMPLEMENTATION
%% ============================================

%% strategy_init(+Config, -State)
%  Initialize fixed-size partitioning strategy
%
%  Configuration options:
%  - strategy_args([rows(N)])    - N rows per partition (default: 1000)
%  - strategy_args([bytes(B)])   - B bytes per partition
%  - strategy_args([items(I)])   - I items per partition (alias for rows)
%
%  @example Initialize with 500 rows per partition
%    ?- strategy_init([strategy_args([rows(500)])], State).
%    State = state(rows, 500, 0).
strategy_init(Config, State) :-
    % Extract strategy arguments
    (   member(strategy_args(Args), Config)
    ->  true
    ;   Args = [rows(1000)]  % Default: 1000 rows
    ),

    % Parse partitioning mode
    (   Args = [rows(N)]
    ->  Mode = rows,
        Size = N
    ;   Args = [bytes(B)]
    ->  Mode = bytes,
        Size = B
    ;   Args = [items(I)]
    ->  Mode = rows,  % items is alias for rows
        Size = I
    ;   % Default to rows
        Mode = rows,
        Size = 1000
    ),

    % Validate size
    (   integer(Size), Size > 0
    ->  true
    ;   throw(error(domain_error(positive_integer, Size),
                    context(strategy_init/2, 'Partition size must be positive integer')))
    ),

    % Initialize state
    State = state(Mode, Size, 0),  % state(mode, size, current_partition_id)
    format('[FixedSize] Initialized: ~w mode, size ~w~n', [Mode, Size]).

%% strategy_partition(+State, +DataStream, -Partitions)
%  Partition data stream into fixed-size chunks
%
%  @arg State Strategy state from strategy_init/2
%  @arg DataStream List of data items
%  @arg Partitions List of partition(ID, Data) terms
strategy_partition(state(Mode, Size, _), DataStream, Partitions) :-
    (   Mode = rows
    ->  partition_by_rows(DataStream, Size, Partitions)
    ;   Mode = bytes
    ->  partition_by_bytes(DataStream, Size, Partitions)
    ;   throw(error(domain_error(partition_mode, Mode),
                    context(strategy_partition/3, 'Unknown partition mode')))
    ).

%% partition_by_rows(+DataStream, +RowsPerPartition, -Partitions)
%  Split data into fixed-size row chunks
partition_by_rows(DataStream, RowsPerPartition, Partitions) :-
    partition_by_rows_helper(DataStream, RowsPerPartition, 0, Partitions).

partition_by_rows_helper([], _, _, []) :- !.
partition_by_rows_helper(DataStream, RowsPerPartition, PartitionID, [partition(PartitionID, Chunk)|Rest]) :-
    % Take up to RowsPerPartition items
    take_n(DataStream, RowsPerPartition, Chunk, Remaining),
    Chunk \= [],  % Ensure we have data
    !,
    NextID is PartitionID + 1,
    partition_by_rows_helper(Remaining, RowsPerPartition, NextID, Rest).

%% partition_by_bytes(+DataStream, +BytesPerPartition, -Partitions)
%  Split data into fixed-size byte chunks
%
%  Note: Estimates size by converting items to strings
partition_by_bytes(DataStream, BytesPerPartition, Partitions) :-
    partition_by_bytes_helper(DataStream, BytesPerPartition, 0, 0, [], Partitions).

partition_by_bytes_helper([], _, PartitionID, _CurrentSize, CurrentChunk, Partitions) :-
    (   CurrentChunk = []
    ->  Partitions = []
    ;   % Final partition with remaining data
        reverse(CurrentChunk, Chunk),
        Partitions = [partition(PartitionID, Chunk)]
    ).
partition_by_bytes_helper([Item|Rest], BytesPerPartition, PartitionID, CurrentSize, CurrentChunk, Partitions) :-
    % Estimate item size
    estimate_item_size(Item, ItemSize),
    NewSize is CurrentSize + ItemSize,

    (   NewSize > BytesPerPartition, CurrentChunk \= []
    ->  % Current partition is full, start new one
        reverse(CurrentChunk, Chunk),
        Partitions = [partition(PartitionID, Chunk)|RestPartitions],
        NextID is PartitionID + 1,
        partition_by_bytes_helper([Item|Rest], BytesPerPartition, NextID, 0, [], RestPartitions)
    ;   % Add item to current partition
        partition_by_bytes_helper(Rest, BytesPerPartition, PartitionID, NewSize, [Item|CurrentChunk], Partitions)
    ).

%% estimate_item_size(+Item, -Size)
%  Estimate size of item in bytes
estimate_item_size(Item, Size) :-
    (   atom(Item)
    ->  atom_length(Item, Size)
    ;   string(Item)
    ->  string_length(Item, Size)
    ;   number(Item)
    ->  Size = 8  % Assume 8 bytes for numbers
    ;   compound(Item)
    ->  % Estimate compound term size
        term_string(Item, Str),
        string_length(Str, Size)
    ;   Size = 1  % Fallback
    ).

%% strategy_assign(+State, +Item, -PartitionID)
%  Assign single item to partition (for streaming)
%
%  In fixed-size strategy, we use round-robin assignment based on item count
strategy_assign(state(rows, Size, CurrentCount), _Item, PartitionID) :-
    PartitionID is CurrentCount // Size.
strategy_assign(state(bytes, Size, _), Item, PartitionID) :-
    % For bytes mode, estimate size and assign
    % This is a simplified streaming version
    estimate_item_size(Item, ItemSize),
    % Use hash of size as simple partitioning
    PartitionID is ItemSize mod Size.

%% strategy_cleanup(+State)
%  Clean up strategy resources
strategy_cleanup(state(Mode, Size, _)) :-
    format('[FixedSize] Cleanup: ~w mode, size ~w~n', [Mode, Size]).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% take_n(+List, +N, -FirstN, -Rest)
%  Take first N elements from List
take_n(List, N, FirstN, Rest) :-
    take_n_helper(List, N, [], FirstN, Rest).

take_n_helper(Rest, 0, Acc, FirstN, Rest) :-
    !,
    reverse(Acc, FirstN).
take_n_helper([], _, Acc, FirstN, []) :-
    !,
    reverse(Acc, FirstN).
take_n_helper([H|T], N, Acc, FirstN, Rest) :-
    N > 0,
    N1 is N - 1,
    take_n_helper(T, N1, [H|Acc], FirstN, Rest).
