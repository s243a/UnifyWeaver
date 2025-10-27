:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% key_based.pl - Key-based partitioning strategy
% Groups data by key value (for aggregation, GROUP BY operations)

:- module(key_based_partitioner, [
    strategy_init/2,
    strategy_partition/3,
    strategy_assign/3,
    strategy_cleanup/1
]).

:- use_module(library(lists)).
:- use_module(library(assoc)).

%% ============================================
%% STRATEGY IMPLEMENTATION
%% ============================================

%% strategy_init(+Config, -State)
%  Initialize key-based partitioning strategy
%
%  Configuration options:
%  - strategy_args([key(KeySpec)])
%
%  KeySpec can be:
%  - column(N) - Extract Nth column (1-indexed)
%  - field(N)  - Alias for column
%  - whole     - Use entire item as key
%
%  @example Initialize with first column as key
%    ?- strategy_init([strategy_args([key(column(1))])], State).
strategy_init(Config, State) :-
    % Extract strategy arguments
    (   member(strategy_args(Args), Config)
    ->  true
    ;   Args = []
    ),

    % Parse key specification
    (   member(key(KeySpec), Args)
    ->  true
    ;   KeySpec = column(1)  % Default: first column
    ),

    % Initialize empty key-to-partition mapping
    empty_assoc(KeyMap),

    % Initialize state
    State = state(KeySpec, KeyMap, 0),  % state(key_spec, key_map, next_partition_id)
    format('[KeyBased] Initialized: key=~w~n', [KeySpec]).

%% strategy_partition(+State, +DataStream, -Partitions)
%  Partition data stream by grouping items with same key
strategy_partition(state(KeySpec, _, _), DataStream, Partitions) :-
    % Build key-to-items mapping
    empty_assoc(KeyMap),
    build_key_groups(DataStream, KeySpec, KeyMap, GroupedMap),

    % Convert to partition list
    assoc_to_partitions(GroupedMap, 0, Partitions).

%% build_key_groups(+Items, +KeySpec, +KeyMap, -GroupedMap)
%  Build mapping from keys to items
build_key_groups([], _, KeyMap, KeyMap).
build_key_groups([Item|Rest], KeySpec, KeyMap, GroupedMap) :-
    % Extract key from item
    extract_key(Item, KeySpec, Key),

    % Add item to key's group
    (   get_assoc(Key, KeyMap, Items)
    ->  % Key exists, prepend item
        put_assoc(Key, KeyMap, [Item|Items], NewKeyMap)
    ;   % New key, create group
        put_assoc(Key, KeyMap, [Item], NewKeyMap)
    ),

    % Continue with rest
    build_key_groups(Rest, KeySpec, NewKeyMap, GroupedMap).

%% assoc_to_partitions(+GroupedMap, +StartID, -Partitions)
%  Convert key-to-items mapping to partition list
assoc_to_partitions(GroupedMap, StartID, Partitions) :-
    assoc_to_list(GroupedMap, KeyValuePairs),
    pairs_to_partitions(KeyValuePairs, StartID, Partitions).

pairs_to_partitions([], _, []).
pairs_to_partitions([Key-Items|Rest], ID, [partition(ID, key(Key), ReversedItems)|RestPartitions]) :-
    % Reverse items (were prepended during building)
    reverse(Items, ReversedItems),
    NextID is ID + 1,
    pairs_to_partitions(Rest, NextID, RestPartitions).

%% strategy_assign(+State, +Item, -PartitionID)
%  Assign single item to partition (for streaming)
%
%  Note: For streaming, we need to maintain state across calls
%  This is a simplified version that uses hash-based assignment
strategy_assign(state(KeySpec, KeyMap, NextID), Item, PartitionID) :-
    extract_key(Item, KeySpec, Key),

    % Check if key already has partition
    (   get_assoc(Key, KeyMap, PartitionID)
    ->  true  % Key already assigned
    ;   % New key, assign next partition ID
        PartitionID = NextID
        % Note: In real streaming, we'd update state with put_assoc
        % But this predicate doesn't modify state (functional approach)
    ).

%% strategy_cleanup(+State)
%  Clean up strategy resources
strategy_cleanup(state(KeySpec, _, _)) :-
    format('[KeyBased] Cleanup: key=~w~n', [KeySpec]).

%% ============================================
%% KEY EXTRACTION
%% ============================================

%% extract_key(+Item, +KeySpec, -Key)
%  Extract key from item based on specification
extract_key(Item, whole, Item) :- !.

extract_key(Item, column(N), Key) :-
    !,
    extract_column(Item, N, Key).

extract_key(Item, field(N), Key) :-
    !,
    extract_column(Item, N, Key).

extract_key(Item, KeySpec, _) :-
    throw(error(domain_error(key_spec, KeySpec),
                context(extract_key/3, 'Unknown key specification'))).

%% extract_column(+Item, +N, -Value)
%  Extract Nth column from item
extract_column(Item, N, Value) :-
    (   compound(Item)
    ->  % Compound term: get Nth argument
        arg(N, Item, Value)
    ;   atom(Item)
    ->  % Atom: split by delimiter and get Nth
        atomic_list_concat(Parts, ':', Item),
        nth1(N, Parts, Value)
    ;   string(Item)
    ->  % String: split by delimiter and get Nth
        split_string(Item, ":", "", Parts),
        nth1(N, Parts, Value)
    ;   is_list(Item)
    ->  % List: get Nth element
        nth1(N, Item, Value)
    ;   % Fallback: use whole item
        Value = Item
    ).
