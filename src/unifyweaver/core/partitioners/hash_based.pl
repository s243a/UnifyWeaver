:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% hash_based.pl - Hash-based partitioning strategy
% Partitions data using hash function (compatible with MapReduce/Hadoop)

:- module(hash_based_partitioner, [
    strategy_init/2,
    strategy_partition/3,
    strategy_assign/3,
    strategy_cleanup/1
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% ============================================
%% STRATEGY IMPLEMENTATION
%% ============================================

%% strategy_init(+Config, -State)
%  Initialize hash-based partitioning strategy
%
%  Configuration options:
%  - strategy_args([key(KeySpec), num_partitions(N), hash_function(F)])
%
%  KeySpec can be:
%  - column(N) - Extract Nth column (1-indexed)
%  - field(N)  - Alias for column
%  - whole    - Hash entire item
%
%  Hash functions:
%  - simple_mod - Simple modulo (fast, good for integers)
%  - term_hash  - SWI-Prolog's term_hash/2 (general purpose)
%  - atom_hash  - Hash after converting to atom
%
%  @example Initialize with 8 partitions, hash first column
%    ?- strategy_init([strategy_args([key(column(1)), num_partitions(8)])], State).
strategy_init(Config, State) :-
    % Extract strategy arguments
    (   member(strategy_args(Args), Config)
    ->  true
    ;   Args = []
    ),

    % Parse key specification
    (   member(key(KeySpec), Args)
    ->  true
    ;   KeySpec = whole  % Default: hash whole item
    ),

    % Parse number of partitions
    (   member(num_partitions(N), Args)
    ->  true
    ;   N = 8  % Default: 8 partitions
    ),

    % Parse hash function
    (   member(hash_function(HashFunc), Args)
    ->  true
    ;   HashFunc = term_hash  % Default: term_hash
    ),

    % Validate
    (   integer(N), N > 0
    ->  true
    ;   throw(error(domain_error(positive_integer, N),
                    context(strategy_init/2, 'num_partitions must be positive integer')))
    ),

    (   member(HashFunc, [simple_mod, term_hash, atom_hash])
    ->  true
    ;   throw(error(domain_error(hash_function, HashFunc),
                    context(strategy_init/2, 'Unknown hash function')))
    ),

    % Initialize state
    State = state(KeySpec, N, HashFunc),
    format('[HashBased] Initialized: key=~w, partitions=~w, hash=~w~n',
           [KeySpec, N, HashFunc]).

%% strategy_partition(+State, +DataStream, -Partitions)
%  Partition data stream using hash function
strategy_partition(state(KeySpec, NumPartitions, HashFunc), DataStream, Partitions) :-
    % Create empty partition buckets
    create_partition_buckets(NumPartitions, Buckets),

    % Assign each item to a partition
    assign_items_to_partitions(DataStream, KeySpec, NumPartitions, HashFunc, Buckets, FilledBuckets),

    % Convert buckets to partition list
    buckets_to_partitions(FilledBuckets, 0, Partitions).

%% create_partition_buckets(+NumPartitions, -Buckets)
%  Create empty partition buckets
create_partition_buckets(NumPartitions, Buckets) :-
    length(Buckets, NumPartitions),
    maplist(=(bucket([])), Buckets).

%% assign_items_to_partitions(+Items, +KeySpec, +NumPartitions, +HashFunc, +Buckets, -FilledBuckets)
%  Assign each item to appropriate partition bucket
assign_items_to_partitions([], _, _, _, Buckets, Buckets).
assign_items_to_partitions([Item|Rest], KeySpec, NumPartitions, HashFunc, Buckets, FilledBuckets) :-
    % Extract key from item
    extract_key(Item, KeySpec, Key),

    % Compute partition ID
    compute_hash_partition(Key, NumPartitions, HashFunc, PartitionID),

    % Add item to bucket
    nth0(PartitionID, Buckets, bucket(Items)),
    replace_at_index(PartitionID, Buckets, bucket([Item|Items]), NewBuckets),

    % Continue with rest
    assign_items_to_partitions(Rest, KeySpec, NumPartitions, HashFunc, NewBuckets, FilledBuckets).

%% replace_at_index(+Index, +List, +NewElement, -NewList)
%  Replace element at Index in List with NewElement
replace_at_index(0, [_|T], Element, [Element|T]) :- !.
replace_at_index(Index, [H|T], Element, [H|NewT]) :-
    Index > 0,
    NextIndex is Index - 1,
    replace_at_index(NextIndex, T, Element, NewT).

%% buckets_to_partitions(+Buckets, +ID, -Partitions)
%  Convert bucket list to partition list, filtering empty partitions
buckets_to_partitions([], _, []).
buckets_to_partitions([bucket(Items)|Rest], ID, Partitions) :-
    NextID is ID + 1,
    buckets_to_partitions(Rest, NextID, RestPartitions),
    (   Items = []
    ->  % Skip empty partition
        Partitions = RestPartitions
    ;   % Reverse items (were prepended) and create partition
        reverse(Items, ReversedItems),
        Partitions = [partition(ID, ReversedItems)|RestPartitions]
    ).

%% strategy_assign(+State, +Item, -PartitionID)
%  Assign single item to partition (for streaming)
strategy_assign(state(KeySpec, NumPartitions, HashFunc), Item, PartitionID) :-
    extract_key(Item, KeySpec, Key),
    compute_hash_partition(Key, NumPartitions, HashFunc, PartitionID).

%% strategy_cleanup(+State)
%  Clean up strategy resources
strategy_cleanup(state(KeySpec, NumPartitions, HashFunc)) :-
    format('[HashBased] Cleanup: key=~w, partitions=~w, hash=~w~n',
           [KeySpec, NumPartitions, HashFunc]).

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
    ;   % Fallback: use whole item
        Value = Item
    ).

%% ============================================
%% HASH COMPUTATION
%% ============================================

%% compute_hash_partition(+Key, +NumPartitions, +HashFunc, -PartitionID)
%  Compute partition ID using hash function
compute_hash_partition(Key, NumPartitions, simple_mod, PartitionID) :-
    !,
    simple_mod_hash(Key, NumPartitions, PartitionID).

compute_hash_partition(Key, NumPartitions, term_hash, PartitionID) :-
    !,
    term_hash(Key, Hash),
    PartitionID is abs(Hash) mod NumPartitions.

compute_hash_partition(Key, NumPartitions, atom_hash, PartitionID) :-
    !,
    atom_hash_impl(Key, Hash),
    PartitionID is Hash mod NumPartitions.

compute_hash_partition(Key, NumPartitions, HashFunc, _) :-
    throw(error(domain_error(hash_function, HashFunc),
                context(compute_hash_partition/4,
                       'Unknown hash function'))).

%% simple_mod_hash(+Key, +NumPartitions, -PartitionID)
%  Simple modulo-based hash (works well for integers)
simple_mod_hash(Key, NumPartitions, PartitionID) :-
    (   integer(Key)
    ->  PartitionID is abs(Key) mod NumPartitions
    ;   atom(Key)
    ->  atom_codes(Key, Codes),
        sum_list(Codes, Sum),
        PartitionID is Sum mod NumPartitions
    ;   string(Key)
    ->  string_codes(Key, Codes),
        sum_list(Codes, Sum),
        PartitionID is Sum mod NumPartitions
    ;   % Convert to term and use term_hash
        term_hash(Key, Hash),
        PartitionID is abs(Hash) mod NumPartitions
    ).

%% atom_hash_impl(+Key, -Hash)
%  Hash implementation for atoms/strings
atom_hash_impl(Key, Hash) :-
    (   atom(Key)
    ->  atom_codes(Key, Codes)
    ;   string(Key)
    ->  string_codes(Key, Codes)
    ;   term_string(Key, KeyStr),
        string_codes(KeyStr, Codes)
    ),
    % Simple polynomial rolling hash
    polynomial_hash(Codes, 0, Hash).

polynomial_hash([], Hash, Hash).
polynomial_hash([Code|Rest], Acc, Hash) :-
    NewAcc is (Acc * 31 + Code) mod 2147483647,  % Large prime
    polynomial_hash(Rest, NewAcc, Hash).
