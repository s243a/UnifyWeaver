:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_dict.pl — Portable dictionary abstraction for the WAM runtime.
%
% Provides a uniform dictionary interface that the WAM runtime uses
% internally and that target transpilers map to native types:
%
%   Prolog:  assoc (AVL tree, O(log n))
%   Haskell: Data.HashMap (O(1) amortized) or Data.Map (O(log n), persistent)
%   Rust:    HashMap (O(1) amortized)
%   Elixir:  Map (O(log n), immutable)
%
% The WAM runtime should use ONLY these predicates for dictionary
% operations, never raw assoc/put_assoc/get_assoc directly. This
% ensures that transpilers can replace the implementation with the
% target language's native dictionary.
%
% Two dictionary flavors:
%   - wam_dict: general-purpose, used for registers, fact tables, labels
%   - wam_binding_dict: optimized for the variable binding table,
%     supports snapshot/restore for backtracking

:- module(wam_dict, [
    % Core dictionary operations
    wam_dict_new/1,           % -Dict
    wam_dict_lookup/3,        % +Key, +Dict, -Value
    wam_dict_lookup/4,        % +Key, +Dict, +Default, -Value
    wam_dict_insert/4,        % +Key, +Value, +DictIn, -DictOut
    wam_dict_remove/3,        % +Key, +DictIn, -DictOut
    wam_dict_has_key/2,       % +Key, +Dict
    wam_dict_from_list/2,     % +Pairs, -Dict
    wam_dict_to_list/2,       % +Dict, -Pairs
    wam_dict_keys/2,          % +Dict, -Keys
    wam_dict_size/2,          % +Dict, -Size
    wam_dict_fold/4,          % +Pred, +Dict, +Acc0, -Acc

    % Grouped dictionary (for fact tables: Key → [Values])
    wam_dict_append/4,        % +Key, +Value, +DictIn, -DictOut
    wam_dict_lookup_all/3,    % +Key, +Dict, -Values

    % Bulk construction
    wam_dict_from_grouped/2   % +KeyValuePairs, -GroupedDict
]).

:- use_module(library(assoc)).
:- use_module(library(lists)).

% ============================================================================
% Core Dictionary Operations
% ============================================================================

%% wam_dict_new(-Dict)
%  Create an empty dictionary.
wam_dict_new(Dict) :-
    empty_assoc(Dict).

%% wam_dict_lookup(+Key, +Dict, -Value)
%  Look up a key. Fails if key not found.
wam_dict_lookup(Key, Dict, Value) :-
    get_assoc(Key, Dict, Value).

%% wam_dict_lookup(+Key, +Dict, +Default, -Value)
%  Look up a key with a default value.
wam_dict_lookup(Key, Dict, Default, Value) :-
    (   get_assoc(Key, Dict, V)
    ->  Value = V
    ;   Value = Default
    ).

%% wam_dict_insert(+Key, +Value, +DictIn, -DictOut)
%  Insert or update a key-value pair.
wam_dict_insert(Key, Value, DictIn, DictOut) :-
    put_assoc(Key, DictIn, Value, DictOut).

%% wam_dict_remove(+Key, +DictIn, -DictOut)
%  Remove a key. If key doesn't exist, returns the same dict.
%  Note: SWI-Prolog assoc doesn't have a direct delete. We rebuild
%  without the key. For the transpilation target, this maps to native
%  dict.remove() which is O(1) or O(log n).
wam_dict_remove(Key, DictIn, DictOut) :-
    (   get_assoc(Key, DictIn, _)
    ->  assoc_to_list(DictIn, Pairs),
        exclude([K-_]>>(K == Key), Pairs, Filtered),
        list_to_assoc(Filtered, DictOut)
    ;   DictOut = DictIn
    ).

%% wam_dict_has_key(+Key, +Dict)
%  True if Key exists in Dict.
wam_dict_has_key(Key, Dict) :-
    get_assoc(Key, Dict, _).

%% wam_dict_from_list(+Pairs, -Dict)
%  Build a dictionary from Key-Value pairs.
%  Pairs is a list of Key-Value terms.
wam_dict_from_list(Pairs, Dict) :-
    empty_assoc(D0),
    foldl(insert_pair, Pairs, D0, Dict).

insert_pair(Key-Value, DIn, DOut) :- !, put_assoc(Key, DIn, Value, DOut).
insert_pair(Key=Value, DIn, DOut) :- !, put_assoc(Key, DIn, Value, DOut).

%% wam_dict_to_list(+Dict, -Pairs)
%  Convert dictionary to a list of Key-Value pairs.
wam_dict_to_list(Dict, Pairs) :-
    assoc_to_list(Dict, Pairs).

%% wam_dict_keys(+Dict, -Keys)
%  Get all keys.
wam_dict_keys(Dict, Keys) :-
    assoc_to_keys(Dict, Keys).

%% wam_dict_size(+Dict, -Size)
%  Get the number of entries.
wam_dict_size(Dict, Size) :-
    assoc_to_keys(Dict, Keys),
    length(Keys, Size).

%% wam_dict_fold(+Pred, +Dict, +Acc0, -Acc)
%  Fold over dictionary entries. Pred is called as Pred(Key, Value, AccIn, AccOut).
wam_dict_fold(Pred, Dict, Acc0, Acc) :-
    assoc_to_list(Dict, Pairs),
    foldl(call_with_pair(Pred), Pairs, Acc0, Acc).

call_with_pair(Pred, Key-Value, AccIn, AccOut) :-
    call(Pred, Key, Value, AccIn, AccOut).

% ============================================================================
% Grouped Dictionary (for fact tables)
% ============================================================================

%% wam_dict_append(+Key, +Value, +DictIn, -DictOut)
%  Append a value to the list at Key. If Key doesn't exist, creates [Value].
wam_dict_append(Key, Value, DictIn, DictOut) :-
    (   get_assoc(Key, DictIn, Existing)
    ->  append(Existing, [Value], NewList),
        put_assoc(Key, DictIn, NewList, DictOut)
    ;   put_assoc(Key, DictIn, [Value], DictOut)
    ).

%% wam_dict_lookup_all(+Key, +Dict, -Values)
%  Look up all values for a key (grouped dict). Returns [] if not found.
wam_dict_lookup_all(Key, Dict, Values) :-
    (   get_assoc(Key, Dict, Vs)
    ->  Values = Vs
    ;   Values = []
    ).

%% wam_dict_from_grouped(+KeyValuePairs, -GroupedDict)
%  Build a grouped dictionary from pairs. Multiple values per key are collected.
%  Input: list of Key-Value pairs (Key may repeat).
%  Output: Dict where each Key maps to a list of Values.
wam_dict_from_grouped(Pairs, Dict) :-
    empty_assoc(D0),
    foldl(group_pair, Pairs, D0, Dict).

group_pair(Key-Value, DIn, DOut) :-
    (   get_assoc(Key, DIn, Existing)
    ->  append(Existing, [Value], NewList),
        put_assoc(Key, DIn, NewList, DOut)
    ;   put_assoc(Key, DIn, [Value], DOut)
    ).
