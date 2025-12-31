:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 UnifyWeaver Contributors
%
% index_analyzer.pl - Analyze and manage database index definitions
%
% This module handles index declarations for predicates, allowing targets
% (like Go/Bbolt) to generate optimized lookup structures.

:- module(index_analyzer, [
    % Core API
    get_indexes/2,          % get_indexes(+Pred, -Indexes)
    has_index/2,            % has_index(+Pred, +Field)
    
    % Declaration API
    declare_index/2,        % declare_index(+Pred, +Field)
    clear_indexes/1,        % clear_indexes(+Pred)
    
    % Testing
    test_index_analyzer/0
]).

:- use_module(library(lists)).

% ============================================================================
% INDEX DECLARATIONS
% ============================================================================
%
% Indexes are stored as: predicate_index(PredIndicator, FieldName).
%
% Usage:
%   :- index(user/2, id).
%   :- index(user/2, email).

:- dynamic predicate_index/2.

%% declare_index(+Pred:pred_indicator, +Field:atom) is det.
%
%  Declares an index on a specific field for a predicate.
%  This instructs the compiler to maintain a secondary index for this field
%  if the target supports it (e.g., Go/Bbolt).
%
%  @param Pred The predicate indicator, e.g., `user/2`.
%  @param Field The field name to index, e.g., `email`.
declare_index(Pred, Field) :-
    % Idempotent insert
    (   predicate_index(Pred, Field)
    ->  true
    ;   assertz(predicate_index(Pred, Field))
    ).

%% clear_indexes(+Pred:pred_indicator) is det.
%
%  Removes all index declarations for a predicate.
%
%  @param Pred The predicate indicator.
clear_indexes(Pred) :-
    retractall(predicate_index(Pred, _)).

%% get_indexes(+Pred:pred_indicator, -Indexes:list) is det.
%
%  Retrieves all declared indexes for a predicate.
%
%  @param Pred The predicate indicator.
%  @param Indexes List of field names that are indexed.
get_indexes(Pred, Indexes) :-
    findall(Field, predicate_index(Pred, Field), Indexes).

%% has_index(+Pred:pred_indicator, +Field:atom) is semidet.
%
%  Checks if a specific field is indexed.
has_index(Pred, Field) :-
    predicate_index(Pred, Field).

% ============================================================================
% PRAGMA DIRECTIVE SUPPORT
% ============================================================================
%
% Allow users to write:
%   :- index(user/2, email).

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- index(Pred, Field)),
    (:- initialization(index_analyzer:declare_index(Pred, Field)))
).

% ============================================================================
% TESTS
% ============================================================================

test_index_analyzer :-
    format('~n=== Testing Index Analyzer ===~n'),

    % Test 1: Declare and query
    format('~nTest 1: Declare and query~n'),
    declare_index(test_user/2, email),
    declare_index(test_user/2, age),
    
    get_indexes(test_user/2, Indexes),
    format('  Indexes: ~w~n', [Indexes]),
    
    (   member(email, Indexes), member(age, Indexes)
    ->  format('  ✓ Found declared indexes~n')
    ;   format('  X Missing indexes~n'), fail
    ),
    
    % Test 2: Check specific index
    format('~nTest 2: Check specific index~n'),
    (   has_index(test_user/2, email)
    ->  format('  ✓ has_index(email) true~n')
    ;   format('  X has_index(email) failed~n'), fail
    ),
    
    % Test 3: Clear indexes
    format('~nTest 3: Clear indexes~n'),
    clear_indexes(test_user/2),
    get_indexes(test_user/2, Empty),
    (   Empty == []
    ->  format('  ✓ Indexes cleared~n')
    ;   format('  X Failed to clear indexes~n'), fail
    ),

    % Cleanup
    clear_indexes(test_user/2),
    
    format('~n=== All index analyzer tests passed! ===~n').
