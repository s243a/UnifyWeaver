:- encoding(utf8).
% constraint_analyzer.pl - Analyze and apply uniqueness and order constraints
%
% This module handles constraint declarations for predicates, determining
% the appropriate deduplication strategy based on uniqueness and order requirements.

:- module(constraint_analyzer, [
    % Core API
    get_constraints/2,              % get_constraints(+Pred, -Constraints)
    constraint_implies_sort_u/1,    % constraint_implies_sort_u(+Constraints)
    constraint_implies_hash/1,      % constraint_implies_hash(+Constraints)

    % Declaration API
    declare_constraint/2,           % declare_constraint(+Pred, +Constraints)
    clear_constraints/1,            % clear_constraints(+Pred)

    % Default configuration
    get_default_constraints/1,      % get_default_constraints(-Constraints)
    set_default_constraints/1,      % set_default_constraints(+Constraints)

    % Testing
    test_constraint_analyzer/0
]).

:- use_module(library(lists)).

% ============================================================================
% DEFAULT CONSTRAINTS
% ============================================================================
%
% Default behavior: unique=true, unordered=true
%
% Rationale:
% - Most Prolog queries don't care about duplicate order
% - Matches current behavior (all examples use `sort -u`)
% - Efficient: allows use of `sort -u` instead of hash tables
% - Easy to override for temporal/ordered data
%
% To change defaults globally:
%   set_default_constraints([unique(true), unordered(false)]).
%
% To override per-predicate:
%   declare_constraint(my_pred/2, [unique(false)]).
%   declare_constraint(temporal_query/2, [unordered(false)]).

:- dynamic default_constraint/1.

% Initialize with explicit defaults
:- assertz(default_constraint(unique(true))).
:- assertz(default_constraint(unordered(true))).

get_default_constraints(Constraints) :-
    findall(C, default_constraint(C), Constraints).

%% set_default_constraints(+NewConstraints:list) is det.
%
%  Sets the global default constraints for all predicates.
%  These defaults are used for any predicate that does not have its
%  own constraints explicitly declared.
%
%  @param NewConstraints A list of constraint terms, e.g., `[unique(true), unordered(false)]`.
set_default_constraints(NewConstraints) :-
    retractall(default_constraint(_)),
    forall(member(C, NewConstraints), assertz(default_constraint(C))).

% ============================================================================
% CONSTRAINT DECLARATIONS
% ============================================================================
%
% Predicates can have constraints declared in multiple ways:
%
% 1. Pragma-style (recommended):
%    :- constraint(grandparent/2, [unique, unordered]).
%    :- constraint(temporal_query/2, [unique, ordered]).
%
% 2. Programmatic:
%    declare_constraint(grandparent/2, [unique, unordered]).

:- dynamic predicate_constraint/2.

%% declare_constraint(+Pred:pred_indicator, +Constraints:list) is det.
%
%  Declares constraints for a specific predicate. This overrides the
%  global defaults for that predicate.
%
%  @param Pred The predicate indicator, e.g., `my_pred/2`.
%  @param Constraints A list of constraint terms. Shorthands like `unique`
%         and `ordered` are normalized.
declare_constraint(Pred, Constraints) :-
    retractall(predicate_constraint(Pred, _)),
    normalize_constraints(Constraints, Normalized),
    assertz(predicate_constraint(Pred, Normalized)).

%% clear_constraints(+Pred:pred_indicator) is det.
%
%  Removes any specific constraints for a predicate, causing it to
%  revert to using the global defaults.
%
%  @param Pred The predicate indicator, e.g., `my_pred/2`.
clear_constraints(Pred) :-
    retractall(predicate_constraint(Pred, _)).

% Normalize constraint list (expand shorthand, validate)
normalize_constraints([], []).
normalize_constraints([unique|Rest], [unique(true)|NormRest]) :- !,
    normalize_constraints(Rest, NormRest).
normalize_constraints([unordered|Rest], [unordered(true)|NormRest]) :- !,
    normalize_constraints(Rest, NormRest).
normalize_constraints([ordered|Rest], [unordered(false)|NormRest]) :- !,
    normalize_constraints(Rest, NormRest).
normalize_constraints([C|Rest], [C|NormRest]) :-
    (C = unique(_) ; C = unordered(_)), !,
    normalize_constraints(Rest, NormRest).
normalize_constraints([Invalid|_], _) :-
    format('ERROR: Invalid constraint: ~w~n', [Invalid]),
    fail.

% ============================================================================
% CONSTRAINT QUERIES
% ============================================================================

%% get_constraints(+Pred:pred_indicator, -Constraints:list) is det.
%
%  Gets the effective constraints for a given predicate. This is the main
%  query predicate used by the compilers.
%
%  It merges the globally defined defaults with any predicate-specific
%  declarations, with the predicate-specific constraints taking precedence.
%
%  @param Pred The predicate indicator, e.g., `my_pred/2`.
%  @param Constraints The resulting list of effective constraint terms.
get_constraints(Pred, Constraints) :-
    (   predicate_constraint(Pred, DeclaredConstraints) ->
        get_default_constraints(Defaults),
        merge_constraints(Defaults, DeclaredConstraints, Constraints)
    ;   get_default_constraints(Constraints)
    ).

% Merge declared constraints with defaults (declared takes precedence)
merge_constraints(Defaults, Declared, Merged) :-
    merge_constraint_list(Defaults, Declared, unique, MergedUnique),
    merge_constraint_list(Defaults, Declared, unordered, MergedOrdered),
    append(MergedUnique, MergedOrdered, Merged).

merge_constraint_list(Defaults, Declared, Key, [Result]) :-
    Constraint =.. [Key, _],
    (   member(Constraint, Declared) ->
        Result = Constraint
    ;   member(Constraint, Defaults) ->
        Result = Constraint
    ;   % This shouldn't happen if defaults are set properly
        format('WARNING: No ~w constraint found, defaulting to true~n', [Key]),
        Result =.. [Key, true]
    ).

% ============================================================================
% DEDUPLICATION STRATEGY SELECTION
% ============================================================================

% Check if constraints imply sort -u can be used
% Requires: unique=true AND unordered=true
constraint_implies_sort_u(Constraints) :-
    member(unique(true), Constraints),
    member(unordered(true), Constraints).

% Check if constraints imply hash-based deduplication
% Requires: unique=true AND unordered=false (ordered)
constraint_implies_hash(Constraints) :-
    member(unique(true), Constraints),
    member(unordered(false), Constraints).

% Get deduplication strategy name (for code generation)
get_dedup_strategy(Constraints, Strategy) :-
    (   constraint_implies_sort_u(Constraints) ->
        Strategy = sort_u
    ;   constraint_implies_hash(Constraints) ->
        Strategy = hash_dedup
    ;   member(unique(false), Constraints) ->
        Strategy = no_dedup
    ;   % Default fallback
        Strategy = sort_u
    ).

% ============================================================================
% PRAGMA DIRECTIVE SUPPORT
% ============================================================================
%
% Allow users to write:
%   :- constraint(predicate/arity, [unique, unordered]).

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- constraint(Pred, Constraints)),
    (:- initialization(constraint_analyzer:declare_constraint(Pred, Constraints)))
).

% ============================================================================
% TESTS
% ============================================================================

test_constraint_analyzer :-
    format('~n=== Testing Constraint Analyzer ===~n'),

    % Test 1: Default constraints
    format('~nTest 1: Default constraints~n'),
    get_default_constraints(Defaults),
    format('  Defaults: ~w~n', [Defaults]),
    assertion(member(unique(true), Defaults)),
    assertion(member(unordered(true), Defaults)),
    format('  ✓ Defaults are unique=true, unordered=true~n'),

    % Test 2: Undeclared predicate uses defaults
    format('~nTest 2: Undeclared predicate~n'),
    get_constraints(undeclared_pred/2, C1),
    format('  undeclared_pred/2: ~w~n', [C1]),
    assertion(constraint_implies_sort_u(C1)),
    format('  ✓ Uses sort -u strategy~n'),

    % Test 3: Declare ordered predicate
    format('~nTest 3: Ordered predicate~n'),
    declare_constraint(temporal_query/2, [unique, ordered]),
    get_constraints(temporal_query/2, C2),
    format('  temporal_query/2: ~w~n', [C2]),
    assertion(constraint_implies_hash(C2)),
    format('  ✓ Uses hash dedup strategy~n'),

    % Test 4: Declare unordered predicate explicitly
    format('~nTest 4: Explicitly unordered~n'),
    declare_constraint(grandparent/2, [unique, unordered]),
    get_constraints(grandparent/2, C3),
    format('  grandparent/2: ~w~n', [C3]),
    assertion(constraint_implies_sort_u(C3)),
    format('  ✓ Uses sort -u strategy~n'),

    % Test 5: Non-unique predicate
    format('~nTest 5: Non-unique predicate~n'),
    declare_constraint(allow_dupes/2, [unique(false)]),
    get_constraints(allow_dupes/2, C4),
    format('  allow_dupes/2: ~w~n', [C4]),
    assertion(\+ constraint_implies_sort_u(C4)),
    assertion(\+ constraint_implies_hash(C4)),
    get_dedup_strategy(C4, Strat4),
    format('  Strategy: ~w~n', [Strat4]),
    format('  ✓ No deduplication~n'),

    % Test 6: Change defaults
    format('~nTest 6: Change defaults~n'),
    set_default_constraints([unique(true), unordered(false)]),
    clear_constraints(new_pred/2),
    get_constraints(new_pred/2, C5),
    format('  new_pred/2 with new defaults: ~w~n', [C5]),
    assertion(constraint_implies_hash(C5)),
    format('  ✓ New defaults applied~n'),

    % Restore original defaults
    set_default_constraints([unique(true), unordered(true)]),

    % Cleanup
    clear_constraints(temporal_query/2),
    clear_constraints(grandparent/2),
    clear_constraints(allow_dupes/2),

    format('~n=== All constraint analyzer tests passed! ===~n').

assertion(Goal) :-
    (   call(Goal) -> true
    ;   format('ASSERTION FAILED: ~w~n', [Goal]),
        fail
    ).
