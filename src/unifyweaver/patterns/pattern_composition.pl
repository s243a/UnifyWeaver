% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% pattern_composition.pl - Advanced Pattern Composition Tools
%
% Provides dependency resolution, conflict detection, and validation
% for composing multiple UI patterns across frameworks.
%
% Key features:
%   - Dependency resolution with topological sorting
%   - Conflict detection (name clashes, incompatible requirements)
%   - Target compatibility validation
%   - Composition graph analysis
%
% Usage:
%   ?- compose_with_deps([app_nav, app_store], react_native, [], Result).
%   ?- detect_conflicts([pattern1, pattern2], Conflicts).
%   ?- validate_composition([patterns], target, Errors).

:- module(pattern_composition, [
    % Dependency resolution
    compose_with_deps/4,            % +Patterns, +Target, +Options, -Result
    resolve_dependencies/2,         % +Patterns, -AllPatterns
    dependency_order/2,             % +Patterns, -OrderedPatterns

    % Conflict detection
    detect_conflicts/2,             % +Patterns, -Conflicts
    pattern_conflicts_with/3,       % +Pattern1, +Pattern2, -Reason

    % Validation
    validate_composition/3,         % +Patterns, +Target, -Errors
    validate_for_target/3,          % +Patterns, +Target, -Errors

    % Analysis
    composition_graph/2,            % +Patterns, -Graph
    required_capabilities/2,        % +Patterns, -Capabilities
    composition_summary/2,          % +Patterns, -Summary

    % Testing
    test_pattern_composition/0
]).

:- use_module(library(lists)).
:- use_module('ui_patterns').

% ============================================================================
% DEPENDENCY RESOLUTION
% ============================================================================

%% compose_with_deps(+Patterns, +Target, +Options, -Result)
%
%  Compose patterns with automatic dependency resolution.
%  Returns a result containing the composed spec or errors.
%
compose_with_deps(Patterns, Target, Options, Result) :-
    % Step 1: Resolve all dependencies
    resolve_dependencies(Patterns, AllPatterns),

    % Step 2: Check for conflicts
    detect_conflicts(AllPatterns, Conflicts),

    % Step 3: Validate for target
    validate_for_target(AllPatterns, Target, TargetErrors),

    % Step 4: Build result
    (   Conflicts = [], TargetErrors = []
    ->  % Success: compose and order patterns
        dependency_order(AllPatterns, Ordered),
        collect_pattern_specs(Ordered, Specs),
        Result = composition(
            patterns(Ordered),
            specs(Specs),
            target(Target),
            options(Options)
        )
    ;   % Failure: return errors
        append(Conflicts, TargetErrors, AllErrors),
        Result = composition_error(AllErrors)
    ).

%% resolve_dependencies(+Patterns, -AllPatterns)
%
%  Resolve all dependencies recursively, returning complete list.
%
resolve_dependencies(Patterns, AllPatterns) :-
    resolve_deps_acc(Patterns, [], AllPatterns).

resolve_deps_acc([], Acc, Acc).
resolve_deps_acc([P|Ps], Acc, Result) :-
    (   member(P, Acc)
    ->  % Already processed
        resolve_deps_acc(Ps, Acc, Result)
    ;   % Get dependencies for this pattern
        pattern_dependencies(P, Deps),
        % Add pattern to accumulator
        append(Acc, [P], Acc1),
        % Recursively resolve dependencies
        append(Deps, Ps, NewPs),
        resolve_deps_acc(NewPs, Acc1, Result)
    ).

%% pattern_dependencies(+PatternName, -Dependencies)
%
%  Get dependencies declared by a pattern.
%
pattern_dependencies(Name, Deps) :-
    (   ui_patterns:pattern(Name, _, Opts),
        member(depends_on(Deps), Opts)
    ->  true
    ;   Deps = []
    ).

%% dependency_order(+Patterns, -OrderedPatterns)
%
%  Sort patterns by dependency order (topological sort).
%  Patterns with no dependencies come first.
%
dependency_order(Patterns, Ordered) :-
    build_dep_graph(Patterns, Graph),
    topological_sort(Graph, Patterns, Ordered).

build_dep_graph(Patterns, Graph) :-
    findall(edge(P, D), (
        member(P, Patterns),
        pattern_dependencies(P, Deps),
        member(D, Deps),
        member(D, Patterns)
    ), Graph).

topological_sort(Graph, Patterns, Sorted) :-
    topo_sort_acc(Patterns, Graph, [], Sorted).

topo_sort_acc([], _, Acc, Acc).
topo_sort_acc(Remaining, Graph, Acc, Sorted) :-
    Remaining \= [],
    % Find patterns with no unsatisfied dependencies
    findall(P, (
        member(P, Remaining),
        \+ (member(edge(P, D), Graph), member(D, Remaining))
    ), Ready),
    (   Ready = []
    ->  % Circular dependency - just append remaining
        append(Acc, Remaining, Sorted)
    ;   % Add ready patterns and continue
        append(Acc, Ready, Acc1),
        subtract(Remaining, Ready, Remaining1),
        topo_sort_acc(Remaining1, Graph, Acc1, Sorted)
    ).

collect_pattern_specs(Patterns, Specs) :-
    findall(spec(P, S), (
        member(P, Patterns),
        ui_patterns:pattern(P, S, _)
    ), Specs).

% ============================================================================
% CONFLICT DETECTION
% ============================================================================

%% detect_conflicts(+Patterns, -Conflicts)
%
%  Detect all conflicts between patterns in the list.
%
detect_conflicts(Patterns, Conflicts) :-
    findall(Conflict, (
        member(P1, Patterns),
        member(P2, Patterns),
        P1 @< P2,  % Avoid duplicates
        pattern_conflicts_with(P1, P2, Conflict)
    ), Conflicts).

%% pattern_conflicts_with(+Pattern1, +Pattern2, -Conflict)
%
%  Check if two patterns conflict. Returns conflict description.
%
pattern_conflicts_with(P1, P2, Conflict) :-
    check_name_conflict(P1, P2, Conflict).
pattern_conflicts_with(P1, P2, Conflict) :-
    check_singleton_conflict(P1, P2, Conflict).
pattern_conflicts_with(P1, P2, Conflict) :-
    check_requirement_conflict(P1, P2, Conflict).
pattern_conflicts_with(P1, P2, Conflict) :-
    check_exclusion_conflict(P1, P2, Conflict).

%% check_name_conflict(+P1, +P2, -Conflict)
%
%  Check for patterns that generate conflicting names.
%
check_name_conflict(P1, P2, conflict(name_clash, P1, P2, Details)) :-
    ui_patterns:pattern(P1, Spec1, _),
    ui_patterns:pattern(P2, Spec2, _),
    spec_generates_name(Spec1, Name),
    spec_generates_name(Spec2, Name),
    format(string(Details), "Both patterns generate '~w'", [Name]).

spec_generates_name(state(global, Shape, _), Name) :-
    member(store(Name), Shape).
spec_generates_name(data(_, Config), Name) :-
    member(name(Name), Config).
spec_generates_name(persistence(_, Config), Name) :-
    member(key(Name), Config).

%% check_singleton_conflict(+P1, +P2, -Conflict)
%
%  Check for multiple singletons that would generate the same artifact.
%  Only conflicts if both are singletons AND generate the same name.
%
check_singleton_conflict(P1, P2, conflict(singleton, P1, P2, Details)) :-
    ui_patterns:pattern(P1, Spec1, Opts1),
    ui_patterns:pattern(P2, Spec2, Opts2),
    member(singleton(true), Opts1),
    member(singleton(true), Opts2),
    same_pattern_type(Spec1, Spec2),
    spec_generates_name(Spec1, Name),
    spec_generates_name(Spec2, Name),
    format(string(Details), "Multiple singleton patterns generate '~w'", [Name]).

same_pattern_type(navigation(T, _, _), navigation(T, _, _)).
same_pattern_type(state(global, _, _), state(global, _, _)).

%% check_requirement_conflict(+P1, +P2, -Conflict)
%
%  Check for conflicting requirements (e.g., different versions).
%
check_requirement_conflict(P1, P2, conflict(requirements, P1, P2, Details)) :-
    ui_patterns:pattern(P1, _, Opts1),
    ui_patterns:pattern(P2, _, Opts2),
    member(requires(Reqs1), Opts1),
    member(requires(Reqs2), Opts2),
    member(Req1, Reqs1),
    member(Req2, Reqs2),
    conflicting_requirements(Req1, Req2),
    format(string(Details), "Conflicting requirements: ~w vs ~w", [Req1, Req2]).

conflicting_requirements(library(L, V1), library(L, V2)) :-
    V1 \= V2.

%% check_exclusion_conflict(+P1, +P2, -Conflict)
%
%  Check for patterns that explicitly exclude each other.
%
check_exclusion_conflict(P1, P2, conflict(exclusion, P1, P2, Details)) :-
    ui_patterns:pattern(P1, _, Opts1),
    member(excludes(Excluded), Opts1),
    member(P2, Excluded),
    format(string(Details), "~w explicitly excludes ~w", [P1, P2]).

% ============================================================================
% VALIDATION
% ============================================================================

%% validate_composition(+Patterns, +Target, -Errors)
%
%  Validate a composition for completeness and correctness.
%
validate_composition(Patterns, Target, Errors) :-
    findall(Error, validate_composition_error(Patterns, Target, Error), Errors).

validate_composition_error(Patterns, _, error(missing_dependency, P, Dep)) :-
    member(P, Patterns),
    pattern_dependencies(P, Deps),
    member(Dep, Deps),
    \+ member(Dep, Patterns),
    \+ ui_patterns:pattern(Dep, _, _).

validate_composition_error(Patterns, _, error(circular_dependency, Cycle)) :-
    find_circular_dependency(Patterns, Cycle),
    Cycle \= [].

validate_composition_error(Patterns, Target, Error) :-
    validate_for_target(Patterns, Target, Errors),
    member(Error, Errors).

%% validate_for_target(+Patterns, +Target, -Errors)
%
%  Validate that all patterns work with the specified target.
%
validate_for_target(Patterns, Target, Errors) :-
    findall(Error, (
        member(P, Patterns),
        ui_patterns:pattern(P, _, Opts),
        member(requires(Reqs), Opts),
        member(Cap, Reqs),
        \+ target_provides_capability(Target, Cap),
        Error = error(missing_capability, P, Target, Cap)
    ), Errors).

%% target_provides_capability(+Target, +Capability)
%
%  Check if a target provides a capability.
%  Uses explicit list to avoid catch-all in ui_patterns.
%
target_provides_capability(react_native, navigation).
target_provides_capability(react_native, state_hooks).
target_provides_capability(react_native, react_query).
target_provides_capability(react_native, async_storage).
target_provides_capability(react_native, zustand).
target_provides_capability(react_native, secure_store).
target_provides_capability(vue, navigation).
target_provides_capability(vue, state_hooks).
target_provides_capability(vue, vue_query).
target_provides_capability(vue, pinia).
target_provides_capability(vue, local_storage).
target_provides_capability(flutter, navigation).
target_provides_capability(flutter, state_hooks).
target_provides_capability(flutter, riverpod).
target_provides_capability(flutter, shared_preferences).
target_provides_capability(flutter, hive).
target_provides_capability(swiftui, navigation).
target_provides_capability(swiftui, state_hooks).
target_provides_capability(swiftui, combine).
target_provides_capability(swiftui, app_storage).
target_provides_capability(swiftui, keychain).

find_circular_dependency(Patterns, Cycle) :-
    member(Start, Patterns),
    find_cycle_from(Start, Patterns, [Start], Cycle),
    !.
find_circular_dependency(_, []).

find_cycle_from(Current, Patterns, Path, Cycle) :-
    pattern_dependencies(Current, Deps),
    member(Next, Deps),
    member(Next, Patterns),
    (   member(Next, Path)
    ->  % Found cycle
        append(Path, [Next], Cycle)
    ;   % Continue searching
        append(Path, [Next], NewPath),
        find_cycle_from(Next, Patterns, NewPath, Cycle)
    ).

% ============================================================================
% ANALYSIS
% ============================================================================

%% composition_graph(+Patterns, -Graph)
%
%  Build a dependency graph for visualization.
%
composition_graph(Patterns, Graph) :-
    findall(node(P, Type, Deps), (
        member(P, Patterns),
        ui_patterns:pattern(P, Spec, _),
        pattern_type(Spec, Type),
        pattern_dependencies(P, Deps)
    ), Nodes),
    findall(edge(From, To), (
        member(P, Patterns),
        pattern_dependencies(P, Deps),
        member(D, Deps),
        From = P,
        To = D
    ), Edges),
    Graph = graph(Nodes, Edges).

pattern_type(navigation(_, _, _), navigation).
pattern_type(state(T, _, _), state(T)).
pattern_type(data(T, _), data(T)).
pattern_type(persistence(T, _), persistence(T)).
pattern_type(composite(_, _), composite).

%% required_capabilities(+Patterns, -Capabilities)
%
%  Get all capabilities required by a set of patterns.
%
required_capabilities(Patterns, Capabilities) :-
    findall(Cap, (
        member(P, Patterns),
        ui_patterns:pattern(P, _, Opts),
        member(requires(Reqs), Opts),
        member(Cap, Reqs)
    ), AllCaps),
    sort(AllCaps, Capabilities).

%% composition_summary(+Patterns, -Summary)
%
%  Generate a summary of a composition.
%
composition_summary(Patterns, Summary) :-
    length(Patterns, Count),
    findall(Type, (
        member(P, Patterns),
        ui_patterns:pattern(P, Spec, _),
        pattern_type(Spec, Type)
    ), Types),
    msort(Types, SortedTypes),
    group_types(SortedTypes, TypeCounts),
    required_capabilities(Patterns, Caps),
    detect_conflicts(Patterns, Conflicts),
    length(Conflicts, ConflictCount),
    Summary = summary(
        pattern_count(Count),
        type_breakdown(TypeCounts),
        required_capabilities(Caps),
        conflict_count(ConflictCount)
    ).

group_types([], []).
group_types([T|Ts], [type(T, N)|Rest]) :-
    count_leading(T, [T|Ts], N, Remaining),
    group_types(Remaining, Rest).

count_leading(_, [], 0, []).
count_leading(T, [T|Ts], N, Remaining) :-
    !,
    count_leading(T, Ts, N1, Remaining),
    N is N1 + 1.
count_leading(_, List, 0, List).

% ============================================================================
% TESTING
% ============================================================================

test_pattern_composition :-
    format('~n=== Pattern Composition Tests ===~n~n'),

    % Setup test patterns
    setup_test_patterns,

    % Test 1: Dependency resolution
    format('Test 1: Dependency resolution...~n'),
    (   resolve_dependencies([test_screen_with_deps], Resolved),
        member(test_base_store, Resolved),
        member(test_screen_with_deps, Resolved)
    ->  format('  PASS: Dependencies resolved~n')
    ;   format('  FAIL: Dependency resolution failed~n')
    ),

    % Test 2: Dependency ordering
    format('~nTest 2: Dependency ordering...~n'),
    (   dependency_order([test_screen_with_deps, test_base_store], Ordered),
        nth0(I1, Ordered, test_base_store),
        nth0(I2, Ordered, test_screen_with_deps),
        I1 < I2
    ->  format('  PASS: Dependencies ordered correctly~n')
    ;   format('  FAIL: Ordering incorrect~n')
    ),

    % Test 3: Conflict detection - name clash
    format('~nTest 3: Conflict detection (name clash)...~n'),
    (   detect_conflicts([test_store1, test_store2], Conflicts1),
        member(conflict(name_clash, _, _, _), Conflicts1)
    ->  format('  PASS: Name clash detected~n')
    ;   format('  FAIL: Name clash not detected~n')
    ),

    % Test 4: No conflicts for compatible patterns
    format('~nTest 4: Compatible patterns...~n'),
    (   detect_conflicts([test_nav, test_query], Conflicts2),
        Conflicts2 = []
    ->  format('  PASS: No false conflicts~n')
    ;   format('  FAIL: False conflicts detected~n')
    ),

    % Test 5: Target validation
    format('~nTest 5: Target validation...~n'),
    (   validate_for_target([test_nav], react_native, Errors1),
        Errors1 = []
    ->  format('  PASS: Valid for target~n')
    ;   format('  FAIL: Validation incorrect~n')
    ),

    % Test 6: Composition summary
    format('~nTest 6: Composition summary...~n'),
    (   composition_summary([test_nav, test_query, test_store1], Summary),
        Summary = summary(pattern_count(3), _, _, _)
    ->  format('  PASS: Summary generated~n')
    ;   format('  FAIL: Summary incorrect~n')
    ),

    % Test 7: Full composition with deps
    format('~nTest 7: Full composition with deps...~n'),
    (   compose_with_deps([test_screen_with_deps], react_native, [], Result),
        Result = composition(patterns(Ps), _, _, _),
        length(Ps, L),
        L >= 2
    ->  format('  PASS: Composition with deps succeeded~n')
    ;   format('  FAIL: Composition failed~n')
    ),

    % Cleanup
    cleanup_test_patterns,

    format('~n=== Tests Complete ===~n').

setup_test_patterns :-
    % Navigation pattern
    ui_patterns:define_pattern(test_nav,
        navigation(tab, [screen(home, 'Home', [])], []),
        [requires([navigation])]),

    % Query pattern
    ui_patterns:define_pattern(test_query,
        data(query, [name(fetchData), endpoint('/api/data')]),
        [requires([react_query])]),

    % Two stores with same name (conflict)
    ui_patterns:define_pattern(test_store1,
        state(global, [store(appStore), slices([])], []),
        [singleton(true)]),
    ui_patterns:define_pattern(test_store2,
        state(global, [store(appStore), slices([])], []),
        [singleton(true)]),

    % Pattern with dependency
    ui_patterns:define_pattern(test_base_store,
        state(global, [store(baseStore), slices([])], []),
        []),
    ui_patterns:define_pattern(test_screen_with_deps,
        navigation(stack, [screen(main, 'Main', [])], []),
        [depends_on([test_base_store])]).

cleanup_test_patterns :-
    retractall(ui_patterns:stored_pattern(test_nav, _, _)),
    retractall(ui_patterns:stored_pattern(test_query, _, _)),
    retractall(ui_patterns:stored_pattern(test_store1, _, _)),
    retractall(ui_patterns:stored_pattern(test_store2, _, _)),
    retractall(ui_patterns:stored_pattern(test_base_store, _, _)),
    retractall(ui_patterns:stored_pattern(test_screen_with_deps, _, _)).

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Pattern composition module loaded~n', [])
), now).
