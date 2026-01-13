% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_pattern_composition.pl - Tests for Pattern Composition Tools
%
% Run with: swipl -g "run_tests" -t halt test_pattern_composition.pl

:- module(test_pattern_composition, []).

:- use_module(library(plunit)).
:- use_module('ui_patterns').
:- use_module('pattern_composition').

% ============================================================================
% Test Setup/Cleanup
% ============================================================================

setup_patterns :-
    % Basic navigation pattern
    ui_patterns:define_pattern(nav_home,
        navigation(stack, [screen(home, 'HomeScreen', [title('Home')])], []),
        [requires([navigation])]),

    % Tab navigation
    ui_patterns:define_pattern(nav_tabs,
        navigation(tab, [
            screen(feed, 'FeedScreen', []),
            screen(profile, 'ProfileScreen', [])
        ], []),
        [requires([navigation])]),

    % Global state stores
    ui_patterns:define_pattern(store_user,
        state(global, [store(userStore), slices([
            slice(user, [field(name, string)], [])
        ])], []),
        [requires([zustand]), singleton(true)]),

    ui_patterns:define_pattern(store_app,
        state(global, [store(appStore), slices([
            slice(theme, [field(mode, string)], [])
        ])], []),
        [requires([zustand]), singleton(true)]),

    % Conflicting store (same name as store_user)
    ui_patterns:define_pattern(store_conflict,
        state(global, [store(userStore), slices([])], []),
        [singleton(true)]),

    % Query patterns
    ui_patterns:define_pattern(query_users,
        data(query, [name(fetchUsers), endpoint('/api/users')]),
        [requires([react_query])]),

    ui_patterns:define_pattern(query_posts,
        data(query, [name(fetchPosts), endpoint('/api/posts')]),
        [requires([react_query])]),

    % Persistence pattern
    ui_patterns:define_pattern(persist_prefs,
        persistence(local, [key(userPrefs), schema('{ theme: string }')]),
        [requires([async_storage])]),

    % Pattern with dependencies
    ui_patterns:define_pattern(screen_user_profile,
        navigation(stack, [screen(profile, 'UserProfile', [])], []),
        [depends_on([store_user, query_users])]),

    % Pattern that depends on another dependent
    ui_patterns:define_pattern(screen_full_app,
        navigation(tab, [
            screen(home, 'Home', []),
            screen(profile, 'Profile', [])
        ], []),
        [depends_on([screen_user_profile, store_app])]),

    % Mutually exclusive patterns
    ui_patterns:define_pattern(auth_jwt,
        state(global, [store(authStore), slices([])], []),
        [excludes([auth_session])]),

    ui_patterns:define_pattern(auth_session,
        state(global, [store(sessionStore), slices([])], []),
        [excludes([auth_jwt])]),

    % Pattern requiring unavailable capability
    ui_patterns:define_pattern(pattern_special,
        state(local, [field(x, 0)], []),
        [requires([special_capability])]).

cleanup_patterns :-
    retractall(ui_patterns:stored_pattern(nav_home, _, _)),
    retractall(ui_patterns:stored_pattern(nav_tabs, _, _)),
    retractall(ui_patterns:stored_pattern(store_user, _, _)),
    retractall(ui_patterns:stored_pattern(store_app, _, _)),
    retractall(ui_patterns:stored_pattern(store_conflict, _, _)),
    retractall(ui_patterns:stored_pattern(query_users, _, _)),
    retractall(ui_patterns:stored_pattern(query_posts, _, _)),
    retractall(ui_patterns:stored_pattern(persist_prefs, _, _)),
    retractall(ui_patterns:stored_pattern(screen_user_profile, _, _)),
    retractall(ui_patterns:stored_pattern(screen_full_app, _, _)),
    retractall(ui_patterns:stored_pattern(auth_jwt, _, _)),
    retractall(ui_patterns:stored_pattern(auth_session, _, _)),
    retractall(ui_patterns:stored_pattern(pattern_special, _, _)).

% ============================================================================
% Tests: Dependency Resolution
% ============================================================================

:- begin_tests(dependency_resolution, [setup(setup_patterns), cleanup(cleanup_patterns)]).

test(resolve_no_deps) :-
    resolve_dependencies([nav_home], Resolved),
    Resolved = [nav_home].

test(resolve_single_dep) :-
    resolve_dependencies([screen_user_profile], Resolved),
    member(store_user, Resolved),
    member(query_users, Resolved),
    member(screen_user_profile, Resolved).

test(resolve_transitive_deps) :-
    resolve_dependencies([screen_full_app], Resolved),
    member(screen_full_app, Resolved),
    member(screen_user_profile, Resolved),
    member(store_user, Resolved),
    member(store_app, Resolved),
    member(query_users, Resolved).

test(resolve_multiple_roots) :-
    resolve_dependencies([nav_home, query_users], Resolved),
    member(nav_home, Resolved),
    member(query_users, Resolved),
    length(Resolved, 2).

test(resolve_no_duplicates) :-
    resolve_dependencies([screen_user_profile, store_user], Resolved),
    msort(Resolved, Sorted),
    length(Resolved, L1),
    length(Sorted, L2),
    L1 = L2.

:- end_tests(dependency_resolution).

% ============================================================================
% Tests: Dependency Ordering
% ============================================================================

:- begin_tests(dependency_ordering, [setup(setup_patterns), cleanup(cleanup_patterns)]).

test(order_independent_patterns) :-
    dependency_order([nav_home, query_users], Ordered),
    length(Ordered, 2).

test(order_deps_before_dependents) :-
    dependency_order([screen_user_profile, store_user, query_users], Ordered),
    nth0(I1, Ordered, store_user),
    nth0(I2, Ordered, query_users),
    nth0(I3, Ordered, screen_user_profile),
    I1 < I3,
    I2 < I3.

test(order_transitive_deps) :-
    resolve_dependencies([screen_full_app], All),
    dependency_order(All, Ordered),
    nth0(IStore, Ordered, store_user),
    nth0(IProfile, Ordered, screen_user_profile),
    nth0(IApp, Ordered, screen_full_app),
    IStore < IProfile,
    IProfile < IApp.

:- end_tests(dependency_ordering).

% ============================================================================
% Tests: Conflict Detection
% ============================================================================

:- begin_tests(conflict_detection, [setup(setup_patterns), cleanup(cleanup_patterns)]).

test(detect_name_conflict) :-
    detect_conflicts([store_user, store_conflict], Conflicts),
    member(conflict(name_clash, _, _, _), Conflicts).

test(detect_singleton_conflict) :-
    detect_conflicts([store_user, store_conflict], Conflicts),
    member(conflict(singleton, _, _, _), Conflicts).

test(detect_exclusion_conflict) :-
    detect_conflicts([auth_jwt, auth_session], Conflicts),
    member(conflict(exclusion, _, _, _), Conflicts).

test(no_conflict_compatible) :-
    detect_conflicts([nav_home, query_users, persist_prefs], Conflicts),
    Conflicts = [].

test(no_conflict_different_stores) :-
    detect_conflicts([store_user, store_app], Conflicts),
    Conflicts = [].

test(conflict_details_present) :-
    detect_conflicts([store_user, store_conflict], Conflicts),
    member(conflict(_, P1, P2, Details), Conflicts),
    atom(P1),
    atom(P2),
    string(Details).

:- end_tests(conflict_detection).

% ============================================================================
% Tests: Target Validation
% ============================================================================

:- begin_tests(target_validation, [setup(setup_patterns), cleanup(cleanup_patterns)]).

test(valid_react_native) :-
    validate_for_target([nav_home, query_users, store_user], react_native, Errors),
    Errors = [].

test(valid_vue) :-
    validate_for_target([nav_home], vue, Errors),
    Errors = [].

test(valid_flutter) :-
    validate_for_target([nav_home], flutter, Errors),
    Errors = [].

test(valid_swiftui) :-
    validate_for_target([nav_home], swiftui, Errors),
    Errors = [].

test(invalid_missing_capability) :-
    validate_for_target([pattern_special], react_native, Errors),
    member(error(missing_capability, pattern_special, react_native, special_capability), Errors).

:- end_tests(target_validation).

% ============================================================================
% Tests: Full Composition
% ============================================================================

:- begin_tests(full_composition, [setup(setup_patterns), cleanup(cleanup_patterns)]).

test(compose_simple) :-
    compose_with_deps([nav_home], react_native, [], Result),
    Result = composition(patterns([nav_home]), _, target(react_native), _).

test(compose_with_deps_resolves) :-
    compose_with_deps([screen_user_profile], react_native, [], Result),
    Result = composition(patterns(Ps), _, _, _),
    member(store_user, Ps),
    member(query_users, Ps).

test(compose_fails_on_conflict) :-
    compose_with_deps([store_user, store_conflict], react_native, [], Result),
    Result = composition_error(Errors),
    Errors \= [].

test(compose_fails_on_exclusion) :-
    compose_with_deps([auth_jwt, auth_session], react_native, [], Result),
    Result = composition_error(_).

test(compose_includes_specs) :-
    compose_with_deps([nav_home], react_native, [], Result),
    Result = composition(_, specs(Specs), _, _),
    member(spec(nav_home, _), Specs).

:- end_tests(full_composition).

% ============================================================================
% Tests: Analysis
% ============================================================================

:- begin_tests(composition_analysis, [setup(setup_patterns), cleanup(cleanup_patterns)]).

test(composition_graph_nodes) :-
    composition_graph([nav_home, store_user], Graph),
    Graph = graph(Nodes, _),
    member(node(nav_home, navigation, _), Nodes),
    member(node(store_user, state(global), _), Nodes).

test(composition_graph_edges) :-
    composition_graph([screen_user_profile, store_user, query_users], Graph),
    Graph = graph(_, Edges),
    member(edge(screen_user_profile, store_user), Edges),
    member(edge(screen_user_profile, query_users), Edges).

test(required_capabilities_collected) :-
    required_capabilities([nav_home, query_users, store_user], Caps),
    member(navigation, Caps),
    member(react_query, Caps),
    member(zustand, Caps).

test(composition_summary_count) :-
    composition_summary([nav_home, query_users, store_user], Summary),
    Summary = summary(pattern_count(3), _, _, _).

test(composition_summary_types) :-
    composition_summary([nav_home, query_users], Summary),
    Summary = summary(_, type_breakdown(Types), _, _),
    member(type(navigation, 1), Types),
    member(type(data(query), 1), Types).

test(composition_summary_conflicts) :-
    composition_summary([store_user, store_conflict], Summary),
    Summary = summary(_, _, _, conflict_count(N)),
    N > 0.

:- end_tests(composition_analysis).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
