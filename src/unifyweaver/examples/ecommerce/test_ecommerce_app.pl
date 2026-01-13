% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_ecommerce_app.pl - plunit tests for E-commerce example
%
% Tests the e-commerce app pattern definitions and project generation.
%
% Run with: swipl -g "run_tests" -t halt test_ecommerce_app.pl

:- module(test_ecommerce_app, []).

:- use_module(library(plunit)).
:- use_module(ecommerce_app).
:- use_module('../../patterns/ui_patterns').

% ============================================================================
% Tests: Pattern Definition
% ============================================================================

:- begin_tests(pattern_definition).

test(defines_main_tabs) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(main_tabs, Spec, _),
    Spec = navigation(tab, Screens, _),
    length(Screens, 4).

test(defines_cart_store) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(cart_store, Spec, _),
    Spec = state(global, _, _).

test(defines_user_store) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(user_store, Spec, _),
    Spec = state(global, _, _).

test(defines_filter_store) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(filter_store, Spec, _),
    Spec = state(global, _, _).

test(defines_product_queries) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(fetch_products, data(query, _), _),
    ui_patterns:stored_pattern(fetch_product, data(query, _), _).

test(defines_order_patterns) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(fetch_orders, data(query, _), _),
    ui_patterns:stored_pattern(create_order, data(mutation, _), _).

test(defines_persistence) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(cart_persistence, persistence(local, _), _),
    ui_patterns:stored_pattern(user_prefs, persistence(local, _), _).

:- end_tests(pattern_definition).

% ============================================================================
% Tests: Pattern Content
% ============================================================================

:- begin_tests(pattern_content).

test(main_tabs_has_home_screen) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(main_tabs, navigation(tab, Screens, _), _),
    member(screen(home, _, _), Screens).

test(main_tabs_has_cart_screen) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(main_tabs, navigation(tab, Screens, _), _),
    member(screen(cart, _, _), Screens).

test(cart_store_has_items_field) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(cart_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(cart, Fields, _), Slices),
    member(field(items, _), Fields).

test(cart_store_has_add_item_action) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(cart_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(cart, _, Actions), Slices),
    member(action(addItem, _), Actions).

test(cart_store_has_clear_cart_action) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(cart_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(cart, _, Actions), Slices),
    member(action(clearCart, _), Actions).

test(user_store_has_auth_fields) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(user_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(user, Fields, _), Slices),
    member(field(isAuthenticated, _), Fields),
    member(field(token, _), Fields).

test(fetch_products_has_endpoint) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(fetch_products, data(query, Config), _),
    member(endpoint('/api/products'), Config).

test(create_order_is_post) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(create_order, data(mutation, Config), _),
    member(method('POST'), Config).

:- end_tests(pattern_content).

% ============================================================================
% Tests: Patterns List
% ============================================================================

:- begin_tests(patterns_list).

test(ecommerce_patterns_not_empty) :-
    ecommerce_app:ecommerce_patterns(Patterns),
    length(Patterns, Len),
    Len > 0.

test(ecommerce_patterns_has_navigation) :-
    ecommerce_app:ecommerce_patterns(Patterns),
    member(main_tabs, Patterns).

test(ecommerce_patterns_has_state) :-
    ecommerce_app:ecommerce_patterns(Patterns),
    member(cart_store, Patterns),
    member(user_store, Patterns).

test(ecommerce_patterns_has_data) :-
    ecommerce_app:ecommerce_patterns(Patterns),
    member(fetch_products, Patterns),
    member(create_order, Patterns).

test(ecommerce_patterns_minimum_count) :-
    ecommerce_app:ecommerce_patterns(Patterns),
    length(Patterns, Len),
    Len >= 15.

:- end_tests(patterns_list).

% ============================================================================
% Tests: Endpoints
% ============================================================================

:- begin_tests(endpoints).

test(endpoints_not_empty) :-
    ecommerce_app:ecommerce_endpoints(Endpoints),
    length(Endpoints, Len),
    Len > 0.

test(endpoints_has_products) :-
    ecommerce_app:ecommerce_endpoints(Endpoints),
    member(endpoint(list_products, get, '/api/products', _), Endpoints).

test(endpoints_has_orders) :-
    ecommerce_app:ecommerce_endpoints(Endpoints),
    member(endpoint(create_order, post, '/api/orders', _), Endpoints).

test(endpoints_has_auth) :-
    ecommerce_app:ecommerce_endpoints(Endpoints),
    member(endpoint(login, post, '/api/auth/login', _), Endpoints),
    member(endpoint(register, post, '/api/auth/register', _), Endpoints).

test(endpoints_has_cart) :-
    ecommerce_app:ecommerce_endpoints(Endpoints),
    member(endpoint(add_to_cart, post, '/api/cart/items', _), Endpoints),
    member(endpoint(remove_from_cart, delete, '/api/cart/items/:id', _), Endpoints).

test(endpoints_minimum_count) :-
    ecommerce_app:ecommerce_endpoints(Endpoints),
    length(Endpoints, Len),
    Len >= 15.

:- end_tests(endpoints).

% ============================================================================
% Tests: Navigation Stacks
% ============================================================================

:- begin_tests(navigation_stacks).

test(product_stack_defined) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(product_stack, navigation(stack, _, _), _).

test(checkout_stack_defined) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(checkout_stack, navigation(stack, _, _), _).

test(order_stack_defined) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(order_stack, navigation(stack, _, _), _).

test(product_stack_has_screens) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(product_stack, navigation(stack, Screens, _), _),
    length(Screens, Len),
    Len >= 2.

:- end_tests(navigation_stacks).

% ============================================================================
% Tests: Filter Store
% ============================================================================

:- begin_tests(filter_store).

test(filter_store_has_query) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(filter_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(filters, Fields, _), Slices),
    member(field(query, _), Fields).

test(filter_store_has_category) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(filter_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(filters, Fields, _), Slices),
    member(field(category, _), Fields).

test(filter_store_has_sort) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(filter_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(filters, Fields, _), Slices),
    member(field(sortBy, _), Fields).

test(filter_store_has_clear_action) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(filter_store, state(global, Shape, _), _),
    member(slices(Slices), Shape),
    member(slice(filters, _, Actions), Slices),
    member(action(clearFilters, _), Actions).

:- end_tests(filter_store).

% ============================================================================
% Tests: Data Queries
% ============================================================================

:- begin_tests(data_queries).

test(fetch_products_has_stale_time) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(fetch_products, data(query, Config), _),
    member(stale_time(_), Config).

test(fetch_product_has_param_endpoint) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(fetch_product, data(query, Config), _),
    member(endpoint('/api/products/:id'), Config).

test(search_products_defined) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(search_products, data(query, Config), _),
    member(endpoint('/api/search'), Config).

test(fetch_categories_defined) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(fetch_categories, data(query, Config), _),
    member(endpoint('/api/categories'), Config).

:- end_tests(data_queries).

% ============================================================================
% Tests: Mutations
% ============================================================================

:- begin_tests(mutations).

test(create_order_invalidates_orders) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(create_order, data(mutation, Config), _),
    member(invalidates(Invalidates), Config),
    member(fetch_orders, Invalidates).

test(sync_cart_is_post) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(sync_cart, data(mutation, Config), _),
    member(method('POST'), Config).

:- end_tests(mutations).

% ============================================================================
% Tests: Persistence
% ============================================================================

:- begin_tests(persistence).

test(cart_persistence_has_key) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(cart_persistence, persistence(local, Config), _),
    member(key(cart), Config).

test(user_prefs_has_schema) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(user_prefs, persistence(local, Config), _),
    member(schema(_), Config).

test(auth_storage_is_secure) :-
    ecommerce_app:define_ecommerce_app,
    ui_patterns:stored_pattern(auth_storage, persistence(secure, _), _).

:- end_tests(persistence).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
