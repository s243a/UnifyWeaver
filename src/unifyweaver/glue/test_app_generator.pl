% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_app_generator.pl - plunit tests for app generator and data integration
%
% Run with: swipl -g "run_tests" -t halt test_app_generator.pl

:- module(test_app_generator, []).

:- use_module(library(plunit)).
:- use_module('app_generator').
:- use_module('data_integration').

% ============================================================================
% Tests: App Spec Accessors
% ============================================================================

:- begin_tests(app_spec_accessors).

test(app_name_extraction) :-
    app_generator:app_name(app(my_app, []), my_app).

test(app_name_with_config) :-
    app_generator:app_name(app(test_app, [theme([]), locales([en])]), test_app).

test(app_theme_present) :-
    app_generator:app_theme(app(test, [theme([colors([primary-'#FF0000'])])]), Theme),
    member(colors(_), Theme).

test(app_theme_missing) :-
    app_generator:app_theme(app(test, []), Theme),
    Theme = [].

test(app_locales_present) :-
    app_generator:app_locales(app(test, [locales([en, es, fr])]), Locales),
    Locales = [en, es, fr].

test(app_locales_missing_defaults_en) :-
    app_generator:app_locales(app(test, []), Locales),
    Locales = [en].

:- end_tests(app_spec_accessors).

% ============================================================================
% Tests: Entry Point Generation
% ============================================================================

:- begin_tests(entry_point_generation).

test(react_native_entry_point) :-
    app_generator:generate_entry_point(
        app(test_app, [navigation(tab, [screen(home, 'Home', [])], [])]),
        react_native, Code),
    sub_atom(Code, _, _, _, 'QueryClientProvider'),
    sub_atom(Code, _, _, _, 'NavigationContainer').

test(vue_entry_point) :-
    app_generator:generate_entry_point(
        app(test_app, []),
        vue, Code),
    sub_atom(Code, _, _, _, 'createApp'),
    sub_atom(Code, _, _, _, 'createPinia').

test(flutter_entry_point) :-
    app_generator:generate_entry_point(
        app(test_app, []),
        flutter, Code),
    sub_atom(Code, _, _, _, 'runApp'),
    sub_atom(Code, _, _, _, 'ProviderScope').

test(swiftui_entry_point) :-
    app_generator:generate_entry_point(
        app(test_app, []),
        swiftui, Code),
    sub_atom(Code, _, _, _, '@main'),
    sub_atom(Code, _, _, _, 'WindowGroup').

:- end_tests(entry_point_generation).

% ============================================================================
% Tests: Theme File Generation
% ============================================================================

:- begin_tests(theme_file_generation).

test(react_native_theme_files) :-
    app_generator:generate_theme_files(
        app(test, [theme([colors([primary-'#6366F1'])])]),
        react_native, Files),
    length(Files, 1),
    member(file(_, Code), Files),
    sub_atom(Code, _, _, _, 'ThemeProvider').

test(vue_theme_files) :-
    app_generator:generate_theme_files(
        app(test, [theme([colors([primary-'#6366F1'])])]),
        vue, Files),
    length(Files, 1),
    member(file(_, Code), Files),
    sub_atom(Code, _, _, _, '--color-primary').

test(flutter_theme_files) :-
    app_generator:generate_theme_files(
        app(test, [theme([colors([primary-'#6366F1'])])]),
        flutter, Files),
    length(Files, 1),
    member(file(_, Code), Files),
    sub_atom(Code, _, _, _, 'themeProvider').

test(swiftui_theme_files) :-
    app_generator:generate_theme_files(
        app(test, [theme([])]),
        swiftui, Files),
    length(Files, 1),
    member(file(_, Code), Files),
    sub_atom(Code, _, _, _, 'struct Theme').

:- end_tests(theme_file_generation).

% ============================================================================
% Tests: Locale File Generation
% ============================================================================

:- begin_tests(locale_file_generation).

test(react_native_locale_files) :-
    app_generator:generate_locale_files(
        app(test, [locales([en, es])]),
        react_native, Files),
    length(Files, 2).

test(vue_locale_files) :-
    app_generator:generate_locale_files(
        app(test, [locales([en])]),
        vue, Files),
    length(Files, 1).

test(flutter_locale_files) :-
    app_generator:generate_locale_files(
        app(test, [locales([en, fr, de])]),
        flutter, Files),
    length(Files, 3).

test(swiftui_locale_files) :-
    app_generator:generate_locale_files(
        app(test, [locales([en])]),
        swiftui, Files),
    length(Files, 1).

:- end_tests(locale_file_generation).

% ============================================================================
% Tests: Navigation File Generation
% ============================================================================

:- begin_tests(navigation_file_generation).

test(react_native_tab_navigation) :-
    app_generator:generate_navigation_file(
        app(test, [navigation(tab, [
            screen(home, 'HomeScreen', []),
            screen(settings, 'SettingsScreen', [])
        ], [])]),
        react_native, Code),
    sub_atom(Code, _, _, _, 'createBottomTabNavigator').

test(vue_router_navigation) :-
    app_generator:generate_navigation_file(
        app(test, [navigation(router, [
            screen(home, 'HomeView', []),
            screen(about, 'AboutView', [])
        ], [])]),
        vue, Code),
    sub_atom(Code, _, _, _, 'createRouter').

test(flutter_go_router) :-
    app_generator:generate_navigation_file(
        app(test, [navigation(tab, [
            screen(home, 'HomeScreen', [])
        ], [])]),
        flutter, Code),
    sub_atom(Code, _, _, _, 'GoRouter').

test(swiftui_tabview) :-
    app_generator:generate_navigation_file(
        app(test, [navigation(tab, [
            screen(home, 'HomeView', [])
        ], [])]),
        swiftui, Code),
    sub_atom(Code, _, _, _, 'TabView').

:- end_tests(navigation_file_generation).

% ============================================================================
% Tests: API Client Generation
% ============================================================================

:- begin_tests(api_client_generation).

test(react_native_api_client) :-
    app_generator:generate_api_client(app(test, []), react_native, Code),
    sub_atom(Code, _, _, _, 'axios'),
    sub_atom(Code, _, _, _, 'create').

test(vue_api_client) :-
    app_generator:generate_api_client(app(test, []), vue, Code),
    sub_atom(Code, _, _, _, 'axios').

test(flutter_api_client) :-
    app_generator:generate_api_client(app(test, []), flutter, Code),
    sub_atom(Code, _, _, _, 'http'),
    sub_atom(Code, _, _, _, 'class ApiClient').

test(swiftui_api_client) :-
    app_generator:generate_api_client(app(test, []), swiftui, Code),
    sub_atom(Code, _, _, _, 'URLSession'),
    sub_atom(Code, _, _, _, 'class APIClient').

:- end_tests(api_client_generation).

% ============================================================================
% Tests: Data Integration - Source Endpoints
% ============================================================================

:- begin_tests(data_integration_endpoints).

test(sqlite_endpoint) :-
    data_integration:source_endpoint(sqlite('test.db', 'SELECT 1'), Endpoint),
    Endpoint = '/api/data'.

test(csv_endpoint) :-
    data_integration:source_endpoint(csv('data/users.csv'), Endpoint),
    sub_atom(Endpoint, _, _, _, 'users').

test(json_endpoint) :-
    data_integration:source_endpoint(json('products.json'), Endpoint),
    sub_atom(Endpoint, _, _, _, 'products').

test(http_endpoint) :-
    data_integration:source_endpoint(http('https://api.example.com'), Endpoint),
    sub_atom(Endpoint, _, _, _, 'proxy').

test(custom_endpoint) :-
    data_integration:source_endpoint(custom(endpoint('/api/custom')), '/api/custom').

:- end_tests(data_integration_endpoints).

% ============================================================================
% Tests: Data Integration - Backend Handlers
% ============================================================================

:- begin_tests(data_integration_backend).

test(fastapi_sqlite_handler) :-
    data_integration:generate_backend_handler(
        binding(tasks, sqlite('app.db', 'SELECT * FROM tasks'), []),
        fastapi, [], Code),
    sub_atom(Code, _, _, _, '@router.get'),
    sub_atom(Code, _, _, _, 'sqlite3.connect').

test(flask_csv_handler) :-
    data_integration:generate_backend_handler(
        binding(users, csv('users.csv'), [endpoint('/api/users')]),
        flask, [], Code),
    sub_atom(Code, _, _, _, '@bp.route'),
    sub_atom(Code, _, _, _, 'DictReader').

test(express_json_handler) :-
    data_integration:generate_backend_handler(
        binding(items, json('items.json'), []),
        express, [], Code),
    sub_atom(Code, _, _, _, 'router.get'),
    sub_atom(Code, _, _, _, 'readFileSync').

test(fastapi_http_handler) :-
    data_integration:generate_backend_handler(
        binding(external, http('https://api.example.com'), []),
        fastapi, [], Code),
    sub_atom(Code, _, _, _, 'httpx.AsyncClient').

test(all_handlers_generation) :-
    data_integration:generate_all_handlers([
        binding(a, sqlite('db.db', 'SELECT 1'), []),
        binding(b, csv('data.csv'), [])
    ], fastapi, [], Code),
    sub_atom(Code, _, _, _, 'get_a'),
    sub_atom(Code, _, _, _, 'get_b').

:- end_tests(data_integration_backend).

% ============================================================================
% Tests: Data Integration - Frontend Queries
% ============================================================================

:- begin_tests(data_integration_frontend).

test(react_native_query_hook) :-
    data_integration:generate_frontend_query(
        binding(tasks, sqlite('db', ''), [endpoint('/api/tasks')]),
        react_native, [], Code),
    sub_atom(Code, _, _, _, 'useTasks'),
    sub_atom(Code, _, _, _, 'useQuery').

test(react_native_mutation_hook) :-
    data_integration:generate_frontend_query(
        binding(tasks, sqlite('db', ''), [endpoint('/api/tasks')]),
        react_native, [], Code),
    sub_atom(Code, _, _, _, 'useTasksMutation'),
    sub_atom(Code, _, _, _, 'useMutation').

test(vue_query_composable) :-
    data_integration:generate_frontend_query(
        binding(users, http('url'), [endpoint('/api/users')]),
        vue, [], Code),
    sub_atom(Code, _, _, _, 'useUsers'),
    sub_atom(Code, _, _, _, 'useQuery').

test(flutter_provider) :-
    data_integration:generate_frontend_query(
        binding(products, json('p.json'), [endpoint('/api/products')]),
        flutter, [], Code),
    sub_atom(Code, _, _, _, 'productsProvider'),
    sub_atom(Code, _, _, _, 'FutureProvider').

test(swiftui_viewmodel) :-
    data_integration:generate_frontend_query(
        binding(orders, sqlite('db', ''), [endpoint('/api/orders')]),
        swiftui, [], Code),
    sub_atom(Code, _, _, _, 'OrdersViewModel'),
    sub_atom(Code, _, _, _, 'ObservableObject').

test(all_queries_generation) :-
    data_integration:generate_all_queries([
        binding(a, sqlite('db', ''), [endpoint('/a')]),
        binding(b, csv('f.csv'), [endpoint('/b')])
    ], react_native, [], Code),
    sub_atom(Code, _, _, _, 'useA'),
    sub_atom(Code, _, _, _, 'useB').

:- end_tests(data_integration_frontend).

% ============================================================================
% Tests: Full Stack Data Layer
% ============================================================================

:- begin_tests(full_stack_data_layer).

test(generates_backend_and_frontend) :-
    data_integration:generate_data_layer([
        binding(tasks, sqlite('app.db', 'SELECT * FROM tasks'), [endpoint('/api/tasks')])
    ], react_native, fastapi, [], Result),
    Result = data_layer(
        backend(fastapi, BackendCode),
        frontend(react_native, FrontendCode)
    ),
    sub_atom(BackendCode, _, _, _, '@router'),
    sub_atom(FrontendCode, _, _, _, 'useQuery').

test(multiple_bindings) :-
    data_integration:generate_data_layer([
        binding(users, csv('users.csv'), [endpoint('/api/users')]),
        binding(products, json('products.json'), [endpoint('/api/products')]),
        binding(orders, sqlite('db.sqlite', 'SELECT * FROM orders'), [endpoint('/api/orders')])
    ], vue, flask, [], Result),
    Result = data_layer(backend(flask, _), frontend(vue, _)).

:- end_tests(full_stack_data_layer).

% ============================================================================
% Tests: Utility Functions
% ============================================================================

:- begin_tests(utility_functions).

test(capitalize_first_lowercase) :-
    app_generator:capitalize_first(hello, Hello),
    Hello = 'Hello'.

test(capitalize_first_already_upper) :-
    app_generator:capitalize_first('World', W),
    W = 'World'.

test(data_integration_capitalize) :-
    data_integration:capitalize_first(tasks, Tasks),
    Tasks = 'Tasks'.

:- end_tests(utility_functions).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
