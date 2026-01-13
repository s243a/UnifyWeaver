% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_project_generator.pl - plunit tests for Project Generator
%
% Tests project file generation and directory structure creation.
%
% Run with: swipl -g "run_tests" -t halt test_project_generator.pl

:- module(test_project_generator, []).

:- use_module(library(plunit)).
:- use_module('project_generator').

% ============================================================================
% Tests: Package.json Generation
% ============================================================================

:- begin_tests(package_json).

test(react_native_has_name) :-
    project_generator:generate_package_json(testapp, react_native, Json),
    sub_string(Json, _, _, _, "testapp").

test(react_native_has_react) :-
    project_generator:generate_package_json(testapp, react_native, Json),
    sub_string(Json, _, _, _, "\"react\":").

test(react_native_has_zustand) :-
    project_generator:generate_package_json(testapp, react_native, Json),
    sub_string(Json, _, _, _, "zustand").

test(react_native_has_react_query) :-
    project_generator:generate_package_json(testapp, react_native, Json),
    sub_string(Json, _, _, _, "@tanstack/react-query").

test(vue_has_name) :-
    project_generator:generate_package_json(testapp, vue, Json),
    sub_string(Json, _, _, _, "testapp").

test(vue_has_vue) :-
    project_generator:generate_package_json(testapp, vue, Json),
    sub_string(Json, _, _, _, "\"vue\":").

test(vue_has_pinia) :-
    project_generator:generate_package_json(testapp, vue, Json),
    sub_string(Json, _, _, _, "pinia").

test(vue_has_router) :-
    project_generator:generate_package_json(testapp, vue, Json),
    sub_string(Json, _, _, _, "vue-router").

:- end_tests(package_json).

% ============================================================================
% Tests: Requirements.txt Generation
% ============================================================================

:- begin_tests(requirements_txt).

test(fastapi_has_fastapi) :-
    project_generator:generate_requirements_txt(fastapi, Content),
    sub_string(Content, _, _, _, "fastapi").

test(fastapi_has_uvicorn) :-
    project_generator:generate_requirements_txt(fastapi, Content),
    sub_string(Content, _, _, _, "uvicorn").

test(fastapi_has_pydantic) :-
    project_generator:generate_requirements_txt(fastapi, Content),
    sub_string(Content, _, _, _, "pydantic").

test(flask_has_flask) :-
    project_generator:generate_requirements_txt(flask, Content),
    sub_string(Content, _, _, _, "flask").

test(flask_has_cors) :-
    project_generator:generate_requirements_txt(flask, Content),
    sub_string(Content, _, _, _, "flask-cors").

:- end_tests(requirements_txt).

% ============================================================================
% Tests: Pubspec.yaml Generation
% ============================================================================

:- begin_tests(pubspec_yaml).

test(pubspec_has_name) :-
    project_generator:generate_pubspec_yaml(testapp, Yaml),
    sub_string(Yaml, _, _, _, "name: testapp").

test(pubspec_has_flutter) :-
    project_generator:generate_pubspec_yaml(testapp, Yaml),
    sub_string(Yaml, _, _, _, "flutter:").

test(pubspec_has_riverpod) :-
    project_generator:generate_pubspec_yaml(testapp, Yaml),
    sub_string(Yaml, _, _, _, "flutter_riverpod").

test(pubspec_has_go_router) :-
    project_generator:generate_pubspec_yaml(testapp, Yaml),
    sub_string(Yaml, _, _, _, "go_router").

:- end_tests(pubspec_yaml).

% ============================================================================
% Tests: TSConfig Generation
% ============================================================================

:- begin_tests(tsconfig).

test(tsconfig_has_strict) :-
    project_generator:generate_tsconfig(Json),
    sub_string(Json, _, _, _, "\"strict\": true").

test(tsconfig_has_es2020) :-
    project_generator:generate_tsconfig(Json),
    sub_string(Json, _, _, _, "ES2020").

test(tsconfig_has_jsx) :-
    project_generator:generate_tsconfig(Json),
    sub_string(Json, _, _, _, "jsx").

:- end_tests(tsconfig).

% ============================================================================
% Tests: React Native App Generation
% ============================================================================

:- begin_tests(react_native_app).

test(app_has_import_react) :-
    project_generator:generate_react_native_app(testapp, Content),
    sub_string(Content, _, _, _, "import React").

test(app_has_navigation_container) :-
    project_generator:generate_react_native_app(testapp, Content),
    sub_string(Content, _, _, _, "NavigationContainer").

test(app_has_query_client_provider) :-
    project_generator:generate_react_native_app(testapp, Content),
    sub_string(Content, _, _, _, "QueryClientProvider").

test(app_has_export_default) :-
    project_generator:generate_react_native_app(testapp, Content),
    sub_string(Content, _, _, _, "export default").

:- end_tests(react_native_app).

% ============================================================================
% Tests: Vue Main Generation
% ============================================================================

:- begin_tests(vue_main).

test(main_has_create_app) :-
    project_generator:generate_vue_main(testapp, Content),
    sub_string(Content, _, _, _, "createApp").

test(main_has_pinia) :-
    project_generator:generate_vue_main(testapp, Content),
    sub_string(Content, _, _, _, "createPinia").

test(main_has_router) :-
    project_generator:generate_vue_main(testapp, Content),
    sub_string(Content, _, _, _, "createRouter").

test(main_has_vue_query) :-
    project_generator:generate_vue_main(testapp, Content),
    sub_string(Content, _, _, _, "VueQueryPlugin").

:- end_tests(vue_main).

% ============================================================================
% Tests: Flutter Main Generation
% ============================================================================

:- begin_tests(flutter_main).

test(main_has_material) :-
    project_generator:generate_flutter_main(testapp, Content),
    sub_string(Content, _, _, _, "package:flutter/material.dart").

test(main_has_riverpod) :-
    project_generator:generate_flutter_main(testapp, Content),
    sub_string(Content, _, _, _, "ProviderScope").

test(main_has_go_router) :-
    project_generator:generate_flutter_main(testapp, Content),
    sub_string(Content, _, _, _, "GoRouter").

test(main_has_app_class) :-
    project_generator:generate_flutter_main(testapp, Content),
    sub_string(Content, _, _, _, "TestappApp").

:- end_tests(flutter_main).

% ============================================================================
% Tests: SwiftUI App Generation
% ============================================================================

:- begin_tests(swiftui_app).

test(app_has_import_swiftui) :-
    project_generator:generate_swiftui_app(testapp, Content),
    sub_string(Content, _, _, _, "import SwiftUI").

test(app_has_main_attribute) :-
    project_generator:generate_swiftui_app(testapp, Content),
    sub_string(Content, _, _, _, "@main").

test(app_has_window_group) :-
    project_generator:generate_swiftui_app(testapp, Content),
    sub_string(Content, _, _, _, "WindowGroup").

test(app_has_navigation_stack) :-
    project_generator:generate_swiftui_app(testapp, Content),
    sub_string(Content, _, _, _, "NavigationStack").

:- end_tests(swiftui_app).

% ============================================================================
% Tests: Swift Package Generation
% ============================================================================

:- begin_tests(swift_package).

test(package_has_name) :-
    project_generator:generate_swift_package(testapp, Content),
    sub_string(Content, _, _, _, "name: \"testapp\"").

test(package_has_ios_platform) :-
    project_generator:generate_swift_package(testapp, Content),
    sub_string(Content, _, _, _, ".iOS").

test(package_has_targets) :-
    project_generator:generate_swift_package(testapp, Content),
    sub_string(Content, _, _, _, "targets:").

:- end_tests(swift_package).

% ============================================================================
% Tests: Component File Generation
% ============================================================================

:- begin_tests(component_files).

test(component_has_interface) :-
    project_generator:generate_component_file(product, react_native, _, Content),
    sub_string(Content, _, _, _, "interface ProductProps").

test(component_has_export) :-
    project_generator:generate_component_file(product, react_native, _, Content),
    sub_string(Content, _, _, _, "export const Product").

test(component_has_styles) :-
    project_generator:generate_component_file(product, react_native, _, Content),
    sub_string(Content, _, _, _, "StyleSheet.create").

:- end_tests(component_files).

% ============================================================================
% Tests: Screen File Generation
% ============================================================================

:- begin_tests(screen_files).

test(screen_has_export) :-
    project_generator:generate_screen_file(home, react_native, _, Content),
    sub_string(Content, _, _, _, "export const HomeScreen").

test(screen_has_view) :-
    project_generator:generate_screen_file(home, react_native, _, Content),
    sub_string(Content, _, _, _, "<View").

:- end_tests(screen_files).

% ============================================================================
% Tests: Store File Generation
% ============================================================================

:- begin_tests(store_files).

test(store_has_create) :-
    project_generator:generate_store_file(cart, react_native, _, Content),
    sub_string(Content, _, _, _, "create").

test(store_has_interface) :-
    project_generator:generate_store_file(cart, react_native, _, Content),
    sub_string(Content, _, _, _, "interface").

:- end_tests(store_files).

% ============================================================================
% Tests: API Client File Generation
% ============================================================================

:- begin_tests(api_client_files).

test(api_has_query_client) :-
    project_generator:generate_api_client_file(react_native, [], Content),
    sub_string(Content, _, _, _, "QueryClient").

test(api_has_base_url) :-
    project_generator:generate_api_client_file(react_native, [], Content),
    sub_string(Content, _, _, _, "BASE_URL").

test(api_has_get_method) :-
    project_generator:generate_api_client_file(react_native, [], Content),
    sub_string(Content, _, _, _, "get:").

test(api_has_post_method) :-
    project_generator:generate_api_client_file(react_native, [], Content),
    sub_string(Content, _, _, _, "post:").

:- end_tests(api_client_files).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
