% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% pattern_glue.pl - Connect UI Patterns to Backend Services
%
% Bridges the ui_patterns module with backend services via glue infrastructure.
% Generates both frontend and backend code for patterns that require server-side data.
%
% Supported Frontend Targets:
%   - React Native (TypeScript) - @tanstack/react-query
%   - Vue 3 (TypeScript) - Vue composables with ref/reactive
%   - Flutter (Dart) - http package with async/await
%   - SwiftUI (Swift) - URLSession with async/await
%
% Supported Backend Targets:
%   - Express (JavaScript) - express.Router handlers
%   - Go (Golang) - net/http handlers
%   - FastAPI (Python) - async handlers with Pydantic
%   - Flask (Python) - classic handlers with flask-cors
%
% Features:
%   - Auto-detect patterns requiring backend services
%   - Generate API endpoints for data patterns
%   - Generate Express routes for query/mutation patterns
%   - Generate API client code for all frontend targets
%   - Cross-runtime support (multiple frontend + backend combinations)

:- module(pattern_glue, [
    % Pattern analysis
    pattern_requires_backend/2,         % +PatternName, -Reason
    analyze_pattern_dependencies/2,     % +PatternName, -Dependencies

    % Backend generation
    generate_backend_for_pattern/4,     % +PatternName, +Target, +Options, -Code
    generate_express_routes/3,          % +Patterns, +Options, -Code
    generate_go_handlers/3,             % +Patterns, +Options, -Code
    generate_fastapi_routes/3,          % +Patterns, +Options, -Code
    generate_flask_routes/3,            % +Patterns, +Options, -Code

    % Full stack generation
    generate_full_stack/4,              % +Patterns, +Options, -Frontend, -Backend
    generate_api_client/3,              % +Endpoints, +Options, -Code

    % Endpoint specification
    endpoint_spec/4,                    % +Name, +Method, +Path, +Handler
    pattern_to_endpoint/3,              % +PatternName, +Options, -EndpointSpec

    % Testing
    test_pattern_glue/0
]).

:- use_module(library(lists)).

% Try to load ui_patterns if available
:- catch(use_module('../patterns/ui_patterns'), _, true).

% Load target modules for multi-framework support
:- catch(use_module('../targets/vue_target', []), _, true).
:- catch(use_module('../targets/flutter_target', []), _, true).
:- catch(use_module('../targets/swiftui_target', []), _, true).

% Load Python backend generators
:- catch(use_module('./fastapi_generator'), _, true).
:- catch(use_module('./flask_generator'), _, true).

% ============================================================================
% PATTERN ANALYSIS
% ============================================================================

%% pattern_requires_backend(+PatternName, -Reason)
%
%  Check if a pattern requires backend services.
%
%  Reasons:
%    - database_access: Pattern fetches from database
%    - file_system: Pattern reads/writes files
%    - computation: Pattern requires heavy computation
%    - authentication: Pattern requires auth
%    - external_api: Pattern calls external services
%
pattern_requires_backend(PatternName, Reason) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    spec_requires_backend(Spec, Reason).

spec_requires_backend(data(query, Config), database_access) :-
    member(endpoint(Endpoint), Config),
    is_api_endpoint(Endpoint).
spec_requires_backend(data(mutation, Config), database_access) :-
    member(endpoint(Endpoint), Config),
    is_api_endpoint(Endpoint).
spec_requires_backend(data(infinite, Config), database_access) :-
    member(endpoint(Endpoint), Config),
    is_api_endpoint(Endpoint).
spec_requires_backend(persistence(secure, _), authentication).

is_api_endpoint(Endpoint) :-
    atom_string(Endpoint, Str),
    sub_string(Str, 0, _, _, "/api/").
is_api_endpoint(Endpoint) :-
    atom_string(Endpoint, Str),
    sub_string(Str, 0, _, _, "http").

%% analyze_pattern_dependencies(+PatternName, -Dependencies)
%
%  Analyze what a pattern depends on.
%
analyze_pattern_dependencies(PatternName, Dependencies) :-
    catch(ui_patterns:pattern(PatternName, Spec, Opts), _, fail),
    findall(Dep, pattern_dependency(Spec, Opts, Dep), Deps),
    sort(Deps, Dependencies).

pattern_dependency(data(query, Config), _, dep(backend, Endpoint)) :-
    member(endpoint(Endpoint), Config).
pattern_dependency(data(mutation, Config), _, dep(backend, Endpoint)) :-
    member(endpoint(Endpoint), Config).
pattern_dependency(_, Opts, dep(pattern, P)) :-
    member(depends_on(Patterns), Opts),
    member(P, Patterns).
pattern_dependency(_, Opts, dep(capability, C)) :-
    member(requires(Caps), Opts),
    member(C, Caps).

% ============================================================================
% BACKEND GENERATION
% ============================================================================

%% generate_backend_for_pattern(+PatternName, +Target, +Options, -Code)
%
%  Generate backend code for a pattern.
%
%  Target: express | go | fastapi | flask
%
generate_backend_for_pattern(PatternName, express, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_express_handler(PatternName, Spec, Options, Code).
generate_backend_for_pattern(PatternName, go, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_go_handler(PatternName, Spec, Options, Code).
generate_backend_for_pattern(PatternName, fastapi, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    catch(fastapi_generator:generate_fastapi_handler(PatternName, Spec, Options, Code), _, fail).
generate_backend_for_pattern(PatternName, flask, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    catch(flask_generator:generate_flask_handler(PatternName, Spec, Options, Code), _, fail).

generate_express_handler(Name, data(query, Config), _Options, Code) :-
    member(endpoint(Endpoint), Config),
    atom_string(Name, NameStr),
    format(string(Code),
"// Handler for ~w query
router.get('~w', async (req, res) => {
  try {
    // TODO: Implement data fetching logic
    const data = await fetchData(req.query);
    res.json({ success: true, data });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});
", [NameStr, Endpoint]).

generate_express_handler(Name, data(mutation, Config), _Options, Code) :-
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    atom_string(Name, NameStr),
    atom_string(Method, MethodLower),
    downcase_atom(MethodLower, MethodStr),
    format(string(Code),
"// Handler for ~w mutation
router.~w('~w', async (req, res) => {
  try {
    // TODO: Implement mutation logic
    const result = await mutateData(req.body);
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});
", [NameStr, MethodStr, Endpoint]).

generate_express_handler(Name, data(infinite, Config), _Options, Code) :-
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'page' ),
    atom_string(Name, NameStr),
    format(string(Code),
"// Handler for ~w paginated query
router.get('~w', async (req, res) => {
  try {
    const ~w = parseInt(req.query.~w) || 1;
    const limit = parseInt(req.query.limit) || 20;

    // TODO: Implement paginated fetching
    const { data, total } = await fetchPaginated(~w, limit);
    const hasMore = ~w * limit < total;

    res.json({
      success: true,
      data,
      nextPage: hasMore ? ~w + 1 : null,
      hasMore
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});
", [NameStr, Endpoint, PageParam, PageParam, PageParam, PageParam, PageParam]).

generate_express_handler(Name, _, _Options, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"// Handler for ~w (generic)
router.get('/api/~w', async (req, res) => {
  res.json({ message: 'Not implemented' });
});
", [NameStr, NameStr]).

generate_go_handler(Name, data(query, Config), _Options, Code) :-
    member(endpoint(Endpoint), Config),
    atom_string(Name, NameStr),
    capitalize_first(NameStr, HandlerName),
    format(string(Code),
"// ~wHandler handles GET ~w
func ~wHandler(w http.ResponseWriter, r *http.Request) {
    // TODO: Implement data fetching logic
    data := map[string]interface{}{
        \"success\": true,
        \"data\":    nil,
    }

    w.Header().Set(\"Content-Type\", \"application/json\")
    json.NewEncoder(w).Encode(data)
}
", [HandlerName, Endpoint, HandlerName]).

generate_go_handler(Name, data(mutation, Config), _Options, Code) :-
    member(endpoint(Endpoint), Config),
    atom_string(Name, NameStr),
    capitalize_first(NameStr, HandlerName),
    format(string(Code),
"// ~wHandler handles POST ~w
func ~wHandler(w http.ResponseWriter, r *http.Request) {
    var input map[string]interface{}
    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // TODO: Implement mutation logic
    result := map[string]interface{}{
        \"success\": true,
        \"data\":    input,
    }

    w.Header().Set(\"Content-Type\", \"application/json\")
    json.NewEncoder(w).Encode(result)
}
", [HandlerName, Endpoint, HandlerName]).

generate_go_handler(Name, _, _Options, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, HandlerName),
    format(string(Code),
"// ~wHandler - generic handler
func ~wHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set(\"Content-Type\", \"application/json\")
    json.NewEncoder(w).Encode(map[string]string{\"message\": \"Not implemented\"})
}
", [HandlerName, HandlerName]).

%% generate_express_routes(+Patterns, +Options, -Code)
%
%  Generate Express router with all pattern handlers.
%
generate_express_routes(Patterns, Options, Code) :-
    option_value(Options, router_name, 'apiRouter', RouterName),
    findall(Handler, (
        member(P, Patterns),
        generate_backend_for_pattern(P, express, Options, Handler)
    ), Handlers),
    atomic_list_concat(Handlers, '\n\n', HandlersStr),
    format(string(Code),
"import express from 'express';

const ~w = express.Router();

~w

export default ~w;
", [RouterName, HandlersStr, RouterName]).

%% generate_go_handlers(+Patterns, +Options, -Code)
%
%  Generate Go HTTP handlers for all patterns.
%
generate_go_handlers(Patterns, Options, Code) :-
    option_value(Options, package_name, 'handlers', PackageName),
    findall(Handler, (
        member(P, Patterns),
        generate_backend_for_pattern(P, go, Options, Handler)
    ), Handlers),
    atomic_list_concat(Handlers, '\n\n', HandlersStr),
    format(string(Code),
"package ~w

import (
    \"encoding/json\"
    \"net/http\"
)

~w
", [PackageName, HandlersStr]).

% ============================================================================
% FULL STACK GENERATION
% ============================================================================

%% generate_full_stack(+Patterns, +Options, -Frontend, -Backend)
%
%  Generate both frontend and backend code for patterns.
%
generate_full_stack(Patterns, Options, Frontend, Backend) :-
    option_value(Options, frontend_target, react_native, FrontendTarget),
    option_value(Options, backend_target, express, BackendTarget),

    % Generate frontend
    generate_frontend_code(Patterns, FrontendTarget, Options, Frontend),

    % Generate backend
    generate_backend_code(Patterns, BackendTarget, Options, Backend).

generate_frontend_code(Patterns, react_native, Options, Code) :-
    findall(PatternCode, (
        member(P, Patterns),
        catch(ui_patterns:compile_pattern(P, react_native, Options, PatternCode), _, fail)
    ), Codes),
    Codes \= [],
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).

generate_frontend_code(Patterns, vue, Options, Code) :-
    findall(PatternCode, (
        member(P, Patterns),
        generate_vue_frontend_for_pattern(P, Options, PatternCode)
    ), Codes),
    Codes \= [],
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).

generate_frontend_code(Patterns, flutter, Options, Code) :-
    findall(PatternCode, (
        member(P, Patterns),
        generate_flutter_frontend_for_pattern(P, Options, PatternCode)
    ), Codes),
    Codes \= [],
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).

generate_frontend_code(Patterns, swiftui, Options, Code) :-
    findall(PatternCode, (
        member(P, Patterns),
        generate_swiftui_frontend_for_pattern(P, Options, PatternCode)
    ), Codes),
    Codes \= [],
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).

generate_frontend_code(_, _, _, "// Frontend code generation not implemented for this target").

%% generate_vue_frontend_for_pattern(+PatternName, +Options, -Code)
%
%  Generate Vue 3 frontend code for a pattern.
%
generate_vue_frontend_for_pattern(PatternName, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_vue_for_spec(PatternName, Spec, Options, Code).

generate_vue_for_spec(_Name, data(query, Config), Options, Code) :-
    catch(vue_target:compile_data_pattern(query, Config, vue, Options, Code), _, fail).
generate_vue_for_spec(_Name, data(mutation, Config), Options, Code) :-
    catch(vue_target:compile_data_pattern(mutation, Config, vue, Options, Code), _, fail).
generate_vue_for_spec(_Name, state(global, Shape, StateConfig), Options, Code) :-
    catch(vue_target:compile_state_pattern(global, Shape, StateConfig, vue, Options, Code), _, fail).
generate_vue_for_spec(_Name, navigation(Type, Screens, NavConfig), Options, Code) :-
    catch(vue_target:compile_navigation_pattern(Type, Screens, NavConfig, vue, Options, Code), _, fail).
generate_vue_for_spec(_Name, persistence(Type, Config), Options, Code) :-
    catch(vue_target:compile_persistence_pattern(Type, Config, vue, Options, Code), _, fail).

%% generate_flutter_frontend_for_pattern(+PatternName, +Options, -Code)
%
%  Generate Flutter/Dart frontend code for a pattern.
%
generate_flutter_frontend_for_pattern(PatternName, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_flutter_for_spec(PatternName, Spec, Options, Code).

generate_flutter_for_spec(_Name, data(query, Config), Options, Code) :-
    catch(flutter_target:compile_data_pattern(query, Config, flutter, Options, Code), _, fail).
generate_flutter_for_spec(_Name, data(mutation, Config), Options, Code) :-
    catch(flutter_target:compile_data_pattern(mutation, Config, flutter, Options, Code), _, fail).
generate_flutter_for_spec(_Name, state(global, Shape, StateConfig), Options, Code) :-
    catch(flutter_target:compile_state_pattern(global, Shape, StateConfig, flutter, Options, Code), _, fail).
generate_flutter_for_spec(_Name, navigation(Type, Screens, NavConfig), Options, Code) :-
    catch(flutter_target:compile_navigation_pattern(Type, Screens, NavConfig, flutter, Options, Code), _, fail).
generate_flutter_for_spec(_Name, persistence(Type, Config), Options, Code) :-
    catch(flutter_target:compile_persistence_pattern(Type, Config, flutter, Options, Code), _, fail).

%% generate_swiftui_frontend_for_pattern(+PatternName, +Options, -Code)
%
%  Generate SwiftUI/Swift frontend code for a pattern.
%
generate_swiftui_frontend_for_pattern(PatternName, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_swiftui_for_spec(PatternName, Spec, Options, Code).

generate_swiftui_for_spec(_Name, data(query, Config), Options, Code) :-
    catch(swiftui_target:compile_data_pattern(query, Config, swiftui, Options, Code), _, fail).
generate_swiftui_for_spec(_Name, data(mutation, Config), Options, Code) :-
    catch(swiftui_target:compile_data_pattern(mutation, Config, swiftui, Options, Code), _, fail).
generate_swiftui_for_spec(_Name, state(global, Shape, StateConfig), Options, Code) :-
    catch(swiftui_target:compile_state_pattern(global, Shape, StateConfig, swiftui, Options, Code), _, fail).
generate_swiftui_for_spec(_Name, navigation(Type, Screens, NavConfig), Options, Code) :-
    catch(swiftui_target:compile_navigation_pattern(Type, Screens, NavConfig, swiftui, Options, Code), _, fail).
generate_swiftui_for_spec(_Name, persistence(Type, Config), Options, Code) :-
    catch(swiftui_target:compile_persistence_pattern(Type, Config, swiftui, Options, Code), _, fail).

generate_backend_code(Patterns, express, Options, Code) :-
    generate_express_routes(Patterns, Options, Code).
generate_backend_code(Patterns, go, Options, Code) :-
    generate_go_handlers(Patterns, Options, Code).
generate_backend_code(Patterns, fastapi, Options, Code) :-
    generate_fastapi_routes(Patterns, Options, Code).
generate_backend_code(Patterns, flask, Options, Code) :-
    generate_flask_routes(Patterns, Options, Code).
generate_backend_code(_, _, _, "// Backend code generation not implemented for this target").

%% generate_fastapi_routes(+Patterns, +Options, -Code)
%
%  Generate FastAPI application with all pattern handlers.
%
generate_fastapi_routes(Patterns, Options, Code) :-
    catch(fastapi_generator:generate_fastapi_app(Patterns, Options, Code), _, fail),
    !.
generate_fastapi_routes(_, _, "# FastAPI generation failed").

%% generate_flask_routes(+Patterns, +Options, -Code)
%
%  Generate Flask application with all pattern handlers.
%
generate_flask_routes(Patterns, Options, Code) :-
    catch(flask_generator:generate_flask_app(Patterns, Options, Code), _, fail),
    !.
generate_flask_routes(_, _, "# Flask generation failed").

%% generate_api_client(+Endpoints, +Options, -Code)
%
%  Generate API client for frontend to call backend.
%  Supports: react_native (TypeScript), vue (TypeScript), flutter (Dart), swiftui (Swift)
%
generate_api_client(Endpoints, Options, Code) :-
    option_value(Options, target, react_native, Target),
    option_value(Options, base_url, 'http://localhost:3000', BaseUrl),
    generate_api_client_for_target(Target, Endpoints, BaseUrl, Options, Code).

%% React Native / TypeScript API client
generate_api_client_for_target(react_native, Endpoints, BaseUrl, _Options, Code) :-
    findall(Method, (
        member(endpoint(Name, HttpMethod, Path, _), Endpoints),
        generate_api_method(Name, HttpMethod, Path, BaseUrl, Method)
    ), Methods),
    atomic_list_concat(Methods, '\n\n', MethodsStr),
    format(string(Code),
"// Auto-generated API client
const BASE_URL = '~w';

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

~w

export const api = {
  // Add methods here
};
", [BaseUrl, MethodsStr]).

%% Vue 3 / TypeScript API client
generate_api_client_for_target(vue, Endpoints, BaseUrl, _Options, Code) :-
    findall(Method, (
        member(endpoint(Name, HttpMethod, Path, _), Endpoints),
        generate_vue_api_method(Name, HttpMethod, Path, BaseUrl, Method)
    ), Methods),
    atomic_list_concat(Methods, '\n\n', MethodsStr),
    format(string(Code),
"// Auto-generated Vue API client
import { ref } from 'vue';
import type { Ref } from 'vue';

const BASE_URL = '~w';

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

interface ApiState<T> {
  data: Ref<T | null>;
  loading: Ref<boolean>;
  error: Ref<string | null>;
}

~w
", [BaseUrl, MethodsStr]).

%% Flutter / Dart API client
generate_api_client_for_target(flutter, Endpoints, BaseUrl, _Options, Code) :-
    findall(Method, (
        member(endpoint(Name, HttpMethod, Path, _), Endpoints),
        generate_flutter_api_method(Name, HttpMethod, Path, BaseUrl, Method)
    ), Methods),
    atomic_list_concat(Methods, '\n\n', MethodsStr),
    format(string(Code),
"// Auto-generated Flutter API client
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiClient {
  static const String baseUrl = '~w';

  static Future<ApiResponse<T>> _handleResponse<T>(
    http.Response response,
    T Function(Map<String, dynamic>) fromJson,
  ) async {
    final body = json.decode(response.body);
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return ApiResponse(success: true, data: fromJson(body));
    }
    return ApiResponse(success: false, error: body['error'] ?? 'Unknown error');
  }

~w
}

class ApiResponse<T> {
  final bool success;
  final T? data;
  final String? error;

  ApiResponse({required this.success, this.data, this.error});
}
", [BaseUrl, MethodsStr]).

%% SwiftUI / Swift API client
generate_api_client_for_target(swiftui, Endpoints, BaseUrl, _Options, Code) :-
    findall(Method, (
        member(endpoint(Name, HttpMethod, Path, _), Endpoints),
        generate_swift_api_method(Name, HttpMethod, Path, BaseUrl, Method)
    ), Methods),
    atomic_list_concat(Methods, '\n\n', MethodsStr),
    format(string(Code),
"// Auto-generated SwiftUI API client
import Foundation

struct ApiResponse<T: Decodable>: Decodable {
    let success: Bool
    let data: T?
    let error: String?
}

enum ApiError: Error {
    case networkError(Error)
    case decodingError(Error)
    case serverError(String)
}

class ApiClient {
    static let shared = ApiClient()
    private let baseURL = \"~w\"

~w
}
", [BaseUrl, MethodsStr]).

generate_api_client_for_target(_, _, _, _, "// API client generation not implemented for this target").

generate_api_method(Name, get, Path, BaseUrl, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"export const ~w = async <T>(params?: Record<string, string>): Promise<ApiResponse<T>> => {
  const url = new URL('~w~w');
  if (params) {
    Object.entries(params).forEach(([key, value]) => url.searchParams.append(key, value));
  }
  const response = await fetch(url.toString());
  return response.json();
};
", [NameStr, BaseUrl, Path]).

generate_api_method(Name, post, Path, BaseUrl, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"export const ~w = async <T>(body: unknown): Promise<ApiResponse<T>> => {
  const response = await fetch('~w~w', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
};
", [NameStr, BaseUrl, Path]).

%% Vue API method generators
generate_vue_api_method(Name, get, Path, BaseUrl, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"export function use~w<T>() {
  const data = ref<T | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const fetch~w = async (params?: Record<string, string>) => {
    loading.value = true;
    error.value = null;
    try {
      const url = new URL('~w~w');
      if (params) {
        Object.entries(params).forEach(([key, value]) => url.searchParams.append(key, value));
      }
      const response = await fetch(url.toString());
      const result: ApiResponse<T> = await response.json();
      if (result.success) {
        data.value = result.data ?? null;
      } else {
        error.value = result.error ?? 'Unknown error';
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading.value = false;
    }
  };

  return { data, loading, error, fetch~w };
}
", [CapName, CapName, BaseUrl, Path, CapName]).

generate_vue_api_method(Name, post, Path, BaseUrl, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"export function use~w<T, TInput>() {
  const data = ref<T | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const mutate = async (body: TInput) => {
    loading.value = true;
    error.value = null;
    try {
      const response = await fetch('~w~w', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const result: ApiResponse<T> = await response.json();
      if (result.success) {
        data.value = result.data ?? null;
      } else {
        error.value = result.error ?? 'Unknown error';
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading.value = false;
    }
  };

  return { data, loading, error, mutate };
}
", [CapName, BaseUrl, Path]).

%% Flutter API method generators
generate_flutter_api_method(Name, get, Path, BaseUrl, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"  static Future<ApiResponse<T>> ~w<T>({
    Map<String, String>? params,
    required T Function(Map<String, dynamic>) fromJson,
  }) async {
    try {
      final uri = Uri.parse('~w~w').replace(queryParameters: params);
      final response = await http.get(uri);
      return _handleResponse(response, fromJson);
    } catch (e) {
      return ApiResponse(success: false, error: e.toString());
    }
  }
", [NameStr, BaseUrl, Path]).

generate_flutter_api_method(Name, post, Path, BaseUrl, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"  static Future<ApiResponse<T>> ~w<T>({
    required Map<String, dynamic> body,
    required T Function(Map<String, dynamic>) fromJson,
  }) async {
    try {
      final uri = Uri.parse('~w~w');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: json.encode(body),
      );
      return _handleResponse(response, fromJson);
    } catch (e) {
      return ApiResponse(success: false, error: e.toString());
    }
  }
", [NameStr, BaseUrl, Path]).

%% Swift API method generators
generate_swift_api_method(Name, get, Path, _BaseUrl, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"    func ~w<T: Decodable>(params: [String: String]? = nil) async throws -> T {
        var components = URLComponents(string: \"\\(baseURL)~w\")!
        if let params = params {
            components.queryItems = params.map { URLQueryItem(name: $0.key, value: $0.value) }
        }

        let (data, _) = try await URLSession.shared.data(from: components.url!)
        let response = try JSONDecoder().decode(ApiResponse<T>.self, from: data)

        if response.success, let result = response.data {
            return result
        }
        throw ApiError.serverError(response.error ?? \"Unknown error\")
    }
", [NameStr, Path]).

generate_swift_api_method(Name, post, Path, _BaseUrl, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"    func ~w<T: Decodable, TInput: Encodable>(body: TInput) async throws -> T {
        var request = URLRequest(url: URL(string: \"\\(baseURL)~w\")!)
        request.httpMethod = \"POST\"
        request.setValue(\"application/json\", forHTTPHeaderField: \"Content-Type\")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(ApiResponse<T>.self, from: data)

        if response.success, let result = response.data {
            return result
        }
        throw ApiError.serverError(response.error ?? \"Unknown error\")
    }
", [NameStr, Path]).

% ============================================================================
% ENDPOINT SPECIFICATION
% ============================================================================

%% endpoint_spec(+Name, +Method, +Path, +Handler)
%
%  Define an API endpoint specification.
%
endpoint_spec(Name, Method, Path, Handler) :-
    atom(Name),
    member(Method, [get, post, put, patch, delete]),
    atom(Path),
    atom(Handler).

%% pattern_to_endpoint(+PatternName, +Options, -EndpointSpec)
%
%  Convert a pattern to an endpoint specification.
%
pattern_to_endpoint(PatternName, _Options, endpoint(PatternName, get, Path, PatternName)) :-
    catch(ui_patterns:pattern(PatternName, data(query, Config), _), _, fail),
    member(endpoint(Path), Config).
pattern_to_endpoint(PatternName, _Options, endpoint(PatternName, post, Path, PatternName)) :-
    catch(ui_patterns:pattern(PatternName, data(mutation, Config), _), _, fail),
    member(endpoint(Path), Config).
pattern_to_endpoint(PatternName, _Options, endpoint(PatternName, get, Path, PatternName)) :-
    catch(ui_patterns:pattern(PatternName, data(infinite, Config), _), _, fail),
    member(endpoint(Path), Config).

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HC]),
    string_chars(Cap, [HC|T]).

downcase_atom(Atom, Lower) :-
    atom_string(Atom, Str),
    string_lower(Str, LowerStr),
    atom_string(Lower, LowerStr).

% ============================================================================
% TESTING
% ============================================================================

test_pattern_glue :-
    format('~n=== Pattern Glue Tests ===~n~n'),

    % Test 1: Pattern analysis
    format('Test 1: Pattern backend detection...~n'),
    (   catch((
            ui_patterns:query_pattern(test_api, '/api/test', [], _),
            pattern_requires_backend(test_api, Reason),
            Reason == database_access
        ), _, fail)
    ->  format('  PASS: Query pattern detected as requiring backend~n')
    ;   format('  SKIP: ui_patterns not loaded~n')
    ),

    % Test 2: Express handler generation
    format('~nTest 2: Express handler generation...~n'),
    (   catch((
            ui_patterns:query_pattern(test_express, '/api/items', [], _),
            generate_backend_for_pattern(test_express, express, [], Code),
            sub_string(Code, _, _, _, "router.get")
        ), _, fail)
    ->  format('  PASS: Generated Express GET handler~n')
    ;   format('  SKIP: ui_patterns not loaded~n')
    ),

    % Test 3: Go handler generation
    format('~nTest 3: Go handler generation...~n'),
    (   catch((
            ui_patterns:mutation_pattern(test_go, '/api/create', [], _),
            generate_backend_for_pattern(test_go, go, [], GoCode),
            sub_string(GoCode, _, _, _, "func")
        ), _, fail)
    ->  format('  PASS: Generated Go handler~n')
    ;   format('  SKIP: ui_patterns not loaded~n')
    ),

    % Test 4: Full stack generation
    format('~nTest 4: Full stack generation...~n'),
    (   catch((
            ui_patterns:query_pattern(full_stack_test, '/api/data', [], _),
            generate_full_stack([full_stack_test], [], Frontend, Backend),
            Frontend \= "",
            Backend \= ""
        ), _, fail)
    ->  format('  PASS: Generated frontend and backend code~n')
    ;   format('  SKIP: ui_patterns not loaded~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Pattern glue module loaded~n', [])
), now).
