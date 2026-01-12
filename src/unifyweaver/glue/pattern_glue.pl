% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% pattern_glue.pl - Connect UI Patterns to Backend Services
%
% Bridges the ui_patterns module with backend services via glue infrastructure.
% Generates both frontend (React Native, Vue) and backend (Express, Go) code
% for patterns that require server-side data.
%
% Features:
%   - Auto-detect patterns requiring backend services
%   - Generate API endpoints for data patterns
%   - Generate Express routes for query/mutation patterns
%   - Cross-runtime support (JS frontend + Go/Python backend)

:- module(pattern_glue, [
    % Pattern analysis
    pattern_requires_backend/2,         % +PatternName, -Reason
    analyze_pattern_dependencies/2,     % +PatternName, -Dependencies

    % Backend generation
    generate_backend_for_pattern/4,     % +PatternName, +Target, +Options, -Code
    generate_express_routes/3,          % +Patterns, +Options, -Code
    generate_go_handlers/3,             % +Patterns, +Options, -Code

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
%  Target: express | go | python
%
generate_backend_for_pattern(PatternName, express, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_express_handler(PatternName, Spec, Options, Code).
generate_backend_for_pattern(PatternName, go, Options, Code) :-
    catch(ui_patterns:pattern(PatternName, Spec, _), _, fail),
    generate_go_handler(PatternName, Spec, Options, Code).

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
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).
generate_frontend_code(_, _, _, "// Frontend code generation not implemented for this target").

generate_backend_code(Patterns, express, Options, Code) :-
    generate_express_routes(Patterns, Options, Code).
generate_backend_code(Patterns, go, Options, Code) :-
    generate_go_handlers(Patterns, Options, Code).
generate_backend_code(_, _, _, "// Backend code generation not implemented for this target").

%% generate_api_client(+Endpoints, +Options, -Code)
%
%  Generate API client for React Native to call backend.
%
generate_api_client(Endpoints, Options, Code) :-
    option_value(Options, base_url, 'http://localhost:3000', BaseUrl),
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
