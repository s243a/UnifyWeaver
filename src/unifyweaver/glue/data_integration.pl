% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% data_integration.pl - Connect UI data patterns to UnifyWeaver data sources
%
% This module bridges UI-layer data bindings with backend data sources,
% generating both backend handlers and frontend query hooks.

:- module(data_integration, [
    % Data binding declarations
    data_binding/3,           % data_binding(Name, Source, Options)

    % Backend generation
    generate_backend_handler/4,   % +Binding, +Target, +Options, -Code
    generate_all_handlers/4,      % +Bindings, +Target, +Options, -Code

    % Frontend generation
    generate_frontend_query/4,    % +Binding, +Target, +Options, -Code
    generate_all_queries/4,       % +Bindings, +Target, +Options, -Code

    % Full stack generation
    generate_data_layer/5,        % +Bindings, +FrontTarget, +BackTarget, +Options, -Result

    % Source type helpers
    source_endpoint/2,            % +Source, -Endpoint
    source_method/2,              % +Source, -Method

    % Testing
    test_data_integration/0
]).

:- use_module(library(lists)).

% ============================================================================
% DATA BINDING DECLARATIONS
% ============================================================================

%! data_binding(+Name, +Source, +Options)
%
%  Declare a data binding that connects a UI data need to a backend source.
%
%  Source can be:
%    - sqlite(DbFile, Query)
%    - http(Url)
%    - csv(File)
%    - json(File)
%    - custom(endpoint(Path))
%
%  Options:
%    - endpoint(Path)     - Override API endpoint path
%    - method(Method)     - HTTP method (get, post, put, delete)
%    - cache(Duration)    - Cache duration in seconds
%    - params(Params)     - Query parameters
%    - transform(Fn)      - Transform function name
%
data_binding(_, _, _).

% ============================================================================
% SOURCE HELPERS
% ============================================================================

%! source_endpoint(+Source, -Endpoint)
%
%  Derive a default API endpoint from a source definition.
%
source_endpoint(sqlite(_, _), '/api/data').
source_endpoint(http(Url), Endpoint) :-
    atom_concat('/api/proxy/', Url, Endpoint).
source_endpoint(csv(File), Endpoint) :-
    file_base_name(File, Base),
    file_name_extension(Name, _, Base),
    atom_concat('/api/', Name, Endpoint).
source_endpoint(json(File), Endpoint) :-
    file_base_name(File, Base),
    file_name_extension(Name, _, Base),
    atom_concat('/api/', Name, Endpoint).
source_endpoint(custom(endpoint(Path)), Path).

%! source_method(+Source, -Method)
%
%  Default HTTP method for a source type.
%
source_method(sqlite(_, _), get).
source_method(http(_), get).
source_method(csv(_), get).
source_method(json(_), get).
source_method(custom(_), get).

% ============================================================================
% BACKEND HANDLER GENERATION
% ============================================================================

%! generate_backend_handler(+Binding, +Target, +Options, -Code)
%
%  Generate a backend handler for a data binding.
%
generate_backend_handler(binding(Name, Source, Opts), fastapi, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    (member(method(Method), Opts) -> true ; source_method(Source, Method)),
    generate_fastapi_handler(Name, Source, Endpoint, Method, Code).

generate_backend_handler(binding(Name, Source, Opts), flask, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    (member(method(Method), Opts) -> true ; source_method(Source, Method)),
    generate_flask_handler(Name, Source, Endpoint, Method, Code).

generate_backend_handler(binding(Name, Source, Opts), express, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    (member(method(Method), Opts) -> true ; source_method(Source, Method)),
    generate_express_handler(Name, Source, Endpoint, Method, Code).

%! generate_all_handlers(+Bindings, +Target, +Options, -Code)
%
%  Generate all backend handlers for a list of bindings.
%
generate_all_handlers(Bindings, Target, Options, Code) :-
    findall(HandlerCode, (
        member(Binding, Bindings),
        generate_backend_handler(Binding, Target, Options, HandlerCode)
    ), HandlerCodes),
    generate_handler_file(Target, HandlerCodes, Code).

% --- FastAPI Handler Generation ---

generate_fastapi_handler(Name, sqlite(DbFile, Query), Endpoint, _Method, Code) :-
    format(atom(Code),
'@router.get("~w")
async def get_~w():
    """Fetch ~w data from SQLite"""
    conn = sqlite3.connect("~w")
    cursor = conn.execute("""~w""")
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return {"data": rows}
', [Endpoint, Name, Name, DbFile, Query]).

generate_fastapi_handler(Name, http(Url), Endpoint, _Method, Code) :-
    format(atom(Code),
'@router.get("~w")
async def get_~w():
    """Proxy request to ~w"""
    async with httpx.AsyncClient() as client:
        response = await client.get("~w")
        return response.json()
', [Endpoint, Name, Url, Url]).

generate_fastapi_handler(Name, csv(File), Endpoint, _Method, Code) :-
    format(atom(Code),
'@router.get("~w")
async def get_~w():
    """Read data from CSV file"""
    import csv
    with open("~w", "r") as f:
        reader = csv.DictReader(f)
        return {"data": list(reader)}
', [Endpoint, Name, File]).

generate_fastapi_handler(Name, json(File), Endpoint, _Method, Code) :-
    format(atom(Code),
'@router.get("~w")
async def get_~w():
    """Read data from JSON file"""
    import json
    with open("~w", "r") as f:
        return json.load(f)
', [Endpoint, Name, File]).

generate_fastapi_handler(Name, custom(endpoint(_)), Endpoint, Method, Code) :-
    format(atom(Code),
'@router.~w("~w")
async def ~w_~w():
    """Custom endpoint handler"""
    # TODO: Implement custom logic
    return {"status": "ok"}
', [Method, Endpoint, Method, Name]).

% --- Flask Handler Generation ---

generate_flask_handler(Name, sqlite(DbFile, Query), Endpoint, _Method, Code) :-
    format(atom(Code),
'@bp.route("~w")
def get_~w():
    """Fetch ~w data from SQLite"""
    conn = sqlite3.connect("~w")
    cursor = conn.execute("""~w""")
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return jsonify({"data": rows})
', [Endpoint, Name, Name, DbFile, Query]).

generate_flask_handler(Name, http(Url), Endpoint, _Method, Code) :-
    format(atom(Code),
'@bp.route("~w")
def get_~w():
    """Proxy request to ~w"""
    response = requests.get("~w")
    return jsonify(response.json())
', [Endpoint, Name, Url, Url]).

generate_flask_handler(Name, csv(File), Endpoint, _Method, Code) :-
    format(atom(Code),
'@bp.route("~w")
def get_~w():
    """Read data from CSV file"""
    import csv
    with open("~w", "r") as f:
        reader = csv.DictReader(f)
        return jsonify({"data": list(reader)})
', [Endpoint, Name, File]).

generate_flask_handler(Name, json(File), Endpoint, _Method, Code) :-
    format(atom(Code),
'@bp.route("~w")
def get_~w():
    """Read data from JSON file"""
    import json
    with open("~w", "r") as f:
        return jsonify(json.load(f))
', [Endpoint, Name, File]).

generate_flask_handler(Name, custom(endpoint(_)), Endpoint, Method, Code) :-
    method_list(Method, Methods),
    format(atom(Code),
'@bp.route("~w", methods=~w)
def ~w():
    """Custom endpoint handler"""
    # TODO: Implement custom logic
    return jsonify({"status": "ok"})
', [Endpoint, Methods, Name]).

method_list(get, "['GET']").
method_list(post, "['POST']").
method_list(put, "['PUT']").
method_list(delete, "['DELETE']").

% --- Express Handler Generation ---

generate_express_handler(Name, sqlite(_DbFile, _Query), Endpoint, _Method, Code) :-
    format(atom(Code),
'router.get("~w", async (req, res) => {
  // ~w handler - SQLite source
  // Note: Requires better-sqlite3 or sqlite3 package
  try {
    const db = new Database(process.env.DB_PATH);
    const rows = db.prepare("SELECT * FROM data").all();
    res.json({ data: rows });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
', [Endpoint, Name]).

generate_express_handler(Name, http(Url), Endpoint, _Method, Code) :-
    format(atom(Code),
'router.get("~w", async (req, res) => {
  // ~w handler - HTTP proxy
  try {
    const response = await fetch("~w");
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
', [Endpoint, Name, Url]).

generate_express_handler(Name, csv(File), Endpoint, _Method, Code) :-
    format(atom(Code),
'router.get("~w", async (req, res) => {
  // ~w handler - CSV source
  const fs = require("fs");
  const csv = require("csv-parse/sync");

  try {
    const content = fs.readFileSync("~w", "utf-8");
    const records = csv.parse(content, { columns: true });
    res.json({ data: records });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
', [Endpoint, Name, File]).

generate_express_handler(Name, json(File), Endpoint, _Method, Code) :-
    format(atom(Code),
'router.get("~w", async (req, res) => {
  // ~w handler - JSON source
  const fs = require("fs");

  try {
    const content = fs.readFileSync("~w", "utf-8");
    res.json(JSON.parse(content));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
', [Endpoint, Name, File]).

generate_express_handler(Name, custom(endpoint(_)), Endpoint, Method, Code) :-
    format(atom(Code),
'router.~w("~w", async (req, res) => {
  // ~w handler - custom endpoint
  // TODO: Implement custom logic
  res.json({ status: "ok" });
});
', [Method, Endpoint, Name]).

% --- Handler File Generation ---

generate_handler_file(fastapi, HandlerCodes, Code) :-
    atomic_list_concat(HandlerCodes, '\n\n', HandlersStr),
    format(atom(Code),
'"""
Auto-generated API handlers
"""
from fastapi import APIRouter
import sqlite3
import httpx

router = APIRouter()

~w
', [HandlersStr]).

generate_handler_file(flask, HandlerCodes, Code) :-
    atomic_list_concat(HandlerCodes, '\n\n', HandlersStr),
    format(atom(Code),
'"""
Auto-generated API handlers
"""
from flask import Blueprint, jsonify
import sqlite3
import requests

bp = Blueprint("api", __name__)

~w
', [HandlersStr]).

generate_handler_file(express, HandlerCodes, Code) :-
    atomic_list_concat(HandlerCodes, '\n\n', HandlersStr),
    format(atom(Code),
'/**
 * Auto-generated API handlers
 */
const express = require("express");
const router = express.Router();

~w

module.exports = router;
', [HandlersStr]).

% ============================================================================
% FRONTEND QUERY GENERATION
% ============================================================================

%! generate_frontend_query(+Binding, +Target, +Options, -Code)
%
%  Generate a frontend query hook for a data binding.
%
generate_frontend_query(binding(Name, Source, Opts), react_native, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    capitalize_first(Name, CapName),
    format(atom(Code),
'export function use~w() {
  return useQuery({
    queryKey: ["~w"],
    queryFn: () => api.get("~w").then(r => r.data),
  });
}

export function use~wMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data) => api.post("~w", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["~w"] });
    },
  });
}
', [CapName, Name, Endpoint, CapName, Endpoint, Name]).

generate_frontend_query(binding(Name, Source, Opts), vue, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    capitalize_first(Name, CapName),
    format(atom(Code),
'export function use~w() {
  return useQuery({
    queryKey: ["~w"],
    queryFn: () => api.get("~w"),
  });
}

export function use~wMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: any) => api.post("~w", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["~w"] });
    },
  });
}
', [CapName, Name, Endpoint, CapName, Endpoint, Name]).

generate_frontend_query(binding(Name, Source, Opts), flutter, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    capitalize_first(Name, CapName),
    format(atom(Code),
'final ~wProvider = FutureProvider<List<dynamic>>((ref) async {
  final response = await http.get(Uri.parse("$baseUrl~w"));
  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    return data["data"] ?? [];
  }
  throw Exception("Failed to load ~w");
});

class ~wNotifier extends StateNotifier<AsyncValue<List<dynamic>>> {
  ~wNotifier() : super(const AsyncValue.loading());

  Future<void> refresh() async {
    state = const AsyncValue.loading();
    try {
      final response = await http.get(Uri.parse("$baseUrl~w"));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        state = AsyncValue.data(data["data"] ?? []);
      }
    } catch (e, st) {
      state = AsyncValue.error(e, st);
    }
  }
}
', [Name, Endpoint, Name, CapName, CapName, Endpoint]).

generate_frontend_query(binding(Name, Source, Opts), swiftui, _Options, Code) :-
    (member(endpoint(Endpoint), Opts) -> true ; source_endpoint(Source, Endpoint)),
    capitalize_first(Name, CapName),
    format(atom(Code),
'class ~wViewModel: ObservableObject {
    @Published var items: [~wItem] = []
    @Published var isLoading = false
    @Published var error: String?

    func fetch() async {
        isLoading = true
        error = nil

        guard let url = URL(string: "\\(baseURL)~w") else {
            error = "Invalid URL"
            isLoading = false
            return
        }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let response = try JSONDecoder().decode(~wResponse.self, from: data)
            await MainActor.run {
                self.items = response.data
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.error = error.localizedDescription
                self.isLoading = false
            }
        }
    }
}
', [CapName, CapName, Endpoint, CapName]).

%! generate_all_queries(+Bindings, +Target, +Options, -Code)
%
%  Generate all frontend queries for a list of bindings.
%
generate_all_queries(Bindings, Target, Options, Code) :-
    findall(QueryCode, (
        member(Binding, Bindings),
        generate_frontend_query(Binding, Target, Options, QueryCode)
    ), QueryCodes),
    generate_query_file(Target, QueryCodes, Code).

% --- Query File Generation ---

generate_query_file(react_native, QueryCodes, Code) :-
    atomic_list_concat(QueryCodes, '\n\n', QueriesStr),
    format(atom(Code),
'/**
 * Auto-generated query hooks
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";

~w
', [QueriesStr]).

generate_query_file(vue, QueryCodes, Code) :-
    atomic_list_concat(QueryCodes, '\n\n', QueriesStr),
    format(atom(Code),
'/**
 * Auto-generated query composables
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/vue-query";
import { api } from "./client";

~w
', [QueriesStr]).

generate_query_file(flutter, QueryCodes, Code) :-
    atomic_list_concat(QueryCodes, '\n\n', QueriesStr),
    format(atom(Code),
'/// Auto-generated data providers
import "dart:convert";
import "package:flutter_riverpod/flutter_riverpod.dart";
import "package:http/http.dart" as http;

const baseUrl = String.fromEnvironment("API_URL", defaultValue: "http://localhost:8000");

~w
', [QueriesStr]).

generate_query_file(swiftui, QueryCodes, Code) :-
    atomic_list_concat(QueryCodes, '\n\n', QueriesStr),
    format(atom(Code),
'/// Auto-generated view models
import Foundation
import SwiftUI

let baseURL = ProcessInfo.processInfo.environment["API_URL"] ?? "http://localhost:8000"

~w
', [QueriesStr]).

% ============================================================================
% FULL STACK GENERATION
% ============================================================================

%! generate_data_layer(+Bindings, +FrontTarget, +BackTarget, +Options, -Result)
%
%  Generate both frontend queries and backend handlers.
%
generate_data_layer(Bindings, FrontTarget, BackTarget, Options, Result) :-
    generate_all_handlers(Bindings, BackTarget, Options, BackendCode),
    generate_all_queries(Bindings, FrontTarget, Options, FrontendCode),
    Result = data_layer(
        backend(BackTarget, BackendCode),
        frontend(FrontTarget, FrontendCode)
    ).

% ============================================================================
% UTILITIES
% ============================================================================

capitalize_first(Atom, Capitalized) :-
    atom_codes(Atom, [First|Rest]),
    (First >= 97, First =< 122 ->
        Upper is First - 32,
        atom_codes(Capitalized, [Upper|Rest])
    ;
        Capitalized = Atom
    ).

% ============================================================================
% TESTING
% ============================================================================

test_data_integration :-
    format('Running data integration tests...~n', []),

    % Test 1: Source endpoint derivation
    (source_endpoint(sqlite('test.db', 'SELECT 1'), Ep1), Ep1 = '/api/data'
    -> format('  Test 1 passed: SQLite endpoint~n', [])
    ; format('  Test 1 FAILED: SQLite endpoint~n', [])),

    % Test 2: CSV endpoint derivation
    (source_endpoint(csv('data/users.csv'), Ep2), sub_atom(Ep2, _, _, _, '/api/users')
    -> format('  Test 2 passed: CSV endpoint~n', [])
    ; format('  Test 2 FAILED: CSV endpoint~n', [])),

    % Test 3: FastAPI handler generation
    (generate_backend_handler(binding(tasks, sqlite('app.db', 'SELECT * FROM tasks'), []), fastapi, [], Code3),
     sub_atom(Code3, _, _, _, '@router.get'),
     sub_atom(Code3, _, _, _, 'get_tasks')
    -> format('  Test 3 passed: FastAPI handler~n', [])
    ; format('  Test 3 FAILED: FastAPI handler~n', [])),

    % Test 4: Flask handler generation
    (generate_backend_handler(binding(users, csv('users.csv'), [endpoint('/api/users')]), flask, [], Code4),
     sub_atom(Code4, _, _, _, '@bp.route'),
     sub_atom(Code4, _, _, _, 'get_users')
    -> format('  Test 4 passed: Flask handler~n', [])
    ; format('  Test 4 FAILED: Flask handler~n', [])),

    % Test 5: Express handler generation
    (generate_backend_handler(binding(items, json('items.json'), []), express, [], Code5),
     sub_atom(Code5, _, _, _, 'router.get'),
     sub_atom(Code5, _, _, _, 'items handler')
    -> format('  Test 5 passed: Express handler~n', [])
    ; format('  Test 5 FAILED: Express handler~n', [])),

    % Test 6: React Native query generation
    (generate_frontend_query(binding(tasks, sqlite('app.db', ''), [endpoint('/api/tasks')]), react_native, [], Code6),
     sub_atom(Code6, _, _, _, 'useTasks'),
     sub_atom(Code6, _, _, _, 'useQuery')
    -> format('  Test 6 passed: React Native query~n', [])
    ; format('  Test 6 FAILED: React Native query~n', [])),

    % Test 7: Vue query generation
    (generate_frontend_query(binding(users, http('https://api.example.com'), [endpoint('/api/users')]), vue, [], Code7),
     sub_atom(Code7, _, _, _, 'useUsers'),
     sub_atom(Code7, _, _, _, 'useQuery')
    -> format('  Test 7 passed: Vue query~n', [])
    ; format('  Test 7 FAILED: Vue query~n', [])),

    % Test 8: Flutter provider generation
    (generate_frontend_query(binding(products, json('products.json'), [endpoint('/api/products')]), flutter, [], Code8),
     sub_atom(Code8, _, _, _, 'productsProvider'),
     sub_atom(Code8, _, _, _, 'FutureProvider')
    -> format('  Test 8 passed: Flutter provider~n', [])
    ; format('  Test 8 FAILED: Flutter provider~n', [])),

    % Test 9: SwiftUI view model generation
    (generate_frontend_query(binding(orders, sqlite('db.sqlite', ''), [endpoint('/api/orders')]), swiftui, [], Code9),
     sub_atom(Code9, _, _, _, 'OrdersViewModel'),
     sub_atom(Code9, _, _, _, 'ObservableObject')
    -> format('  Test 9 passed: SwiftUI view model~n', [])
    ; format('  Test 9 FAILED: SwiftUI view model~n', [])),

    % Test 10: Full data layer generation
    (generate_data_layer([
        binding(tasks, sqlite('app.db', 'SELECT * FROM tasks'), [endpoint('/api/tasks')]),
        binding(users, csv('users.csv'), [endpoint('/api/users')])
     ], react_native, fastapi, [], Result),
     Result = data_layer(backend(fastapi, _), frontend(react_native, _))
    -> format('  Test 10 passed: Full data layer~n', [])
    ; format('  Test 10 FAILED: Full data layer~n', [])),

    format('All 10 data integration tests completed!~n', []).

% Run tests on load if main
:- initialization((
    (current_prolog_flag(argv, [File|_]),
     sub_atom(File, _, _, 0, 'data_integration.pl'))
    -> test_data_integration
    ; true
), main).
