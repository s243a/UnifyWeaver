:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% typescript_target.pl - TypeScript Code Generation Target
%
% Compiles Prolog predicates to TypeScript for:
% - Type-safe JavaScript with interfaces
% - Node.js, Deno, Bun, and browser runtimes
% - React/Next.js integration
% - Runtime selection via js_runtime_choice/2

:- module(typescript_target, [
    % Standard interface
    target_info/1,                  % -Info
    compile_predicate/3,            % +Pred/Arity, +Options, -Code
    compile_facts/3,                % +Pred, +Arity, -Code
    compile_recursion/3,            % +Pred/Arity, +Options, -Code
    compile_module/3,               % +Predicates, +Options, -Code
    write_typescript_module/2,      % +Code, +Filename
    init_typescript_target/0,

    % Binding system exports
    clear_binding_imports/0,        % Clear collected binding imports
    collect_binding_import/1,       % Collect an import from bindings
    get_collected_imports/1,        % Get imports collected from bindings

    % Component system exports
    collect_declared_component/2,   % Record that a component is used

    % Service generation exports (Express/HTTP)
    compile_express_service/2,      % +Service, -Code
    compile_express_router/2,       % +RouterSpec, -Code
    compile_http_client/2,          % +ClientSpec, -Code

    % Legacy compatibility
    compile_predicate_to_typescript/3
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% Binding system integration
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('../bindings/typescript_bindings').
:- use_module('typescript_runtime/custom_typescript', []).

% Track required imports from bindings
:- dynamic required_binding_import/1.
:- dynamic collected_component/2.

%% ============================================
%% TARGET INFO
%% ============================================

target_info(info{
    name: "TypeScript",
    family: javascript,
    file_extension: ".ts",
    runtime: auto,              % node, deno, bun, browser
    features: [types, generics, async, modules, interfaces],
    recursion_patterns: [tail_recursion, linear_recursion, list_fold, transitive_closure],
    compile_command: "npx tsc"
}).

%% ============================================
%% INITIALIZATION
%% ============================================

%% init_typescript_target
%
%  Initialize TypeScript target with bindings and clear state.
%
init_typescript_target :-
    retractall(required_binding_import(_)),
    retractall(collected_component(_, _)),
    init_typescript_bindings,
    format('[TypeScript Target] Initialized with bindings~n', []).

%% ============================================
%% IMPORT COLLECTION SYSTEM
%% ============================================

%% collect_binding_import(+Import)
%
%  Record that an import is required (e.g., 'fs', 'path', 'express').
%
collect_binding_import(Import) :-
    (   required_binding_import(Import)
    ->  true
    ;   assertz(required_binding_import(Import))
    ).

%% clear_binding_imports
%
%  Clear all collected binding imports.
%
clear_binding_imports :-
    retractall(required_binding_import(_)).

%% get_collected_imports(-Imports)
%
%  Get all collected imports from bindings.
%
get_collected_imports(Imports) :-
    findall(I, required_binding_import(I), Imports).

%% format_binding_imports(+Imports, -FormattedStr)
%
%  Format a list of import names for TypeScript import statements.
%
format_binding_imports([], "").
format_binding_imports(Imports, FormattedStr) :-
    Imports \= [],
    sort(Imports, UniqueImports),
    findall(Formatted,
        (   member(Import, UniqueImports),
            format_single_import(Import, Formatted)
        ),
        FormattedList),
    atomic_list_concat(FormattedList, '\n', FormattedStr).

%% format_single_import(+Import, -Formatted)
%
%  Format a single import. Handles different import types.
%
format_single_import(Import, Formatted) :-
    atom_string(Import, ImportStr),
    (   sub_string(ImportStr, 0, 1, _, ".")
    ->  % Relative import (e.g., './rpyc_bridge')
        format(string(Formatted), "import { * } from '~w';", [ImportStr])
    ;   sub_string(ImportStr, 0, 1, _, "@")
    ->  % Scoped package (e.g., '@types/node')
        format(string(Formatted), "import * as ~w from '~w';", [make_import_alias(ImportStr), ImportStr])
    ;   % Node.js built-in or npm package
        format(string(Formatted), "import * as ~w from '~w';", [ImportStr, ImportStr])
    ).

%% make_import_alias(+ScopedName, -Alias)
%
%  Create an alias for scoped package names.
%
make_import_alias(Name, Alias) :-
    atom_string(Name, NameStr),
    (   sub_string(NameStr, _Before, 1, After, "/")
    ->  sub_string(NameStr, _, After, 0, Alias)
    ;   Alias = NameStr
    ).

%% ============================================
%% COMPONENT COLLECTION SYSTEM
%% ============================================

%% collect_declared_component(+Category, +Name)
%
%  Record that a component is used in the code.
%
collect_declared_component(Category, Name) :-
    (   collected_component(Category, Name)
    ->  true
    ;   assertz(collected_component(Category, Name))
    ).

%% ============================================
%% MAIN DISPATCH
%% ============================================

compile_predicate(Pred/Arity, Options, Code) :-
    compile_predicate_to_typescript(Pred/Arity, Options, Code).

compile_predicate_to_typescript(Pred/Arity, Options, Code) :-
    option(type(Type), Options, facts),
    (   Type == facts
    ->  compile_facts(Pred, Arity, Code)
    ;   Type == recursion
    ->  compile_recursion(Pred/Arity, Options, Code)
    ;   Type == module
    ->  compile_module([pred(Pred, Arity, facts)], Options, Code)
    ;   compile_facts(Pred, Arity, Code)
    ).

%% ============================================
%% FACTS → TYPED ARRAYS
%% ============================================

compile_facts(Pred, Arity, Code) :-
    atom_string(Pred, PredStr),
    capitalize(PredStr, TypeName),
    
    % Gather facts
    findall(FactData, (
        functor(Goal, Pred, Arity),
        call(Goal),
        Goal =.. [_|Args],
        format_ts_tuple(Args, FactData)
    ), Facts),
    
    % Generate field names
    generate_field_names(Arity, FieldNames),
    generate_interface_fields(FieldNames, InterfaceFields),
    
    % Generate fact array
    atomic_list_concat(Facts, ',\n  ', FactList),
    atomic_list_concat(FieldNames, ', ', FieldsStr),
    
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Predicate: ~w/~w

export interface ~wFact {
~w
}

export const ~wFacts: ~wFact[] = [
  ~w
];

export const query~w = (~w: Partial<~wFact>): ~wFact[] => {
  return ~wFacts.filter(fact => {
    return Object.entries(~w).every(([key, value]) => 
      (fact as any)[key] === value
    );
  });
};

export const is~w = (...args: string[]): boolean => {
  const [~w] = args;
  return ~wFacts.some(f => ~w);
};
', [PredStr, Arity, TypeName, InterfaceFields, 
    PredStr, TypeName, FactList, 
    TypeName, 'criteria', TypeName, TypeName,
    PredStr, 'criteria',
    TypeName, FieldsStr, PredStr, generate_match_expr(FieldNames)]).

%% ============================================
%% RECURSION → FUNCTIONS
%% ============================================

compile_recursion(Pred/_Arity, Options, Code) :-
    atom_string(Pred, PredStr),
    option(pattern(Pattern), Options, tail_recursion),
    option(module_name(_ModName), Options, PredStr),

    (   Pattern == tail_recursion
    ->  generate_tail_recursion(PredStr, Code)
    ;   Pattern == list_fold
    ->  generate_list_fold(PredStr, Code)
    ;   Pattern == linear_recursion
    ->  generate_linear_recursion(PredStr, Code)
    ;   generate_tail_recursion(PredStr, Code)
    ).

generate_tail_recursion(Name, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Pattern: tail_recursion

export const ~w = (n: number, acc: number = 0): number => {
  if (n <= 0) return acc;
  return ~w(n - 1, acc + n);
};

// Strict version for guaranteed TCO
export const ~wStrict = (n: number, acc: number = 0): number => {
  let current = n;
  let result = acc;
  while (current > 0) {
    result += current;
    current--;
  }
  return result;
};
', [Name, Name, Name]).

generate_list_fold(Name, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Pattern: list_fold

export const ~w = (items: number[]): number => {
  return items.reduce((acc, item) => acc + item, 0);
};

// Explicit fold version
export const ~wFold = <T, R>(
  items: T[],
  initial: R,
  fn: (acc: R, item: T) => R
): R => {
  return items.reduce(fn, initial);
};
', [Name, Name]).

generate_linear_recursion(Name, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Pattern: linear_recursion (fibonacci)

const ~wMemo = new Map<number, number>();

export const ~w = (n: number): number => {
  if (n <= 0) return 0;
  if (n === 1) return 1;
  
  if (~wMemo.has(n)) {
    return ~wMemo.get(n)!;
  }
  
  const result = ~w(n - 1) + ~w(n - 2);
  ~wMemo.set(n, result);
  return result;
};
', [Name, Name, Name, Name, Name, Name, Name]).

%% ============================================
%% MODULE COMPILATION
%% ============================================

compile_module(Predicates, Options, Code) :-
    option(module_name(ModName), Options, 'Generated'),

    % Generate exports (future use for explicit export statements)
    findall(Export, (
        member(pred(Name, _Arity, _Type), Predicates),
        atom_string(Name, Export)
    ), Exports),
    atomic_list_concat(Exports, ', ', _ExportList),
    
    % Generate code for each predicate
    findall(PredCode, (
        member(pred(Name, Arity, Type), Predicates),
        generate_pred_code_ts(Name, Arity, Type, PredCode)
    ), PredCodes),
    atomic_list_concat(PredCodes, '\n\n', PredsSection),
    
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Module: ~w

~w
', [ModName, PredsSection]).

generate_pred_code_ts(Name, _Arity, tail_recursion, Code) :-
    atom_string(Name, NameStr),
    generate_tail_recursion(NameStr, Code).

generate_pred_code_ts(Name, _Arity, list_fold, Code) :-
    atom_string(Name, NameStr),
    generate_list_fold(NameStr, Code).

generate_pred_code_ts(Name, _Arity, linear_recursion, Code) :-
    atom_string(Name, NameStr),
    generate_linear_recursion(NameStr, Code).

generate_pred_code_ts(Name, _Arity, factorial, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
'// ~w (factorial)
export const ~w = (n: number): number => {
  if (n <= 1) return 1;
  return n * ~w(n - 1);
};
', [NameStr, NameStr, NameStr]).

%% ============================================
%% HELPERS
%% ============================================

capitalize(Str, Cap) :-
    string_chars(Str, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HC]),
    string_chars(Cap, [HC|T]).

format_ts_tuple(Args, Str) :-
    maplist(format_ts_arg, Args, ArgStrs),
    generate_field_names(length(Args, L), L, FieldNames),
    maplist(format_field_value, FieldNames, ArgStrs, Pairs),
    atomic_list_concat(Pairs, ', ', Inner),
    format(string(Str), '{ ~w }', [Inner]).

format_ts_arg(Arg, Str) :-
    (   atom(Arg) -> format(string(Str), '"~w"', [Arg])
    ;   number(Arg) -> number_string(Arg, Str)
    ;   string(Arg) -> format(string(Str), '"~w"', [Arg])
    ;   format(string(Str), '"~w"', [Arg])
    ).

format_field_value(Field, Value, Pair) :-
    format(string(Pair), '~w: ~w', [Field, Value]).

generate_field_names(Arity, Names) :-
    findall(Name, (
        between(1, Arity, N),
        format(string(Name), 'arg~w', [N])
    ), Names).

generate_field_names(_, 0, []) :- !.
generate_field_names(_, N, Names) :-
    generate_field_names(N, Names).

generate_interface_fields(FieldNames, Fields) :-
    maplist([F, Line]>>format(string(Line), '  ~w: string;', [F]), FieldNames, Lines),
    atomic_list_concat(Lines, '\n', Fields).

generate_match_expr(FieldNames) :-
    maplist([F, Expr]>>format(string(Expr), 'f.~w === ~w', [F, F]), FieldNames, Exprs),
    atomic_list_concat(Exprs, ' && ', _Match).

generate_match_expr(FieldNames, Match) :-
    maplist([F, Expr]>>format(string(Expr), 'f.~w === ~w', [F, F]), FieldNames, Exprs),
    atomic_list_concat(Exprs, ' && ', Match).

%% ============================================
%% EXPRESS SERVICE GENERATION
%% ============================================

%% compile_express_service(+Service, -Code)
%
%  Generate an Express.js service from a service specification.
%
%  Service format:
%    service(Name, [
%        port(Port),
%        endpoints([...]),
%        middleware([...])
%    ])
%
compile_express_service(service(Name, Config), Code) :-
    atom_string(Name, NameStr),
    option(port(Port), Config, 3000),
    option(endpoints(Endpoints), Config, []),
    option(middleware(Middleware), Config, [cors, json]),

    % Collect imports
    collect_binding_import(express),
    (member(cors, Middleware) -> collect_binding_import(cors) ; true),

    % Generate middleware setup
    generate_middleware_setup(Middleware, MiddlewareCode),

    % Generate endpoints
    generate_express_endpoints(Endpoints, EndpointsCode),

    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Service: ~w

import express, { Request, Response } from "express";
import cors from "cors";

const app = express();

// Middleware
~w

// Endpoints
~w

// Start server
const PORT = process.env.PORT || ~w;
app.listen(PORT, () => {
  console.log(`~w service running on port ${PORT}`);
});

export default app;
', [NameStr, MiddlewareCode, EndpointsCode, Port, NameStr]).

%% generate_middleware_setup(+Middleware, -Code)
%
%  Generate Express middleware setup code.
%
generate_middleware_setup(Middleware, Code) :-
    findall(Line, (
        member(MW, Middleware),
        middleware_to_code(MW, Line)
    ), Lines),
    atomic_list_concat(Lines, '\n', Code).

middleware_to_code(cors, 'app.use(cors());').
middleware_to_code(json, 'app.use(express.json());').
middleware_to_code(urlencoded, 'app.use(express.urlencoded({ extended: true }));').
middleware_to_code(static(Path), Line) :-
    format(string(Line), 'app.use(express.static("~w"));', [Path]).
middleware_to_code(limit(Size), Line) :-
    format(string(Line), 'app.use(express.json({ limit: "~w" }));', [Size]).

%% generate_express_endpoints(+Endpoints, -Code)
%
%  Generate Express endpoint handlers.
%
generate_express_endpoints(Endpoints, Code) :-
    findall(EndpointCode, (
        member(Endpoint, Endpoints),
        generate_single_endpoint(Endpoint, EndpointCode)
    ), Codes),
    atomic_list_concat(Codes, '\n\n', Code).

%% generate_single_endpoint(+Endpoint, -Code)
%
%  Generate a single Express endpoint.
%
%  Endpoint format:
%    endpoint(Path, Method, Handler)
%    endpoint(Path, Method, [body(Schema), handler(Code)])
%
generate_single_endpoint(endpoint(Path, Method, Handler), Code) :-
    atom_string(Method, MethodStr),
    string_lower(MethodStr, MethodLower),
    (   atom(Handler)
    ->  atom_string(Handler, HandlerStr),
        format(string(Code),
'app.~w("~w", async (req: Request, res: Response) => {
  try {
    const result = await ~w(req);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({ success: false, error: String(error) });
  }
});', [MethodLower, Path, HandlerStr])
    ;   is_list(Handler)
    ->  option(handler(HandlerCode), Handler, 'res.json({ ok: true })'),
        option(body(BodySchema), Handler, none),
        generate_endpoint_with_validation(Path, MethodLower, BodySchema, HandlerCode, Code)
    ;   format(string(Code),
'app.~w("~w", (req: Request, res: Response) => {
  res.json({ message: "Not implemented" });
});', [MethodLower, Path])
    ).

generate_endpoint_with_validation(Path, Method, none, HandlerCode, Code) :-
    format(string(Code),
'app.~w("~w", async (req: Request, res: Response) => {
  try {
    ~w
  } catch (error) {
    res.status(500).json({ success: false, error: String(error) });
  }
});', [Method, Path, HandlerCode]).

generate_endpoint_with_validation(Path, Method, Schema, HandlerCode, Code) :-
    Schema \= none,
    format(string(Code),
'app.~w("~w", async (req: Request, res: Response) => {
  try {
    const body = req.body;
    // TODO: Validate against schema: ~w
    ~w
  } catch (error) {
    res.status(500).json({ success: false, error: String(error) });
  }
});', [Method, Path, Schema, HandlerCode]).

%% compile_express_router(+RouterSpec, -Code)
%
%  Generate an Express Router for modular route handling.
%
compile_express_router(router(Name, Endpoints), Code) :-
    atom_string(Name, NameStr),
    generate_express_endpoints(Endpoints, EndpointsCode),

    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Router: ~w

import { Router, Request, Response } from "express";

export const ~wRouter = Router();

~w
', [NameStr, NameStr, EndpointsCode]).

%% ============================================
%% HTTP CLIENT GENERATION
%% ============================================

%% compile_http_client(+ClientSpec, -Code)
%
%  Generate a typed HTTP client for API consumption.
%
compile_http_client(client(Name, Config), Code) :-
    atom_string(Name, NameStr),
    option(base_url(BaseUrl), Config, ''),
    option(endpoints(Endpoints), Config, []),

    % Generate client methods
    generate_client_methods(Endpoints, MethodsCode),

    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// HTTP Client: ~w

const BASE_URL = "~w";

export interface ApiResponse<T> {
  success: boolean;
  result?: T;
  error?: string;
}

~w

export const ~wClient = {
  baseUrl: BASE_URL,
  // Add methods here
};
', [NameStr, BaseUrl, MethodsCode, NameStr]).

generate_client_methods([], '').
generate_client_methods([Endpoint|Rest], Code) :-
    generate_client_method(Endpoint, MethodCode),
    generate_client_methods(Rest, RestCode),
    format(string(Code), '~w\n\n~w', [MethodCode, RestCode]).

generate_client_method(endpoint(Path, get, Name), Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
'export const ~w = async (): Promise<ApiResponse<unknown>> => {
  const response = await fetch(`${BASE_URL}~w`);
  return response.json();
};', [NameStr, Path]).

generate_client_method(endpoint(Path, post, Name), Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
'export const ~w = async (data: unknown): Promise<ApiResponse<unknown>> => {
  const response = await fetch(`${BASE_URL}~w`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
};', [NameStr, Path]).

%% ============================================
%% FILE OUTPUT
%% ============================================

write_typescript_module(Code, Filename) :-
    open(Filename, write, Stream),
    write(Stream, Code),
    close(Stream),
    format('TypeScript module written to: ~w~n', [Filename]),
    format('Compile with: npx tsc ~w~n', [Filename]).
