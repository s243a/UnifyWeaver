% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Express Generator - Declarative API Endpoint Generation
%
% This module provides declarative API endpoint definitions and generates
% Express.js router code from Prolog specifications.
%
% Usage:
%   % Define an endpoint
%   api_endpoint('/numpy/mean', [
%       method(post),
%       module(numpy),
%       function(mean),
%       input_schema([data: array(number)]),
%       output_schema(number),
%       description("Calculate arithmetic mean")
%   ]).
%
%   % Generate Express router
%   ?- generate_express_router(python_api, Code).

:- module(express_generator, [
    % Endpoint declaration
    api_endpoint/2,                     % api_endpoint(+Path, +Config)
    api_endpoint_group/2,               % api_endpoint_group(+Name, +Endpoints)

    % Endpoint management
    declare_endpoint/2,                 % declare_endpoint(+Path, +Config)
    declare_endpoint_group/2,           % declare_endpoint_group(+Name, +Endpoints)
    clear_endpoints/0,                  % clear_endpoints

    % Code generation
    generate_express_router/2,          % generate_express_router(+Name, -Code)
    generate_express_router/3,          % generate_express_router(+Name, +Options, -Code)
    generate_express_app/2,             % generate_express_app(+Name, -Code)
    generate_express_app/3,             % generate_express_app(+Name, +Options, -Code)
    generate_endpoint_types/2,          % generate_endpoint_types(+Name, -Code)

    % Integration with rpyc_security
    generate_secure_router/2,           % generate_secure_router(+Name, -Code)

    % Utilities
    endpoints_for_module/2,             % endpoints_for_module(+Module, -Endpoints)
    all_endpoints/1,                    % all_endpoints(-Endpoints)

    % Testing
    test_express_generator/0
]).

:- use_module(library(lists)).

% Conditionally load rpyc_security if available
:- if(exists_source('./rpyc_security')).
:- use_module('./rpyc_security', [
    rpyc_allowed_module/2,
    is_call_allowed/3,
    is_attr_allowed/3
]).
:- endif.

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic api_endpoint/2.
:- dynamic api_endpoint_group/2.

% ============================================================================
% DEFAULT ENDPOINTS (Examples)
% ============================================================================

% Math module endpoints
api_endpoint('/math/sqrt', [
    method(post),
    module(math),
    function(sqrt),
    input_schema([value: number]),
    output_schema(number),
    description("Calculate square root")
]).

api_endpoint('/math/pow', [
    method(post),
    module(math),
    function(pow),
    input_schema([base: number, exp: number]),
    output_schema(number),
    description("Calculate power (base^exp)")
]).

api_endpoint('/math/pi', [
    method(get),
    module(math),
    attr(pi),
    output_schema(number),
    description("Get pi constant")
]).

api_endpoint('/math/e', [
    method(get),
    module(math),
    attr(e),
    output_schema(number),
    description("Get e constant (Euler's number)")
]).

% NumPy endpoints
api_endpoint('/numpy/mean', [
    method(post),
    module(numpy),
    function(mean),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate arithmetic mean")
]).

api_endpoint('/numpy/std', [
    method(post),
    module(numpy),
    function(std),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate standard deviation")
]).

api_endpoint('/numpy/sum', [
    method(post),
    module(numpy),
    function(sum),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate sum of array")
]).

api_endpoint('/numpy/min', [
    method(post),
    module(numpy),
    function(min),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Find minimum value")
]).

api_endpoint('/numpy/max', [
    method(post),
    module(numpy),
    function(max),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Find maximum value")
]).

% Statistics endpoints
api_endpoint('/statistics/mean', [
    method(post),
    module(statistics),
    function(mean),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate mean using statistics module")
]).

api_endpoint('/statistics/median', [
    method(post),
    module(statistics),
    function(median),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate median")
]).

api_endpoint('/statistics/stdev', [
    method(post),
    module(statistics),
    function(stdev),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate standard deviation")
]).

% Endpoint groups
api_endpoint_group(math_endpoints, [
    '/math/sqrt',
    '/math/pow',
    '/math/pi',
    '/math/e'
]).

api_endpoint_group(numpy_endpoints, [
    '/numpy/mean',
    '/numpy/std',
    '/numpy/sum',
    '/numpy/min',
    '/numpy/max'
]).

api_endpoint_group(statistics_endpoints, [
    '/statistics/mean',
    '/statistics/median',
    '/statistics/stdev'
]).

% ============================================================================
% ENDPOINT MANAGEMENT
% ============================================================================

%% declare_endpoint(+Path, +Config)
%  Dynamically declare a new endpoint.
declare_endpoint(Path, Config) :-
    (   api_endpoint(Path, _)
    ->  retract(api_endpoint(Path, _))
    ;   true
    ),
    assertz(api_endpoint(Path, Config)).

%% declare_endpoint_group(+Name, +Endpoints)
%  Dynamically declare an endpoint group.
declare_endpoint_group(Name, Endpoints) :-
    (   api_endpoint_group(Name, _)
    ->  retract(api_endpoint_group(Name, _))
    ;   true
    ),
    assertz(api_endpoint_group(Name, Endpoints)).

%% clear_endpoints
%  Clear all dynamic endpoints.
clear_endpoints :-
    retractall(api_endpoint(_, _)),
    retractall(api_endpoint_group(_, _)).

%% all_endpoints(-Endpoints)
%  Get all defined endpoints as Path-Config pairs.
all_endpoints(Endpoints) :-
    findall(Path-Config, api_endpoint(Path, Config), Endpoints).

%% endpoints_for_module(+Module, -Endpoints)
%  Get all endpoints for a specific Python module.
endpoints_for_module(Module, Endpoints) :-
    findall(Path-Config, (
        api_endpoint(Path, Config),
        member(module(Module), Config)
    ), Endpoints).

% ============================================================================
% EXPRESS ROUTER GENERATION
% ============================================================================

%% generate_express_router(+Name, -Code)
%  Generate Express router with default options.
generate_express_router(Name, Code) :-
    generate_express_router(Name, [], Code).

%% generate_express_router(+Name, +Options, -Code)
%  Generate Express router with options.
%
%  Options:
%    - endpoints(List)      - List of endpoint paths or group names
%    - bridge_import(Path)  - Import path for RPyC bridge
%    - include_validation(Bool) - Include validation (default: true)
%
generate_express_router(Name, Options, Code) :-
    atom_string(Name, NameStr),

    % Get endpoints to include
    get_endpoints_to_generate(Options, EndpointPairs),

    % Get bridge import path
    (member(bridge_import(BridgePath), Options) -> true ; BridgePath = './rpyc_bridge'),

    % Generate route handlers
    generate_route_handlers(EndpointPairs, RouteHandlers),
    atomic_list_concat(RouteHandlers, '\n\n', RoutesCode),

    format(atom(Code), '/**
 * Express Router: ~w
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import { Router, Request, Response } from ''express'';
import { bridge } from ''~w'';
import { validateCall, validateAttr } from ''./validator'';

export const ~wRouter = Router();

~w

export default ~wRouter;
', [NameStr, BridgePath, NameStr, RoutesCode, NameStr]).

%% get_endpoints_to_generate(+Options, -EndpointPairs)
%  Get list of endpoint Path-Config pairs based on options.
get_endpoints_to_generate(Options, EndpointPairs) :-
    (   member(endpoints(Specs), Options)
    ->  expand_endpoint_specs(Specs, EndpointPairs)
    ;   all_endpoints(EndpointPairs)
    ).

%% expand_endpoint_specs(+Specs, -EndpointPairs)
%  Expand endpoint specifications (paths or group names) to Path-Config pairs.
expand_endpoint_specs([], []).
expand_endpoint_specs([Spec|Rest], AllPairs) :-
    (   api_endpoint_group(Spec, Paths)
    ->  % It's a group name - expand to paths
        findall(Path-Config, (
            member(Path, Paths),
            api_endpoint(Path, Config)
        ), GroupPairs)
    ;   atom(Spec), api_endpoint(Spec, Config)
    ->  % It's a path atom
        GroupPairs = [Spec-Config]
    ;   string(Spec), atom_string(PathAtom, Spec), api_endpoint(PathAtom, Config)
    ->  % It's a path string
        GroupPairs = [PathAtom-Config]
    ;   GroupPairs = []
    ),
    expand_endpoint_specs(Rest, RestPairs),
    append(GroupPairs, RestPairs, AllPairs).

%% generate_route_handlers(+EndpointPairs, -Handlers)
%  Generate Express route handler code for each endpoint.
generate_route_handlers([], []).
generate_route_handlers([Path-Config|Rest], [Handler|Handlers]) :-
    generate_single_route_handler(Path, Config, Handler),
    generate_route_handlers(Rest, Handlers).

%% generate_single_route_handler(+Path, +Config, -Handler)
%  Generate a single Express route handler.
generate_single_route_handler(Path, Config, Handler) :-
    atom_string(Path, PathStr),

    % Get method (default: post)
    (member(method(Method), Config) -> true ; Method = post),
    atom_string(Method, MethodStr),

    % Get module
    member(module(Module), Config),
    atom_string(Module, ModuleStr),

    % Get description
    (member(description(Desc), Config) -> true ; Desc = ""),

    % Check if it's a function call or attribute access
    (   member(function(Func), Config)
    ->  atom_string(Func, FuncStr),
        generate_function_handler(PathStr, MethodStr, ModuleStr, FuncStr, Desc, Config, Handler)
    ;   member(attr(Attr), Config)
    ->  atom_string(Attr, AttrStr),
        generate_attr_handler(PathStr, MethodStr, ModuleStr, AttrStr, Desc, Handler)
    ;   Handler = ""
    ).

%% generate_function_handler(+Path, +Method, +Module, +Func, +Desc, +Config, -Handler)
%  Generate handler for a function call endpoint.
generate_function_handler(PathStr, MethodStr, ModuleStr, FuncStr, Desc, Config, Handler) :-
    % Generate input extraction based on schema
    (member(input_schema(Schema), Config)
    ->  generate_input_extraction(Schema, InputExtraction, ArgsList)
    ;   InputExtraction = "const args = req.body.args || [];",
        ArgsList = "args"
    ),

    format(atom(Handler), '// ~w
~wRouter.~w(''~w'', async (req: Request, res: Response) => {
  try {
    ~w

    // Validate the call
    const validation = validateCall(''~w'', ''~w'', ~w);
    if (!validation.valid) {
      return res.status(400).json({ success: false, error: validation.error });
    }

    // Execute the call via RPyC bridge
    const result = await bridge.call(''~w'', ''~w'', ~w);

    res.json({ success: true, result });
  } catch (error) {
    console.error(''Error in ~w:'', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : ''Unknown error'',
    });
  }
});', [Desc, PathStr, MethodStr, PathStr, InputExtraction, ModuleStr, FuncStr, ArgsList,
       ModuleStr, FuncStr, ArgsList, PathStr]).

%% generate_attr_handler(+Path, +Method, +Module, +Attr, +Desc, -Handler)
%  Generate handler for an attribute access endpoint.
generate_attr_handler(PathStr, MethodStr, ModuleStr, AttrStr, Desc, Handler) :-
    format(atom(Handler), '// ~w
~wRouter.~w(''~w'', async (req: Request, res: Response) => {
  try {
    // Validate the attribute access
    const validation = validateAttr(''~w'', ''~w'');
    if (!validation.valid) {
      return res.status(400).json({ success: false, error: validation.error });
    }

    // Get the attribute via RPyC bridge
    const result = await bridge.getAttr(''~w'', ''~w'');

    res.json({ success: true, result });
  } catch (error) {
    console.error(''Error in ~w:'', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : ''Unknown error'',
    });
  }
});', [Desc, PathStr, MethodStr, PathStr, ModuleStr, AttrStr, ModuleStr, AttrStr, PathStr]).

%% generate_input_extraction(+Schema, -Code, -ArgsList)
%  Generate code to extract inputs from request body.
generate_input_extraction(Schema, Code, ArgsList) :-
    findall(Name, member(Name:_, Schema), Names),
    maplist(atom_string, Names, NameStrs),
    atomic_list_concat(NameStrs, ', ', DestructureList),
    maplist(wrap_in_brackets, NameStrs, BracketedNames),
    atomic_list_concat(BracketedNames, ', ', ArgsList),
    format(atom(Code), 'const { ~w } = req.body;', [DestructureList]).

wrap_in_brackets(Name, Bracketed) :-
    format(atom(Bracketed), '~w', [Name]).

% ============================================================================
% EXPRESS APP GENERATION
% ============================================================================

%% generate_express_app(+Name, -Code)
%  Generate complete Express app with default options.
generate_express_app(Name, Code) :-
    generate_express_app(Name, [], Code).

%% generate_express_app(+Name, +Options, -Code)
%  Generate complete Express app with options.
generate_express_app(Name, Options, Code) :-
    atom_string(Name, NameStr),

    % Get port (default: 3001)
    (member(port(Port), Options) -> true ; Port = 3001),

    % Generate router
    generate_express_router(Name, Options, RouterCode),

    format(atom(Code), '/**
 * Express Application: ~w
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import express from ''express'';
import cors from ''cors'';

const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: ''10mb'' }));

// Health check
app.get(''/health'', (req, res) => {
  res.json({ status: ''ok'', service: ''~w'' });
});

// ============================================================================
// ROUTER
// ============================================================================

~w

// Mount router
app.use(''/api'', ~wRouter);

// Error handler
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error(''Unhandled error:'', err);
  res.status(500).json({ success: false, error: ''Internal server error'' });
});

// Start server
const PORT = process.env.PORT || ~w;
app.listen(PORT, () => {
  console.log(`~w service running on port ${PORT}`);
});

export default app;
', [NameStr, NameStr, RouterCode, NameStr, Port, NameStr]).

% ============================================================================
% SECURE ROUTER GENERATION (with rpyc_security integration)
% ============================================================================

%% generate_secure_router(+Name, -Code)
%  Generate Express router with security middleware integration.
generate_secure_router(Name, Code) :-
    atom_string(Name, NameStr),
    all_endpoints(EndpointPairs),
    generate_route_handlers(EndpointPairs, RouteHandlers),
    atomic_list_concat(RouteHandlers, '\n\n', RoutesCode),

    format(atom(Code), '/**
 * Secure Express Router: ~w
 * Generated by UnifyWeaver - DO NOT EDIT
 *
 * Includes rate limiting, validation, and security middleware.
 */

import { Router, Request, Response } from ''express'';
import { bridge } from ''./rpyc_bridge'';
import { validateCall, validateAttr } from ''./validator'';
import {
  rateLimiter,
  timeoutMiddleware,
  securityMiddleware,
} from ''./security_middleware'';

export const ~wRouter = Router();

// Apply security middleware to all routes
~wRouter.use(securityMiddleware);

~w

export default ~wRouter;
', [NameStr, NameStr, NameStr, RoutesCode, NameStr]).

% ============================================================================
% TYPE GENERATION
% ============================================================================

%% generate_endpoint_types(+Name, -Code)
%  Generate TypeScript types for API endpoints.
generate_endpoint_types(Name, Code) :-
    atom_string(Name, NameStr),
    all_endpoints(EndpointPairs),
    generate_type_definitions(EndpointPairs, TypeDefs),
    atomic_list_concat(TypeDefs, '\n\n', TypesCode),

    format(atom(Code), '/**
 * API Types: ~w
 * Generated by UnifyWeaver - DO NOT EDIT
 */

~w

// Generic API response wrapper
export interface ApiResponse<T> {
  success: boolean;
  result?: T;
  error?: string;
}
', [NameStr, TypesCode]).

%% generate_type_definitions(+EndpointPairs, -TypeDefs)
%  Generate TypeScript type definitions for endpoints.
generate_type_definitions([], []).
generate_type_definitions([Path-Config|Rest], [TypeDef|TypeDefs]) :-
    generate_single_type_def(Path, Config, TypeDef),
    generate_type_definitions(Rest, TypeDefs).

%% generate_single_type_def(+Path, +Config, -TypeDef)
%  Generate TypeScript type for a single endpoint.
generate_single_type_def(Path, Config, TypeDef) :-
    % Convert path to type name (e.g., /numpy/mean -> NumpyMean)
    atom_string(Path, PathStr),
    path_to_type_name(PathStr, TypeName),

    % Generate input type
    (   member(input_schema(Schema), Config)
    ->  generate_schema_type(Schema, InputType)
    ;   InputType = "void"
    ),

    % Generate output type
    (   member(output_schema(OutSchema), Config)
    ->  schema_to_ts_type(OutSchema, OutputType)
    ;   OutputType = "unknown"
    ),

    format(atom(TypeDef), '// ~w
export interface ~wInput {
  ~w
}

export type ~wOutput = ~w;', [PathStr, TypeName, InputType, TypeName, OutputType]).

%% path_to_type_name(+Path, -TypeName)
%  Convert API path to TypeScript type name.
path_to_type_name(Path, TypeName) :-
    % Remove leading slash and split by /
    atom_string(PathAtom, Path),
    atom_codes(PathAtom, Codes),
    (Codes = [0'/|Rest] -> NameCodes = Rest ; NameCodes = Codes),
    atom_codes(CleanPath, NameCodes),
    atomic_list_concat(Parts, '/', CleanPath),
    maplist(capitalize_first, Parts, CapParts),
    atomic_list_concat(CapParts, '', TypeName).

%% capitalize_first(+Atom, -Capitalized)
%  Capitalize first letter of an atom.
capitalize_first(Atom, Capitalized) :-
    atom_string(Atom, Str),
    (   Str = ""
    ->  Capitalized = ''
    ;   string_codes(Str, [First|Rest]),
        to_upper(First, Upper),
        string_codes(CapStr, [Upper|Rest]),
        atom_string(Capitalized, CapStr)
    ).

%% generate_schema_type(+Schema, -TSType)
%  Generate TypeScript type from schema.
generate_schema_type([], "").
generate_schema_type([Name:Type|Rest], TSType) :-
    atom_string(Name, NameStr),
    schema_to_ts_type(Type, TSTypeStr),
    generate_schema_type(Rest, RestType),
    (   RestType = ""
    ->  format(atom(TSType), '~w: ~w;', [NameStr, TSTypeStr])
    ;   format(atom(TSType), '~w: ~w;~n  ~w', [NameStr, TSTypeStr, RestType])
    ).

%% schema_to_ts_type(+Schema, -TSType)
%  Convert schema type to TypeScript type.
schema_to_ts_type(number, "number").
schema_to_ts_type(string, "string").
schema_to_ts_type(boolean, "boolean").
schema_to_ts_type(array(Inner), TSType) :-
    schema_to_ts_type(Inner, InnerTS),
    format(atom(TSType), '~w[]', [InnerTS]).
schema_to_ts_type(object, "Record<string, unknown>").
schema_to_ts_type(any, "unknown").
schema_to_ts_type(_, "unknown").

% ============================================================================
% TESTING
% ============================================================================

test_express_generator :-
    format('~n=== Express Generator Tests ===~n~n'),

    % Test endpoint queries
    format('Endpoint Queries:~n'),
    all_endpoints(AllEps),
    length(AllEps, EpCount),
    format('  Total endpoints: ~w~n', [EpCount]),

    endpoints_for_module(math, MathEps),
    length(MathEps, MathCount),
    format('  Math endpoints: ~w~n', [MathCount]),

    endpoints_for_module(numpy, NumpyEps),
    length(NumpyEps, NumpyCount),
    format('  NumPy endpoints: ~w~n', [NumpyCount]),

    % Test code generation
    format('~nCode Generation:~n'),
    (   generate_express_router(test_api, RouterCode),
        atom_length(RouterCode, RouterLen),
        format('  Router: ~d chars~n', [RouterLen])
    ;   format('  Router: FAILED~n')
    ),

    (   generate_express_app(test_app, AppCode),
        atom_length(AppCode, AppLen),
        format('  App: ~d chars~n', [AppLen])
    ;   format('  App: FAILED~n')
    ),

    (   generate_endpoint_types(test_types, TypesCode),
        atom_length(TypesCode, TypesLen),
        format('  Types: ~d chars~n', [TypesLen])
    ;   format('  Types: FAILED~n')
    ),

    format('~n=== Tests Complete ===~n').
