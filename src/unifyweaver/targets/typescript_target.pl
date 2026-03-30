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
:- use_module('../core/clause_body_analysis').

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

% Try native clause body lowering first
compile_predicate_to_typescript(Pred/Arity, _Options, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    \+ (member(_-Body, Clauses), Body == true),
    native_ts_clause_body(Pred/Arity, Clauses, FuncBody),
    !,
    atom_string(Pred, PredStr),
    Arity1 is Arity - 1,
    build_ts_arg_list(Arity1, ArgList),
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Native Clause Lowering
// Predicate: ~w/~w

function ~w(~w): string {
~w
}

// CLI entry point
if (process.argv.length > 2) {
    console.log(~w(parseInt(process.argv[2])));
}
', [PredStr, Arity, PredStr, ArgList, FuncBody, PredStr]).

% Fallback to type-based dispatch
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
%% NATIVE CLAUSE BODY LOWERING
%% ============================================

%% build_ts_arg_list(+N, -ArgList)
build_ts_arg_list(0, "") :- !.
build_ts_arg_list(N, ArgList) :-
    findall(ArgDecl, (
        between(1, N, I),
        format(string(ArgDecl), 'arg~w: number', [I])
    ), ArgDecls),
    atomic_list_concat(ArgDecls, ', ', ArgList).

%% native_ts_clause_body(+PredSpec, +Clauses, -Code)

% Single clause
native_ts_clause_body(PredSpec, [Head-Body], Code) :-
    native_ts_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !,
    (   Condition == "true"
    ->  format(string(Code), '    return ~w;', [ClauseCode])
    ;   format(string(Code),
'    if (~w) {
        return ~w;
    }
    throw new Error("No matching clause for ~w");', [Condition, ClauseCode, PredSpec])
    ).

% Multi-clause → if/else if/else
native_ts_clause_body(PredSpec, Clauses, Code) :-
    Clauses = [_|[_|_]],
    maplist(native_ts_clause_pair(PredSpec), Clauses, Branches),
    Branches \= [],
    branches_to_ts_if_chain(Branches, PredSpec, Code).

native_ts_clause_pair(PredSpec, Head-Body, branch(Condition, ClauseCode)) :-
    native_ts_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !.

%% native_ts_clause(+PredSpec, +Head, +Body, -Condition, -Code)
native_ts_clause(_PredSpec, Head, Body, Condition, Code) :-
    Head =.. [_Pred|HeadArgs],
    length(HeadArgs, Arity),
    build_head_varmap(HeadArgs, 1, VarMap),
    (   Arity > 1
    ->  append(_InputHeadArgs, [OutputHeadArg], HeadArgs),
        ts_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ;   OutputHeadArg = _,
        ts_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ),
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  ts_resolve_value(VarMap, OutputHeadArg, Code),
        GoalConditions = []
    ;   (   Arity > 1, nonvar(OutputHeadArg)
        ->  clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
            maplist(ts_guard_condition(VarMap), GuardGoals, GoalConditions),
            (   OutputGoals == []
            ->  ts_literal(OutputHeadArg, Code)
            ;   ts_output_goals(OutputGoals, VarMap, Code)
            )
        ;   native_ts_goal_sequence(Goals, VarMap, GoalConditions, Code)
        )
    ),
    append(HeadConditions, GoalConditions, AllConditions),
    combine_ts_conditions(AllConditions, Condition).

%% ts_head_conditions(+HeadArgs, +Index, +Arity, -Conditions)
ts_head_conditions([], _, _, []).
ts_head_conditions([_], _, Arity, []) :- Arity > 1, !.
ts_head_conditions([HeadArg|Rest], Index, Arity, Conditions) :-
    (   var(HeadArg)
    ->  Conditions = RestConditions
    ;   format(string(ArgName), 'arg~w', [Index]),
        ts_literal(HeadArg, Literal),
        format(string(Cond), '~w === ~w', [ArgName, Literal]),
        Conditions = [Cond|RestConditions]
    ),
    NextIndex is Index + 1,
    ts_head_conditions(Rest, NextIndex, Arity, RestConditions).

%% native_ts_goal_sequence(+Goals, +VarMap, -Conditions, -Code)
%  Uses classify_goal_sequence for advanced pattern detection.
%  Falls back to clause_guard_output_split if classification fails.
native_ts_goal_sequence(Goals, VarMap, Conditions, Code) :-
    classify_goal_sequence(Goals, VarMap, ClassifiedGoals),
    ClassifiedGoals \= [],
    ts_render_classified_goals(ClassifiedGoals, VarMap, Conditions, Lines),
    Lines \= [],
    atomic_list_concat(Lines, '\n', Code),
    !.
native_ts_goal_sequence(Goals, VarMap, Conditions, Code) :-
    clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
    maplist(ts_guard_condition(VarMap), GuardGoals, Conditions),
    ts_output_goals(OutputGoals, VarMap, Code).

%% ts_render_classified_goals(+ClassifiedGoals, +VarMap, -Conditions, -Lines)
ts_render_classified_goals([], _VarMap, [], []).
ts_render_classified_goals([Classified], VarMap, Conds, Lines) :-
    !,
    ts_render_classified_last(Classified, VarMap, Conds, Lines).
%% Guarded tail: output followed by guard(s)
ts_render_classified_goals([output(Goal, _, _)|Rest], VarMap, [], Lines) :-
    Rest = [guard(_, _)|_],
    !,
    ts_output_goal(Goal, VarMap, AssignLine, VarMap1),
    ts_collect_trailing_guards(Rest, VarMap1, GuardGoals, _Remaining),
    maplist(ts_guard_condition(VarMap1), GuardGoals, GuardConds),
    atomic_list_concat(GuardConds, ' && ', GuardExpr),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMap1, OutName)
    ->  true
    ;   OutName = "undefined"
    ),
    format(string(IfLine), '  if (~w) {', [GuardExpr]),
    format(string(RetLine), '    return ~w;', [OutName]),
    CloseLine = '  }',
    Lines = [AssignLine, IfLine, RetLine, CloseLine].
ts_render_classified_goals([Classified|Rest], VarMap, Conds, Lines) :-
    ts_render_classified_mid(Classified, VarMap, MidConds, MidLines, VarMap1),
    ts_render_classified_goals(Rest, VarMap1, RestConds, RestLines),
    append(MidConds, RestConds, Conds),
    append(MidLines, RestLines, Lines).

%% ts_render_classified_mid(+Classified, +VarMap, -Conds, -Lines, -VarMapOut)
ts_render_classified_mid(guard(Goal, _), VarMap, [Cond], [], VarMap) :-
    ts_guard_condition(VarMap, Goal, Cond).
ts_render_classified_mid(output(Goal, _, _), VarMap0, [], [Line], VarMapOut) :-
    ts_output_goal(Goal, VarMap0, Line, VarMapOut).
ts_render_classified_mid(output_ite(If, Then, Else, _SharedVars), VarMap0, [], Lines, VarMap0) :-
    ts_guard_condition(VarMap0, If, Cond),
    ts_branch_value(Then, VarMap0, ThenExpr),
    ts_branch_value(Else, VarMap0, ElseExpr),
    format(string(IfLine), '  if (~w) {', [Cond]),
    format(string(ThenLine), '    return ~w;', [ThenExpr]),
    ElseLine = '  } else {',
    format(string(ElseRetLine), '    return ~w;', [ElseExpr]),
    Lines = [IfLine, ThenLine, ElseLine, ElseRetLine, '  }'].
ts_render_classified_mid(passthrough(Goal), VarMap0, [], [Line], VarMapOut) :-
    ts_output_goal(Goal, VarMap0, Line, VarMapOut).
ts_render_classified_mid(_, VarMap, [], [], VarMap).

%% ts_render_classified_last(+Classified, +VarMap, -Conds, -Lines)
ts_render_classified_last(guard(Goal, _), VarMap, [Cond], []) :-
    ts_guard_condition(VarMap, Goal, Cond).
ts_render_classified_last(output(Goal, _, _), VarMap, [], Lines) :-
    ts_output_goal_last_lines(Goal, VarMap, Lines).
ts_render_classified_last(output_ite(If, Then, Else, _), VarMap, [], Lines) :-
    ts_guard_condition(VarMap, If, Cond),
    ts_branch_value(Then, VarMap, ThenExpr),
    ts_branch_value(Else, VarMap, ElseExpr),
    format(string(IfLine), '  if (~w) {', [Cond]),
    format(string(ThenLine), '    return ~w;', [ThenExpr]),
    ElseLine = '  } else {',
    format(string(ElseRetLine), '    return ~w;', [ElseExpr]),
    Lines = [IfLine, ThenLine, ElseLine, ElseRetLine, '  }'].
ts_render_classified_last(output_disj(Alternatives, _SharedVars), VarMap, [], Lines) :-
    ts_disj_if_chain(Alternatives, VarMap, Lines).
ts_render_classified_last(passthrough(Goal), VarMap, [], Lines) :-
    ts_output_goal_last_lines(Goal, VarMap, Lines).
ts_render_classified_last(_, _, [], []).

%% ts_output_goal_last_lines(+Goal, +VarMap, -Lines)
ts_output_goal_last_lines(Goal, VarMap, [Line]) :-
    ts_output_goal(Goal, VarMap, AssignLine, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, OutName)
    ->  format(string(RetPart), '\n  return ~w;', [OutName]),
        atom_concat(AssignLine, RetPart, Line)
    ;   Line = AssignLine
    ).
ts_output_goal_last_lines(Goal, VarMap, [Line]) :-
    ts_branch_value(Goal, VarMap, Expr),
    format(string(Line), '  return ~w;', [Expr]).

%% ts_collect_trailing_guards(+ClassifiedGoals, +VarMap, -GuardGoals, -Remaining)
ts_collect_trailing_guards([guard(Goal, _)|Rest], VarMap, [Goal|Guards], Remaining) :-
    !, ts_collect_trailing_guards(Rest, VarMap, Guards, Remaining).
ts_collect_trailing_guards(Remaining, _, [], Remaining).

%% ts_disj_if_chain(+Alternatives, +VarMap, -Lines)
ts_disj_if_chain([], _, []).
ts_disj_if_chain([Alt], VarMap, [ElseLine, RetLine, CloseLine]) :-
    !,
    ts_branch_value(Alt, VarMap, ValExpr),
    ElseLine = '  } else {',
    format(string(RetLine), '    return ~w;', [ValExpr]),
    CloseLine = '  }'.
ts_disj_if_chain([Alt|Rest], VarMap, Lines) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(ts_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' && ', CondExpr)
    ;   CondExpr = "true"
    ),
    ts_branch_value(Alt, VarMap, ValExpr),
    format(string(IfLine), '  if (~w) {', [CondExpr]),
    format(string(RetLine), '    return ~w;', [ValExpr]),
    ts_disj_else_if_chain(Rest, VarMap, RestLines),
    append([IfLine, RetLine], RestLines, Lines).

ts_disj_else_if_chain([], _, []).
ts_disj_else_if_chain([Alt], VarMap, [ElseLine, RetLine, CloseLine]) :-
    !,
    ts_branch_value(Alt, VarMap, ValExpr),
    ElseLine = '  } else {',
    format(string(RetLine), '    return ~w;', [ValExpr]),
    CloseLine = '  }'.
ts_disj_else_if_chain([Alt|Rest], VarMap, [ElseIfLine, RetLine|RestLines]) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(ts_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' && ', CondExpr)
    ;   CondExpr = "true"
    ),
    ts_branch_value(Alt, VarMap, ValExpr),
    format(string(ElseIfLine), '  } else if (~w) {', [CondExpr]),
    format(string(RetLine), '    return ~w;', [ValExpr]),
    ts_disj_else_if_chain(Rest, VarMap, RestLines).

%% ts_guard_condition(+VarMap, +Goal, -Condition)
ts_guard_condition(VarMap, _Module:Goal, Condition) :-
    !, ts_guard_condition(VarMap, Goal, Condition).
ts_guard_condition(VarMap, Goal, Condition) :-
    compound(Goal),
    Goal =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    ts_expr(Left, VarMap, TLeft),
    ts_expr(Right, VarMap, TRight),
    ts_op(StdOp, TOp),
    format(string(Condition), '~w ~w ~w', [TLeft, TOp, TRight]).

%% ts_output_goals(+Goals, +VarMap, -Code)
ts_output_goals([], _VarMap, '"error"') :- !.
ts_output_goals([Goal], VarMap, Code) :-
    !, ts_output_goal_last(Goal, VarMap, Code).
ts_output_goals([Goal|Rest], VarMap0, Code) :-
    ts_output_goal(Goal, VarMap0, _Line, VarMap1),
    ts_output_goals(Rest, VarMap1, Code).

%% ts_output_goal_last — produce the return expression
ts_output_goal_last(_Module:Goal, VarMap, Code) :-
    !, ts_output_goal_last(Goal, VarMap, Code).
ts_output_goal_last(Goal, VarMap, Code) :-
    if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    !,
    ts_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code).
ts_output_goal_last(=(Var, Expr), VarMap, Code) :-
    var(Var), !,
    ts_expr(Expr, VarMap, Code).
ts_output_goal_last(is(Var, Expr), VarMap, Code) :-
    var(Var), !,
    ts_expr(Expr, VarMap, Code).

%% ts_output_goal — produce a const assignment (not used as return)
ts_output_goal(_Module:Goal, VarMap0, Line, VarMapOut) :-
    !, ts_output_goal(Goal, VarMap0, Line, VarMapOut).
ts_output_goal(=(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    ts_expr(Expr, VarMap0, TExpr),
    format(string(Line), 'const ~w = ~w;', [VarName, TExpr]).
ts_output_goal(is(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    ts_expr(Expr, VarMap0, TExpr),
    format(string(Line), 'const ~w = ~w;', [VarName, TExpr]).

%% ts_if_then_else_output — generate ternary expressions
ts_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code) :-
    flatten_ts_if_branches(IfGoal, ThenGoal, ElseGoal, Branches, DefaultGoal),
    ts_branches_to_ternary(Branches, DefaultGoal, VarMap, Code).

flatten_ts_if_branches(If, Then, Else, [branch(If, Then)|RestBranches], Default) :-
    if_then_else_goal(Else, If2, Then2, Else2),
    !,
    flatten_ts_if_branches(If2, Then2, Else2, RestBranches, Default).
flatten_ts_if_branches(If, Then, Else, [branch(If, Then)], Else).

ts_branches_to_ternary([branch(If, Then)], DefaultGoal, VarMap, Code) :-
    !,
    ts_guard_condition(VarMap, If, IfCond),
    ts_branch_value(Then, VarMap, ThenVal),
    ts_branch_value(DefaultGoal, VarMap, ElseVal),
    format(string(Code), '(~w) ? ~w : ~w', [IfCond, ThenVal, ElseVal]).
ts_branches_to_ternary([branch(If, Then)|Rest], DefaultGoal, VarMap, Code) :-
    ts_guard_condition(VarMap, If, IfCond),
    ts_branch_value(Then, VarMap, ThenVal),
    ts_branches_to_ternary(Rest, DefaultGoal, VarMap, ElseCode),
    format(string(Code), '(~w) ? ~w : ~w', [IfCond, ThenVal, ElseCode]).

%% ts_branch_value — extract result value from a branch
ts_branch_value(_Module:Goal, VarMap, Value) :-
    !, ts_branch_value(Goal, VarMap, Value).
ts_branch_value(Goal, VarMap, Value) :-
    if_then_else_goal(Goal, If, Then, Else),
    !,
    ts_guard_condition(VarMap, If, Cond),
    ts_branch_value(Then, VarMap, ThenVal),
    ts_branch_value(Else, VarMap, ElseVal),
    format(string(Value), '(~w) ? ~w : ~w', [Cond, ThenVal, ElseVal]).
ts_branch_value((A, B), VarMap, Value) :-
    !,
    normalize_goals((A, B), Goals),
    last(Goals, LastGoal),
    ts_branch_value(LastGoal, VarMap, Value).
ts_branch_value(=(_, Expr), VarMap, Value) :-
    !, ts_expr(Expr, VarMap, Value).
ts_branch_value(is(_, Expr), VarMap, Value) :-
    !, ts_expr(Expr, VarMap, Value).
ts_branch_value(Goal, VarMap, Value) :-
    ts_expr(Goal, VarMap, Value).

% ============================================================================
% MULTIFILE HOOKS — Register TypeScript renderers for shared compile_expression
% ============================================================================

clause_body_analysis:render_output_goal(typescript, Goal, VarMap, Line, VarName, VarMapOut) :-
    ts_output_goal(Goal, VarMap, Line, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, VarName)
    ->  true
    ;   VarName = "_"
    ).

clause_body_analysis:render_guard_condition(typescript, Goal, VarMap, CondStr) :-
    ts_guard_condition(VarMap, Goal, CondStr).

clause_body_analysis:render_branch_value(typescript, Branch, VarMap, ExprStr) :-
    ts_branch_value(Branch, VarMap, ExprStr).

clause_body_analysis:render_ite_block(typescript, Cond, ThenLines, ElseLines, Indent, _ReturnVars, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Indent, Cond]),
    ts_indent_lines(ThenLines, Indent, IndentedThen),
    (   ElseLines \= []
    ->  format(string(ElseLine), '~w} else {', [Indent]),
        ts_indent_lines(ElseLines, Indent, IndentedElse),
        format(string(EndLine), '~w}', [Indent]),
        append([IfLine|IndentedThen], [ElseLine|IndentedElse], PreEnd),
        append(PreEnd, [EndLine], Lines)
    ;   format(string(EndLine), '~w}', [Indent]),
        append([IfLine|IndentedThen], [EndLine], Lines)
    ).

ts_indent_lines([], _, []).
ts_indent_lines([Line|Rest], Indent, [Indented|RestIndented]) :-
    format(string(Indented), '~w    ~w', [Indent, Line]),
    ts_indent_lines(Rest, Indent, RestIndented).

%% ts_expr — convert Prolog expression to TypeScript syntax
ts_expr(Var, VarMap, TExpr) :-
    var(Var), !,
    (   lookup_var(Var, VarMap, Name)
    ->  TExpr = Name
    ;   term_string(Var, TExpr)
    ).
ts_expr(Expr, VarMap, TExpr) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    ts_expr(Left, VarMap, TLeft),
    ts_expr(Right, VarMap, TRight),
    ts_op(StdOp, TOp),
    format(string(TExpr), '(~w ~w ~w)', [TLeft, TOp, TRight]).
ts_expr(-Expr, VarMap, TExpr) :-
    !,
    ts_expr(Expr, VarMap, Inner),
    format(string(TExpr), '(-~w)', [Inner]).
ts_expr(abs(Expr), VarMap, TExpr) :-
    !,
    ts_expr(Expr, VarMap, Inner),
    format(string(TExpr), 'Math.abs(~w)', [Inner]).
ts_expr(Atom, _VarMap, TExpr) :-
    atom(Atom), !,
    ts_literal(Atom, TExpr).
ts_expr(Number, _VarMap, TExpr) :-
    number(Number), !,
    format(string(TExpr), '~w', [Number]).
ts_expr(String, _VarMap, TExpr) :-
    string(String), !,
    format(string(TExpr), '"~w"', [String]).

%% ts_literal — convert Prolog value to TypeScript literal
ts_literal(Value, '""') :- var(Value), !.
ts_literal(true, '"true"') :- !.
ts_literal(false, '"false"') :- !.
ts_literal(Value, TsLiteral) :-
    number(Value), !,
    format(string(TsLiteral), 'String(~w)', [Value]).
ts_literal(Value, TsLiteral) :-
    atom(Value), !,
    format(string(TsLiteral), '"~w"', [Value]).
ts_literal(Value, TsLiteral) :-
    string(Value), !,
    format(string(TsLiteral), '"~w"', [Value]).
ts_literal(Value, TsLiteral) :-
    term_string(Value, S),
    format(string(TsLiteral), '"~w"', [S]).

%% ts_resolve_value — resolve variable or constant to TypeScript expression
ts_resolve_value(VarMap, Var, TExpr) :-
    var(Var), !,
    lookup_var(Var, VarMap, TExpr).
ts_resolve_value(_VarMap, Value, TExpr) :-
    ts_literal(Value, TExpr).

%% ts_op — map standard operator to TypeScript syntax
ts_op('>', '>').
ts_op('<', '<').
ts_op('>=', '>=').
ts_op('<=', '<=').
ts_op('==', '===').
ts_op('!=', '!==').
ts_op('+', '+').
ts_op('-', '-').
ts_op('*', '*').
ts_op('/', '/').
ts_op('%', '%').
ts_op('&&', '&&').
ts_op('||', '||').

%% combine_ts_conditions — join conditions with &&
combine_ts_conditions([], "true") :- !.
combine_ts_conditions([Condition], Condition) :- !.
combine_ts_conditions(Conditions, Combined) :-
    atomic_list_concat(Conditions, ' && ', Combined).

%% branches_to_ts_if_chain — build TypeScript if/else if/else chain
branches_to_ts_if_chain(Branches, PredSpec, Code) :-
    branches_to_ts_if_lines(Branches, PredSpec, Lines),
    atomic_list_concat(Lines, '\n', Code).

branches_to_ts_if_lines([branch(Condition, ClauseCode)], PredSpec, [IfLine, RetLine, ElseLine, ErrLine, CloseLine]) :-
    !,
    format(string(IfLine), '    if (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    ElseLine = '    } else {',
    format(string(ErrLine), '        throw new Error("No matching clause for ~w");', [PredSpec]),
    CloseLine = '    }'.
branches_to_ts_if_lines([branch(Condition, ClauseCode)|Rest], PredSpec, [IfLine, RetLine|RestLines]) :-
    format(string(IfLine), '    if (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    branches_to_ts_elif_lines(Rest, PredSpec, RestLines).

branches_to_ts_elif_lines([branch(Condition, ClauseCode)], PredSpec, [ElifLine, RetLine, ElseLine, ErrLine, CloseLine]) :-
    !,
    format(string(ElifLine), '    } else if (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    ElseLine = '    } else {',
    format(string(ErrLine), '        throw new Error("No matching clause for ~w");', [PredSpec]),
    CloseLine = '    }'.
branches_to_ts_elif_lines([branch(Condition, ClauseCode)|Rest], PredSpec, [ElifLine, RetLine|RestLines]) :-
    format(string(ElifLine), '    } else if (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    branches_to_ts_elif_lines(Rest, PredSpec, RestLines).

%% ============================================
%% FILE OUTPUT
%% ============================================

write_typescript_module(Code, Filename) :-
    open(Filename, write, Stream),
    write(Stream, Code),
    close(Stream),
    format('TypeScript module written to: ~w~n', [Filename]),
    format('Compile with: npx tsc ~w~n', [Filename]).

% ============================================================================
% ADVANCED RECURSION - Multifile dispatch registrations
% ============================================================================

% ============================================================================
% TAIL RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(typescript, PredStr, Arity, _BaseClauses, _RecClauses, _AccPos, StepOp, _ExitAfterResult, Code) :-
    step_op_to_ts(StepOp, TsStepExpr),
    (   Arity =:= 3 ->
        format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Tail Recursion (list, multifile dispatch)
// Predicate: ~w/~w

const ~w = (items: number[]): number => {
  let acc = 0;
  for (const item of items) {
    ~w;
  }
  return acc;
};

if (process.argv[2]) {
  const items = process.argv[2].split(",").map(Number);
  console.log(~w(items));
}
', [PredStr, Arity, PredStr, TsStepExpr, PredStr])
    ;   Arity =:= 2 ->
        format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Tail Recursion (multifile dispatch)
// Predicate: ~w/~w

const ~w = (items: number[]): number => {
  return items.length;
};

if (process.argv[2]) {
  const items = process.argv[2].split(",").map(Number);
  console.log(~w(items));
}
', [PredStr, Arity, PredStr, PredStr])
    ;   fail
    ).

step_op_to_ts(arithmetic(Expr), TsExpr) :- tail_expr_to_ts(Expr, TsExpr).
step_op_to_ts(unknown, 'acc += 1').

tail_expr_to_ts(_ + Const, TsExpr) :- integer(Const), !, format(atom(TsExpr), 'acc += ~w', [Const]).
tail_expr_to_ts(_ + _, 'acc += item') :- !.
tail_expr_to_ts(_ - _, 'acc -= item') :- !.
tail_expr_to_ts(_ * _, 'acc *= item') :- !.
tail_expr_to_ts(_, 'acc += 1').

% ============================================================================
% LINEAR RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/linear_recursion').
:- multifile linear_recursion:compile_linear_pattern/8.

linear_recursion:compile_linear_pattern(typescript, PredStr, Arity, BaseClauses, _RecClauses, _MemoEnabled, _MemoStrategy, Code) :-
    atom_string(Pred, PredStr),
    linear_recursion:extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    linear_recursion:detect_input_type(BaseInput, InputType),
    % Extract recursive clauses once (used by both numeric and list branches)
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), AllClauses),
    partition(linear_recursion:is_recursive_clause(Pred), AllClauses, ActualRec, _),
    (   InputType = numeric ->
        % Extract fold expression
        (   ActualRec = [clause(RH, RBody)|_],
            RH =.. [_, InputVar, _],
            linear_recursion:find_recursive_call(RBody, RecCall),
            RecCall =.. [_, _, AccVar],
            linear_recursion:find_last_is_expression(RBody, _ is FoldExpr)
        ->  translate_fold_expr_typescript(FoldExpr, InputVar, AccVar, TsOp)
        ;   TsOp = "acc + current"
        ),
        % Extract step size
        (   linear_recursion:extract_step_info_for(Pred/Arity, Step, _Dir) -> true ; Step = 1 ),
        format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Linear Recursion (numeric, multifile dispatch)
// Predicate: ~w/~w

const ~wMemo = new Map<number, number>();

const ~w = (n: number): number => {
  if (~wMemo.has(n)) return ~wMemo.get(n)!;
  if (n === ~w) return ~w;
  const current = n;
  const acc = ~w(n - ~w);
  const result = ~w;
  ~wMemo.set(n, result);
  return result;
};

if (process.argv[2]) {
  console.log(~w(parseInt(process.argv[2])));
}
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr, BaseInput, BaseOutput, PredStr, Step, TsOp, PredStr, PredStr])
    ;   InputType = list ->
        % Re-extract fold with head variable for list patterns
        (   ActualRec = [clause(LRHead, LRBody)|_] ->
            linear_recursion:find_last_is_expression(LRBody, _ is LFoldExpr),
            linear_recursion:find_recursive_call(LRBody, LRecCall),
            LRecCall =.. [_, _, LAccVar],
            LRHead =.. [_, [LHeadVar|_], _],
            translate_list_fold_typescript(LFoldExpr, LHeadVar, LAccVar, ListTsOp)
        ;   ListTsOp = "acc + current"
        ),
        format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Linear Recursion (list, multifile dispatch)
// Predicate: ~w/2

const ~w = (lst: number[]): number => {
  if (lst.length === 0) return ~w;
  return lst.reduce((acc, current) => ~w, ~w);
};

if (process.argv[2]) {
  console.log(~w(process.argv[2].split(",").map(Number)));
}
', [PredStr, PredStr, BaseOutput, ListTsOp, BaseOutput, PredStr])
    ;   linear_generic_typescript(PredStr, Arity, Code)
    ).

linear_generic_typescript(PredStr, Arity, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Linear Recursion (generic, multifile dispatch)
// Predicate: ~w/~w

const ~wMemo = new Map<number, number>();

const ~w = (n: number): number => {
  if (~wMemo.has(n)) return ~wMemo.get(n)!;
  if (n <= 0) return 0;
  if (n === 1) return 1;
  const result = ~w(n - 1) + n;
  ~wMemo.set(n, result);
  return result;
};

if (process.argv[2]) {
  console.log(~w(parseInt(process.argv[2])));
}
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr]).

%% translate_fold_expr_typescript(+PrologExpr, +InputVar, +AccVar, -TsExpr)
translate_fold_expr_typescript(A * B, InputVar, AccVar, Expr) :-
    translate_ts_term(A, InputVar, AccVar, AT),
    translate_ts_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w * ~w', [AT, BT]).
translate_fold_expr_typescript(A + B, InputVar, AccVar, Expr) :-
    translate_ts_term(A, InputVar, AccVar, AT),
    translate_ts_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w + ~w', [AT, BT]).
translate_fold_expr_typescript(A - B, InputVar, AccVar, Expr) :-
    translate_ts_term(A, InputVar, AccVar, AT),
    translate_ts_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w - ~w', [AT, BT]).
translate_fold_expr_typescript(Term, InputVar, AccVar, Expr) :-
    translate_ts_term(Term, InputVar, AccVar, Expr).

translate_ts_term(Term, InputVar, _AccVar, 'current') :- Term == InputVar, !.
translate_ts_term(Term, _InputVar, AccVar, 'acc') :- Term == AccVar, !.
translate_ts_term(Number, _, _, TsTerm) :- integer(Number), !,
    format(string(TsTerm), '~w', [Number]).
translate_ts_term(Atom, _, _, TsTerm) :-
    format(string(TsTerm), '~w', [Atom]).

%% translate_list_fold_typescript(+PrologExpr, +HeadVar, +AccVar, -TsExpr)
%  Like translate_fold_expr_typescript but maps HeadVar → 'current' (reduce callback
%  parameter for each element) and AccVar → 'acc' (reduce accumulator parameter).
translate_list_fold_typescript(A * B, HV, AV, E) :-
    translate_list_term_ts(A, HV, AV, AT), translate_list_term_ts(B, HV, AV, BT),
    format(string(E), '~w * ~w', [AT, BT]).
translate_list_fold_typescript(A + B, HV, AV, E) :-
    translate_list_term_ts(A, HV, AV, AT), translate_list_term_ts(B, HV, AV, BT),
    format(string(E), '~w + ~w', [AT, BT]).
translate_list_fold_typescript(A - B, HV, AV, E) :-
    translate_list_term_ts(A, HV, AV, AT), translate_list_term_ts(B, HV, AV, BT),
    format(string(E), '~w - ~w', [AT, BT]).
translate_list_fold_typescript(T, HV, AV, E) :- translate_list_term_ts(T, HV, AV, E).

translate_list_term_ts(T, HV, _, 'current') :- T == HV, !.
translate_list_term_ts(T, _, AV, 'acc') :- T == AV, !.
translate_list_term_ts(N, _, _, S) :- integer(N), !, format(string(S), '~w', [N]).
translate_list_term_ts(A, _, _, S) :- format(string(S), '~w', [A]).

% ============================================================================
% TREE RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

tree_recursion:compile_tree_pattern(typescript, _Pattern, Pred, _Arity, _UseMemo, TsCode) :-
    atom_string(Pred, PredStr),
    format(string(TsCode),
'// Generated by UnifyWeaver TypeScript Target - Tree Recursion (multifile dispatch)

const ~wMemo = new Map<number, number>();

const ~w = (n: number): number => {
  if (~wMemo.has(n)) return ~wMemo.get(n)!;
  if (n <= 0) return 0;
  if (n === 1) return 1;
  const result = ~w(n - 1) + ~w(n - 2);
  ~wMemo.set(n, result);
  return result;
};

if (process.argv[2]) {
  console.log(~w(parseInt(process.argv[2])));
}
', [PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MULTICALL LINEAR RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/multicall_linear_recursion').
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.

multicall_linear_recursion:compile_multicall_pattern(typescript, PredStr, BaseClauses, _RecClauses, _MemoEnabled, TsCode) :-
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '  if (n === ~w) return ~w;', [BInput, BOutput])
    ), BaseCaseCodes0),
    sort(BaseCaseCodes0, BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),
    format(string(TsCode),
'// Generated by UnifyWeaver TypeScript Target - Multicall Linear Recursion (multifile dispatch)

const ~wMemo = new Map<number, number>();

const ~w = (n: number): number => {
  if (~wMemo.has(n)) return ~wMemo.get(n)!;
~w
  const result = ~w(n - 1) + ~w(n - 2);
  ~wMemo.set(n, result);
  return result;
};

if (process.argv[2]) {
  console.log(~w(parseInt(process.argv[2])));
}
', [PredStr, PredStr, PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% DIRECT MULTICALL RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/direct_multi_call_recursion').
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.

direct_multi_call_recursion:compile_direct_multicall_pattern(typescript, PredStr, BaseClauses, _RecClause, TsCode) :-
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '  if (n === ~w) { ~wMemo.set(~w, ~w); return ~w; }', [BInput, PredStr, BInput, BOutput, BOutput])
    ), BaseCaseCodes0),
    sort(BaseCaseCodes0, BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),
    format(string(TsCode),
'// Generated by UnifyWeaver TypeScript Target - Direct Multicall Recursion (multifile dispatch)

const ~wMemo = new Map<number, number>();

const ~w = (n: number): number => {
  if (~wMemo.has(n)) return ~wMemo.get(n)!;
~w
  const result = ~w(n - 1) + ~w(n - 2);
  ~wMemo.set(n, result);
  return result;
};

if (process.argv[2]) {
  console.log(~w(parseInt(process.argv[2])));
}
', [PredStr, PredStr, PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MUTUAL RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/mutual_recursion').
:- multifile mutual_recursion:compile_mutual_pattern/5.

mutual_recursion:compile_mutual_pattern(typescript, Predicates, MemoEnabled, _MemoStrategy, TsCode) :-
    mutual_functions_typescript(Predicates, Predicates, MemoEnabled, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    mutual_dispatch_typescript(Predicates, DispatchCode),
    format(string(TsCode),
'// Generated by UnifyWeaver TypeScript Target - Mutual Recursion (multifile dispatch)

const mutualMemo = new Map<string, boolean>();

~w

if (process.argv[2] && process.argv[3]) {
  const func = process.argv[2];
  const n = parseInt(process.argv[3]);
~w
}
', [FunctionsCode, DispatchCode]).

mutual_functions_typescript([], _AllPreds, _MemoEnabled, []).
mutual_functions_typescript([Pred/Arity|Rest], AllPreds, MemoEnabled, [FuncCode|RestCodes]) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(mutual_recursion:is_mutual_recursive_clause(AllPreds), Clauses, RecClauses, BaseClauses),
    findall(BaseLine, (
        member(clause(BHead, true), BaseClauses),
        BHead =.. [_P, BValue],
        format(string(BaseLine), '  if (n === ~w) return true;', [BValue])
    ), BaseLines),
    atomic_list_concat(BaseLines, '\n', BaseCode),
    (   RecClauses = [clause(_RHead, RBody)|_] ->
        extract_mutual_rec_info_typescript(RBody, Guard, CalledPred, Step),
        atom_string(CalledPred, CalledStr),
        (   Guard = (N > Threshold), var(N) ->
            (   MemoEnabled = true ->
                format(string(RecCode),
'  if (n > ~w) {
    const key = "~w:" + n;
    if (mutualMemo.has(key)) return mutualMemo.get(key)!;
    const result = ~w(n ~w);
    mutualMemo.set(key, result);
    return result;
  }
  return false;', [Threshold, PredStr, CalledStr, Step])
            ;   format(string(RecCode),
'  return n > ~w ? ~w(n ~w) : false;', [Threshold, CalledStr, Step])
            )
        ;   format(string(RecCode), '  return ~w(n ~w);', [CalledStr, Step])
        )
    ;   RecCode = '  return false;'
    ),
    format(string(FuncCode),
'const ~w = (n: number): boolean => {
~w
~w
};', [PredStr, BaseCode, RecCode]),
    mutual_functions_typescript(Rest, AllPreds, MemoEnabled, RestCodes).

mutual_dispatch_typescript(Predicates, Code) :-
    findall(DispatchLine, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr),
        format(string(DispatchLine), '  if (func === "~w") console.log(~w(n));', [PredStr, PredStr])
    ), Lines),
    atomic_list_concat(Lines, '\n', Code).

extract_mutual_rec_info_typescript(Body, Guard, CalledPred, Step) :-
    extract_goals_typescript(Body, Goals),
    (   member(Guard, Goals), Guard = (_ > _) -> true
    ;   Guard = none
    ),
    member(Call, Goals),
    compound(Call),
    Call \= (_ is _), Call \= (_ > _), Call \= (_ < _),
    Call \= (_ >= _), Call \= (_ =< _),
    functor(Call, CalledPred, _),
    (   member(_ is _ - K, Goals), integer(K) ->
        format(string(Step), '- ~w', [K])
    ;   member(_ is _ + K, Goals), integer(K), K < 0 ->
        AbsK is abs(K),
        format(string(Step), '- ~w', [AbsK])
    ;   Step = "- 1"
    ).

extract_goals_typescript((A, B), Goals) :- !,
    extract_goals_typescript(A, GA),
    extract_goals_typescript(B, GB),
    append(GA, GB, Goals).
extract_goals_typescript(true, []) :- !.
extract_goals_typescript(Goal, [Goal]).

% ============================================================================
% GENERAL RECURSIVE PATTERN (visited-set cycle detection)
% ============================================================================

:- multifile advanced_recursive_compiler:compile_general_recursive_pattern/6.

%% Arity-2: wrapper + worker with base case check and recursive accumulation
advanced_recursive_compiler:compile_general_recursive_pattern(typescript, PredStr, 2, BaseClauses, RecClauses, Code) :-
    %% Build camelCase worker name
    atom_string(PredAtom, PredStr),
    atom_concat(PredAtom, 'Worker', WorkerAtom),
    atom_string(WorkerAtom, WorkerStr),
    %% Extract base case key/value from first base clause
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_, BaseKey, BaseVal],
        format(string(BaseCheck),
            '    if (arg1 === "~w") return ["~w"];', [BaseKey, BaseVal])
    ;   BaseCheck = '    // no base case extracted'
    ),
    %% Extract recursive step from first recursive clause
    (   RecClauses = [(_, RecBody)|_]
    ->  extract_rec_call_typescript(RecBody, PredStr, WorkerStr, RecCallExpr)
    ;   format(string(RecCallExpr), '~w(arg1, visited)', [WorkerStr])
    ),
    format(string(Code),
'// General recursive: ~w (with cycle detection)\n\c
function ~w(arg1: string): string[] {\n\c
    return ~w(arg1, new Set<string>());\n\c
}\n\c
\n\c
function ~w(arg1: string, visited: Set<string>): string[] {\n\c
    if (visited.has(arg1)) return [];\n\c
    visited.add(arg1);\n\c
~w\n\c
    const sub = ~w;\n\c
    return [...sub];\n\c
}\n',
    [PredStr, PredStr, WorkerStr, WorkerStr, BaseCheck, RecCallExpr]).

%% Arity-3: wrapper + worker with counter/output style
advanced_recursive_compiler:compile_general_recursive_pattern(typescript, PredStr, 3, BaseClauses, RecClauses, Code) :-
    atom_string(PredAtom, PredStr),
    atom_concat(PredAtom, 'Worker', WorkerAtom),
    atom_string(WorkerAtom, WorkerStr),
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_, BaseKey, _, BaseVal],
        format(string(BaseCheck),
            '    if (arg1 === "~w") return ["~w"];', [BaseKey, BaseVal])
    ;   BaseCheck = '    // no base case extracted'
    ),
    (   RecClauses = [(_, RecBody)|_]
    ->  extract_rec_call_typescript(RecBody, PredStr, WorkerStr, RecCallExpr)
    ;   format(string(RecCallExpr), '~w(arg1, visited)', [WorkerStr])
    ),
    format(string(Code),
'// General recursive: ~w (with cycle detection)\n\c
function ~w(arg1: string): string[] {\n\c
    return ~w(arg1, new Set<string>());\n\c
}\n\c
\n\c
function ~w(arg1: string, visited: Set<string>): string[] {\n\c
    if (visited.has(arg1)) return [];\n\c
    visited.add(arg1);\n\c
~w\n\c
    return ~w;\n\c
}\n',
    [PredStr, PredStr, WorkerStr, WorkerStr, BaseCheck, RecCallExpr]).

extract_rec_call_typescript((A, B), PredStr, WorkerStr, Expr) :-
    nonvar(A),
    functor(A, Pred, _),
    atom_string(Pred, PredStr), !,
    A =.. [_|CallArgs],
    (   CallArgs = [Arg1|_]
    ->  format(string(Expr), '~w(~w, visited)', [WorkerStr, Arg1])
    ;   format(string(Expr), '~w(arg1, visited)', [WorkerStr])
    ).
extract_rec_call_typescript((_, B), PredStr, WorkerStr, Expr) :- !,
    extract_rec_call_typescript(B, PredStr, WorkerStr, Expr).
extract_rec_call_typescript(Goal, PredStr, WorkerStr, Expr) :-
    nonvar(Goal),
    functor(Goal, Pred, _),
    atom_string(Pred, PredStr), !,
    Goal =.. [_|CallArgs],
    (   CallArgs = [Arg1|_]
    ->  format(string(Expr), '~w(~w, visited)', [WorkerStr, Arg1])
    ;   format(string(Expr), '~w(arg1, visited)', [WorkerStr])
    ).
extract_rec_call_typescript(_, _PredStr, WorkerStr, Expr) :-
    format(string(Expr), '~w(arg1, visited)', [WorkerStr]).
