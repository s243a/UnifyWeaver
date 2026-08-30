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
    compile_collected_components/1, % Compile all collected components to code

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
% custom_chart self-registers its component type via :- initialization(now);
% loading it here (empty import list) triggers that registration so the
% Chart.js component type is available to declare_component/4 (G-P4).
:- use_module('typescript_runtime/custom_chart', []).
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

%% compile_collected_components(-Code)
%
%  Generate TypeScript code for all collected components by delegating to
%  component_registry:compile_component/4 for each. Mirrors python_target's
%  emit loop (python_target.pl:~195). Returns '' when no components were
%  collected, so component-free modules are unchanged (G-P4).
%
compile_collected_components(Code) :-
    findall(CompCode, (
        collected_component(Category, Name),
        component_registry:compile_component(Category, Name, [], CompCode)
    ), CompCodes),
    (   CompCodes = []
    ->  Code = ''
    ;   atomic_list_concat(CompCodes, '\n\n', Code)
    ).

%% ============================================
%% MAIN DISPATCH
%% ============================================

compile_predicate(Pred/Arity, Options, Code) :-
    compile_predicate_to_typescript(Pred/Arity, Options, Code).

% Streaming / generator emit mode (G-P8). When a streaming option is present
% (`mode(generator)` / `mode(pipeline)`, or the clojure-style aliases
% `generator_mode(true)` / `pipeline_input(true)`) and the predicate is a
% recognised transform/filter shape, emit a TS program that reads stdin line by
% line (Node's built-in `readline`, no npm dependency), applies the predicate,
% and streams results to stdout. Tried first so streaming wins over the batch
% paths; if the shape does NOT qualify, compile_streaming_typescript/3 fails and
% we fall through to the normal (batch) clauses below — so predicates compiled
% WITHOUT a streaming option are byte-for-byte unchanged.
compile_predicate_to_typescript(Pred/Arity, Options, Code) :-
    ts_streaming_option(Options, Mode),
    compile_streaming_typescript(Pred/Arity, Mode, Code),
    !.

% Aggregate compilation (G-P3): recognise aggregate_all/3, aggregate/3 and
% findall/3 goals in a clause body and lower them into a self-contained TS
% reducer over the solution set of the inner goal. Tried first so aggregate
% bodies are handled specially rather than falling through to the generic
% native clause path (which cannot express fold-over-solutions).
compile_predicate_to_typescript(Pred/Arity, _Options, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    ts_aggregate_predicate(Pred/Arity, Clauses, Code),
    !.

% Structural (list) recursion lowering — derive real TS from list-shaped
% clauses (member/append/reverse/list-length etc.). This is DERIVED from the
% actual clause heads/bodies, not a canned template (G-P2). Tried before the
% numeric native path; it only fires for genuine list-recursive predicates.
compile_predicate_to_typescript(Pred/Arity, _Options, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses \= [],
    native_ts_structural(Pred/Arity, Clauses, Code),
    !.

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
%% STREAMING / GENERATOR EMIT MODE (G-P8)
%% ============================================
%%
%% Emits a self-contained TypeScript CLI that reads input incrementally from
%% stdin (one record per line) using Node's built-in `readline`, applies the
%% predicate to each record, and streams matching/derived results to stdout —
%% instead of the batch/all-at-once form. Runs on stock node
%% (`node --experimental-strip-types`, node >= 22) with no npm dependency.
%%
%% Option shapes (the first that matches wins):
%%   mode(generator) / generator_mode(true)  -> generator mode
%%   mode(pipeline)  / pipeline_input(true)  -> pipeline mode
%% (`mode/1` is the primary spelling; the `*_mode`/`*_input` aliases match the
%%  option names clojure_target already uses so a pipeline spec can target
%%  either family with the same options.)
%%
%% Qualifying shapes (single-clause; anything else FALLS BACK to batch):
%%   - Filter  pred(X)    :- Guard1, Guard2, ...     (0+ comparison guards)
%%       generator: emits the numeric value when all guards hold.
%%       pipeline : passes the original input line through when all guards hold.
%%   - Transform pred(X, Y) :- [Guards,] Y is Expr   (or Y = Expr)
%%       both modes: emits String(Expr) for each X for which the guards hold
%%       (a guard failure drops the record — generator yields nothing / pipeline
%%        filters it out).
%% Numeric records are the target (input parsed with Number()); non-numeric or
%% multi-clause / nondeterministic predicates deliberately fall back to batch.

%% ts_streaming_option(+Options, -Mode)
%  Extract the streaming mode from the options, or fail if none is present.
ts_streaming_option(Options, Mode) :-
    (   option(mode(M), Options), memberchk(M, [generator, pipeline])
    ->  Mode = M
    ;   option(generator_mode(true), Options)
    ->  Mode = generator
    ;   option(pipeline_input(true), Options)
    ->  Mode = pipeline
    ;   fail
    ).

%% compile_streaming_typescript(+Pred/Arity, +Mode, -Code)
%  Fails (→ batch fallback) unless the predicate is a single-clause filter
%  (arity 1) or transform (arity 2) as described above.
compile_streaming_typescript(Pred/Arity, Mode, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    Clauses = [SingleHead-SingleBody],
    atom_string(Pred, PredStr),
    (   Arity =:= 1
    ->  ts_streaming_filter(SingleHead, SingleBody, TestExpr),
        ts_streaming_filter_module(PredStr, Mode, TestExpr, Code)
    ;   Arity =:= 2
    ->  ts_streaming_transform(SingleHead, SingleBody, GuardExpr, RetExpr),
        ts_streaming_transform_module(PredStr, Mode, GuardExpr, RetExpr, Code)
    ;   fail
    ).

%% ts_streaming_filter(+Head, +Body, -TestExpr)
%  Arity-1 filter: the body must be a (possibly empty) conjunction of comparison
%  guards over the single input variable. TestExpr is the combined TS boolean.
ts_streaming_filter(Head, Body, TestExpr) :-
    Head =.. [_, In],
    var(In),
    VM = [In-"x"],
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  TestExpr = "true"
    ;   maplist(ts_guard_condition(VM), Goals, Conds),
        atomic_list_concat(Conds, ' && ', TestExpr)
    ).

%% ts_streaming_transform(+Head, +Body, -GuardExpr, -RetExpr)
%  Arity-2 transform: body = guards + exactly one output goal binding the head's
%  second (output) argument via `is`/`=`. GuardExpr is `none` or `guard(TS)`.
ts_streaming_transform(Head, Body, GuardExpr, RetExpr) :-
    Head =.. [_, In, Out],
    var(In), var(Out),
    VM = [In-"x"],
    normalize_goals(Body, Goals),
    clause_guard_output_split(Goals, VM, Guards, Outputs),
    Outputs = [OutGoal],
    goal_output_var(OutGoal, OV), OV == Out,
    ts_output_goal_last(OutGoal, VM, RetExpr),
    (   Guards == []
    ->  GuardExpr = none
    ;   maplist(ts_guard_condition(VM), Guards, Conds),
        atomic_list_concat(Conds, ' && ', GStr),
        GuardExpr = guard(GStr)
    ).

%% ts_streaming_emit(+Mode, -EmitExpr)
%  What a passing filter record writes to stdout: the derived value (generator)
%  or the untouched input line (pipeline, record pass-through).
ts_streaming_emit(generator, "String(x)").
ts_streaming_emit(pipeline, "trimmed").

%% ts_mode_label(+Mode, -Label)
ts_mode_label(generator, "generator").
ts_mode_label(pipeline, "pipeline").

%% ts_streaming_filter_module(+PredStr, +Mode, +TestExpr, -Code)
ts_streaming_filter_module(PredStr, Mode, TestExpr, Code) :-
    ts_mode_label(Mode, Label),
    ts_streaming_emit(Mode, EmitExpr),
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Streaming (~w mode)
// Predicate: ~w/1  (filter over an input stream)

import { createInterface } from "node:readline";

export function ~wTest(x: number): boolean {
  return (~w);
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (trimmed.length === 0) return;
  const x = Number(trimmed);
  if (~wTest(x)) {
    console.log(~w);
  }
});
', [Label, PredStr, PredStr, TestExpr, PredStr, EmitExpr]).

%% ts_streaming_transform_module(+PredStr, +Mode, +GuardExpr, +RetExpr, -Code)
ts_streaming_transform_module(PredStr, Mode, GuardExpr, RetExpr, Code) :-
    ts_mode_label(Mode, Label),
    (   GuardExpr = guard(GStr)
    ->  format(string(GuardLine), '  if (!(~w)) return [];\n', [GStr])
    ;   GuardLine = ""
    ),
    % Transform yields an array of 0+ results (empty when a guard fails), so the
    % same shape covers generator (mapcat/flatMap) and pipeline (keep/drop)
    % semantics. `: number[]` also survives the vanilla_js type-strip and the
    % annotated_js JSDoc rewrite, so streaming flows cleanly by inheritance.
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Streaming (~w mode)
// Predicate: ~w/2  (transform over an input stream)

import { createInterface } from "node:readline";

export function ~wTransform(x: number): number[] {
~w  return [~w];
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (trimmed.length === 0) return;
  const x = Number(trimmed);
  for (const result of ~wTransform(x)) {
    console.log(String(result));
  }
});
', [Label, PredStr, PredStr, GuardLine, RetExpr, PredStr]).

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
    generate_match_expr(FieldNames, MatchExpr),

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
    TypeName, FieldsStr, PredStr, MatchExpr]).

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
    ;   Pattern == transitive_closure
    ->  generate_transitive_closure(PredStr, Code)
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

%% generate_transitive_closure(+Name, -Code)
%
%  Emit real transitive-closure logic (breadth-first reachability over a base
%  relation), matching the BFS algorithm used by the shared TypeScript TC
%  template (templates/targets/typescript/transitive_closure.mustache).
%
%  Produces two exported functions parameterized by the base relation edges:
%    - <Name>Closure(edges, start): all nodes reachable from start
%    - <Name>(edges, start, target): whether target is reachable from start
%
generate_transitive_closure(Name, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Pattern: transitive_closure

const build~wRelation = (edges: [string, string][]): Map<string, string[]> => {
  const baseRelation = new Map<string, string[]>();
  for (const [from, to] of edges) {
    if (!baseRelation.has(from)) baseRelation.set(from, []);
    baseRelation.get(from)!.push(to);
  }
  return baseRelation;
};

// All nodes reachable from `start` via the transitive closure of the base relation.
export const ~wClosure = (edges: [string, string][], start: string): string[] => {
  const baseRelation = build~wRelation(edges);
  const results: string[] = [];
  const visited = new Set<string>([start]);
  const queue: string[] = [start];
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const next of baseRelation.get(current) || []) {
      if (!visited.has(next)) {
        visited.add(next);
        queue.push(next);
        results.push(next);
      }
    }
  }
  return results;
};

// True iff `target` is reachable from `start` (excluding the trivial start === target).
export const ~w = (edges: [string, string][], start: string, target: string): boolean => {
  if (start === target) return false;
  const baseRelation = build~wRelation(edges);
  const visited = new Set<string>([start]);
  const queue: string[] = [start];
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const next of baseRelation.get(current) || []) {
      if (next === target) return true;
      if (!visited.has(next)) {
        visited.add(next);
        queue.push(next);
      }
    }
  }
  return false;
};
', [Name, Name, Name, Name, Name]).

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

    % Emit any declared components (G-P4). compile_collected_components/1
    % yields '' when none were collected, keeping component-free modules
    % byte-for-byte unchanged.
    compile_collected_components(ComponentsCode),
    (   ComponentsCode == ''
    ->  Body = PredsSection
    ;   format(string(Body), '~w\n\n~w', [PredsSection, ComponentsCode])
    ),

    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target
// Module: ~w

~w
', [ModName, Body]).

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
    length(Args, L),
    generate_field_names(L, FieldNames),
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

generate_interface_fields(FieldNames, Fields) :-
    maplist([F, Line]>>format(string(Line), '  ~w: string;', [F]), FieldNames, Lines),
    atomic_list_concat(Lines, '\n', Fields).

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
%% AGGREGATE COMPILATION (G-P3)
%% ============================================
%%
%% Compiles aggregate GOALS in a clause body into self-contained TypeScript
%% reducers over the solution set of the inner goal. Supported templates:
%%
%%   aggregate_all(count,      Goal, N)   -> number
%%   aggregate_all(sum(Expr),  Goal, S)   -> number
%%   aggregate_all(max(Expr),  Goal, M)   -> number
%%   aggregate_all(min(Expr),  Goal, M)   -> number
%%   aggregate_all(bag(Tmpl),  Goal, L)   -> any[]
%%   aggregate_all(set(Tmpl),  Goal, L)   -> any[] (sorted, deduped)
%%   aggregate(...)                       -> normalised to aggregate_all
%%   findall(Tmpl, Goal, L)               -> any[]
%%
%% The inner Goal is a single extensional relation goal, optionally followed by
%% arithmetic (`V is Expr`) computations and comparison guards. Bound head
%% inputs act as group/filter keys (the ordinary free-variable grouping case).
%% The inner relation's facts are embedded as a function-local const so the
%% emitted function is fully self-contained and runs on stock node.

%% ts_aggregate_predicate(+Pred/Arity, +Clauses, -Code)
%  Succeeds only when the predicate is a single clause whose body is an
%  aggregate/findall goal binding the head's output argument.
ts_aggregate_predicate(Pred/_Arity, [Head-Body], Code) :-
    ts_extract_aggregate(Body, agg(Op, Template, InnerGoal, Result)),
    Head =.. [_|HeadArgs],
    append(InputArgs, [OutArg], HeadArgs),
    var(OutArg),
    Result == OutArg,
    ts_aggregate_code(Pred, InputArgs, Op, Template, InnerGoal, Code).

%% ts_extract_aggregate(+Body, -Agg)
%  Body must be exactly one aggregate/findall goal.
ts_extract_aggregate(Body, Agg) :-
    normalize_goals(Body, [Goal]),
    ts_normalize_aggregate(Goal, Agg).

ts_normalize_aggregate(_Module:Goal, Agg) :-
    !, ts_normalize_aggregate(Goal, Agg).
ts_normalize_aggregate(aggregate_all(Template, Goal, Result),
                       agg(Op, VTemplate, Goal, Result)) :-
    ts_agg_template(Template, Op, VTemplate).
ts_normalize_aggregate(aggregate(Template, Goal, Result), Agg) :-
    ts_normalize_aggregate(aggregate_all(Template, Goal, Result), Agg).
ts_normalize_aggregate(findall(Template, Goal, Result),
                       agg(findall, Template, Goal, Result)).

%% ts_agg_template(+Template, -Op, -ValueExpr)
ts_agg_template(count,  count, 1) :- !.
ts_agg_template(sum(E), sum,   E) :- !.
ts_agg_template(max(E), max,   E) :- !.
ts_agg_template(min(E), min,   E) :- !.
ts_agg_template(bag(E), bag,   E) :- !.
ts_agg_template(set(E), set,   E) :- !.

%% ts_aggregate_code(+Pred, +InputArgs, +Op, +Template, +InnerGoal, -Code)
ts_aggregate_code(Pred, InputArgs, Op, Template, InnerGoal, Code) :-
    atom_string(Pred, PredStr),
    normalize_goals(InnerGoal, InnerGoals),
    InnerGoals = [RelGoal0|PostGoals],
    ts_strip_module(RelGoal0, RelGoal),
    RelGoal =.. [RelPred|RelArgs],
    length(RelArgs, RelArity),
    % Embed the inner relation's facts as a local const array.
    ts_agg_fact_array(RelPred, RelArity, FactsInner),
    ts_agg_field_type(RelArity, FieldType),
    % Bound inputs -> equality filters; free vars -> value bindings.
    ts_inner_bindings(RelArgs, InputArgs, VarMap0, KeyFilters),
    % Post goals (arithmetic + guards) become in-loop statements.
    ts_agg_post_stmts(PostGoals, VarMap0, PostStmts, VarMap1),
    % Bound-key filters render as continue-guards at the top of the loop.
    findall(FLine,
            ( member(Cond, KeyFilters),
              format(string(FLine), '        if (!(~w)) continue;', [Cond]) ),
            KeyLines),
    append(KeyLines, PostStmts, LoopHead),
    % Fold fragments for this aggregate operator.
    ts_agg_fold(Op, Template, VarMap1, RetType, InitLine, UpdateLine, FinishLine),
    atomic_list_concat(LoopHead, '\n', LoopHeadStr),
    ts_agg_params(InputArgs, ParamStr),
    ( LoopHeadStr == ''
    ->  format(string(LoopBody), '~w', [UpdateLine])
    ;   format(string(LoopBody), '~w\n~w', [LoopHeadStr, UpdateLine])
    ),
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Aggregate Lowering (G-P3)
// Predicate: ~w  (~w)

export function ~w(~w): ~w {
    const facts: ~w[] = [~w];
~w
    for (const f of facts) {
~w
    }
~w
}

// CLI entry point
console.log(JSON.stringify(~w(...process.argv.slice(2))));
', [PredStr, Op, PredStr, ParamStr, RetType, FieldType, FactsInner,
    InitLine, LoopBody, FinishLine, PredStr]).

%% ts_strip_module(+Goal, -Goal)
ts_strip_module(_Module:Goal, Goal) :- !.
ts_strip_module(Goal, Goal).

%% ts_agg_fact_array(+RelPred, +RelArity, -FactsInner)
%  Collect the inner relation's extensional facts from `user` as a TS array
%  literal body (comma-separated object literals). Empty when no facts exist.
ts_agg_fact_array(RelPred, RelArity, FactsInner) :-
    functor(RelTemplate, RelPred, RelArity),
    findall(Tuple,
            ( user:clause(RelTemplate, true),
              RelTemplate =.. [_|FArgs],
              format_ts_tuple(FArgs, Tuple) ),
            Tuples),
    atomic_list_concat(Tuples, ', ', FactsInner).

%% ts_agg_field_type(+Arity, -TypeStr) — object type for a fact row
ts_agg_field_type(Arity, TypeStr) :-
    generate_field_names(Arity, Names),
    maplist([N, F]>>format(string(F), '~w: any', [N]), Names, Fields),
    atomic_list_concat(Fields, ', ', Inner),
    format(string(TypeStr), '{ ~w }', [Inner]).

%% ts_agg_params(+InputArgs, -ParamStr) — function parameter list
ts_agg_params(InputArgs, ParamStr) :-
    length(InputArgs, N),
    findall(P, ( between(1, N, I), format(string(P), 'arg~w: any', [I]) ), Params),
    atomic_list_concat(Params, ', ', ParamStr).

%% ts_inner_bindings(+RelArgs, +InputArgs, -VarMap, -KeyFilters)
%  Map each inner relation argument: bound head inputs and constants become
%  equality filters; free variables are bound to their `f.argN` field ref so
%  post-goals and the value template can reference them.
ts_inner_bindings(RelArgs, InputArgs, VarMap, KeyFilters) :-
    ts_inner_bindings_(RelArgs, 1, InputArgs, [], VarMap, [], KeyFilters).

ts_inner_bindings_([], _, _, VM, VM, KF, KF).
ts_inner_bindings_([A|As], I, InputArgs, VM0, VM, KF0, KF) :-
    (   var(A)
    ->  format(string(FieldRef), 'f.arg~w', [I]),
        VM1 = [A-FieldRef|VM0],
        (   ts_input_index(A, InputArgs, K)
        ->  format(string(Cond), 'String(f.arg~w) === String(arg~w)', [I, K]),
            KF1 = [Cond|KF0]
        ;   KF1 = KF0
        )
    ;   VM1 = VM0,
        format_ts_arg(A, Lit),
        format(string(Cond), 'String(f.arg~w) === String(~w)', [I, Lit]),
        KF1 = [Cond|KF0]
    ),
    I1 is I + 1,
    ts_inner_bindings_(As, I1, InputArgs, VM1, VM, KF1, KF).

%% ts_input_index(+Var, +InputArgs, -Index) — position of Var among head inputs
ts_input_index(Var, InputArgs, K) :-
    nth1(K, InputArgs, A), A == Var, !.

%% ts_agg_post_stmts(+PostGoals, +VarMap0, -Stmts, -VarMapOut)
%  Lower `V is Expr` to a const line and comparison guards to continue-guards,
%  preserving order and threading the VarMap.
ts_agg_post_stmts([], VM, [], VM).
ts_agg_post_stmts([G0|Gs], VM0, [Stmt|Rest], VM) :-
    ts_strip_module(G0, G),
    (   G = is(V, Expr), var(V)
    ->  ensure_var(VM0, V, Name, VM1),
        ts_expr(Expr, VM0, TExpr),
        format(string(Stmt), '        const ~w = ~w;', [Name, TExpr])
    ;   ts_guard_condition(VM0, G, Cond),
        VM1 = VM0,
        format(string(Stmt), '        if (!(~w)) continue;', [Cond])
    ),
    ts_agg_post_stmts(Gs, VM1, Rest, VM).

%% ts_agg_fold(+Op, +Template, +VarMap, -RetType, -Init, -Update, -Finish)
ts_agg_fold(count, _Template, _VM, "number",
            "    let acc = 0;",
            "        acc += 1;",
            "    return acc;") :- !.
ts_agg_fold(sum, Template, VM, "number",
            "    let acc = 0;",
            Update,
            "    return acc;") :- !,
    ts_expr(Template, VM, ValExpr),
    format(string(Update), '        acc += Number(~w);', [ValExpr]).
ts_agg_fold(max, Template, VM, "number",
            "    let acc: number | undefined = undefined;",
            Update,
            "    if (acc === undefined) throw new Error(\"aggregate_all(max): no solutions\");\n    return acc;") :- !,
    ts_expr(Template, VM, ValExpr),
    format(string(Update),
'        { const _v = Number(~w); if (acc === undefined || _v > acc) acc = _v; }',
           [ValExpr]).
ts_agg_fold(min, Template, VM, "number",
            "    let acc: number | undefined = undefined;",
            Update,
            "    if (acc === undefined) throw new Error(\"aggregate_all(min): no solutions\");\n    return acc;") :- !,
    ts_expr(Template, VM, ValExpr),
    format(string(Update),
'        { const _v = Number(~w); if (acc === undefined || _v < acc) acc = _v; }',
           [ValExpr]).
ts_agg_fold(bag, Template, VM, "any[]",
            "    const acc: any[] = [];",
            Update,
            "    return acc;") :- !,
    ts_agg_value_expr(Template, VM, ValExpr),
    format(string(Update), '        acc.push(~w);', [ValExpr]).
ts_agg_fold(findall, Template, VM, "any[]",
            "    const acc: any[] = [];",
            Update,
            "    return acc;") :- !,
    ts_agg_value_expr(Template, VM, ValExpr),
    format(string(Update), '        acc.push(~w);', [ValExpr]).
ts_agg_fold(set, Template, VM, "any[]",
            "    const acc: any[] = [];",
            Update,
            "    acc.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));\n    return acc;") :- !,
    ts_agg_value_expr(Template, VM, ValExpr),
    format(string(Update),
'        { const _v = ~w; if (!acc.some(x => x === _v)) acc.push(_v); }',
           [ValExpr]).

%% ts_agg_value_expr(+Template, +VarMap, -Expr)
%  Resolve a collection template (bag/set/findall) to a TS value expression.
%  A bound variable resolves to its field ref; a constant to a literal.
ts_agg_value_expr(Template, VM, Expr) :-
    (   var(Template), lookup_var(Template, VM, Expr)
    ->  true
    ;   ts_expr(Template, VM, Expr)
    ).

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
        % Arity-2 tail/accumulator over a list: fold the derived step
        % operation across the elements (NOT a canned `items.length` stub).
        format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Tail Recursion (arity-2 accumulator loop, derived)
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
% DERIVED NUMERIC MULTI-CALL RECURSION (G-P1)
%
% Shared by the tree / multicall / direct-multicall hooks. Instead of emitting
% a hardcoded `<pred>(n-1) + <pred>(n-2)` Fibonacci body for every predicate,
% this derives the base cases, recursive-call argument offsets and the
% aggregation expression FROM THE ACTUAL CLAUSE BODY. When no recursive clause
% is available to derive from (e.g. a bare dispatch-shape call with no user
% clauses), it falls back to the canned Fibonacci body so shape-only callers
% still get valid, memoized code.
% ============================================================================

%% ts_is_rec_clause(+Pred, +ClauseTerm)
%  True if the clause body calls Pred (any arity).
ts_is_rec_clause(Pred, clause(_, Body)) :- !, ts_body_calls(Body, Pred).
ts_is_rec_clause(Pred, _-Body)          :- !, ts_body_calls(Body, Pred).
ts_is_rec_clause(Pred, (_ :- Body))     :- !, ts_body_calls(Body, Pred).

%% ts_body_calls(+Body, +Pred) — Pred appears as a goal functor in Body.
ts_body_calls((A, B), P) :- !, ( ts_body_calls(A, P) ; ts_body_calls(B, P) ).
ts_body_calls((A ; B), P) :- !, ( ts_body_calls(A, P) ; ts_body_calls(B, P) ).
ts_body_calls((A -> B), P) :- !, ( ts_body_calls(A, P) ; ts_body_calls(B, P) ).
ts_body_calls(_M:G, P) :- !, ts_body_calls(G, P).
ts_body_calls(G, P) :- compound(G), functor(G, P, _).

%% ts_clause_hb(+ClauseTerm, -Head, -Body)
ts_clause_hb(clause(H, B), H, B) :- !.
ts_clause_hb((H :- B), H, B) :- !.
ts_clause_hb(H-B, H, B) :- !.
ts_clause_hb((H, B), H, B) :- callable(H), !.
ts_clause_hb(H, H, true).

%% ts_numeric_recursion_or_fallback(+Pred, +PredStr, +Label, +BaseClauses, +RecClauses, -Code)
ts_numeric_recursion_or_fallback(Pred, PredStr, Label, BaseClauses, RecClauses, Code) :-
    (   RecClauses \= [],
        ts_derive_numeric_recursion(Pred, BaseClauses, RecClauses, BaseCaseStr, ResultExpr)
    ->  true
    ;   ts_fallback_basecases(BaseClauses, BaseCaseStr),
        format(string(ResultExpr), '~w(n - 1) + ~w(n - 2)', [PredStr, PredStr])
    ),
    ts_numeric_recursion_code(PredStr, Label, BaseCaseStr, ResultExpr, Code).

%% ts_fallback_basecases(+BaseClauses, -BaseCaseStr)
%  Derive base-case checks from base clauses if numeric, else the fib default.
ts_fallback_basecases(BaseClauses, BaseCaseStr) :-
    findall(Line, (
        member(BC, BaseClauses), ts_clause_hb(BC, BHead, _),
        BHead =.. [_, BIn, BOut], number(BIn), number(BOut),
        format(string(Line), '  if (n === ~w) return ~w;', [BIn, BOut])
    ), Lines0),
    sort(Lines0, Lines),
    (   Lines == []
    ->  BaseCaseStr = "  if (n <= 0) return 0;\n  if (n === 1) return 1;"
    ;   atomic_list_concat(Lines, '\n', BaseCaseStr)
    ).

%% ts_numeric_recursion_code(+PredStr, +Label, +BaseCaseStr, +ResultExpr, -Code)
ts_numeric_recursion_code(PredStr, Label, BaseCaseStr, ResultExpr, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - ~w (derived from clause body)

const ~wMemo = new Map<number, number>();

const ~w = (n: number): number => {
  if (~wMemo.has(n)) return ~wMemo.get(n)!;
~w
  const result = ~w;
  ~wMemo.set(n, result);
  return result;
};

if (process.argv[2]) {
  console.log(~w(parseInt(process.argv[2])));
}
', [Label, PredStr, PredStr, PredStr, PredStr, BaseCaseStr, ResultExpr, PredStr, PredStr]).

%% ts_derive_numeric_recursion(+Pred, +BaseClauses, +RecClauses, -BaseCaseStr, -ResultExpr)
%  Derive base cases and the recursive result expression from the real clauses.
ts_derive_numeric_recursion(Pred, BaseClauses, RecClauses, BaseCaseStr, ResultExpr) :-
    RecClauses = [RC0|_],
    ts_clause_hb(RC0, RHead, RBody),
    RHead =.. [_, InputVar, _OutVar],
    var(InputVar),                       % numeric/scalar input (not a structure)
    % Base cases (numeric input -> numeric output)
    findall(BLine, (
        member(BC, BaseClauses), ts_clause_hb(BC, BHead, _),
        BHead =.. [_, BIn, BOut], number(BIn), number(BOut),
        format(string(BLine), '  if (n === ~w) return ~w;', [BIn, BOut])
    ), BaseLines0),
    sort(BaseLines0, BaseLines),
    BaseLines \= [],
    atomic_list_concat(BaseLines, '\n', BaseCaseStr),
    % Decompose the recursive body
    ts_body_components(RBody, Pred, Computations, RecCalls, Aggregation),
    RecCalls = [_|_],
    ts_build_callmap(RecCalls, Pred, Computations, InputVar, CallMap),
    (   Aggregation = (_ is AggExpr)
    ->  ts_num_agg(AggExpr, CallMap, InputVar, ResultExpr)
    ;   RecCalls = [Call1|_], Call1 =.. [_, _, OutV1],
        ts_lookup_callmap(OutV1, CallMap, ResultExpr)
    ).

%% ts_body_components(+Body, +Pred, -Computations, -RecCalls, -Aggregation)
%  Split a recursive-clause body into is/2 computations, recursive calls, and
%  the aggregation goal (the `_ is Expr` whose Expr uses a recursive output).
ts_body_components(Body, Pred, Comps, RecCalls, Aggregation) :-
    ts_extract_goals(Body, Goals0),
    maplist(ts_strip_mod, Goals0, Goals),
    partition(ts_is_rec_call(Pred), Goals, RecCalls, NonRec),
    include(ts_is_is_goal, NonRec, IsGoals),
    (   select(AggG, IsGoals, Comps0),
        AggG = (_ is AExpr),
        term_variables(AExpr, EVs),
        member(EV, EVs),
        member(Call, RecCalls), Call =.. [_|CA], last(CA, OV), EV == OV
    ->  Aggregation = AggG, Comps = Comps0
    ;   Comps = IsGoals, Aggregation = none
    ).

ts_extract_goals((A, B), Gs) :- !, ts_extract_goals(A, GA), ts_extract_goals(B, GB), append(GA, GB, Gs).
ts_extract_goals(true, []) :- !.
ts_extract_goals(_M:G, Gs) :- !, ts_extract_goals(G, Gs).
ts_extract_goals(G, [G]).

ts_strip_mod(_M:G, G) :- !.
ts_strip_mod(G, G).

ts_is_rec_call(Pred, G) :- compound(G), functor(G, Pred, _).
ts_is_is_goal(_ is _).

%% ts_build_callmap(+RecCalls, +Pred, +Computations, +InputVar, -CallMap)
%  Map each recursive call's output variable to its TS call expression, with
%  the input argument derived from the is/2 computation feeding that call.
ts_build_callmap([], _, _, _, []).
ts_build_callmap([Call|Rest], Pred, Comps, InVar, [OutV-CE|Map]) :-
    Call =.. [_, CallIn, OutV],
    (   CallIn == InVar
    ->  ArgStr = "n"
    ;   ts_find_comp(CallIn, Comps, Expr)
    ->  ts_num_offset(Expr, InVar, ArgStr)
    ;   ArgStr = "n"
    ),
    atom_string(Pred, PredStr),
    format(string(CE), "~w(~w)", [PredStr, ArgStr]),
    ts_build_callmap(Rest, Pred, Comps, InVar, Map).

ts_find_comp(V, [(V0 is E)|_], E) :- V0 == V, !.
ts_find_comp(V, [_|T], E) :- ts_find_comp(V, T, E).

ts_lookup_callmap(V, [V0-CE|_], CE) :- V0 == V, !.
ts_lookup_callmap(V, [_|T], CE) :- ts_lookup_callmap(V, T, CE).

%% ts_num_offset(+Expr, +InputVar, -TsExpr) — arithmetic on the scalar input.
ts_num_offset(V, IV, "n") :- var(V), V == IV, !.
ts_num_offset(V, _, S) :- var(V), !, term_string(V, S).
ts_num_offset(N, _, S) :- number(N), !, format(string(S), '~w', [N]).
ts_num_offset(A+K, IV, S) :- number(K), K < 0, !, ts_num_offset(A, IV, SA), K1 is -K, format(string(S), '~w - ~w', [SA, K1]).
ts_num_offset(A+B, IV, S) :- !, ts_num_offset(A, IV, SA), ts_num_offset(B, IV, SB), format(string(S), '~w + ~w', [SA, SB]).
ts_num_offset(A-B, IV, S) :- !, ts_num_offset(A, IV, SA), ts_num_offset(B, IV, SB), format(string(S), '~w - ~w', [SA, SB]).
ts_num_offset(A*B, IV, S) :- !, ts_num_offset(A, IV, SA), ts_num_offset(B, IV, SB), format(string(S), '~w * ~w', [SA, SB]).
ts_num_offset(A/B, IV, S) :- !, ts_num_offset(A, IV, SA), ts_num_offset(B, IV, SB), format(string(S), '~w / ~w', [SA, SB]).

%% ts_num_agg(+Expr, +CallMap, +InputVar, -TsExpr)
%  Translate the aggregation expression: recursive output vars become their
%  call expressions, the input var becomes `n`, operators map to JS.
ts_num_agg(V, Map, _, CE) :- var(V), ts_lookup_callmap(V, Map, CE), !.
ts_num_agg(V, _, IV, "n") :- var(V), V == IV, !.
ts_num_agg(V, _, _, S) :- var(V), !, term_string(V, S).
ts_num_agg(N, _, _, S) :- number(N), !, format(string(S), '~w', [N]).
ts_num_agg(A+B, M, IV, S) :- !, ts_num_agg(A, M, IV, SA), ts_num_agg(B, M, IV, SB), format(string(S), '~w + ~w', [SA, SB]).
ts_num_agg(A-B, M, IV, S) :- !, ts_num_agg(A, M, IV, SA), ts_num_agg(B, M, IV, SB), format(string(S), '~w - ~w', [SA, SB]).
ts_num_agg(A*B, M, IV, S) :- !, ts_num_agg(A, M, IV, SA), ts_num_agg(B, M, IV, SB), format(string(S), '~w * ~w', [SA, SB]).
ts_num_agg(A/B, M, IV, S) :- !, ts_num_agg(A, M, IV, SA), ts_num_agg(B, M, IV, SB), format(string(S), '~w / ~w', [SA, SB]).

% ============================================================================
% STRUCTURAL (LIST) RECURSION LOWERING (G-P2)
%
% Compiles list-shaped predicates (member/2, append/3, list-length,
% reverse-accumulator, etc.) into real recursive TypeScript functions over JS
% arrays, DERIVED from the actual clause heads and bodies. Deterministic
% functional shapes are emitted as a first-match if-chain; pure tests (no
% output argument) return a boolean.
% ============================================================================

%% native_ts_structural(+Pred/Arity, +Clauses, -Code)
native_ts_structural(Pred/Arity, Clauses, Code) :-
    ts_struct_detect(Pred, Clauses, Arity, _DPos, Mode),
    atom_string(Pred, PredStr),
    ts_struct_inputs(Arity, Mode, InPositions),
    maplist(ts_struct_param(Clauses), InPositions, ParamDecls),
    atomic_list_concat(ParamDecls, ', ', ParamList),
    ( Mode = function(_) -> RetType = "any" ; RetType = "boolean" ),
    maplist(ts_struct_clause(Pred, Arity, Mode), Clauses, Blocks),
    atomic_list_concat(Blocks, '\n', BodyBlocks),
    (   Mode = function(_)
    ->  format(string(Default), '  throw new Error("no matching clause for ~w/~w");', [PredStr, Arity])
    ;   Default = "  return false;"
    ),
    format(string(Code),
'// Generated by UnifyWeaver TypeScript Target - Structural Recursion (native clause lowering)
// Predicate: ~w/~w

export function ~w(~w): ~w {
~w
~w
}
', [PredStr, Arity, PredStr, ParamList, RetType, BodyBlocks, Default]).

%% ts_struct_detect(+Pred, +Clauses, +Arity, -DPos, -Mode)
ts_struct_detect(Pred, Clauses, Arity, DPos, Mode) :-
    % At least one clause recurses on Pred (rules out plain facts).
    once(( member(_-Body, Clauses), ts_body_calls(Body, Pred) )),
    % Decomposition position: a cons pattern at some position of a rec-clause head.
    (   member(RH-RBody, Clauses),
        ts_body_calls(RBody, Pred),
        between(1, Arity, DPos),
        arg(DPos, RH, RA), ts_is_cons(RA)
    ->  true
    ),
    % Every clause head must carry a list-compatible term at DPos.
    forall(member(H-_, Clauses), ( arg(DPos, H, DA), ts_list_pos_ok(DA) )),
    ( DPos =:= Arity -> Mode = test ; Mode = function(Arity) ).

ts_is_cons(T) :- nonvar(T), T = [_|_].
ts_list_pos_ok(T) :- var(T), !.
ts_list_pos_ok([]) :- !.
ts_list_pos_ok(T) :- nonvar(T), T = [_|_].
ts_islisty(T) :- nonvar(T), ( T = [] ; T = [_|_] ).

%% ts_struct_inputs(+Arity, +Mode, -InputPositions)
ts_struct_inputs(Arity, test, Ps) :- numlist(1, Arity, Ps).
ts_struct_inputs(Arity, function(Out), Ps) :-
    numlist(1, Arity, All), exclude(==(Out), All, Ps).

%% ts_struct_param(+Clauses, +Pos, -Decl)
ts_struct_param(Clauses, Pos, Decl) :-
    ( ( member(H-_, Clauses), arg(Pos, H, A), ts_islisty(A) ) -> Ty = "any[]" ; Ty = "any" ),
    format(string(Decl), "a~w: ~w", [Pos, Ty]).

%% ts_struct_clause(+Pred, +Arity, +Mode, +Head-Body, -Block)
ts_struct_clause(Pred, _Arity, Mode, Head-Body, Block) :-
    Head =.. [_|HeadArgs],
    ( Mode = function(OutPos) -> true ; OutPos = 0 ),
    ts_head_positions(HeadArgs, 1, OutPos, [], Bind1, [], Conds0),
    reverse(Conds0, HeadConds),
    ts_struct_body(Body, Pred, Mode, Bind1, Bind2, GuardConds, StmtLines, Tail),
    append(HeadConds, GuardConds, AllConds),
    ( AllConds == [] -> CondStr = "true" ; atomic_list_concat(AllConds, ' && ', CondStr) ),
    (   Mode = test
    ->  ( Tail = yes(R) -> RetExpr = R ; RetExpr = "true" )
    ;   nth1(OutPos, HeadArgs, OutArg),
        ts_term_expr(OutArg, Bind2, RetExpr)
    ),
    (   StmtLines == []
    ->  format(string(Block), '  if (~w) {\n    return ~w;\n  }', [CondStr, RetExpr])
    ;   atomic_list_concat(StmtLines, '\n', StmtBlock),
        format(string(Block), '  if (~w) {\n~w\n    return ~w;\n  }', [CondStr, StmtBlock, RetExpr])
    ).

%% ts_head_positions(+HeadArgs, +Idx, +OutPos, +Bind0, -Bind, +Conds0, -Conds)
ts_head_positions([], _, _, B, B, C, C).
ts_head_positions([Arg|Rest], Idx, OutPos, B0, B, C0, C) :-
    (   Idx =:= OutPos
    ->  B1 = B0, C1 = C0
    ;   format(string(PName), "a~w", [Idx]),
        ts_match(PName, Arg, B0, B1, C0, C1)
    ),
    Idx1 is Idx + 1,
    ts_head_positions(Rest, Idx1, OutPos, B1, B, C1, C).

%% ts_match(+Expr, +Pattern, +Bind0, -Bind, +Conds0, -Conds)
ts_match(Expr, V, B0, B, C0, C) :- var(V), !,
    (   ts_bget(V, B0, Prev)
    ->  B = B0, format(string(Cond), "~w === ~w", [Expr, Prev]), C = [Cond|C0]
    ;   B = [V-Expr|B0], C = C0
    ).
ts_match(Expr, [], B, B, C0, [Cond|C0]) :- !,
    format(string(Cond), "~w.length === 0", [Expr]).
ts_match(Expr, [H|T], B0, B, C0, C) :- !,
    format(string(NonEmpty), "~w.length > 0", [Expr]),
    format(string(HeadE), "~w[0]", [Expr]),
    format(string(TailE), "~w.slice(1)", [Expr]),
    ts_match(HeadE, H, B0, B1, [NonEmpty|C0], C1),
    ts_match(TailE, T, B1, B, C1, C).
ts_match(Expr, N, B, B, C0, [Cond|C0]) :- number(N), !,
    format(string(Cond), "~w === ~w", [Expr, N]).
ts_match(Expr, A, B, B, C0, [Cond|C0]) :- atom(A), !,
    format(string(Cond), '~w === "~w"', [Expr, A]).

ts_bget(V, [V0-E|_], E) :- V0 == V, !.
ts_bget(V, [_|T], E) :- ts_bget(V, T, E).

%% ts_term_expr(+Term, +Bind, -Expr) — Prolog term (list/var/atom/number) -> TS.
ts_term_expr(V, B, E) :- var(V), !, ( ts_bget(V, B, E0) -> E = E0 ; E = "undefined" ).
ts_term_expr([], _, "[]") :- !.
ts_term_expr([H|T], B, E) :- !,
    ts_term_expr(H, B, HE), ts_term_expr(T, B, TE),
    format(string(E), "[~w, ...~w]", [HE, TE]).
ts_term_expr(N, _, E) :- number(N), !, format(string(E), "~w", [N]).
ts_term_expr(A, _, E) :- atom(A), !, format(string(E), '"~w"', [A]).

%% ts_struct_body(+Body, +Pred, +Mode, +B0, -B, -GuardConds, -StmtLines, -Tail)
ts_struct_body(Body, Pred, Mode, B0, B, Guards, Stmts, Tail) :-
    ts_extract_goals(Body, Goals0),
    maplist(ts_strip_mod, Goals0, Goals),
    ts_struct_goals(Goals, Pred, Mode, B0, B, [], GuardsR, [], StmtsR, no, Tail),
    reverse(GuardsR, Guards),
    reverse(StmtsR, Stmts).

ts_struct_goals(Goals, Pred, Mode, B0, B, G0, G, S0, S, T0, T) :-
    ts_struct_goals(Goals, Pred, Mode, B0, B, G0, G, S0, S, T0, T, 0).
ts_struct_goals([], _, _, B, B, G, G, S, S, T, T, _).
ts_struct_goals([Goal|Rest], Pred, Mode, B0, B, G0, G, S0, S, T0, T, N0) :-
    ts_struct_goal(Goal, Pred, Mode, B0, B1, G0, G1, S0, S1, T0, T1, N0, N1),
    ts_struct_goals(Rest, Pred, Mode, B1, B, G1, G, S1, S, T1, T, N1).

%% ts_struct_goal(+Goal, +Pred, +Mode, +B0,-B, +G0,-G, +S0,-S, +T0,-T, +N0,-N)
ts_struct_goal(true, _, _, B, B, G, G, S, S, T, T, N, N) :- !.
ts_struct_goal(Goal, _Pred, _Mode, B, B, G0, [Cond|G0], S, S, T, T, N, N) :-
    Goal =.. [Op, L, R], ts_cmp_op(Op, JsOp), !,
    ts_arith(L, B, LS), ts_arith(R, B, RS),
    format(string(Cond), "~w ~w ~w", [LS, JsOp, RS]).
ts_struct_goal(is(V, Expr), _Pred, _Mode, B0, [V-Nm|B0], G, G, S0, [Stmt|S0], T, T, N0, N1) :-
    var(V), !,
    ts_arith(Expr, B0, ES),
    format(string(Nm), "_s~w", [N0]), N1 is N0 + 1,
    format(string(Stmt), "    const ~w = ~w;", [Nm, ES]).
ts_struct_goal(=(V, Term), _Pred, _Mode, B0, [V-E|B0], G, G, S, S, T, T, N, N) :-
    var(V), !,
    ts_term_expr(Term, B0, E).
ts_struct_goal(Goal, Pred, function(_), B0, B, G, G, S0, [Stmt|S0], T, T, N0, N1) :-
    compound(Goal), functor(Goal, Pred, _), !,
    Goal =.. [_|Args],
    append(InArgs, [OutArg], Args),
    var(OutArg),
    maplist(ts_term_expr_b(B0), InArgs, InEs),
    atomic_list_concat(InEs, ', ', ArgStr),
    atom_string(Pred, PredStr),
    format(string(Nm), "_s~w", [N0]), N1 is N0 + 1,
    format(string(Stmt), "    const ~w = ~w(~w);", [Nm, PredStr, ArgStr]),
    B = [OutArg-Nm|B0].
ts_struct_goal(Goal, Pred, test, B, B, G, G, S, S, _T0, yes(R), N, N) :-
    compound(Goal), functor(Goal, Pred, _), !,
    Goal =.. [_|Args],
    maplist(ts_term_expr_b(B), Args, Es),
    atomic_list_concat(Es, ', ', ArgStr),
    atom_string(Pred, PredStr),
    format(string(R), "~w(~w)", [PredStr, ArgStr]).

ts_term_expr_b(B, T, E) :- ts_term_expr(T, B, E).

ts_cmp_op(>, ">").
ts_cmp_op(<, "<").
ts_cmp_op(>=, ">=").
ts_cmp_op(=<, "<=").
ts_cmp_op(=:=, "===").
ts_cmp_op(=\=, "!==").
ts_cmp_op(==, "===").
ts_cmp_op(\==, "!==").

%% ts_arith(+Expr, +Bind, -TsExpr)
ts_arith(V, B, S) :- var(V), !, ( ts_bget(V, B, S) -> true ; term_string(V, S) ).
ts_arith(N, _, S) :- number(N), !, format(string(S), "~w", [N]).
ts_arith(A+B, Bi, S) :- !, ts_arith(A, Bi, SA), ts_arith(B, Bi, SB), format(string(S), "(~w + ~w)", [SA, SB]).
ts_arith(A-B, Bi, S) :- !, ts_arith(A, Bi, SA), ts_arith(B, Bi, SB), format(string(S), "(~w - ~w)", [SA, SB]).
ts_arith(A*B, Bi, S) :- !, ts_arith(A, Bi, SA), ts_arith(B, Bi, SB), format(string(S), "(~w * ~w)", [SA, SB]).
ts_arith(A/B, Bi, S) :- !, ts_arith(A, Bi, SA), ts_arith(B, Bi, SB), format(string(S), "(~w / ~w)", [SA, SB]).
ts_arith(A mod B, Bi, S) :- !, ts_arith(A, Bi, SA), ts_arith(B, Bi, SB), format(string(S), "(~w % ~w)", [SA, SB]).
ts_arith(A, _, S) :- atom(A), !, format(string(S), '"~w"', [A]).

% ============================================================================
% TREE RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

tree_recursion:compile_tree_pattern(typescript, _Pattern, Pred, Arity, _UseMemo, TsCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(ts_is_rec_clause(Pred), Clauses, RecClauses, BaseClauses),
    ts_numeric_recursion_or_fallback(Pred, PredStr, "Tree Recursion",
                                     BaseClauses, RecClauses, TsCode).

% ============================================================================
% MULTICALL LINEAR RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/multicall_linear_recursion').
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.

multicall_linear_recursion:compile_multicall_pattern(typescript, PredStr, BaseClauses, RecClauses, _MemoEnabled, TsCode) :-
    atom_string(Pred, PredStr),
    ts_numeric_recursion_or_fallback(Pred, PredStr, "Multicall Linear Recursion",
                                     BaseClauses, RecClauses, TsCode).

% ============================================================================
% DIRECT MULTICALL RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/direct_multi_call_recursion').
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.

direct_multi_call_recursion:compile_direct_multicall_pattern(typescript, PredStr, BaseClauses, RecClause, TsCode) :-
    atom_string(Pred, PredStr),
    ( is_list(RecClause) -> RecClauses = RecClause ; RecClauses = [RecClause] ),
    ts_numeric_recursion_or_fallback(Pred, PredStr, "Direct Multicall Recursion",
                                     BaseClauses, RecClauses, TsCode).

% ============================================================================
% MUTUAL RECURSION - TypeScript target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/mutual_recursion').
:- use_module('../core/advanced/pattern_matchers', [is_per_path_visited_pattern/4]).
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

%% No-visited-pattern — plain recursive without cycle detection
advanced_recursive_compiler:compile_general_recursive_pattern(typescript, PredStr, Arity, BaseClauses, RecClauses, Code) :-
    atom_string(Pred, PredStr),
    append(BaseClauses, RecClauses, AllClauses),
    \+ is_per_path_visited_pattern(Pred, Arity, AllClauses, _),
    !,
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_|BaseArgs], last(BaseArgs, BaseVal),
        BaseArgs = [BaseKey|_],
        format(string(BaseCheck), '    if (arg1 === "~w") return ["~w"];', [BaseKey, BaseVal])
    ;   BaseCheck = '    // no base case extracted'
    ),
    format(string(Code),
'// General recursive: ~w (plain, no visited pattern)\n\c
function ~w(arg1: string): string[] {\n\c
~w\n\c
    return [...~w(arg1)];\n\c
}\n',
    [PredStr, PredStr, BaseCheck, PredStr]).

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

extract_rec_call_typescript((A, _), PredStr, WorkerStr, Expr) :-
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
