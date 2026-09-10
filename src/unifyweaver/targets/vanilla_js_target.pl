:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% vanilla_js_target.pl - Plain (untyped) JavaScript Code Generation Target
%
% A *variant* of the TypeScript target (typescript_target.pl) that emits
% plain vanilla JavaScript with no type annotations. Follows the same
% inheritance pattern the ClojureScript/Python families use: this module
% `use_module`s its base target and only overrides the TS->JS difference,
% which is the removal of TypeScript's compile-time-only type syntax.
%
%   vanilla_js_target : typescript_target  ::  clojurescript_target : clojure_target
%
% The bulk of the codegen (native clause lowering, recursion patterns,
% expression translation, all the advanced multifile recursion hooks) is
% reused unchanged from typescript_target. The single override is the
% type-stripping rewrite `vanilla_js_type_strip/2` (mirroring
% clojurescript_target's `clojurescript_interop_rewrite/2`): it takes the
% TypeScript source produced by the base target and rewrites it into valid
% ES-module JavaScript that runs on stock Node / in a browser with no build
% step and no runtime dependency.
%
% What is stripped (the TS-only surface):
%   - inline type annotations       (`: number`, `: string[]`, `: Set<string>`)
%   - function return types         (`): number =>`  ->  `) =>`)
%   - `interface` declaration blocks (dropped entirely)
%   - generic type arguments        (`new Map<number, number>()` -> `new Map()`,
%                                     `<T, R>`)
%   - non-null assertions           (`x.get(n)!` -> `x.get(n)`)
%   - `as` type assertions          (`(fact as any)` -> `(fact)`)
%
% Example:
%   ?- compile_predicate_to_vanilla_js(double/2, [], Code).

:- module(vanilla_js_target, [
    % Standard interface
    target_info/1,                    % -Info
    compile_predicate/3,              % +Pred/Arity, +Options, -Code (registry dispatch)
    compile_predicate_to_vanilla_js/3,% +Pred/Arity, +Options, -Code
    compile_facts/3,                  % +Pred, +Arity, -Code
    compile_recursion/3,              % +Pred/Arity, +Options, -Code
    compile_module/3,                 % +Predicates, +Options, -Code
    write_vanilla_js_module/2,        % +Code, +Filename
    init_vanilla_js_target/0,

    % The core TS -> JS override
    vanilla_js_type_strip/2,          % +TypeScriptCode, -JavaScriptCode

    % Binding system exports (delegated to the TypeScript base)
    clear_binding_imports/0,
    collect_binding_import/1,
    get_collected_imports/1
]).

:- use_module(library(lists)).
:- use_module(library(pcre)).

% Inherit the TypeScript target. Import nothing into this namespace (we define
% our own contract predicates and call the base module-qualified), but loading
% it registers all of typescript_target's codegen plus its advanced multifile
% recursion hooks, which we reuse unchanged.
:- use_module(typescript_target, []).

%% ============================================
%% TARGET INFO
%% ============================================

target_info(info{
    name: "VanillaJS",
    family: javascript,
    file_extension: ".js",
    runtime: auto,              % node, browser, deno, bun
    features: [plain, modules, async],
    recursion_patterns: [tail_recursion, linear_recursion, list_fold, transitive_closure],
    compile_command: "node"
}).

%% ============================================
%% INITIALIZATION
%% ============================================

%% init_vanilla_js_target
%  Initialize the vanilla JS target (delegates to the TypeScript base).
init_vanilla_js_target :-
    typescript_target:init_typescript_target,
    format('[VanillaJS Target] Initialized (variant of typescript_target)~n', []).

%% ============================================
%% MAIN DISPATCH (delegate to TS base, then strip types)
%% ============================================

%% compile_predicate(+Pred/Arity, +Options, -Code)
%  Thin wrapper so target_registry's compile_to_target/4 can dispatch here.
compile_predicate(PredIndicator, Options, Code) :-
    compile_predicate_to_vanilla_js(PredIndicator, Options, Code).

%% compile_predicate_to_vanilla_js(+Pred/Arity, +Options, -Code)
%  Compile a predicate to plain JavaScript by delegating to the TypeScript
%  base target and stripping the TypeScript-only type syntax.
compile_predicate_to_vanilla_js(PredIndicator, Options, Code) :-
    typescript_target:compile_predicate_to_typescript(PredIndicator, Options, TsCode),
    vanilla_js_type_strip(TsCode, Code).

%% compile_facts(+Pred, +Arity, -Code)
compile_facts(Pred, Arity, Code) :-
    typescript_target:compile_facts(Pred, Arity, TsCode),
    vanilla_js_type_strip(TsCode, Code).

%% compile_recursion(+Pred/Arity, +Options, -Code)
compile_recursion(PredIndicator, Options, Code) :-
    typescript_target:compile_recursion(PredIndicator, Options, TsCode),
    vanilla_js_type_strip(TsCode, Code).

%% compile_module(+Predicates, +Options, -Code)
compile_module(Predicates, Options, Code) :-
    typescript_target:compile_module(Predicates, Options, TsCode),
    vanilla_js_type_strip(TsCode, Code).

%% ============================================
%% BINDING HOOKS (delegate to the TypeScript base)
%% ============================================

clear_binding_imports :-
    typescript_target:clear_binding_imports.

collect_binding_import(Import) :-
    typescript_target:collect_binding_import(Import).

get_collected_imports(Imports) :-
    typescript_target:get_collected_imports(Imports).

%% ============================================
%% TYPE STRIP (the core TS -> JS override)
%% ============================================

%% vanilla_js_type_strip(+TypeScriptCode, -JavaScriptCode)
%  Turn TypeScript source (produced by typescript_target) into valid vanilla
%  ES-module JavaScript by removing all compile-time-only type syntax.
%
%  Centralized single rewrite predicate, mirroring
%  clojurescript_target:clojurescript_interop_rewrite/2. Order matters: the
%  rules that need parentheses/brackets intact run before the generic type
%  annotation stripper, and generic type arguments are removed last (to a
%  fixpoint, so nested generics collapse fully).
vanilla_js_type_strip(In, Out) :-
    ( string(In) -> S0 = In ; atom_string(In, S0) ),
    js_strip_rules(Rules),
    foldl(apply_js_rule, Rules, S0, S1),
    strip_generics_fixpoint(S1, Out).

apply_js_rule(re(Pattern, With), In, Out) :-
    re_replace(Pattern, With, In, Out).

%% js_strip_rules(-Rules)
%  Ordered list of re(Pattern/Flags, Replacement) rewrites.
js_strip_rules([
    % 1. Drop `interface` declaration blocks entirely (no nested braces emitted).
    re("(?:export )?interface \\w+ \\{[^}]*\\}\\n?"/g, ""),

    % 2. Remove `as` type assertions actually emitted by the base target.
    re(" as (?:any|unknown)\\b"/g, ""),

    % 3. Remove function-type parameter annotations, e.g.
    %    `fn: (acc: R, item: T) => R`  (must run before rule 6, parens intact).
    re(":\\s*\\([^)]*\\)\\s*=>\\s*[A-Za-z0-9_\\[\\]<>]+"/g, ""),

    % 4. Remove arrow-function generic type-parameter lists: `= <T, R>(` -> `= (`.
    re("=\\s*<[^<>]*>\\s*\\("/g, "= ("),

    % 5. Remove tuple / array-literal type annotations, e.g. `: [string, string][]`
    %    and G-A3-9's `: [any, any]`.
    %
    %    The bracket contents are matched as a TYPE LIST, not as "anything up to
    %    the first `]`". The looser form ate the `args: [` of an object literal:
    %    G-A3-12's compound representation is `{$: "schema", args: [[], []]}`, in
    %    which `: [[]` is followed by `,` and satisfied the old lookahead exactly,
    %    so `{$: "-", args, ["name"]]}` reached node -- a syntax error inside an
    %    otherwise-correct module. No tuple annotation this target emits contains
    %    anything but type names.
    re(":\\s*\\[\\s*(?:any|number|string|boolean|unknown|void|null|undefined)\
(?:\\s*,\\s*(?:any|number|string|boolean|unknown|void|null|undefined))*\
\\s*\\](?:\\[\\])?(?=\\s*(?:=>|[,);{=]))"/g, ""),

    % 6. Remove keyword / named type annotations, e.g. `: number`, `: string[]`,
    %    `: Set<string>`, `: Partial<FooFact>`, `: Promise<...>`, return types,
    %    AND union types `: T1 | T2 | ...` (e.g. `: number | null`,
    %    `: string | undefined`, `: A<X> | B[]`). The leading type atom uses the
    %    unambiguous named set (never bare `null`/`undefined`, so a ternary
    %    `cond ? a : null` is untouched); subsequent union members after a `|`
    %    additionally admit `null`/`undefined`. Applies in both param position
    %    (lookahead `,`/`)`) and return position (lookahead `=>`/`{`/`=`/`;`).
    re(":\\s*(?:number|string|boolean|void|unknown|any|T|R|Request|Response|Partial|Promise|Set|Map|ApiResponse|[A-Z][A-Za-z0-9_]*Fact)(?:<[^;{}()]*>)?(?:\\[\\])?(?:\\s*\\|\\s*(?:number|string|boolean|void|unknown|any|null|undefined|T|R|Request|Response|Partial|Promise|Set|Map|ApiResponse|[A-Z][A-Za-z0-9_]*Fact)(?:<[^;{}()]*>)?(?:\\[\\])?)*(?=\\s*(?:=>|[,);{=]))"/g, ""),

    % 7. Remove non-null assertions: `x.get(n)!` -> `x.get(n)`.
    re("\\)!"/g, ")")
]).

%% strip_generics_fixpoint(+In, -Out)
%  Remove generic type arguments that are attached to an identifier
%  (`new Map<number, number>()`, `new Set<string>()`, leftover `Foo<Bar>`),
%  repeatedly until stable so nested generics collapse. The lookbehind on a
%  word character keeps real `<` / `<=` comparison operators (always
%  space-separated in generated code) untouched.
strip_generics_fixpoint(In, Out) :-
    re_replace("(?<=[A-Za-z0-9_])<[^<>]*>"/g, "", In, Mid),
    ( Mid == In
    ->  Out = In
    ;   strip_generics_fixpoint(Mid, Out)
    ).

%% ============================================
%% ADVANCED RECURSION — inherit TS, then strip types
%% ============================================
%
% The advanced recursion compiler (core/advanced/advanced_recursive_compiler.pl)
% dispatches on the target atom via these multifile hook predicates. Mirror the
% annotated_js target exactly: register a clause per pattern that delegates to
% the `typescript` clause at runtime (so we inherit every TS recursion fix) and
% then applies our TS->JS type-strip. Without these clauses any caller driving
% the advanced compiler with target(vanilla_js) finds no clause and fails.

:- multifile tail_recursion:compile_tail_pattern/9.
:- multifile linear_recursion:compile_linear_pattern/8.
:- multifile tree_recursion:compile_tree_pattern/6.
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.
:- multifile mutual_recursion:compile_mutual_pattern/5.
:- multifile advanced_recursive_compiler:compile_general_recursive_pattern/6.

tail_recursion:compile_tail_pattern(vanilla_js, PredStr, Arity, Base, Rec, AccPos, StepOp, Exit, Code) :-
    tail_recursion:compile_tail_pattern(typescript, PredStr, Arity, Base, Rec, AccPos, StepOp, Exit, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

linear_recursion:compile_linear_pattern(vanilla_js, PredStr, Arity, Base, Rec, Memo, Strat, Code) :-
    linear_recursion:compile_linear_pattern(typescript, PredStr, Arity, Base, Rec, Memo, Strat, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

tree_recursion:compile_tree_pattern(vanilla_js, Pattern, Pred, Arity, UseMemo, Code) :-
    tree_recursion:compile_tree_pattern(typescript, Pattern, Pred, Arity, UseMemo, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

multicall_linear_recursion:compile_multicall_pattern(vanilla_js, PredStr, Base, Rec, Memo, Code) :-
    multicall_linear_recursion:compile_multicall_pattern(typescript, PredStr, Base, Rec, Memo, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

direct_multi_call_recursion:compile_direct_multicall_pattern(vanilla_js, PredStr, Base, Rec, Code) :-
    direct_multi_call_recursion:compile_direct_multicall_pattern(typescript, PredStr, Base, Rec, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

mutual_recursion:compile_mutual_pattern(vanilla_js, Preds, Memo, Strat, Code) :-
    mutual_recursion:compile_mutual_pattern(typescript, Preds, Memo, Strat, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

advanced_recursive_compiler:compile_general_recursive_pattern(vanilla_js, PredStr, Arity, Base, Rec, Code) :-
    advanced_recursive_compiler:compile_general_recursive_pattern(typescript, PredStr, Arity, Base, Rec, TSCode),
    vanilla_js_target:vanilla_js_type_strip(TSCode, Code).

%% ============================================
%% FILE OUTPUT
%% ============================================

write_vanilla_js_module(Code, Filename) :-
    open(Filename, write, Stream),
    write(Stream, Code),
    close(Stream),
    format('Vanilla JS module written to: ~w~n', [Filename]),
    format('Run with: node ~w~n', [Filename]).
