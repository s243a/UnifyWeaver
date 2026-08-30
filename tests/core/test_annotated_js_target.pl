% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% test_annotated_js_target.pl - plunit tests for the AnnotatedJS target
%
% Run:
%   swipl -q -g test_annotated_js_target -t halt tests/core/test_annotated_js_target.pl

:- module(test_annotated_js_target, [test_annotated_js_target/0]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/annotated_js_target').

test_annotated_js_target :-
    run_tests([annotated_js_target]).

:- begin_tests(annotated_js_target).

has(Code, Substr) :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

%% Output is JSDoc-annotated JS: has /** and no TS-only syntax.
assert_annotated_js(Code) :-
    has(Code, "/**"),
    hasnt(Code, ": number"),
    hasnt(Code, "interface "),
    hasnt(Code, "<T>").

% ============================================================================
% target_info
% ============================================================================

test(target_info) :-
    annotated_js_target:target_info(Info),
    Info.name == "AnnotatedJS",
    Info.family == javascript,
    Info.file_extension == ".js",
    once(member(jsdoc, Info.features)),
    once(member(tsc_checked, Info.features)),
    once(member(tail_recursion, Info.recursion_patterns)),
    once(member(linear_recursion, Info.recursion_patterns)),
    once(member(list_fold, Info.recursion_patterns)),
    once(member(transitive_closure, Info.recursion_patterns)).

% ============================================================================
% Rewrite unit: TS inline types → JSDoc
% ============================================================================

test(rewrite_arrow_params) :-
    annotated_js_target:ts_to_annotated_js(
        "export const sum = (n: number, acc: number = 0): number => {\n  return n;\n};\n",
        Out),
    has(Out, "@param {number} n"),
    has(Out, "@returns {number}"),
    has(Out, "export const sum = (n, acc = 0) => {"),
    assert_annotated_js(Out).

test(rewrite_interface_to_typedef) :-
    annotated_js_target:ts_to_annotated_js(
        "export interface PersonFact {\n  arg1: string;\n  arg2: string;\n}\n",
        Out),
    has(Out, "@typedef {Object} PersonFact"),
    has(Out, "@property {string} arg1"),
    hasnt(Out, "interface "),
    has(Out, "/**").

test(rewrite_generic_arrow) :-
    annotated_js_target:ts_to_annotated_js(
        "export const fold = <T, R>(\n  items: T[],\n  initial: R,\n  fn: (acc: R, item: T) => R\n): R => {\n  return items.reduce(fn, initial);\n};\n",
        Out),
    has(Out, "@template T"),
    has(Out, "@template R"),
    has(Out, "export const fold = ("),
    hasnt(Out, "<T>"),
    hasnt(Out, ": number").

test(rewrite_nonnull_and_map) :-
    annotated_js_target:ts_to_annotated_js(
        "const fibMemo = new Map<number, number>();\nexport const fib = (n: number): number => {\n  return fibMemo.get(n)!;\n};\n",
        Out),
    has(Out, "new Map("),
    has(Out, "@type {Map<number, number>}"),
    hasnt(Out, "get(n)!"),
    hasnt(Out, ": number").

% ============================================================================
% G-P14: rewrite-robustness (union types, namespace imports, inline arrows)
% ============================================================================

% Namespace imports are NOT type annotations: `import * as X from '...'` must
% survive intact (the old `strip_as_casts` fired on `* as X` and mangled it).
test(namespace_import_intact) :-
    annotated_js_target:ts_to_annotated_js(
        "import * as readline from \"readline\";\nconst rl = readline.createInterface({ input: process.stdin });\n",
        Out),
    has(Out, "import * as readline from \"readline\";"),
    hasnt(Out, "@type {readline"),
    hasnt(Out, "import *;").

% Aliased named imports/exports also survive.
test(named_import_alias_intact) :-
    annotated_js_target:ts_to_annotated_js(
        "import { readFile as rf } from \"node:fs\";\n",
        Out),
    has(Out, "import { readFile as rf } from \"node:fs\";").

% A genuine value cast still becomes a JSDoc @type cast (regression).
test(genuine_as_cast_still_rewritten) :-
    annotated_js_target:ts_to_annotated_js(
        "  return (fact as any).arg1;\n",
        Out),
    has(Out, "@type {any}"),
    has(Out, "(fact)"),
    hasnt(Out, " as any").

% Inline arrow-parameter annotation is stripped cleanly (callback position).
test(inline_arrow_param_stripped) :-
    annotated_js_target:ts_to_annotated_js(
        "rl.on(\"line\", (line: string) => {\n  addFact(line);\n});\n",
        Out),
    has(Out, "(line) =>"),
    hasnt(Out, ": string").

% Union return type → JSDoc union; signature body carries no type.
test(union_return_to_jsdoc) :-
    annotated_js_target:ts_to_annotated_js(
        "export const f = (n: number): number | null => {\n  return n;\n};\n",
        Out),
    has(Out, "@returns {number | null}"),
    has(Out, "export const f = (n) => {"),
    hasnt(Out, ": number"),
    hasnt(Out, ": null =>").

% ============================================================================
% Fact compilation
% ============================================================================

test(compile_fact, [setup(assert_person_facts), cleanup(retract_person_facts)]) :-
    annotated_js_target:compile_facts(person, 2, Code),
    has(Code, "@typedef {Object} PersonFact"),
    has(Code, "personFacts"),
    assert_annotated_js(Code).

assert_person_facts :-
    assertz(typescript_target:person(tom, 25)),
    assertz(typescript_target:person(bob, 30)).

retract_person_facts :-
    retractall(typescript_target:person(_, _)).

% ============================================================================
% Recursion patterns (inherited from TypeScript, then annotated)
% ============================================================================

test(tail_recursion) :-
    annotated_js_target:compile_recursion(sum/2, [pattern(tail_recursion)], Code),
    has(Code, "export const sum"),
    has(Code, "sum(n - 1, acc + n)"),
    has(Code, "@param {number} n"),
    assert_annotated_js(Code).

test(linear_recursion) :-
    annotated_js_target:compile_recursion(fib/2, [pattern(linear_recursion)], Code),
    has(Code, "fib(n - 1)"),
    has(Code, "fibMemo"),
    has(Code, "@param {number} n"),
    assert_annotated_js(Code).

test(list_fold) :-
    annotated_js_target:compile_recursion(listSum/2, [pattern(list_fold)], Code),
    has(Code, "reduce"),
    has(Code, "export const listSum"),
    has(Code, "@param {number[]} items"),
    assert_annotated_js(Code).

test(transitive_closure) :-
    annotated_js_target:compile_recursion(ancestor/2, [pattern(transitive_closure)], Code),
    has(Code, "ancestor"),
    once((has(Code, "baseRelation") ; has(Code, "visited"))),
    assert_annotated_js(Code).

% ============================================================================
% Multi-predicate module
% ============================================================================

test(compile_module) :-
    annotated_js_target:compile_module(
        [pred(sum, 2, tail_recursion),
         pred(factorial, 1, factorial),
         pred(fib, 2, linear_recursion),
         pred(listSum, 2, list_fold)],
        [module_name('PrologMath')],
        Code),
    has(Code, "Module: PrologMath"),
    has(Code, "export const sum"),
    has(Code, "export const factorial"),
    has(Code, "export const fib"),
    has(Code, "export const listSum"),
    assert_annotated_js(Code).

% ============================================================================
% Binding hooks delegate to TypeScript
% ============================================================================

test(binding_hooks_delegate) :-
    annotated_js_target:clear_binding_imports,
    annotated_js_target:collect_binding_import(fs),
    annotated_js_target:get_collected_imports(Imports),
    once(member(fs, Imports)),
    annotated_js_target:clear_binding_imports,
    annotated_js_target:get_collected_imports(Empty),
    Empty == [].

:- end_tests(annotated_js_target).
