% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_typescript_target.pl - plunit suite for the TypeScript pattern target
%
% Covers the public compilation surface of typescript_target.pl:
%   - facts        -> typed interface + fact array + query/membership helpers
%   - recursion    -> each pattern declared in target_info/1
%                     (tail_recursion, linear_recursion, list_fold,
%                      transitive_closure)
%   - modules      -> multi-predicate compile_module/3
%   - bindings     -> registration + import lookup (typescript_bindings.pl)
%
% Run: swipl -q -g test_typescript_target -t halt tests/core/test_typescript_target.pl

:- module(test_typescript_target, [test_typescript_target/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/typescript_target').
:- use_module('../../src/unifyweaver/bindings/typescript_bindings').

test_typescript_target :-
    run_tests([typescript_target]).

:- begin_tests(typescript_target).

% Helpers: deterministic substring checks
has(Code, Substr)   :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

% ============================================================================
% target_info/1 metadata
% ============================================================================

test(target_info_declares_js_family) :-
    typescript_target:target_info(Info),
    get_dict(family, Info, Family), Family == javascript,
    get_dict(file_extension, Info, Ext), Ext == ".ts".

test(target_info_declares_four_recursion_patterns) :-
    typescript_target:target_info(Info),
    get_dict(recursion_patterns, Info, Patterns),
    memberchk(tail_recursion, Patterns),
    memberchk(linear_recursion, Patterns),
    memberchk(list_fold, Patterns),
    memberchk(transitive_closure, Patterns).

% ============================================================================
% Facts -> typed arrays + interfaces
% ============================================================================
% NOTE: compile_facts emits the interface, fact array scaffold and the
% query/membership helpers. We assert the structural TypeScript surface here.

test(facts_binary_predicate, [setup(assert_parent_facts), cleanup(retract_parent_facts)]) :-
    typescript_target:compile_facts(parent, 2, Code),
    has(Code, "export interface ParentFact"),
    has(Code, "arg1: string;"),
    has(Code, "arg2: string;"),
    has(Code, "export const parentFacts: ParentFact[]"),
    has(Code, "export const queryParent"),
    has(Code, "export const isParent").

test(facts_unary_predicate, [setup(assert_color_facts), cleanup(retract_color_facts)]) :-
    typescript_target:compile_facts(color, 1, Code),
    has(Code, "export interface ColorFact"),
    has(Code, "arg1: string;"),
    has(Code, "export const colorFacts: ColorFact[]").

% compile_facts/3 gathers facts with a bare call/1 executed in the
% typescript_target module, so we assert the sample facts there directly.
assert_parent_facts :-
    assertz(typescript_target:parent(tom, bob)),
    assertz(typescript_target:parent(bob, ann)).
retract_parent_facts :-
    retractall(typescript_target:parent(_, _)).
assert_color_facts :-
    assertz(typescript_target:color(red)),
    assertz(typescript_target:color(blue)).
retract_color_facts :-
    retractall(typescript_target:color(_)).

% ============================================================================
% Recursion patterns (each pattern declared in target_info/1)
% ============================================================================

test(recursion_tail) :-
    typescript_target:compile_recursion(sumTail/2, [pattern(tail_recursion)], Code),
    has(Code, "Pattern: tail_recursion"),
    has(Code, "export const sumTail"),
    has(Code, "acc"),
    has(Code, "=>").

test(recursion_list_fold) :-
    typescript_target:compile_recursion(listSum/2, [pattern(list_fold)], Code),
    has(Code, "Pattern: list_fold"),
    has(Code, "export const listSum"),
    has(Code, ".reduce(").

test(recursion_linear) :-
    typescript_target:compile_recursion(fib/2, [pattern(linear_recursion)], Code),
    has(Code, "Pattern: linear_recursion"),
    has(Code, "export const fib"),
    has(Code, "Map<number, number>").

% transitive_closure is declared in target_info/1 but not given a dedicated
% generator in compile_recursion/3: it currently routes through the
% tail_recursion fallback. We exercise the dispatch and assert it yields a
% non-empty arrow-function body. (Flagged for INT-0 in the integration patch.)
test(recursion_transitive_closure_fallback) :-
    typescript_target:compile_recursion(reachable/2, [pattern(transitive_closure)], Code),
    string_length(Code, Len), Len > 0,
    has(Code, "export const reachable"),
    has(Code, "=>").

% Genuine transitive-closure lowering: the general-recursive multifile hook
% emits a recursive traversal for a base + recursive clause pair.
test(general_recursive_arity2_emits_function) :-
    advanced_recursive_compiler:compile_general_recursive_pattern(
        typescript, "reaches", 2,
        [(reaches(a, b), true)],
        [(reaches(X, Y), (edge(X, Z), reaches(Z, Y)))],
        Code),
    has(Code, "function reaches(arg1: string)"),
    has(Code, "string[]").

% ============================================================================
% Module compilation (multi-predicate)
% ============================================================================

test(module_multi_predicate) :-
    typescript_target:compile_module(
        [ pred(sumTail, 2, tail_recursion),
          pred(listSum, 2, list_fold),
          pred(fib,     2, linear_recursion),
          pred(fact,    1, factorial) ],
        [module_name('PrologMath')],
        Code),
    has(Code, "// Module: PrologMath"),
    has(Code, "export const sumTail"),
    has(Code, "export const listSum"),
    has(Code, ".reduce("),
    has(Code, "fibMemo"),
    has(Code, "export const fact"),
    has(Code, "(factorial)").

% ============================================================================
% Bindings: registration + import collection
% ============================================================================

test(bindings_init_and_lookup, [setup(typescript_bindings:init_typescript_bindings)]) :-
    % Stdlib bindings register
    once((
        typescript_bindings:ts_binding(sqrt/2, 'Math.sqrt', [number], [number], _),
        typescript_bindings:ts_binding(string_length/2, '.length', [string], [number], _),
        typescript_bindings:ts_binding(json_stringify/2, 'JSON.stringify', [any], [string], _)
    )).

test(bindings_new_collection_and_number, [setup(typescript_bindings:init_typescript_bindings)]) :-
    % Bindings added by TS-1 (Map/Set collections + Number formatting)
    once((
        typescript_bindings:ts_binding(map_get/3, '.get', [map, any], [any], _),
        typescript_bindings:ts_binding(set_add/3, '.add', _, _, _),
        typescript_bindings:ts_binding(number_to_fixed/3, '.toFixed', _, _, _)
    )).

test(bindings_import_lookup, [setup(typescript_bindings:init_typescript_bindings)]) :-
    % Node built-ins flow their required import through the Options list
    once(typescript_bindings:ts_binding_import(read_file_sync/2, 'fs')),
    once(typescript_bindings:ts_binding_import(path_join/3, 'path')).

test(binding_import_collection_roundtrip) :-
    % The target's import-collection API dedups and returns collected imports
    typescript_target:clear_binding_imports,
    typescript_target:collect_binding_import('fs'),
    typescript_target:collect_binding_import('path'),
    typescript_target:collect_binding_import('fs'),   % duplicate ignored
    typescript_target:get_collected_imports(Imports),
    sort(Imports, Sorted),
    Sorted == ['fs', 'path'].

:- end_tests(typescript_target).
