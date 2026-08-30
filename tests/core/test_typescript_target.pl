:- encoding(utf8).
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
% Run: swipl -q -g test_typescript_target_core -t halt tests/core/test_typescript_target.pl

:- module(test_typescript_target_core, [test_typescript_target_core/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module('../../src/unifyweaver/targets/typescript_target').
:- use_module('../../src/unifyweaver/bindings/typescript_bindings').
:- use_module('../../src/unifyweaver/core/advanced/tree_recursion',
              [ compile_tree_recursion/3 ]).

test_typescript_target_core :-
    run_tests([typescript_target]).

:- begin_tests(typescript_target).

% Helpers: deterministic substring checks
has(Code, Substr)   :- once(sub_string(Code, _, _, _, Substr)).
hasnt(Code, Substr) :- \+ sub_string(Code, _, _, _, Substr).

% node availability gate (native TS type-stripping, node >= 22)
node_available :-
    catch(( process_create(path(node), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

% Write TS Code to a temp .ts file, run it under node --experimental-strip-types
% with Argv, return trimmed stdout as an atom.
ts_write_run(Code, Argv, Out) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.ts', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Code), close(W) ),
        ts_node_exec(File, Argv, Out),
        catch(delete_file(File), _, true)).

ts_node_exec(File, Argv, Out) :-
    append(['--experimental-strip-types', File], Argv, Args),
    process_create(path(node), Args,
                   [stdout(pipe(O)), stderr(null), process(P)]),
    read_string(O, _, Str), close(O), process_wait(P, _),
    normalize_space(atom(Out), Str).

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

% ============================================================================
% G-P1: DERIVED (non-canned) numeric multi-call recursion
%
% The tree/multicall/direct hooks previously emitted a HARDCODED
% `<pred>(n - 1) + <pred>(n - 2)` Fibonacci body for ANY predicate. These
% tests assert the emitted body is DERIVED from the actual clause: a non-fib
% recursion must NOT contain the fib-shaped body, and fib itself must produce
% exactly its own shape (proving derivation, not a hardcoded template).
% ============================================================================

% comb(N) = 2*comb(N-1) + comb(N-2), base comb(0)=1, comb(1)=1 — NOT fibonacci.
% Asserted with @/2 so the clause bodies live in `user` unwrapped (the mature
% multicall/direct pattern matchers partition on unqualified body goals).
assert_comb :-
    @(assertz(comb(0, 1)), user),
    @(assertz(comb(1, 1)), user),
    @(assertz((comb(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                             comb(N1, A), comb(N2, B), R is 2*A + B)), user).
retract_comb :- retractall(user:comb(_, _)).

% fib(N) = fib(N-1) + fib(N-2) — the genuine fib shape.
assert_tfib :-
    @(assertz(tfib(0, 0)), user),
    @(assertz(tfib(1, 1)), user),
    @(assertz((tfib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                             tfib(N1, A), tfib(N2, B), R is A + B)), user).
retract_tfib :- retractall(user:tfib(_, _)).

test(tree_derives_nonfib_body, [setup(assert_comb), cleanup(retract_comb)]) :-
    compile_tree_recursion(comb/2, [target(typescript)], Code),
    % base cases derived from the actual base clauses
    has(Code, "if (n === 0) return 1;"),
    has(Code, "if (n === 1) return 1;"),
    % recursive body derived (weight 2, real offsets), NOT the canned fib body
    has(Code, "const result = 2 * comb(n - 1) + comb(n - 2);"),
    hasnt(Code, "result = comb(n - 1) + comb(n - 2)").  % canned fib shape absent

test(tree_derives_fib_shape_when_it_is_fib, [setup(assert_tfib), cleanup(retract_tfib)]) :-
    compile_tree_recursion(tfib/2, [target(typescript)], Code),
    has(Code, "if (n === 0) return 0;"),
    has(Code, "if (n === 1) return 1;"),
    % fib genuinely IS this shape — derived from its clause, not hardcoded
    has(Code, "const result = tfib(n - 1) + tfib(n - 2);").

test(multicall_derives_nonfib_body, [setup(assert_comb), cleanup(retract_comb)]) :-
    multicall_linear_recursion:compile_multicall_linear_recursion(
        comb/2, [target(typescript)], Code),
    has(Code, "const result = 2 * comb(n - 1) + comb(n - 2);"),
    hasnt(Code, "result = comb(n - 1) + comb(n - 2)").

test(direct_multicall_derives_nonfib_body, [setup(assert_comb), cleanup(retract_comb)]) :-
    direct_multi_call_recursion:compile_direct_multi_call(
        comb/2, [target(typescript)], Code),
    has(Code, "const result = 2 * comb(n - 1) + comb(n - 2);"),
    hasnt(Code, "result = comb(n - 1) + comb(n - 2)").

% Shape-only fallback: with no user clauses to derive from, the hook still
% yields valid, memoized, fib-shaped code (backward compatible).
test(tree_fallback_when_no_clauses) :-
    tree_recursion:compile_tree_pattern(typescript, fib, nofib, 2, false, Code),
    has(Code, "const nofib = (n: number): number =>"),
    has(Code, "new Map<number, number>()"),
    has(Code, "nofib(n - 1) + nofib(n - 2)").

% Derived code runs under node and matches the SWI oracle.
test(tree_nonfib_runs_under_node,
     [setup(assert_comb), cleanup(retract_comb), condition(node_available)]) :-
    compile_tree_recursion(comb/2, [target(typescript)], Code),
    forall(member(N, [0,1,2,5,7]), (
        once(user:comb(N, Exp)),
        ts_write_run(Code, [N], Got),
        atom_number(Got, GotN),
        GotN =:= Exp
    )).

% ============================================================================
% G-P2: STRUCTURAL (list) recursion — real TS derived from list clauses
% ============================================================================

assert_cmem :-
    assertz(user:(cmem(X, [X|_]))),
    assertz((user:cmem(X, [_|T]) :- cmem(X, T))).
retract_cmem :- retractall(user:cmem(_, _)).

assert_capp :-
    assertz(user:(capp([], L, L))),
    assertz((user:capp([H|T], L, [H|R]) :- capp(T, L, R))).
retract_capp :- retractall(user:capp(_, _, _)).

assert_clen :-
    assertz(user:(clen([], 0))),
    assertz((user:clen([_|T], N) :- clen(T, N1), N is N1 + 1)).
retract_clen :- retractall(user:clen(_, _)).

test(structural_member_is_derived, [setup(assert_cmem), cleanup(retract_cmem)]) :-
    typescript_target:compile_predicate(cmem/2, [], Code),
    has(Code, "export function cmem(a1: any, a2: any[]): boolean"),
    % element-equality against the decomposed head, plus tail recursion
    has(Code, "a2[0] === a1"),
    has(Code, "return cmem(a1, a2.slice(1));"),
    has(Code, "return false;").

test(structural_append_is_derived, [setup(assert_capp), cleanup(retract_capp)]) :-
    typescript_target:compile_predicate(capp/3, [], Code),
    has(Code, "export function capp("),
    has(Code, "if (a1.length === 0)"),
    has(Code, "const _s0 = capp(a1.slice(1), a2);"),
    has(Code, "return [a1[0], ..._s0];").

test(structural_length_is_derived, [setup(assert_clen), cleanup(retract_clen)]) :-
    typescript_target:compile_predicate(clen/2, [], Code),
    has(Code, "export function clen(a1: any[])"),
    has(Code, "const _s0 = clen(a1.slice(1));"),
    has(Code, "_s0 + 1").

% Structural predicates run under node and match the SWI oracle.
test(structural_member_runs_under_node,
     [setup(assert_cmem), cleanup(retract_cmem), condition(node_available)]) :-
    typescript_target:compile_predicate(cmem/2, [], Code0),
    string_concat(Code0,
        "\nconst el = process.argv[2]; const lst = JSON.parse(process.argv[3]); console.log(cmem(el, lst));\n",
        Code),
    ts_write_run(Code, [a, '["a","b","c"]'], O1), O1 == 'true',
    ts_write_run(Code, [z, '["a","b","c"]'], O2), O2 == 'false'.

test(structural_length_runs_under_node,
     [setup(assert_clen), cleanup(retract_clen), condition(node_available)]) :-
    typescript_target:compile_predicate(clen/2, [], Code0),
    string_concat(Code0,
        "\nconsole.log(clen(JSON.parse(process.argv[2])));\n", Code),
    ts_write_run(Code, ['["x","y","z","w"]'], O),
    O == '4'.

:- end_tests(typescript_target).
