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
:- use_module(library(http/json)).
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

% ============================================================================
% G-P4: COMPONENT EMISSION — declared components are compiled INTO the module
%
% Before this fix, compile_module/3 collected declared components (via
% collect_declared_component/2) but never called
% component_registry:compile_component/4, so the component code was silently
% dropped. These tests assert the emitted module now CONTAINS the component
% output, and that a component-free module is unaffected (behavior-preserving).
% ============================================================================

% Declare + collect a raw-inject custom_typescript component and a custom_chart
% component, so compile_module has something to emit.
setup_component_emission :-
    typescript_target:init_typescript_target,   % clears collected_component/2
    component_registry:declare_component(source, ts_validator, custom_typescript,
        [ code("return input * 2;"),
          input_type('number'),
          output_type('number') ]),
    typescript_target:collect_declared_component(source, ts_validator),
    component_registry:declare_component(source, sine_chart, custom_chart,
        [ chart_type(line), title("Sine") ]),
    typescript_target:collect_declared_component(source, sine_chart).

cleanup_component_emission :-
    retractall(typescript_target:collected_component(_, _)),
    catch(component_registry:retract_component(source, ts_validator), _, true),
    catch(component_registry:retract_component(source, sine_chart), _, true).

test(component_emission_includes_declared,
     [setup(setup_component_emission), cleanup(cleanup_component_emission)]) :-
    typescript_target:compile_module(
        [pred(fact, 1, factorial)],
        [module_name('WithComponents')],
        Code),
    % predicate code still emitted alongside components
    has(Code, "export const fact"),
    % custom_typescript raw-inject component now present (previously dropped)
    has(Code, "export const ts_validator = ("),
    has(Code, "input: number"),
    has(Code, "return input * 2;"),
    % custom_chart component now present (previously orphaned + dropped)
    has(Code, "Custom Chart Component: sine_chart"),
    has(Code, "export function createsine_chartChart("),
    has(Code, "type: 'line'"),
    % title-display flag is a real boolean, not the old unevaluated term
    has(Code, "display: true"),
    hasnt(Code, "-> true;false").

% A module that declares NO components must be byte-for-byte unchanged:
% no component markers leak in, and the plain predicate output is intact.
test(component_free_module_unchanged,
     [setup(typescript_target:init_typescript_target)]) :-
    typescript_target:compile_module(
        [pred(fact, 1, factorial)],
        [module_name('NoComponents')],
        Code),
    has(Code, "// Module: NoComponents"),
    has(Code, "export const fact"),
    hasnt(Code, "Custom Chart Component"),
    hasnt(Code, "export const ts_validator").

% ============================================================================
% G-P3: AGGREGATE COMPILATION — aggregate_all/3 and findall/3 goals
%
% The TS pattern target previously compiled ZERO aggregate goals (vs go 36 /
% python 10 / rust 10). These tests assert that aggregate_all(count/sum/max/
% min/bag/set, ...) and findall/3 now lower to a self-contained TS reducer over
% the inner goal's solution set, whose result matches the SWI oracle when run
% under node.
% ============================================================================

assert_agg_parent :-
    assertz(user:parent(tom, bob)),
    assertz(user:parent(tom, liz)),
    assertz(user:parent(bob, ann)),
    assertz(user:parent(bob, pat)),
    assertz((user:count_children(P, N) :- aggregate_all(count, parent(P, _), N))),
    assertz((user:children_of(P, L)   :- findall(C, parent(P, C), L))),
    assertz((user:kids_bag(P, L)      :- aggregate_all(bag(C), parent(P, C), L))).
retract_agg_parent :-
    retractall(user:parent(_, _)),
    retractall(user:count_children(_, _)),
    retractall(user:children_of(_, _)),
    retractall(user:kids_bag(_, _)).

assert_agg_person :-
    assertz(user:person(alice, 30)),
    assertz(user:person(bob,   25)),
    assertz(user:person(carol, 40)),
    assertz((user:total_age(T)  :- aggregate_all(sum(A), person(_, A), T))),
    assertz((user:double_sum(T) :- aggregate_all(sum(W), (person(_, A), W is A * 2), T))),
    assertz((user:max_age(M)    :- aggregate_all(max(A), person(_, A), M))),
    assertz((user:min_age(M)    :- aggregate_all(min(A), person(_, A), M))).
retract_agg_person :-
    retractall(user:person(_, _)),
    retractall(user:total_age(_)),
    retractall(user:double_sum(_)),
    retractall(user:max_age(_)),
    retractall(user:min_age(_)).

assert_agg_color :-
    assertz(user:col(red)),
    assertz(user:col(blue)),
    assertz(user:col(red)),
    assertz(user:col(green)),
    assertz((user:uniq_colors(L) :- aggregate_all(set(X), col(X), L))).
retract_agg_color :-
    retractall(user:col(_)),
    retractall(user:uniq_colors(_)).

% -- Structural (non-node) recognition: the emitted TS is a real reducer -------

test(agg_count_lowers_to_reducer, [setup(assert_agg_parent), cleanup(retract_agg_parent)]) :-
    typescript_target:compile_predicate(count_children/2, [], Code),
    has(Code, "export function count_children(arg1: any): number"),
    has(Code, "for (const f of facts)"),
    has(Code, "acc += 1;"),
    % grouped by the bound head input
    has(Code, "String(f.arg1) === String(arg1)").

test(agg_sum_lowers_to_reducer, [setup(assert_agg_person), cleanup(retract_agg_person)]) :-
    typescript_target:compile_predicate(total_age/1, [], Code),
    has(Code, "export function total_age(): number"),
    has(Code, "acc += Number(f.arg2);").

test(agg_sum_with_arithmetic_expr, [setup(assert_agg_person), cleanup(retract_agg_person)]) :-
    typescript_target:compile_predicate(double_sum/1, [], Code),
    % the `W is A * 2` post-goal is lowered into the loop body
    has(Code, "const v1 = (f.arg2 * 2);"),
    has(Code, "acc += Number(v1);").

test(agg_findall_lowers_to_array, [setup(assert_agg_parent), cleanup(retract_agg_parent)]) :-
    typescript_target:compile_predicate(children_of/2, [], Code),
    has(Code, "export function children_of(arg1: any): any[]"),
    has(Code, "acc.push(f.arg2);").

% -- Node execution vs SWI oracle ---------------------------------------------

test(agg_count_runs_under_node,
     [setup(assert_agg_parent), cleanup(retract_agg_parent), condition(node_available)]) :-
    typescript_target:compile_predicate(count_children/2, [], Code),
    forall(member(P, [tom, bob]), (
        aggregate_all(count, user:parent(P, _), Exp),
        ts_write_run(Code, [P], Got),
        atom_number(Got, GotN),
        GotN =:= Exp
    )).

test(agg_sum_runs_under_node,
     [setup(assert_agg_person), cleanup(retract_agg_person), condition(node_available)]) :-
    typescript_target:compile_predicate(total_age/1, [], Code),
    aggregate_all(sum(A), user:person(_, A), Exp),
    ts_write_run(Code, [], Got),
    atom_number(Got, GotN),
    GotN =:= Exp.

test(agg_sum_arith_runs_under_node,
     [setup(assert_agg_person), cleanup(retract_agg_person), condition(node_available)]) :-
    typescript_target:compile_predicate(double_sum/1, [], Code),
    aggregate_all(sum(W), (user:person(_, A), W is A * 2), Exp),
    ts_write_run(Code, [], Got),
    atom_number(Got, GotN),
    GotN =:= Exp.

test(agg_max_runs_under_node,
     [setup(assert_agg_person), cleanup(retract_agg_person), condition(node_available)]) :-
    typescript_target:compile_predicate(max_age/1, [], Code),
    aggregate_all(max(A), user:person(_, A), Exp),
    ts_write_run(Code, [], Got),
    atom_number(Got, GotN),
    GotN =:= Exp.

test(agg_min_runs_under_node,
     [setup(assert_agg_person), cleanup(retract_agg_person), condition(node_available)]) :-
    typescript_target:compile_predicate(min_age/1, [], Code),
    aggregate_all(min(A), user:person(_, A), Exp),
    ts_write_run(Code, [], Got),
    atom_number(Got, GotN),
    GotN =:= Exp.

test(agg_bag_runs_under_node,
     [setup(assert_agg_parent), cleanup(retract_agg_parent), condition(node_available)]) :-
    typescript_target:compile_predicate(kids_bag/2, [], Code),
    aggregate_all(bag(C), user:parent(bob, C), Exp),
    ts_write_run(Code, [bob], Got),
    atom_json_term(Got, GotJson, []),
    GotJson == Exp.

test(agg_set_runs_under_node,
     [setup(assert_agg_color), cleanup(retract_agg_color), condition(node_available)]) :-
    typescript_target:compile_predicate(uniq_colors/1, [], Code),
    aggregate_all(set(X), user:col(X), Exp),
    ts_write_run(Code, [], Got),
    atom_json_term(Got, GotJson, []),
    GotJson == Exp.

test(agg_findall_runs_under_node,
     [setup(assert_agg_parent), cleanup(retract_agg_parent), condition(node_available)]) :-
    typescript_target:compile_predicate(children_of/2, [], Code),
    findall(C, user:parent(tom, C), Exp),
    ts_write_run(Code, [tom], Got),
    atom_json_term(Got, GotJson, []),
    GotJson == Exp.

% ============================================================================
% G-P8: STREAMING / GENERATOR EMIT MODE
%
% The TS pattern target previously had NO streaming/pipeline/generator mode
% (0 refs vs python ~432 / go ~390 / rust ~325). These tests assert that a
% single-clause filter/transform predicate now compiles, under mode(generator)/
% mode(pipeline) (and the clojure-style aliases generator_mode/pipeline_input),
% to a TS program that reads stdin line-by-line via Node's built-in `readline`,
% applies the predicate, and streams results to stdout. Default (non-streaming)
% compiles must be unchanged.
% ============================================================================

% Pipe Input (an atom/string, newline-separated) to a compiled streaming TS
% program under node, returning the non-empty stdout lines as a list of strings.
ts_write_run_stdin(Code, Input, Lines) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.ts', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Code), close(W) ),
        ts_node_exec_stdin(File, Input, Lines),
        catch(delete_file(File), _, true)).

ts_node_exec_stdin(File, Input, Lines) :-
    process_create(path(node), ['--experimental-strip-types', File],
                   [stdin(pipe(In)), stdout(pipe(O)), stderr(null), process(P)]),
    format(In, '~w', [Input]), close(In),
    read_string(O, _, Str), close(O), process_wait(P, _),
    split_string(Str, "\n", " \t\r", Parts),
    exclude(==(""), Parts, Lines).

assert_stream_big :-
    assertz((user:big(X) :- X > 5)).
retract_stream_big :- retractall(user:big(_)).

assert_stream_posdoub :-
    assertz((user:posdoub(X, Y) :- X > 0, Y is X * 2)).
retract_stream_posdoub :- retractall(user:posdoub(_, _)).

assert_stream_doub :-
    assertz((user:doub(X, Y) :- Y is X * 2)).
retract_stream_doub :- retractall(user:doub(_, _)).

% -- Structural recognition ---------------------------------------------------

test(stream_filter_pipeline_shape,
     [setup(assert_stream_big), cleanup(retract_stream_big)]) :-
    typescript_target:compile_predicate(big/1, [mode(pipeline)], Code),
    has(Code, "Streaming (pipeline mode)"),
    has(Code, "import { createInterface } from \"node:readline\""),
    has(Code, "export function bigTest(x: number): boolean"),
    has(Code, "return (x > 5);"),
    has(Code, "createInterface({ input: process.stdin"),
    has(Code, "if (bigTest(x)) {"),
    % pipeline mode passes the original line through
    has(Code, "console.log(trimmed);").

test(stream_filter_generator_emits_value,
     [setup(assert_stream_big), cleanup(retract_stream_big)]) :-
    typescript_target:compile_predicate(big/1, [mode(generator)], Code),
    has(Code, "Streaming (generator mode)"),
    % generator mode emits the derived numeric value
    has(Code, "console.log(String(x));"),
    hasnt(Code, "console.log(trimmed);").

test(stream_transform_generator_shape,
     [setup(assert_stream_posdoub), cleanup(retract_stream_posdoub)]) :-
    typescript_target:compile_predicate(posdoub/2, [mode(generator)], Code),
    % transform yields an array of 0+ results (rewrite-safe `number[]`)
    has(Code, "export function posdoubTransform(x: number): number[]"),
    % guard lowered into an early empty-array return (drops the record)
    has(Code, "if (!(x > 0)) return [];"),
    has(Code, "return [(x * 2)];"),
    has(Code, "for (const result of posdoubTransform(x)) {").

test(stream_transform_unguarded_has_no_guard_line,
     [setup(assert_stream_doub), cleanup(retract_stream_doub)]) :-
    typescript_target:compile_predicate(doub/2, [mode(pipeline)], Code),
    has(Code, "export function doubTransform(x: number): number[]"),
    has(Code, "return [(x * 2)];"),
    hasnt(Code, "return [];").

% clojure-style option aliases select the same modes
test(stream_alias_generator_mode,
     [setup(assert_stream_doub), cleanup(retract_stream_doub)]) :-
    typescript_target:compile_predicate(doub/2, [generator_mode(true)], Code),
    has(Code, "Streaming (generator mode)"),
    has(Code, "doubTransform").

test(stream_alias_pipeline_input,
     [setup(assert_stream_big), cleanup(retract_stream_big)]) :-
    typescript_target:compile_predicate(big/1, [pipeline_input(true)], Code),
    has(Code, "Streaming (pipeline mode)"),
    has(Code, "bigTest").

% Default (no streaming option) is UNCHANGED: batch native-clause lowering,
% never the streaming readline scaffold.
test(stream_default_compile_unchanged,
     [setup(assert_stream_doub), cleanup(retract_stream_doub)]) :-
    typescript_target:compile_predicate(doub/2, [], Code),
    has(Code, "Native Clause Lowering"),
    hasnt(Code, "readline"),
    hasnt(Code, "Streaming (").

% Multi-clause / non-qualifying predicate falls back to batch even WITH a
% streaming option (behavior-preserving: streaming never emits wrong code).
test(stream_multiclause_falls_back_to_batch,
     [setup(assert_stream_multi), cleanup(retract_stream_multi)]) :-
    typescript_target:compile_predicate(sgn/2, [mode(generator)], Code),
    hasnt(Code, "readline"),
    hasnt(Code, "Streaming (").

assert_stream_multi :-
    assertz((user:sgn(X, 1)  :- X > 0)),
    assertz((user:sgn(X, -1) :- X < 0)).
retract_stream_multi :- retractall(user:sgn(_, _)).

% -- Node execution vs SWI oracle ---------------------------------------------

test(stream_filter_pipeline_runs_under_node,
     [setup(assert_stream_big), cleanup(retract_stream_big), condition(node_available)]) :-
    typescript_target:compile_predicate(big/1, [mode(pipeline)], Code),
    Input = "3\n7\n10\n2\n6\n",
    Xs = [3,7,10,2,6],
    ts_write_run_stdin(Code, Input, GotLines),
    % SWI oracle over the SAME input: lines that satisfy big/1, passed through
    findall(S, (member(X, Xs), user:big(X), number_string(X, S)), ExpLines),
    GotLines == ExpLines.

test(stream_transform_generator_runs_under_node,
     [setup(assert_stream_posdoub), cleanup(retract_stream_posdoub), condition(node_available)]) :-
    typescript_target:compile_predicate(posdoub/2, [mode(generator)], Code),
    Input = "-3\n4\n0\n5\n8\n",
    Xs = [-3,4,0,5,8],
    ts_write_run_stdin(Code, Input, GotLines),
    % SWI oracle: derived Y=X*2 for each X>0, in stream order
    findall(S, (member(X, Xs), user:posdoub(X, Y), number_string(Y, S)), ExpLines),
    GotLines == ExpLines.

% ============================================================================
% G-P7: NEGATION + TYPE-CHECK GUARD CODEGEN
%
% ts_guard_condition/3 previously handled ONLY binary comparisons. Guards that
% the shared clause_body_analysis classifier routes to the guard renderer —
% negation (\+/not) and type-check predicates (integer/1, atom/1, is_list/1,
% ...) — had NO clause and FAILED at render, aborting compilation. These tests
% assert the two new clause families render correctly and run under node
% matching the SWI oracle.
% ============================================================================

assert_gp7_qpos :- assertz((user:qpos(X) :- integer(X), X > 0)).
retract_gp7_qpos :- retractall(user:qpos(_)).

assert_gp7_pnm :- assertz((user:pnm(X) :- \+ member(X, [1,2,3]))).
retract_gp7_pnm :- retractall(user:pnm(_)).

assert_gp7_pgt :- assertz((user:pgt(X) :- \+ (X > 5))).
retract_gp7_pgt :- retractall(user:pgt(_)).

% -- Structural: the guards now render (batch path no longer fails) -----------

test(gp7_typecheck_renders_in_batch,
     [setup(assert_gp7_qpos), cleanup(retract_gp7_qpos)]) :-
    % integer/1 → Number.isInteger; comparison guard still works alongside it
    typescript_target:compile_predicate(qpos/1, [], Code),
    has(Code, "Number.isInteger(arg1)"),
    has(Code, "arg1 > 0").

test(gp7_negation_member_renders_in_batch,
     [setup(assert_gp7_pnm), cleanup(retract_gp7_pnm)]) :-
    % \+ member(X, [1,2,3]) → !( [1, 2, 3].includes(arg1) )
    typescript_target:compile_predicate(pnm/1, [], Code),
    has(Code, "!([1, 2, 3].includes(arg1))").

test(gp7_negation_comparison_renders_in_batch,
     [setup(assert_gp7_pgt), cleanup(retract_gp7_pgt)]) :-
    % \+ (X > 5) → !(arg1 > 5)
    typescript_target:compile_predicate(pgt/1, [], Code),
    has(Code, "!(arg1 > 5)").

% -- Pipeline (filter) shape: clean boolean function over the guards ----------

test(gp7_typecheck_pipeline_shape,
     [setup(assert_gp7_qpos), cleanup(retract_gp7_qpos)]) :-
    typescript_target:compile_predicate(qpos/1, [mode(pipeline)], Code),
    has(Code, "export function qposTest(x: number): boolean"),
    has(Code, "return (Number.isInteger(x) && x > 0);").

test(gp7_negation_member_pipeline_shape,
     [setup(assert_gp7_pnm), cleanup(retract_gp7_pnm)]) :-
    typescript_target:compile_predicate(pnm/1, [mode(pipeline)], Code),
    has(Code, "return (!([1, 2, 3].includes(x)));").

% -- Node execution vs SWI oracle --------------------------------------------

test(gp7_typecheck_runs_under_node,
     [setup(assert_gp7_qpos), cleanup(retract_gp7_qpos), condition(node_available)]) :-
    typescript_target:compile_predicate(qpos/1, [mode(pipeline)], Code),
    Input = "3\n-1\n7\n0\n", Xs = [3,-1,7,0],
    ts_write_run_stdin(Code, Input, GotLines),
    findall(S, (member(X, Xs), user:qpos(X), number_string(X, S)), ExpLines),
    GotLines == ExpLines.

test(gp7_negation_member_runs_under_node,
     [setup(assert_gp7_pnm), cleanup(retract_gp7_pnm), condition(node_available)]) :-
    typescript_target:compile_predicate(pnm/1, [mode(pipeline)], Code),
    Input = "1\n2\n4\n7\n", Xs = [1,2,4,7],
    ts_write_run_stdin(Code, Input, GotLines),
    findall(S, (member(X, Xs), user:pnm(X), number_string(X, S)), ExpLines),
    GotLines == ExpLines.

test(gp7_negation_comparison_runs_under_node,
     [setup(assert_gp7_pgt), cleanup(retract_gp7_pgt), condition(node_available)]) :-
    typescript_target:compile_predicate(pgt/1, [mode(generator)], Code),
    Input = "3\n5\n6\n10\n", Xs = [3,5,6,10],
    ts_write_run_stdin(Code, Input, GotLines),
    findall(S, (member(X, Xs), user:pgt(X), number_string(X, S)), ExpLines),
    GotLines == ExpLines.

% ============================================================================
% G-P7 follow-up: REGEX match/2[,3] GUARD CODEGEN
%
% ts_guard_condition/3 now renders match/2 and match/3 (UnifyWeaver's regex-match
% predicate: subject FIRST, pattern SECOND, optional 3rd = regex type) as
% `new RegExp("<pattern>").test(x)`. Boolean truthiness mirrors Python's
% unanchored re.search. The arity-1 streaming filter reads its input line as a
% string when the body match-tests it, so text lines flow correctly under node.
% \+ match(...) composes via the existing negation clause. Oracle: library(pcre)
% re_match/2 (match/2 is a compile-time DSL marker, not executable in SWI).
% ============================================================================

:- use_module(library(pcre)).

assert_gp7m_starts_a  :- assertz((user:starts_a(X)  :- match(X, '^a'))).
retract_gp7m_starts_a :- retractall(user:starts_a(_)).

assert_gp7m_not_a  :- assertz((user:not_a(X)  :- \+ match(X, '^a'))).
retract_gp7m_not_a :- retractall(user:not_a(_)).

assert_gp7m_digit  :- assertz((user:has_digit(X) :- match(X, '\\d+'))).
retract_gp7m_digit :- retractall(user:has_digit(_)).

assert_gp7m_typed  :- assertz((user:typed(X, R) :- (match(X, 'foo', pcre) -> R = 'yes' ; R = 'no'))).
retract_gp7m_typed :- retractall(user:typed(_, _)).

% -- Structural: match renders as a RegExp test (batch had no RegExp before) ---

test(gp7m_match_renders_in_pipeline,
     [setup(assert_gp7m_starts_a), cleanup(retract_gp7m_starts_a)]) :-
    typescript_target:compile_predicate(starts_a/1, [mode(pipeline)], Code),
    % match(X, '^a') → new RegExp("^a").test(x); input line typed as string
    has(Code, "new RegExp(\"^a\").test(x)"),
    has(Code, "export function starts_aTest(x: string): boolean"),
    has(Code, "const x = trimmed;").

test(gp7m_negation_match_renders_in_pipeline,
     [setup(assert_gp7m_not_a), cleanup(retract_gp7m_not_a)]) :-
    % \+ match(X, '^a') → !(new RegExp("^a").test(x)) (negation composes)
    typescript_target:compile_predicate(not_a/1, [mode(pipeline)], Code),
    has(Code, "!(new RegExp(\"^a\").test(x))").

test(gp7m_match_backslash_escaped,
     [setup(assert_gp7m_digit), cleanup(retract_gp7m_digit)]) :-
    % '\d+' must survive as the JS regex \d+  →  string literal "\\d+" (two
    % literal backslashes in the emitted code = four in this Prolog literal)
    typescript_target:compile_predicate(has_digit/1, [mode(pipeline)], Code),
    has(Code, "new RegExp(\"\\\\d+\").test(x)").

test(gp7m_match3_renders,
     [setup(assert_gp7m_typed), cleanup(retract_gp7m_typed)]) :-
    % match/3 (regex type advisory) renders the same RegExp test, here inside
    % a batch if-then-else guard position (composes with control flow)
    typescript_target:compile_predicate(typed/2, [], Code),
    has(Code, "new RegExp(\"foo\").test(").

% -- Node execution vs pcre oracle -------------------------------------------

test(gp7m_match_runs_under_node,
     [setup(assert_gp7m_starts_a), cleanup(retract_gp7m_starts_a), condition(node_available)]) :-
    typescript_target:compile_predicate(starts_a/1, [mode(pipeline)], Code),
    Input = "apple\nbanana\navocado\ncherry\n",
    Lines = ["apple","banana","avocado","cherry"],
    ts_write_run_stdin(Code, Input, GotLines),
    findall(L, (member(L, Lines), re_match("^a", L)), ExpLines),
    GotLines == ExpLines.

test(gp7m_negation_match_runs_under_node,
     [setup(assert_gp7m_not_a), cleanup(retract_gp7m_not_a), condition(node_available)]) :-
    typescript_target:compile_predicate(not_a/1, [mode(pipeline)], Code),
    Input = "apple\nbanana\navocado\ncherry\n",
    Lines = ["apple","banana","avocado","cherry"],
    ts_write_run_stdin(Code, Input, GotLines),
    findall(L, (member(L, Lines), \+ re_match("^a", L)), ExpLines),
    GotLines == ExpLines.

% ============================================================================
% G-P-dedup: UNIQUENESS / ORDER CONSTRAINT HANDLING
%
% compile_facts previously emitted the raw fact array regardless of any
% declared unique/unordered constraint, so a predicate declared unique still
% carried duplicate rows. It now consults constraint_analyzer:get_constraints/2
% (declaration merged over the global defaults unique=true/unordered=true) and,
% mirroring the rust/go dedup semantics, wraps the fact array:
%   - unique(false)           -> raw array (no dedup)
%   - unique(true), ordered   -> order-preserving dedup (Set over JSON keys)
%   - unique(true), unordered -> dedup + sort (sort-based dedup)
% queryX/isX read that array, so the whole facts surface inherits it.
% Oracle: SWI setof/sort over the same clauses.
% ============================================================================

:- use_module('../../src/unifyweaver/core/constraint_analyzer',
              [ declare_constraint/2, clear_constraints/1 ]).

% Facts with a genuine duplicate (dup(a,b) asserted twice), in the
% typescript_target module (compile_facts gathers via a bare call/1 there).
assert_dup_facts :-
    assertz(typescript_target:dup(a, b)),
    assertz(typescript_target:dup(a, b)),
    assertz(typescript_target:dup(c, d)).
retract_dup_facts :-
    retractall(typescript_target:dup(_, _)),
    clear_constraints(dup/2).

% -- Structural recognition ---------------------------------------------------

% Default (no declaration) => unique(true), unordered(true) => sort-based dedup.
test(dedup_default_emits_sort_dedup,
     [setup(assert_dup_facts), cleanup(retract_dup_facts)]) :-
    typescript_target:compile_facts(dup, 2, Code),
    % Set-over-JSON-keys dedup + .sort() (sort-based), reading back with JSON.parse
    has(Code, "new Set("),
    has(Code, "JSON.stringify(f)"),
    has(Code, ".sort()"),
    has(Code, "JSON.parse(s)").

% unique(false) => raw array, byte-for-byte the historical shape (no dedup expr).
test(dedup_unique_false_leaves_raw_array,
     [setup(assert_dup_facts), cleanup(retract_dup_facts)]) :-
    declare_constraint(dup/2, [unique(false)]),
    typescript_target:compile_facts(dup, 2, Code),
    has(Code, "export const dupFacts: DupFact[] = ["),
    hasnt(Code, "new Set("),
    hasnt(Code, "JSON.stringify").

% unique(true), ordered => order-preserving dedup, NO sort.
test(dedup_ordered_emits_order_preserving,
     [setup(assert_dup_facts), cleanup(retract_dup_facts)]) :-
    declare_constraint(dup/2, [unique(true), ordered]),
    typescript_target:compile_facts(dup, 2, Code),
    has(Code, "a.indexOf(s) === i"),
    hasnt(Code, ".sort()").

% -- Node execution vs SWI oracle ---------------------------------------------

% Default: duplicates removed. Compare the emitted set against SWI setof.
test(dedup_default_runs_under_node,
     [setup(assert_dup_facts), cleanup(retract_dup_facts), condition(node_available)]) :-
    typescript_target:compile_facts(dup, 2, Code0),
    string_concat(Code0,
        "\nconsole.log(JSON.stringify(dupFacts.map(f => [f.arg1, f.arg2])));\n",
        Code),
    ts_write_run(Code, [], Got),
    atom_json_term(Got, GotJson, []),   % JSON strings decode to atoms
    % SWI oracle: the DISTINCT (a,b) pairs, sorted (setof gives sorted, no dups)
    setof([A, B], typescript_target:dup(A, B), Pairs),
    GotJson == Pairs.

% unique(false): duplicates retained (the raw multiset, in assertion order).
test(dedup_unique_false_runs_under_node,
     [setup(assert_dup_facts), cleanup(retract_dup_facts), condition(node_available)]) :-
    declare_constraint(dup/2, [unique(false)]),
    typescript_target:compile_facts(dup, 2, Code0),
    string_concat(Code0,
        "\nconsole.log(dupFacts.length);\n", Code),
    ts_write_run(Code, [], Got),
    atom_number(Got, N),
    % three asserted facts, one a duplicate -> raw length is 3
    N =:= 3.

:- end_tests(typescript_target).
