:- module(test_prolog_target, [run_prolog_target_tests/0]).

:- use_module(library(plunit)).
:- use_module(library(gensym)).
:- use_module('../../src/unifyweaver/core/constraint_analyzer').
:- use_module('../../src/unifyweaver/targets/prolog_target').

run_prolog_target_tests :-
    run_tests([prolog_target]).

:- begin_tests(prolog_target).

setup_branch_pruning_fixture :-
    cleanup_branch_pruning_fixture,
    assertz(user:test_ppv_edge(a, b)),
    assertz(user:test_ppv_edge(b, c)),
    assertz(user:test_ppv_edge(b, blocked)),
    assertz(user:(test_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_ppv_edge(Cat, Ancestor),
        \+ member(Ancestor, Visited),
        Hops = 1)),
    assertz(user:(test_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_ppv_edge(Cat, Mid),
        Mid \= blocked,
        \+ member(Mid, Visited),
        test_ppv_reach(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)),
    assertz(user:mode(test_ppv_reach(-, +, -, +))).

cleanup_branch_pruning_fixture :-
    retractall(user:test_ppv_edge(_, _)),
    retractall(user:test_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_ppv_reach(_, _, _, _))).

setup_no_mode_fixture :-
    setup_branch_pruning_fixture,
    retractall(user:mode(test_ppv_reach(_, _, _, _))).

setup_execution_branch_pruning_fixture :-
    cleanup_execution_branch_pruning_fixture,
    assertz(user:test_exec_edge(a, b)),
    assertz(user:test_exec_edge(b, c)),
    assertz(user:test_exec_edge(a, d)),
    assertz(user:test_exec_edge(d, e)),
    assertz(user:test_exec_edge(e, c)),
    assertz(user:(test_exec_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_exec_edge(Cat, Ancestor),
        \+ member(Ancestor, Visited),
        Hops = 1)),
    assertz(user:(test_exec_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_exec_edge(Cat, Mid),
        \+ member(Mid, Visited),
        test_exec_ppv_reach(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)),
    assertz(user:mode(test_exec_ppv_reach(-, +, -, +))).

cleanup_execution_branch_pruning_fixture :-
    retractall(user:test_exec_edge(_, _)),
    retractall(user:test_exec_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_exec_ppv_reach(_, _, _, _))).

setup_noncanonical_step_fixture :-
    cleanup_noncanonical_step_fixture,
    assertz(user:test_noncanonical_guard(a, warmed)),
    assertz(user:test_noncanonical_guard(b, warmed)),
    assertz(user:test_noncanonical_edge(a, b)),
    assertz(user:test_noncanonical_edge(b, c)),
    assertz(user:(test_noncanonical_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_noncanonical_edge(Cat, Ancestor),
        \+ member(Ancestor, Visited),
        Hops = 1)),
    assertz(user:(test_noncanonical_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_noncanonical_guard(Cat, warmed),
        test_noncanonical_edge(Cat, Mid),
        \+ member(Mid, Visited),
        test_noncanonical_ppv_reach(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)),
    assertz(user:mode(test_noncanonical_ppv_reach(-, +, -, +))).

cleanup_noncanonical_step_fixture :-
    retractall(user:test_noncanonical_guard(_, _)),
    retractall(user:test_noncanonical_edge(_, _)),
    retractall(user:test_noncanonical_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_noncanonical_ppv_reach(_, _, _, _))).

setup_min_closure_fixture :-
    cleanup_min_closure_fixture,
    assertz(user:test_min_edge(a, b)),
    assertz(user:test_min_edge(b, c)),
    assertz(user:test_min_edge(a, d)),
    assertz(user:test_min_edge(d, e)),
    assertz(user:test_min_edge(e, c)),
    assertz(user:max_depth(4)),
    assertz(user:(test_min_ppv_reach(Cat, Parent, 1, Visited) :-
        test_min_edge(Cat, Parent),
        \+ member(Parent, Visited))),
    assertz(user:(test_min_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        max_depth(MaxD),
        length(Visited, Depth), Depth < MaxD, !,
        test_min_edge(Cat, Mid),
        \+ member(Mid, Visited),
        test_min_ppv_reach(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)),
    assertz(user:mode(test_min_ppv_reach(-, +, -, +))).

cleanup_min_closure_fixture :-
    retractall(user:test_min_edge(_, _)),
    retractall(user:max_depth(_)),
    retractall(user:test_min_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_min_ppv_reach(_, _, _, _))).

setup_weighted_min_closure_fixture :-
    cleanup_weighted_min_closure_fixture,
    assertz(user:test_weighted_min_edge(a, b)),
    assertz(user:test_weighted_min_edge(a, c)),
    assertz(user:test_weighted_min_edge(b, d)),
    assertz(user:test_weighted_min_edge(c, d)),
    assertz(user:test_weighted_min_cost(a, 1)),
    assertz(user:test_weighted_min_cost(b, 5)),
    assertz(user:test_weighted_min_cost(c, 2)),
    assertz(user:max_depth(4)),
    assertz(user:(test_weighted_min_ppv_reach(Cat, Parent, Cost, Visited) :-
        test_weighted_min_edge(Cat, Parent),
        test_weighted_min_cost(Cat, Step),
        Step > 0,
        Cost is Step,
        \+ member(Parent, Visited))),
    assertz(user:(test_weighted_min_ppv_reach(Cat, Ancestor, Cost, Visited) :-
        max_depth(MaxD),
        length(Visited, Depth), Depth < MaxD, !,
        test_weighted_min_edge(Cat, Mid),
        test_weighted_min_cost(Cat, Step),
        Step > 0,
        \+ member(Mid, Visited),
        test_weighted_min_ppv_reach(Mid, Ancestor, PrevCost, [Mid|Visited]),
        Cost is PrevCost + Step)),
    assertz(user:mode(test_weighted_min_ppv_reach(-, +, -, +))).

cleanup_weighted_min_closure_fixture :-
    retractall(user:test_weighted_min_edge(_, _)),
    retractall(user:test_weighted_min_cost(_, _)),
    retractall(user:max_depth(_)),
    retractall(user:test_weighted_min_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_weighted_min_ppv_reach(_, _, _, _))),
    clear_constraints(test_weighted_min_ppv_reach/4).

setup_weighted_min_metadata_fixture :-
    cleanup_weighted_min_closure_fixture,
    assertz(user:test_weighted_min_edge(a, b)),
    assertz(user:test_weighted_min_edge(a, c)),
    assertz(user:test_weighted_min_edge(b, d)),
    assertz(user:test_weighted_min_edge(c, d)),
    assertz(user:test_weighted_min_cost(a, 1)),
    assertz(user:test_weighted_min_cost(b, 5)),
    assertz(user:test_weighted_min_cost(c, 2)),
    assertz(user:max_depth(4)),
    assertz(user:(test_weighted_min_ppv_reach(Cat, Parent, Cost, Visited) :-
        test_weighted_min_edge(Cat, Parent),
        test_weighted_min_cost(Cat, Step),
        Cost is Step,
        \+ member(Parent, Visited))),
    assertz(user:(test_weighted_min_ppv_reach(Cat, Ancestor, Cost, Visited) :-
        max_depth(MaxD),
        length(Visited, Depth), Depth < MaxD, !,
        test_weighted_min_edge(Cat, Mid),
        test_weighted_min_cost(Cat, Step),
        \+ member(Mid, Visited),
        test_weighted_min_ppv_reach(Mid, Ancestor, PrevCost, [Mid|Visited]),
        Cost is PrevCost + Step)),
    assertz(user:mode(test_weighted_min_ppv_reach(-, +, -, +))),
    declare_constraint(test_weighted_min_ppv_reach/4, [positive_step(3)]).

setup_weighted_min_unproven_fixture :-
    cleanup_weighted_min_closure_fixture,
    assertz(user:test_weighted_min_edge(a, b)),
    assertz(user:test_weighted_min_edge(a, c)),
    assertz(user:test_weighted_min_edge(b, d)),
    assertz(user:test_weighted_min_edge(c, d)),
    assertz(user:test_weighted_min_cost(a, 1)),
    assertz(user:test_weighted_min_cost(b, 5)),
    assertz(user:test_weighted_min_cost(c, 2)),
    assertz(user:max_depth(4)),
    assertz(user:(test_weighted_min_ppv_reach(Cat, Parent, Cost, Visited) :-
        test_weighted_min_edge(Cat, Parent),
        test_weighted_min_cost(Cat, Step),
        Cost is Step,
        \+ member(Parent, Visited))),
    assertz(user:(test_weighted_min_ppv_reach(Cat, Ancestor, Cost, Visited) :-
        max_depth(MaxD),
        length(Visited, Depth), Depth < MaxD, !,
        test_weighted_min_edge(Cat, Mid),
        test_weighted_min_cost(Cat, Step),
        \+ member(Mid, Visited),
        test_weighted_min_ppv_reach(Mid, Ancestor, PrevCost, [Mid|Visited]),
        Cost is PrevCost + Step)),
    assertz(user:mode(test_weighted_min_ppv_reach(-, +, -, +))).

setup_effective_distance_accumulation_fixture :-
    cleanup_effective_distance_accumulation_fixture,
    assertz(user:test_effective_edge(a, b)),
    assertz(user:test_effective_edge(b, c)),
    assertz(user:test_effective_edge(a, d)),
    assertz(user:test_effective_edge(d, e)),
    assertz(user:test_effective_edge(e, c)),
    assertz(user:dimension_n(5)),
    assertz(user:max_depth(4)),
    assertz(user:(test_effective_ppv_reach(Cat, Parent, 1, Visited) :-
        test_effective_edge(Cat, Parent),
        \+ member(Parent, Visited))),
    assertz(user:(test_effective_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        max_depth(MaxD),
        length(Visited, Depth), Depth < MaxD, !,
        test_effective_edge(Cat, Mid),
        \+ member(Mid, Visited),
        test_effective_ppv_reach(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)),
    assertz(user:mode(test_effective_ppv_reach(-, +, -, +))).

cleanup_effective_distance_accumulation_fixture :-
    retractall(user:test_effective_edge(_, _)),
    retractall(user:dimension_n(_)),
    retractall(user:max_depth(_)),
    retractall(user:test_effective_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_effective_ppv_reach(_, _, _, _))).

build_execution_runtime_source(ModuleName, PredicateCode, Source) :-
    atomic_list_concat([
        'test_exec_edge(a, b).',
        'test_exec_edge(b, c).',
        'test_exec_edge(a, d).',
        'test_exec_edge(d, e).',
        'test_exec_edge(e, c).'
    ], '\n', FactsCode),
    format(atom(Source),
        ':- module(~q, [test_exec_ppv_reach/4]).~n:- use_module(library(lists)).~n~w~n~n~w~n',
        [ModuleName, FactsCode, PredicateCode]).

write_temp_module_source(Source, Path) :-
    tmp_file_stream(text, Path, Stream),
    write(Stream, Source),
    nl(Stream),
    close(Stream).

cleanup_temp_module_source(Path) :-
    catch(unload_file(Path), _, true),
    catch(delete_file(Path), _, true).

collect_generated_execution_hops(Options, PredicateCode, Hops) :-
    once(prolog_target:generate_predicate_code(test_exec_ppv_reach/4, Options, PredicateCode)),
    gensym(test_exec_ppv_runtime_, ModuleName),
    build_execution_runtime_source(ModuleName, PredicateCode, Source),
    write_temp_module_source(Source, Path),
    setup_call_cleanup(
        load_files(Path, []),
        (
            Goal =.. [test_exec_ppv_reach, a, c, H, [a]],
            findall(H, ModuleName:Goal, Hops0),
            sort(Hops0, Hops)
        ),
        cleanup_temp_module_source(Path)
    ).

build_min_runtime_source(ModuleName, PredicateCode, Source) :-
    atomic_list_concat([
        'test_min_edge(a, b).',
        'test_min_edge(b, c).',
        'test_min_edge(a, d).',
        'test_min_edge(d, e).',
        'test_min_edge(e, c).',
        'max_depth(4).'
    ], '\n', FactsCode),
    format(atom(Source),
        ':- module(~q, []).~n:- use_module(library(lists)).~n~w~n~n~w~n',
        [ModuleName, FactsCode, PredicateCode]).

collect_generated_min_hops(Options, PredicateCode, Hops) :-
    once(prolog_target:generate_predicate_code(test_min_ppv_reach/4, Options, PredicateCode)),
    gensym(test_min_ppv_runtime_, ModuleName),
    build_min_runtime_source(ModuleName, PredicateCode, Source),
    write_temp_module_source(Source, Path),
    setup_call_cleanup(
        load_files(Path, []),
        (
            Goal =.. ['test_min_ppv_reach$min', a, c, H],
            findall(H, ModuleName:Goal, Hops0),
            sort(Hops0, Hops)
        ),
        cleanup_temp_module_source(Path)
    ).

build_weighted_min_runtime_source(ModuleName, PredicateCode, Source) :-
    atomic_list_concat([
        'test_weighted_min_edge(a, b).',
        'test_weighted_min_edge(a, c).',
        'test_weighted_min_edge(b, d).',
        'test_weighted_min_edge(c, d).',
        'test_weighted_min_cost(a, 1).',
        'test_weighted_min_cost(b, 5).',
        'test_weighted_min_cost(c, 2).',
        'max_depth(4).'
    ], '\n', FactsCode),
    format(atom(Source),
        ':- module(~q, []).~n:- use_module(library(lists)).~n~w~n~n~w~n',
        [ModuleName, FactsCode, PredicateCode]).

collect_generated_weighted_min_costs(Options, PredicateCode, Costs) :-
    once(prolog_target:generate_predicate_code(test_weighted_min_ppv_reach/4, Options, PredicateCode)),
    gensym(test_weighted_min_ppv_runtime_, ModuleName),
    build_weighted_min_runtime_source(ModuleName, PredicateCode, Source),
    write_temp_module_source(Source, Path),
    setup_call_cleanup(
        load_files(Path, []),
        (
            Goal =.. ['test_weighted_min_ppv_reach$min', a, d, Cost],
            findall(Cost, ModuleName:Goal, Costs0),
            sort(Costs0, Costs)
        ),
        cleanup_temp_module_source(Path)
    ).

build_effective_distance_runtime_source(ModuleName, PredicateCode, Source) :-
    atomic_list_concat([
        'test_effective_edge(a, b).',
        'test_effective_edge(b, c).',
        'test_effective_edge(a, d).',
        'test_effective_edge(d, e).',
        'test_effective_edge(e, c).',
        'dimension_n(5).',
        'max_depth(4).'
    ], '\n', FactsCode),
    format(atom(Source),
        ':- module(~q, []).~n:- use_module(library(lists)).~n:- use_module(library(aggregate)).~n~w~n~n~w~n',
        [ModuleName, FactsCode, PredicateCode]).

collect_generated_effective_distance_sums(Options, PredicateCode, Sums) :-
    once(prolog_target:generate_predicate_code(test_effective_ppv_reach/4, Options, PredicateCode)),
    gensym(test_effective_ppv_runtime_, ModuleName),
    build_effective_distance_runtime_source(ModuleName, PredicateCode, Source),
    write_temp_module_source(Source, Path),
    setup_call_cleanup(
        load_files(Path, []),
        (
            Goal =.. ['test_effective_ppv_reach$effective_distance_sum', a, c, Sum],
            findall(Sum, ModuleName:Goal, Sums0),
            sort(Sums0, Sums)
        ),
        cleanup_temp_module_source(Path)
    ).

test(emits_branch_pruning_helpers_for_parameterized_ppv,
        [setup(setup_branch_pruning_fixture),
         cleanup(cleanup_branch_pruning_fixture)]) :-
    once(generate_prolog_script([test_ppv_reach/4], [dialect(swi)], Code)),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$prune')),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$prune_guard')),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$prune_cache')),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$pruned')),
    once(sub_atom(Code, _, _, _, ':- table')).

test(skips_branch_pruning_without_mode_signal,
        [setup(setup_no_mode_fixture),
         cleanup(cleanup_branch_pruning_fixture)]) :-
    once(generate_prolog_script([test_ppv_reach/4], [dialect(swi)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$prune'),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$pruned').

test(skips_branch_pruning_for_non_swi_dialect,
        [setup(setup_branch_pruning_fixture),
         cleanup(cleanup_branch_pruning_fixture)]) :-
    once(generate_prolog_script([test_ppv_reach/4], [dialect(gnu)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$prune'),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$pruned').

test(skips_branch_pruning_when_disabled_explicitly,
        [setup(setup_branch_pruning_fixture),
         cleanup(cleanup_branch_pruning_fixture)]) :-
    once(generate_prolog_script([test_ppv_reach/4], [dialect(swi), branch_pruning(false)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$prune'),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$pruned').

test(rejects_noncanonical_ppv_leading_goal,
        [setup(setup_noncanonical_step_fixture),
         cleanup(cleanup_noncanonical_step_fixture)]) :-
    once(generate_prolog_script([test_noncanonical_ppv_reach/4], [dialect(swi)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_noncanonical_ppv_reach$prune'),
    \+ sub_atom(Code, _, _, _, 'test_noncanonical_ppv_reach$pruned').

test(strips_only_codegen_module_qualifiers) :-
    prolog_target:strip_codegen_module_qualifiers(user:(foo(X), prolog_target:bar(X), other:baz(X)), Goal),
    Goal = (foo(X), bar(X), other:baz(X)).

test(rename_recursive_calls_preserves_foreign_module_qualifiers) :-
    prolog_target:rename_recursive_calls(
        (user:test_ppv_reach(A, B, C, D), other:test_ppv_reach(A, B, C, D)),
        test_ppv_reach,
        4,
        'test_ppv_reach$worker',
        Goal
    ),
    Goal = ('test_ppv_reach$worker'(A, B, C, D), other:'test_ppv_reach$worker'(A, B, C, D)).

test(exec_generated_branch_pruned_code_preserves_results,
        [setup(setup_execution_branch_pruning_fixture),
         cleanup(cleanup_execution_branch_pruning_fixture)]) :-
    collect_generated_execution_hops([dialect(swi)], PredicateCode, Actual),
    once(sub_atom(PredicateCode, _, _, _, 'test_exec_ppv_reach$pruned')),
    findall(H, user:test_exec_ppv_reach(a, c, H, [a]), Expected0),
    sort(Expected0, Expected),
    Expected == [2, 3],
    Actual == Expected.

test(exec_generated_disabled_branch_pruning_matches_enabled_results,
        [setup(setup_execution_branch_pruning_fixture),
         cleanup(cleanup_execution_branch_pruning_fixture)]) :-
    collect_generated_execution_hops([dialect(swi)], EnabledCode, Enabled),
    collect_generated_execution_hops([dialect(swi), branch_pruning(false)], DisabledCode, Disabled),
    once(sub_atom(EnabledCode, _, _, _, 'test_exec_ppv_reach$pruned')),
    \+ sub_atom(DisabledCode, _, _, _, 'test_exec_ppv_reach$pruned'),
    Enabled == [2, 3],
    Disabled == Enabled.

test(emits_bounded_min_closure_helper_for_counted_ppv,
        [setup(setup_min_closure_fixture),
         cleanup(cleanup_min_closure_fixture)]) :-
    collect_generated_min_hops([dialect(swi)], PredicateCode, Actual),
    once(sub_atom(PredicateCode, _, _, _, 'test_min_ppv_reach$min')),
    once(sub_atom(PredicateCode, _, _, _, 'test_min_ppv_reach$min_budget')),
    Actual == [2].

test(skips_min_closure_helper_when_disabled_explicitly,
        [setup(setup_min_closure_fixture),
         cleanup(cleanup_min_closure_fixture)]) :-
    once(generate_prolog_script([test_min_ppv_reach/4], [dialect(swi), min_closure(false)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_min_ppv_reach$min').

test(emits_weighted_min_closure_helper_for_positive_weighted_ppv,
        [setup(setup_weighted_min_closure_fixture),
         cleanup(cleanup_weighted_min_closure_fixture)]) :-
    collect_generated_weighted_min_costs([dialect(swi)], PredicateCode, Actual),
    once(sub_atom(PredicateCode, _, _, _, 'test_weighted_min_ppv_reach$min')),
    once(sub_atom(PredicateCode, _, _, _, 'test_weighted_min_ppv_reach$min_budget')),
    Actual == [3].

test(emits_weighted_min_closure_helper_from_positive_step_metadata,
        [setup(setup_weighted_min_metadata_fixture),
         cleanup(cleanup_weighted_min_closure_fixture)]) :-
    collect_generated_weighted_min_costs([dialect(swi)], PredicateCode, Actual),
    once(sub_atom(PredicateCode, _, _, _, 'test_weighted_min_ppv_reach$min')),
    Actual == [3].

test(skips_weighted_min_closure_without_positive_step_proof,
        [setup(setup_weighted_min_unproven_fixture),
         cleanup(cleanup_weighted_min_closure_fixture)]) :-
    once(generate_prolog_script([test_weighted_min_ppv_reach/4], [dialect(swi)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_weighted_min_ppv_reach$min').

test(emits_effective_distance_accumulation_helper_for_counted_ppv,
        [setup(setup_effective_distance_accumulation_fixture),
         cleanup(cleanup_effective_distance_accumulation_fixture)]) :-
    Options = [
        dialect(swi),
        min_closure(false),
        branch_pruning(false),
        effective_distance_accumulation(auto),
        predicates([dimension_n/1, max_depth/1, test_effective_ppv_reach/4])
    ],
    collect_generated_effective_distance_sums(Options, PredicateCode, [Actual]),
    once(sub_atom(PredicateCode, _, _, _, 'test_effective_ppv_reach$effective_distance_sum')),
    Expected is (3 ** -5) + (4 ** -5),
    Delta is abs(Actual - Expected),
    Delta < 1.0e-12.

test(skips_effective_distance_accumulation_helper_when_disabled_explicitly,
        [setup(setup_effective_distance_accumulation_fixture),
         cleanup(cleanup_effective_distance_accumulation_fixture)]) :-
    once(generate_prolog_script(
        [dimension_n/1, max_depth/1, test_effective_ppv_reach/4],
        [dialect(swi), effective_distance_accumulation(false)],
        Code)),
    \+ sub_atom(Code, _, _, _, 'test_effective_ppv_reach$effective_distance_sum').

:- end_tests(prolog_target).
