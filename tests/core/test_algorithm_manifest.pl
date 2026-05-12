:- encoding(utf8).
%% Test suite for algorithm_manifest.pl
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_algorithm_manifest.pl

:- use_module('../../src/unifyweaver/core/algorithm_manifest').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("Algorithm Manifest Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    %% Each test isolates its own manifest state.
    algorithm_manifest:reset_manifest,
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

pass(Name) :- format("[PASS] ~w~n", [Name]).
fail_test(Name, Reason) :- format("[FAIL] ~w: ~w~n", [Name, Reason]), fail.

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_manifest_absent_options_unchanged).
test(test_manifest_absent_adds_sentinel).
test(test_manifest_present_merges_options).
test(test_caller_wins_on_conflict).
test(test_multiple_optimization_facts_concat).
test(test_optimization_options_preserves_nested_structure).
test(test_orphan_optimization_emits_warning_and_no_merge).
test(test_duplicate_decl_algorithm_throws).
test(test_decl_algorithm_validates_name_is_atom).
test(test_decl_algorithm_validates_opts_is_list).
test(test_load_manifest_is_idempotent).
test(test_manifest_algorithm_accessor).
test(test_manifest_optimization_options_accessor).
test(test_reset_manifest_clears_state).

%% ========================================================================
%% Tests: manifest absent
%% ========================================================================

test_manifest_absent_options_unchanged :-
    Test = 'manifest absent: caller options pass through (sans sentinel)',
    Options0 = [foo(1), bar(2)],
    algorithm_manifest:load_algorithm_manifest(Options0, Result),
    %% Strip the sentinel for comparison
    select(algorithm_manifest_loaded(true), Result, StrippedResult),
    (   StrippedResult == Options0
    ->  pass(Test)
    ;   fail_test(Test, format_atom('expected ~w, got ~w', [Options0, StrippedResult]))
    ).

test_manifest_absent_adds_sentinel :-
    Test = 'manifest absent: sentinel still added',
    algorithm_manifest:load_algorithm_manifest([foo(1)], Result),
    (   memberchk(algorithm_manifest_loaded(true), Result)
    ->  pass(Test)
    ;   fail_test(Test, 'sentinel missing from result')
    ).

%% ========================================================================
%% Tests: manifest present
%% ========================================================================

test_manifest_present_merges_options :-
    Test = 'manifest present: optimization options merge in',
    algorithm_manifest:decl_algorithm(test_alg, [kernel(foo/2)]),
    algorithm_manifest:decl_algorithm_optimization(test_alg, [
        cache_strategy(auto),
        working_set_fraction(0.001)
    ]),
    Options0 = [foo(1)],
    algorithm_manifest:load_algorithm_manifest(Options0, Result),
    (   memberchk(cache_strategy(auto), Result),
        memberchk(working_set_fraction(0.001), Result),
        memberchk(foo(1), Result)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('merged options missing keys: ~w', [Result]))
    ).

test_caller_wins_on_conflict :-
    Test = 'caller wins: caller option value precedes manifest value',
    algorithm_manifest:decl_algorithm(test_alg, [kernel(foo/2)]),
    algorithm_manifest:decl_algorithm_optimization(test_alg, [
        cache_strategy(auto)
    ]),
    %% Caller specifies cache_strategy(manual); should win over manifest's auto.
    Options0 = [cache_strategy(manual)],
    algorithm_manifest:load_algorithm_manifest(Options0, Result),
    %% option/3 returns first match
    option(cache_strategy(Picked), Result),
    (   Picked == manual
    ->  pass(Test)
    ;   fail_test(Test, format_atom('manifest beat caller: got ~w', [Picked]))
    ).

test_multiple_optimization_facts_concat :-
    Test = 'multiple decl_algorithm_optimization/2 facts concatenate',
    algorithm_manifest:decl_algorithm(test_alg, [kernel(foo/2)]),
    algorithm_manifest:decl_algorithm_optimization(test_alg, [a(1), b(2)]),
    algorithm_manifest:decl_algorithm_optimization(test_alg, [c(3), d(4)]),
    algorithm_manifest:manifest_optimization_options(test_alg, Concat),
    (   Concat == [a(1), b(2), c(3), d(4)]
    ->  pass(Test)
    ;   fail_test(Test, format_atom('expected [a(1),b(2),c(3),d(4)], got ~w', [Concat]))
    ).

test_optimization_options_preserves_nested_structure :-
    %% Regression test for the flatten/2 trap. An option whose value
    %% is itself a list (like tree_cost_function/2's parameters) must
    %% survive the concat step intact.
    Test = 'nested option terms survive manifest concat (no flatten)',
    algorithm_manifest:decl_algorithm(test_alg, [kernel(foo/2)]),
    algorithm_manifest:decl_algorithm_optimization(test_alg, [
        tree_cost_function(flux, [iterations(1), parent_decay(0.5)])
    ]),
    algorithm_manifest:manifest_optimization_options(test_alg, Concat),
    (   memberchk(tree_cost_function(flux, [iterations(1), parent_decay(0.5)]),
                  Concat)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('nested params got flattened: ~w', [Concat]))
    ).

%% ========================================================================
%% Tests: orphans and errors
%% ========================================================================

test_orphan_optimization_emits_warning_and_no_merge :-
    %% Optimization declared but no matching algorithm. Should emit a
    %% warning (we can't easily assert on stderr; just verify it
    %% doesn't merge into the result).
    Test = 'orphan optimization: no algorithm → ignored',
    algorithm_manifest:decl_algorithm_optimization(orphaned_alg, [
        cache_strategy(auto)
    ]),
    Options0 = [foo(1)],
    algorithm_manifest:load_algorithm_manifest(Options0, Result),
    (   \+ memberchk(cache_strategy(auto), Result),
        memberchk(foo(1), Result)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('orphan optimization leaked in: ~w', [Result]))
    ).

test_duplicate_decl_algorithm_throws :-
    Test = 'duplicate decl_algorithm/2 throws error',
    algorithm_manifest:decl_algorithm(dup_alg, [kernel(a/1)]),
    catch(
        ( algorithm_manifest:decl_algorithm(dup_alg, [kernel(b/2)]),
          Caught = no
        ),
        error(domain_error(unique_algorithm_decl, dup_alg), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'duplicate decl_algorithm did not throw')
    ).

test_decl_algorithm_validates_name_is_atom :-
    Test = 'decl_algorithm/2 rejects non-atom Name',
    catch(
        ( algorithm_manifest:decl_algorithm("not_an_atom", [foo(1)]),
          Caught = no
        ),
        error(type_error(atom, _), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'string Name was accepted')
    ).

test_decl_algorithm_validates_opts_is_list :-
    Test = 'decl_algorithm/2 rejects non-list AlgorithmOpts',
    catch(
        ( algorithm_manifest:decl_algorithm(test_alg, not_a_list),
          Caught = no
        ),
        error(type_error(list, _), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'non-list Opts was accepted')
    ).

%% ========================================================================
%% Tests: idempotency and accessors
%% ========================================================================

test_load_manifest_is_idempotent :-
    Test = 'load_algorithm_manifest is idempotent (sentinel-driven)',
    algorithm_manifest:decl_algorithm(test_alg, [kernel(foo/2)]),
    algorithm_manifest:decl_algorithm_optimization(test_alg, [cache_strategy(auto)]),
    Options0 = [caller_opt(x)],
    algorithm_manifest:load_algorithm_manifest(Options0, Once),
    algorithm_manifest:load_algorithm_manifest(Once, Twice),
    (   Once == Twice
    ->  pass(Test)
    ;   fail_test(Test, format_atom('second call mutated: ~w vs ~w', [Once, Twice]))
    ).

test_manifest_algorithm_accessor :-
    Test = 'manifest_algorithm/2 returns declared algorithm',
    algorithm_manifest:decl_algorithm(my_alg, [kernel(foo/4), max_depth(7)]),
    algorithm_manifest:manifest_algorithm(Name, Opts),
    (   Name == my_alg,
        memberchk(kernel(foo/4), Opts),
        memberchk(max_depth(7), Opts)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('got name=~w opts=~w', [Name, Opts]))
    ).

test_manifest_optimization_options_accessor :-
    Test = 'manifest_optimization_options/2 returns empty list for algorithm with no optimizations',
    algorithm_manifest:decl_algorithm(bare_alg, [kernel(foo/2)]),
    algorithm_manifest:manifest_optimization_options(bare_alg, Opts),
    (   Opts == []
    ->  pass(Test)
    ;   fail_test(Test, format_atom('expected [], got ~w', [Opts]))
    ).

test_reset_manifest_clears_state :-
    Test = 'reset_manifest/0 clears registered state',
    algorithm_manifest:decl_algorithm(clear_alg, [kernel(foo/2)]),
    algorithm_manifest:decl_algorithm_optimization(clear_alg, [foo(1)]),
    algorithm_manifest:reset_manifest,
    (   \+ algorithm_manifest:manifest_algorithm(_, _),
        algorithm_manifest:manifest_optimization_options(clear_alg, [])
    ->  pass(Test)
    ;   fail_test(Test, 'state survived reset')
    ).

%% ========================================================================
%% Helper
%% ========================================================================

format_atom(Format, Args, Atom) :-
    format(string(S), Format, Args),
    atom_string(Atom, S).
format_atom(Format, Args) :-
    format_atom(Format, Args, A),
    A == A.
