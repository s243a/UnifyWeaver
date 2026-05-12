:- encoding(utf8).
%% Test suite for cost_function.pl
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_cost_function.pl

:- use_module('../../src/unifyweaver/core/cost_function').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("Cost Function Tests~n"),
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
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

pass(Name) :- format("[PASS] ~w~n", [Name]).
fail_test(Name, Reason) :- format("[FAIL] ~w: ~w~n", [Name, Reason]), fail.

%% ========================================================================
%% Tests
%% ========================================================================

test(test_registry_has_three_entries).
test(test_hop_distance_registered).
test(test_flux_registered).
test(test_semantic_similarity_registered).
test(test_unregistered_name_throws).

test(test_validate_hop_distance_with_max_hops).
test(test_validate_hop_distance_without_params).
test(test_validate_flux_with_params).
test(test_validate_flux_without_params).
test(test_validate_semantic_requires_embedding_path).
test(test_validate_rejects_wrong_shape).
test(test_validate_rejects_unregistered_name).
test(test_validate_rejects_wrong_param_type).
test(test_validate_accepts_unknown_param_keys).

test(test_defaults_hop_distance).
test(test_defaults_flux).
test(test_defaults_semantic_omits_required).

test(test_with_defaults_fills_missing).
test(test_with_defaults_preserves_caller_values).
test(test_with_defaults_validates_first).

test(test_is_cost_function_term_accepts_valid).
test(test_is_cost_function_term_rejects_invalid).

%% ========================================================================
%% Registry tests
%% ========================================================================

test_registry_has_three_entries :-
    Test = 'registry has exactly three entries',
    findall(N, cost_function:cost_function_name(N), Names),
    length(Names, 3),
    (   sort(Names, [flux, hop_distance, semantic_similarity])
    ->  pass(Test)
    ;   fail_test(Test, format_atom('got ~w', [Names]))
    ).

test_hop_distance_registered :-
    Test = 'hop_distance is in the registry',
    cost_function:cost_function_param_schema(hop_distance, Schema),
    (   memberchk(param_spec(max_hops, positive_integer, default(5)), Schema)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('schema=~w', [Schema]))
    ).

test_flux_registered :-
    Test = 'flux is in the registry',
    cost_function:cost_function_param_schema(flux, Schema),
    (   memberchk(param_spec(iterations,   positive_integer, default(1)), Schema),
        memberchk(param_spec(parent_decay, float, default(0.5)), Schema),
        memberchk(param_spec(child_decay,  float, default(0.3)), Schema),
        memberchk(param_spec(flux_merge,   atom, default(sum)), Schema)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('schema=~w', [Schema]))
    ).

test_semantic_similarity_registered :-
    Test = 'semantic_similarity has embedding_path as required',
    cost_function:cost_function_param_schema(semantic_similarity, Schema),
    (   memberchk(param_spec(embedding_path, atom, required), Schema),
        memberchk(param_spec(dim, positive_integer, default(128)), Schema)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('schema=~w', [Schema]))
    ).

test_unregistered_name_throws :-
    Test = 'cost_function_param_schema throws for unknown name',
    catch(
        ( cost_function:cost_function_param_schema(no_such_function, _),
          Caught = no
        ),
        error(domain_error(cost_function_name, no_such_function), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'unknown name did not throw')
    ).

%% ========================================================================
%% Validation tests
%% ========================================================================

test_validate_hop_distance_with_max_hops :-
    Test = 'validate_cost_function accepts hop_distance with max_hops',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(hop_distance, [max_hops(10)])),
          Result = ok
        ),
        _, Result = error
    ),
    (   Result == ok
    ->  pass(Test)
    ;   fail_test(Test, 'rejected valid hop_distance')
    ).

test_validate_hop_distance_without_params :-
    Test = 'validate_cost_function accepts hop_distance with empty params',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(hop_distance, [])),
          Result = ok
        ),
        _, Result = error
    ),
    (   Result == ok
    ->  pass(Test)
    ;   fail_test(Test, 'rejected empty params for hop_distance')
    ).

test_validate_flux_with_params :-
    Test = 'validate_cost_function accepts flux with all params',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(flux, [
                  iterations(2),
                  parent_decay(0.6),
                  child_decay(0.4),
                  flux_merge(max)
              ])),
          Result = ok
        ),
        _, Result = error
    ),
    (   Result == ok
    ->  pass(Test)
    ;   fail_test(Test, 'rejected valid flux')
    ).

test_validate_flux_without_params :-
    Test = 'validate_cost_function accepts flux with no params (all default)',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(flux, [])),
          Result = ok
        ),
        _, Result = error
    ),
    (   Result == ok
    ->  pass(Test)
    ;   fail_test(Test, 'rejected empty params for flux')
    ).

test_validate_semantic_requires_embedding_path :-
    Test = 'validate_cost_function rejects semantic_similarity missing embedding_path',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(semantic_similarity, [dim(64)])),
          Caught = no
        ),
        error(domain_error(required_param, embedding_path), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'missing embedding_path did not throw')
    ).

test_validate_rejects_wrong_shape :-
    Test = 'validate_cost_function rejects non-tree_cost_function term',
    catch(
        ( cost_function:validate_cost_function(some_other_term),
          Caught = no
        ),
        error(domain_error(cost_function_term, _), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'wrong shape was accepted')
    ).

test_validate_rejects_unregistered_name :-
    Test = 'validate_cost_function rejects unregistered name',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(bogus_function, [])),
          Caught = no
        ),
        error(domain_error(cost_function_name, bogus_function), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'unregistered name was accepted')
    ).

test_validate_rejects_wrong_param_type :-
    Test = 'validate_cost_function rejects wrong param type',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(hop_distance, [max_hops(not_an_integer)])),
          Caught = no
        ),
        error(type_error(positive_integer, not_an_integer), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'wrong param type was accepted')
    ).

test_validate_accepts_unknown_param_keys :-
    %% Unknown keys are silently accepted (future-compat).
    Test = 'validate_cost_function accepts unknown param keys',
    catch(
        ( cost_function:validate_cost_function(
              tree_cost_function(hop_distance, [
                  max_hops(5),
                  future_param(42)
              ])),
          Result = ok
        ),
        _, Result = error
    ),
    (   Result == ok
    ->  pass(Test)
    ;   fail_test(Test, 'unknown param key was rejected')
    ).

%% ========================================================================
%% Defaults tests
%% ========================================================================

test_defaults_hop_distance :-
    Test = 'cost_function_default_params returns hop_distance defaults',
    cost_function:cost_function_default_params(hop_distance, Defaults),
    (   memberchk(max_hops(5), Defaults)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('got ~w', [Defaults]))
    ).

test_defaults_flux :-
    Test = 'cost_function_default_params returns flux defaults',
    cost_function:cost_function_default_params(flux, Defaults),
    (   memberchk(iterations(1), Defaults),
        memberchk(parent_decay(0.5), Defaults),
        memberchk(child_decay(0.3), Defaults),
        memberchk(flux_merge(sum), Defaults)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('got ~w', [Defaults]))
    ).

test_defaults_semantic_omits_required :-
    Test = 'cost_function_default_params omits required-no-default params',
    cost_function:cost_function_default_params(semantic_similarity, Defaults),
    (   memberchk(dim(128), Defaults),
        \+ memberchk(embedding_path(_), Defaults)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('got ~w', [Defaults]))
    ).

%% ========================================================================
%% with_defaults tests
%% ========================================================================

test_with_defaults_fills_missing :-
    Test = 'cost_function_with_defaults fills missing params',
    cost_function:cost_function_with_defaults(
        tree_cost_function(flux, [iterations(3)]),
        tree_cost_function(flux, FullParams)),
    (   memberchk(iterations(3), FullParams),
        memberchk(parent_decay(0.5), FullParams),
        memberchk(child_decay(0.3), FullParams),
        memberchk(flux_merge(sum), FullParams)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('got ~w', [FullParams]))
    ).

test_with_defaults_preserves_caller_values :-
    %% Caller-provided value should NOT get overwritten by default.
    Test = 'cost_function_with_defaults preserves caller values',
    cost_function:cost_function_with_defaults(
        tree_cost_function(flux, [parent_decay(0.9)]),
        tree_cost_function(flux, FullParams)),
    (   memberchk(parent_decay(0.9), FullParams),
        \+ memberchk(parent_decay(0.5), FullParams)
    ->  pass(Test)
    ;   fail_test(Test, format_atom('caller value lost: ~w', [FullParams]))
    ).

test_with_defaults_validates_first :-
    Test = 'cost_function_with_defaults validates before filling',
    catch(
        ( cost_function:cost_function_with_defaults(
              tree_cost_function(hop_distance, [max_hops(bogus)]),
              _),
          Caught = no
        ),
        error(type_error(positive_integer, bogus), _),
        Caught = yes
    ),
    (   Caught == yes
    ->  pass(Test)
    ;   fail_test(Test, 'invalid input not caught')
    ).

%% ========================================================================
%% is_cost_function_term tests
%% ========================================================================

test_is_cost_function_term_accepts_valid :-
    Test = 'is_cost_function_term accepts valid shape',
    (   cost_function:is_cost_function_term(
            tree_cost_function(hop_distance, [max_hops(5)]))
    ->  pass(Test)
    ;   fail_test(Test, 'rejected valid term')
    ).

test_is_cost_function_term_rejects_invalid :-
    Test = 'is_cost_function_term rejects wrong shape',
    (   \+ cost_function:is_cost_function_term(some_other_term),
        \+ cost_function:is_cost_function_term(tree_cost_function(bogus_name, []))
    ->  pass(Test)
    ;   fail_test(Test, 'accepted invalid term')
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
