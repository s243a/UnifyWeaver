:- encoding(utf8).
% Test suite for the relation_policy/2 declaration registry.
% Phase 1: parser + registry + lookup. No backend enforcement.
% Usage: swipl -g run_tests -t halt tests/test_relation_policy.pl

:- use_module('../src/unifyweaver/core/relation_policy').

:- begin_tests(relation_policy).

% Wipe the registry before every test so they're order-independent.
% plunit's setup hook runs once per test.
setup_fresh :- clear_relation_policies.

test(parse_basic, [setup(setup_fresh)]) :-
    relation_policy(edge/2,
        [ key([arg(1), arg(2)]),
          order(natural),
          unique(true),
          on_duplicate(throw),
          determinism(semidet),
          cardinality(medium) ]),
    get_relation_policy(edge/2, key, [arg(1), arg(2)]),
    get_relation_policy(edge/2, order, natural),
    get_relation_policy(edge/2, unique, true),
    get_relation_policy(edge/2, on_duplicate, throw),
    get_relation_policy(edge/2, determinism, semidet),
    get_relation_policy(edge/2, cardinality, medium).

test(lookup_missing_fails, [setup(setup_fresh)]) :-
    \+ get_relation_policy(no_such/2, unique, _).

test(default_returns_when_unset, [setup(setup_fresh)]) :-
    get_relation_policy(p/2, unique, V, false),
    V == false.

test(default_skipped_when_set, [setup(setup_fresh)]) :-
    relation_policy(p/2, [unique(true)]),
    get_relation_policy(p/2, unique, V, false),
    V == true.

% Effective policy: source-level override wins over the declaration
% which wins over the built-in default.
test(effective_override_wins, [setup(setup_fresh)]) :-
    relation_policy(p/2, [on_duplicate(throw)]),
    get_effective_policy(p/2, [on_duplicate(warn)], on_duplicate, V),
    V == warn.

test(effective_declaration_when_no_override, [setup(setup_fresh)]) :-
    relation_policy(p/2, [on_duplicate(throw)]),
    get_effective_policy(p/2, [], on_duplicate, V),
    V == throw.

test(effective_falls_back_to_default, [setup(setup_fresh)]) :-
    get_effective_policy(p/2, [], on_duplicate, V),
    V == keep_all.

test(effective_default_for_unknown_key_via_doc_defaults,
     [setup(setup_fresh)]) :-
    get_effective_policy(p/2, [], order, V),         V  == natural,
    get_effective_policy(p/2, [], unique, V2),       V2 == false,
    get_effective_policy(p/2, [], cardinality, V3),  V3 == unknown,
    get_effective_policy(p/2, [], determinism, V4),  V4 == nondet,
    get_effective_policy(p/2, [], key, V5),          V5 == all.

% Latest-wins per key, NOT per declaration. Re-declaring some
% keys leaves others untouched.
test(redeclare_overrides_per_key, [setup(setup_fresh)]) :-
    relation_policy(q/2, [unique(true), determinism(det)]),
    relation_policy(q/2, [unique(false)]),     % only unique
    get_relation_policy(q/2, unique, U),
    get_relation_policy(q/2, determinism, D),
    U == false, D == det.

test(enumerate_via_current, [setup(setup_fresh)]) :-
    relation_policy(r/2, [unique(true), order(natural)]),
    findall(K-V, current_relation_policy(r/2, K, V), Pairs),
    sort(Pairs, Sorted),
    Sorted == [order-natural, unique-true].

% Validation: bad predicate indicator.
test(reject_non_pred_indicator, [setup(setup_fresh)]) :-
    catch(relation_policy(not_a_pi, [unique(true)]),
          error(type_error(predicate_indicator, not_a_pi), _),
          true).

test(reject_negative_arity, [setup(setup_fresh)]) :-
    catch(relation_policy(p/(-1), [unique(true)]),
          error(type_error(nonneg, -1), _),
          true).

% Validation: unknown / malformed options.
test(reject_unknown_option_key, [setup(setup_fresh)]) :-
    catch(relation_policy(p/2, [bogus(x)]),
          error(domain_error(relation_policy_key, bogus), _),
          true).

test(reject_bad_unique_value, [setup(setup_fresh)]) :-
    catch(relation_policy(p/2, [unique(maybe)]),
          error(type_error(boolean, maybe), _),
          true).

test(reject_bad_order_value, [setup(setup_fresh)]) :-
    catch(relation_policy(p/2, [order(weird)]),
          error(domain_error(relation_policy_order_spec, weird), _),
          true).

test(reject_bad_dup_policy, [setup(setup_fresh)]) :-
    catch(relation_policy(p/2, [on_duplicate(unknown)]),
          error(domain_error(relation_policy_on_duplicate, unknown), _),
          true).

test(reject_bad_determinism, [setup(setup_fresh)]) :-
    catch(relation_policy(p/2, [determinism(superdet)]),
          error(domain_error(relation_policy_determinism, superdet), _),
          true).

% Accept all SWI-style key, order, and on_duplicate values.
test(accept_all_dup_policies, [setup(setup_fresh)]) :-
    forall(member(P, [throw, warn, overwrite, first_wins, keep_all]),
           relation_policy(test/2, [on_duplicate(P)])),
    relation_policy(test/2, [on_duplicate(fallback(warn))]).

test(accept_order_directions, [setup(setup_fresh)]) :-
    relation_policy(p/2, [order([asc(arg(1)), desc(arg(2)), arg(3)])]),
    get_relation_policy(p/2, order, [asc(arg(1)), desc(arg(2)), arg(3)]).

test(accept_cardinality_number, [setup(setup_fresh)]) :-
    relation_policy(p/2, [cardinality(42)]),
    get_relation_policy(p/2, cardinality, 42).

test(option_keys_complete) :-
    relation_policy_option_keys(Keys),
    sort(Keys, Sorted),
    Sorted == [cardinality, determinism, key, on_duplicate,
               order, unique].

:- end_tests(relation_policy).
