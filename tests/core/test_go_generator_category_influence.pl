:- module(test_go_generator_category_influence, [
    test_go_generator_category_influence/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/go_target.pl').

setup_category_influence_helper_example :-
    cleanup_all,
    assertz(user:influence_dimension(5)),
    assertz(user:article_category(a1, physics)),
    assertz(user:root_category(science)),
    assertz(user:category_ancestor(physics, science, 2)),
    assertz(user:(article_root_weight(Article, Root, 1.0) :-
        article_category(Article, Root),
        root_category(Root))),
    assertz(user:(article_root_weight(Article, Root, Weight) :-
        influence_dimension(N),
        article_category(Article, Cat),
        root_category(Root),
        Cat \= Root,
        category_ancestor(Cat, Root, Hops),
        Distance is Hops + 1,
        Weight is (Distance ** (-N)))),
    assertz(user:(category_influence(Root, Score) :-
        root_category(Root),
        aggregate_all(sum(W), article_root_weight(_, Root, W), Score),
        Score > 0)).

cleanup_all :-
    catch(abolish(user:influence_dimension/1), _, true),
    catch(abolish(user:article_category/2), _, true),
    catch(abolish(user:root_category/1), _, true),
    catch(abolish(user:category_ancestor/3), _, true),
    catch(abolish(user:article_root_weight/3), _, true),
    catch(abolish(user:category_influence/2), _, true).

:- begin_tests(go_generator_category_influence).

test(compile_weighted_helper_with_computed_builtins, [
    setup(setup_category_influence_helper_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(article_root_weight/3, [mode(generator)], Code),
    sub_string(Code, _, _, _, '"math"'),
    sub_string(Code, _, _, _, 'math.Pow((toFloat64Must('),
    sub_string(Code, _, _, _, 'j3.Args["arg0"]'),
    sub_string(Code, _, _, _, 'fact.Args["arg1"] != j2.Args["arg0"]'),
    \+ sub_string(Code, _, _, _, '"arg2": nil').

test(compile_outer_category_influence_aggregate, [
    setup(setup_category_influence_helper_example),
    cleanup(cleanup_all)
]) :-
    compile_predicate_to_go(category_influence/2, [mode(generator)], Code),
    sub_string(Code, _, _, _, 'if fact.Relation != "root_category"'),
    sub_string(Code, _, _, _, 'if f.Relation == "article_root_weight" && (f.Args["arg1"] == fact.Args["arg0"])'),
    sub_string(Code, _, _, _, 'agg := 0.0; for _, v := range values { agg += v }').

:- end_tests(go_generator_category_influence).

test_go_generator_category_influence :-
    run_tests([go_generator_category_influence]).
