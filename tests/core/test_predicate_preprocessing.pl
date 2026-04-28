:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/core/predicate_preprocessing').

:- begin_tests(predicate_preprocessing).

test(preprocess_mode_normalization) :-
    preprocess_mode(artifact, artifact),
    preprocess_mode(artifact([format(grouped_tsv_arg1)]), artifact),
    preprocess_mode(exact_hash_index([key([1])]), artifact),
    preprocess_mode(adjacency_index([access([adjacency])]), artifact),
    preprocess_mode(relation_rows([format(edn_rows)]), sidecar),
    preprocess_mode(inline_data([]), inline),
    preprocess_mode(benchmark_mode(sidecar), sidecar).

test(shared_declaration_lookup) :-
    setup_call_cleanup(
        assertz(user:preprocess(article_category/2, exact_hash_index([key([1]), values([2])]))),
        ( declared_preprocess(article_category/2, Mode, preprocess_info(Kind, Options)),
          assertion(Mode == artifact),
          assertion(Kind == exact_hash_index),
          assertion(member(key([1]), Options)),
          assertion(member(values([2]), Options))
        ),
        maybe_abolish_test_predicate(preprocess/2)
    ).

test(shared_declaration_metadata) :-
    setup_call_cleanup(
        assertz(user:preprocess(category_parent/2,
                                exact_hash_index([key([1]), values([2])]))),
        ( declared_preprocess_metadata(category_parent/2, Mode,
                                       preprocess_info(Kind, Options),
                                       Metadata),
          assertion(Mode == artifact),
          assertion(Kind == exact_hash_index),
          assertion(member(key([1]), Options)),
          assertion(member(values([2]), Options)),
          assertion(Metadata.format == exact_hash_index),
          assertion(member(exact_key_lookup, Metadata.access_contracts)),
          assertion(member(arg_position_lookup(1), Metadata.access_contracts)),
          assertion(member(grouped_values_lookup([2]), Metadata.access_contracts))
        ),
        maybe_abolish_test_predicate(preprocess/2)
    ).

:- end_tests(predicate_preprocessing).

maybe_abolish_test_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).
