:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputPath, VariantAtom]
    ->  true
    ;   format(user_error,
            'Usage: swipl -q -s generate_prolog_category_influence_benchmark.pl -- <facts.pl> <output-script> <seeded|accumulated>~n',
            []),
        halt(1)
    ),
    parse_variant(VariantAtom, Variant, Options),
    benchmark_workload_path(WorkloadPath),
    absolute_file_name(FactsPath, FactsAbsPath, [access(read)]),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    Predicates = [
        dimension_n/1,
        max_depth/1,
        category_ancestor/4
    ],
    prolog_target:generate_prolog_script(Predicates, Options, BaseScriptCode),
    benchmark_driver_code(Variant, FactsAbsPath, DriverCode),
    atomic_list_concat([BaseScriptCode, DriverCode], '\n\n', ScriptCode),
    prolog_target:write_prolog_script(ScriptCode, OutputPath).

parse_variant('seeded', seeded, [
        dialect(swi),
        entry_point(run_benchmark),
        branch_pruning(false),
        min_closure(false),
        seeded_accumulation(false)
    ]) :-
    !.
parse_variant('accumulated', accumulated, [
        dialect(swi),
        entry_point(run_benchmark),
        branch_pruning(false),
        min_closure(false),
        seeded_accumulation(auto)
    ]) :-
    !.
parse_variant(Atom, _, _) :-
    format(user_error,
        'Unsupported category-influence benchmark variant: ~w (expected seeded or accumulated)~n',
        [Atom]),
    halt(1).

benchmark_driver_code(Variant, FactsPath, Code) :-
    format(atom(FactsPathLine), 'facts_path(~q).', [FactsPath]),
    benchmark_index_build_line(Variant, IndexBuildLine),
    benchmark_results_line(Variant, ResultsLine),
    benchmark_weight_sum_query_line(Variant, WeightSumQueryLine),
    benchmark_mode_line(Variant, ModeLine),
    Lines = [
        ':- use_module(library(aggregate)).',
        ':- use_module(library(lists)).',
        ':- dynamic bench_seed_hops/3.',
        ':- dynamic bench_seed_weight_sum/3.',
        FactsPathLine,
        '',
        'load_benchmark_facts :-',
        '    facts_path(Path),',
        '    load_files(Path, [silent(true)]).',
        '',
        'collect_seed_categories(Seeds) :-',
        '    setof(Cat, Article^article_category(Article, Cat), Seeds).',
        '',
        'clear_seed_indexes :-',
        '    retractall(bench_seed_hops(_, _, _)),',
        '    retractall(bench_seed_weight_sum(_, _, _)).',
        '',
        'seed_hops_query(Cat, Root, Hops) :-',
        '    category_ancestor(Cat, Root, Hops, [Cat]).',
        '',
        'seed_weight_sum_query(Cat, Root, WeightSum) :-',
        WeightSumQueryLine,
        '',
        'build_seed_hops_index(SeedCount, TupleCount) :-',
        '    clear_seed_indexes,',
        '    collect_seed_categories(Seeds),',
        '    length(Seeds, SeedCount),',
        '    forall(',
        '        member(Cat, Seeds),',
        '        forall(',
        '            seed_hops_query(Cat, Root, Hops),',
        '            assertz(bench_seed_hops(Cat, Root, Hops))',
        '        )',
        '    ),',
        '    aggregate_all(count, bench_seed_hops(_, _, _), TupleCount).',
        '',
        'build_seed_weight_sum_index(SeedCount, TupleCount) :-',
        '    clear_seed_indexes,',
        '    collect_seed_categories(Seeds),',
        '    length(Seeds, SeedCount),',
        '    forall(',
        '        member(Cat, Seeds),',
        '        forall(',
        '            seed_weight_sum_query(Cat, Root, WeightSum),',
        '            assertz(bench_seed_weight_sum(Cat, Root, WeightSum))',
        '        )',
        '    ),',
        '    aggregate_all(count, bench_seed_weight_sum(_, _, _), TupleCount).',
        '',
        'seeded_article_root_weight_from_hops(Article, Root, 1.0) :-',
        '    article_category(Article, Root).',
        'seeded_article_root_weight_from_hops(Article, Root, Weight) :-',
        '    dimension_n(N),',
        '    article_category(Article, Cat),',
        '    Cat \\= Root,',
        '    bench_seed_hops(Cat, Root, Hops),',
        '    Distance is Hops + 1,',
        '    Weight is Distance ** (-N).',
        '',
        'seeded_article_root_weight_from_sums(Article, Root, 1.0) :-',
        '    article_category(Article, Root).',
        'seeded_article_root_weight_from_sums(Article, Root, Weight) :-',
        '    article_category(Article, Cat),',
        '    Cat \\= Root,',
        '    bench_seed_weight_sum(Cat, Root, Weight).',
        '',
        'compute_category_influence_results_from_hops(Results) :-',
        '    setof(Root, root_category(Root), Roots),',
        '    findall(Score-Root,',
        '        (   member(Root, Roots),',
        '            aggregate_all(sum(W), seeded_article_root_weight_from_hops(_, Root, W), Score),',
        '            Score > 0',
        '        ),',
        '        Pairs),',
        '    sort(1, @>=, Pairs, Results).',
        '',
        'compute_category_influence_results_from_sums(Results) :-',
        '    setof(Root, root_category(Root), Roots),',
        '    findall(Score-Root,',
        '        (   member(Root, Roots),',
        '            aggregate_all(sum(W), seeded_article_root_weight_from_sums(_, Root, W), Score),',
        '            Score > 0',
        '        ),',
        '        Pairs),',
        '    sort(1, @>=, Pairs, Results).',
        '',
        'print_category_influence_results(Results) :-',
        '    format("root_category\\tinfluence_score~n"),',
        '    forall(',
        '        member(Score-Root, Results),',
        '        format("~w\\t~12f~n", [Root, Score])',
        '    ).',
        '',
        'run_benchmark :-',
        '    statistics(walltime, [StartMs, _]),',
        '    load_benchmark_facts,',
        '    statistics(walltime, [AfterLoadMs, _]),',
        '    LoadMs is AfterLoadMs - StartMs,',
        '    statistics(inferences, InferencesBefore),',
        IndexBuildLine,
        '    statistics(walltime, [AfterQueryMs, _]),',
        '    QueryMs is AfterQueryMs - AfterLoadMs,',
        ResultsLine,
        '    statistics(walltime, [AfterAggregationMs, _]),',
        '    AggregationMs is AfterAggregationMs - AfterQueryMs,',
        '    print_category_influence_results(Results),',
        '    statistics(inferences, InferencesAfter),',
        '    TotalMs is AfterAggregationMs - StartMs,',
        '    Inferences is InferencesAfter - InferencesBefore,',
        '    length(Results, RootCount),',
        ModeLine,
        '    format(user_error, ''load_ms=~w~n'', [LoadMs]),',
        '    format(user_error, ''query_ms=~w~n'', [QueryMs]),',
        '    format(user_error, ''aggregation_ms=~w~n'', [AggregationMs]),',
        '    format(user_error, ''total_ms=~w~n'', [TotalMs]),',
        '    format(user_error, ''inferences=~w~n'', [Inferences]),',
        '    format(user_error, ''seed_count=~w~n'', [SeedCount]),',
        '    format(user_error, ''tuple_count=~w~n'', [TupleCount]),',
        '    format(user_error, ''root_count=~w~n'', [RootCount]).'
    ],
    atomic_list_concat(Lines, '\n', Code).

benchmark_index_build_line(seeded, '    build_seed_hops_index(SeedCount, TupleCount),').
benchmark_index_build_line(accumulated, '    build_seed_weight_sum_index(SeedCount, TupleCount),').

benchmark_results_line(seeded, '    compute_category_influence_results_from_hops(Results),').
benchmark_results_line(accumulated, '    compute_category_influence_results_from_sums(Results),').

benchmark_weight_sum_query_line(seeded, '    fail.').
benchmark_weight_sum_query_line(accumulated, '    ''category_ancestor$power_sum_selected''(Cat, Root, WeightSum).').

benchmark_mode_line(seeded, '    format(user_error, ''mode=seeded~n'', []),').
benchmark_mode_line(accumulated, '    format(user_error, ''mode=accumulated~n'', []),').
