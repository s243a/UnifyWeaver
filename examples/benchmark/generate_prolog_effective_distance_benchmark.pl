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
            'Usage: swipl -q -s generate_prolog_effective_distance_benchmark.pl -- <facts.pl> <output-script> <seeded|pruned|accumulated>~n',
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
        min_closure(false)
    ]) :-
    !.
parse_variant('pruned', pruned, [
        dialect(swi),
        entry_point(run_benchmark),
        branch_pruning(auto),
        min_closure(false)
    ]) :-
    !.
parse_variant('accumulated', accumulated, [
        dialect(swi),
        entry_point(run_benchmark),
        branch_pruning(false),
        min_closure(false),
        effective_distance_accumulation(auto)
    ]) :-
    !.
parse_variant(Atom, _, _) :-
    format(user_error,
        'Unsupported effective-distance benchmark variant: ~w (expected seeded, pruned, or accumulated)~n',
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
        'collect_articles(Articles) :-',
        '    setof(Article, Cat^article_category(Article, Cat), Articles).',
        '',
        'clear_seed_indexes :-',
        '    retractall(bench_seed_hops(_, _, _)),',
        '    retractall(bench_seed_weight_sum(_, _, _)).',
        '',
        'seed_hops_query(Cat, Root, Hops) :-',
        '    category_ancestor(Cat, Root, Hops, [Cat]).',
        '',
        'build_seed_hops_index(Root, SeedCount, TupleCount) :-',
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
        'build_seed_weight_sum_index(Root, SeedCount, TupleCount) :-',
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
        'seeded_article_root_hops(Article, Root, 1) :-',
        '    article_category(Article, Root).',
        'seeded_article_root_hops(Article, Root, Hops) :-',
        '    article_category(Article, Cat),',
        '    Cat \\= Root,',
        '    bench_seed_hops(Cat, Root, CatHops),',
        '    Hops is CatHops + 1.',
        '',
        'seeded_article_root_weight(Article, Root, 1.0) :-',
        '    article_category(Article, Root).',
        'seeded_article_root_weight(Article, Root, Weight) :-',
        '    article_category(Article, Cat),',
        '    Cat \\= Root,',
        '    bench_seed_weight_sum(Cat, Root, Weight).',
        '',
        'seed_weight_sum_query(Cat, Root, WeightSum) :-',
        WeightSumQueryLine,
        '',
        'compute_effective_distance_results_from_hops(Root, Results) :-',
        '    collect_articles(Articles),',
        '    dimension_n(N),',
        '    NegN is -N,',
        '    findall(Deff-Article,',
        '        (   member(Article, Articles),',
        '            aggregate_all(sum(W),',
        '                (   seeded_article_root_hops(Article, Root, Hops),',
        '                    W is Hops ** NegN',
        '                ),',
        '                WeightSum),',
        '            WeightSum > 0,',
        '            InvN is -1 / N,',
        '            Deff is WeightSum ** InvN',
        '        ),',
        '        Pairs),',
        '    sort(Pairs, Results).',
        '',
        'compute_effective_distance_results_from_weight_sums(Root, Results) :-',
        '    collect_articles(Articles),',
        '    dimension_n(N),',
        '    InvN is -1 / N,',
        '    findall(Deff-Article,',
        '        (   member(Article, Articles),',
        '            aggregate_all(sum(W),',
        '                seeded_article_root_weight(Article, Root, W),',
        '                WeightSum),',
        '            WeightSum > 0,',
        '            Deff is WeightSum ** InvN',
        '        ),',
        '        Pairs),',
        '    sort(Pairs, Results).',
        '',
        'print_effective_distance_results(Root, Results) :-',
        '    format("article\\troot_category\\teffective_distance~n"),',
        '    forall(',
        '        member(Distance-Article, Results),',
        '        format("~w\\t~w\\t~6f~n", [Article, Root, Distance])',
        '    ).',
        '',
        'run_benchmark :-',
        '    statistics(walltime, [LoadStartMs, _]),',
        '    load_benchmark_facts,',
        '    statistics(walltime, [LoadEndMs, _]),',
        '    statistics(inferences, InferencesBefore),',
        '    root_category(Root),',
        IndexBuildLine,
        '    statistics(walltime, [QueryEndMs, _]),',
        ResultsLine,
        '    statistics(walltime, [AggEndMs, _]),',
        '    print_effective_distance_results(Root, Results),',
        '    statistics(inferences, InferencesAfter),',
        '    LoadMs is LoadEndMs - LoadStartMs,',
        '    QueryMs is QueryEndMs - LoadEndMs,',
        '    AggregationMs is AggEndMs - QueryEndMs,',
        '    TotalMs is AggEndMs - LoadStartMs,',
        '    Inferences is InferencesAfter - InferencesBefore,',
        '    length(Results, ArticleCount),',
        ModeLine,
        '    format(user_error, ''load_ms=~w~n'', [LoadMs]),',
        '    format(user_error, ''query_ms=~w~n'', [QueryMs]),',
        '    format(user_error, ''aggregation_ms=~w~n'', [AggregationMs]),',
        '    format(user_error, ''total_ms=~w~n'', [TotalMs]),',
        '    format(user_error, ''inferences=~w~n'', [Inferences]),',
        '    format(user_error, ''seed_count=~w~n'', [SeedCount]),',
        '    format(user_error, ''tuple_count=~w~n'', [TupleCount]),',
        '    format(user_error, ''article_count=~w~n'', [ArticleCount]).'
    ],
    atomic_list_concat(Lines, '\n', Code).

benchmark_index_build_line(seeded, '    build_seed_hops_index(Root, SeedCount, TupleCount),').
benchmark_index_build_line(pruned, '    build_seed_hops_index(Root, SeedCount, TupleCount),').
benchmark_index_build_line(accumulated, '    build_seed_weight_sum_index(Root, SeedCount, TupleCount),').

benchmark_results_line(seeded, '    compute_effective_distance_results_from_hops(Root, Results),').
benchmark_results_line(pruned, '    compute_effective_distance_results_from_hops(Root, Results),').
benchmark_results_line(accumulated, '    compute_effective_distance_results_from_weight_sums(Root, Results),').

benchmark_weight_sum_query_line(seeded, '    fail.').
benchmark_weight_sum_query_line(pruned, '    fail.').
benchmark_weight_sum_query_line(accumulated, '    ''category_ancestor$effective_distance_sum''(Cat, Root, WeightSum).').

benchmark_mode_line(seeded, '    format(user_error, ''mode=seeded~n'', []),').
benchmark_mode_line(pruned, '    format(user_error, ''mode=pruned~n'', []),').
benchmark_mode_line(accumulated, '    format(user_error, ''mode=accumulated~n'', []),').
