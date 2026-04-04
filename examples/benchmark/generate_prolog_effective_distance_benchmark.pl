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
            'Usage: swipl -q -s generate_prolog_effective_distance_benchmark.pl -- <facts.pl> <output-script> <seeded|pruned>~n',
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
parse_variant(Atom, _, _) :-
    format(user_error,
        'Unsupported effective-distance benchmark variant: ~w (expected seeded or pruned)~n',
        [Atom]),
    halt(1).

benchmark_driver_code(Variant, FactsPath, Code) :-
    format(atom(FactsPathLine), 'facts_path(~q).', [FactsPath]),
    benchmark_mode_line(Variant, ModeLine),
    Lines = [
        ':- use_module(library(aggregate)).',
        ':- use_module(library(lists)).',
        ':- dynamic bench_seed_hops/3.',
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
        'clear_seed_hops_index :-',
        '    retractall(bench_seed_hops(_, _, _)).',
        '',
        'seed_hops_query(Cat, Root, Hops) :-',
        '    category_ancestor(Cat, Root, Hops, [Cat]).',
        '',
        'build_seed_hops_index(Root, SeedCount, TupleCount) :-',
        '    clear_seed_hops_index,',
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
        'seeded_article_root_hops(Article, Root, 1) :-',
        '    article_category(Article, Root).',
        'seeded_article_root_hops(Article, Root, Hops) :-',
        '    article_category(Article, Cat),',
        '    Cat \\= Root,',
        '    bench_seed_hops(Cat, Root, CatHops),',
        '    Hops is CatHops + 1.',
        '',
        'compute_effective_distance_results(Root, Results) :-',
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
        '    build_seed_hops_index(Root, SeedCount, TupleCount),',
        '    statistics(walltime, [QueryEndMs, _]),',
        '    compute_effective_distance_results(Root, Results),',
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

benchmark_mode_line(seeded, '    format(user_error, ''mode=seeded~n'', []),').
benchmark_mode_line(pruned, '    format(user_error, ''mode=pruned~n'', []),').
