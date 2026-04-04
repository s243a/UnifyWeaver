:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'shortest_path_to_root.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputPath, VariantAtom]
    ->  true
    ;   format(user_error,
            'Usage: swipl -q -s generate_prolog_shortest_path_seeded_benchmark.pl -- <facts.pl> <output-script> <all|min>~n',
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
        max_depth/1,
        category_ancestor/4,
        category_ancestor/3
    ],
    prolog_target:generate_prolog_script(Predicates, Options, BaseScriptCode),
    benchmark_driver_code(Variant, FactsAbsPath, DriverCode),
    atomic_list_concat([BaseScriptCode, DriverCode], '\n\n', ScriptCode),
    prolog_target:write_prolog_script(ScriptCode, OutputPath).

parse_variant('all', all, [
        dialect(swi),
        entry_point(run_benchmark),
        branch_pruning(false),
        min_closure(false)
    ]) :-
    !.
parse_variant('min', min, [
        dialect(swi),
        entry_point(run_benchmark),
        branch_pruning(false),
        min_closure(auto)
    ]) :-
    !.
parse_variant(Atom, _, _) :-
    format(user_error,
        'Unsupported seeded benchmark variant: ~w (expected all or min)~n',
        [Atom]),
    halt(1).

benchmark_driver_code(Variant, FactsPath, Code) :-
    format(atom(FactsPathLine), 'facts_path(~q).', [FactsPath]),
    benchmark_query_line(Variant, QueryLine),
    benchmark_mode_line(Variant, ModeLine),
    Lines = [
        ':- use_module(library(aggregate)).',
        ':- dynamic bench_seed_distance/3.',
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
        'clear_seed_distance_index :-',
        '    retractall(bench_seed_distance(_, _, _)).',
        '',
        'seed_distance_query(Cat, Root, Hops) :-',
        QueryLine,
        '',
        'build_seed_distance_index(Root, SeedCount, TupleCount) :-',
        '    clear_seed_distance_index,',
        '    collect_seed_categories(Seeds),',
        '    length(Seeds, SeedCount),',
        '    forall(',
        '        member(Cat, Seeds),',
        '        forall(',
        '            seed_distance_query(Cat, Root, Hops),',
        '            assertz(bench_seed_distance(Cat, Root, Hops))',
        '        )',
        '    ),',
        '    aggregate_all(count, bench_seed_distance(_, _, _), TupleCount).',
        '',
        'seeded_article_root_distance(Article, Root, 1) :-',
        '    article_category(Article, Root).',
        'seeded_article_root_distance(Article, Root, Distance) :-',
        '    article_category(Article, Cat),',
        '    Cat \\= Root,',
        '    bench_seed_distance(Cat, Root, Hops),',
        '    Distance is Hops + 1.',
        '',
        'compute_shortest_path_results(Root, Results) :-',
        '    collect_articles(Articles),',
        '    findall(MinHops-Article,',
        '        (   member(Article, Articles),',
        '            aggregate_all(min(Dist),',
        '                seeded_article_root_distance(Article, Root, Dist),',
        '                MinHops),',
        '            MinHops \\= inf',
        '        ),',
        '        Pairs),',
        '    sort(Pairs, Results).',
        '',
        'print_shortest_path_results(Root, Results) :-',
        '    format("article\\troot_category\\tshortest_path~n"),',
        '    forall(',
        '        member(Hops-Article, Results),',
        '        format("~w\\t~w\\t~w~n", [Article, Root, Hops])',
        '    ).',
        '',
        'run_benchmark :-',
        '    get_time(TLoad0),',
        '    load_benchmark_facts,',
        '    get_time(TLoad1),',
        '    statistics(inferences, InferencesBefore),',
        '    root_category(Root),',
        '    build_seed_distance_index(Root, SeedCount, TupleCount),',
        '    get_time(T1),',
        '    compute_shortest_path_results(Root, Results),',
        '    get_time(T2),',
        '    print_shortest_path_results(Root, Results),',
        '    statistics(inferences, InferencesAfter),',
        '    LoadMs is round((TLoad1 - TLoad0) * 1000),',
        '    QueryMs is round((T1 - TLoad1) * 1000),',
        '    AggregationMs is round((T2 - T1) * 1000),',
        '    TotalMs is round((T2 - TLoad0) * 1000),',
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

benchmark_query_line(all, '    category_ancestor(Cat, Root, Hops).').
benchmark_query_line(min, '    ''category_ancestor$min''(Cat, Root, Hops).').

benchmark_mode_line(all, '    format(user_error, ''mode=all~n'', []),').
benchmark_mode_line(min, '    format(user_error, ''mode=min~n'', []),').
