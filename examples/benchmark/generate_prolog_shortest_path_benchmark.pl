:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'shortest_path_to_root.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputPath, BranchSettingAtom]
    ->  true
    ;   format(user_error,
            'Usage: swipl -q -s generate_prolog_shortest_path_benchmark.pl -- <facts.pl> <output-script> <auto|false>~n',
            []),
        halt(1)
    ),
    parse_branch_setting(BranchSettingAtom, BranchSetting),
    benchmark_workload_path(WorkloadPath),
    load_files(FactsPath, [silent(true)]),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    Predicates = [
        category_parent/2,
        article_category/2,
        root_category/1,
        max_depth/1,
        category_ancestor/4,
        category_ancestor/3,
        shortest_path/3,
        article_root_distance/3,
        run/0
    ],
    Options = [
        dialect(swi),
        entry_point(run),
        branch_pruning(BranchSetting)
    ],
    prolog_target:generate_prolog_script(Predicates, Options, ScriptCode),
    prolog_target:write_prolog_script(ScriptCode, OutputPath).

parse_branch_setting('auto', auto) :- !.
parse_branch_setting('false', false) :- !.
parse_branch_setting(Atom, _) :-
    format(user_error,
        'Unsupported branch_pruning setting: ~w (expected auto or false)~n',
        [Atom]),
    halt(1).
