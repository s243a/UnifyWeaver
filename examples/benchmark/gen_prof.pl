:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, [_, OutputDir]),
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    Predicates = [user:dimension_n/1, user:max_depth/1, user:category_ancestor/4, user:power_sum_bound/4],
    Options = [module_name('wam-prof'), no_kernels(true), emit_mode(interpreter)],
    write_wam_haskell_project(Predicates, Options, OutputDir),
    halt(0).
main :- halt(1).

power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).
