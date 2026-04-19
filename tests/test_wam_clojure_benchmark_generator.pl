:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../examples/benchmark/generate_wam_clojure_optimized_benchmark').

:- begin_tests(wam_clojure_benchmark_generator).

test(kernel_mode_options) :-
    parse_kernel_mode(kernels_on, OnOptions),
    parse_kernel_mode(kernels_off, OffOptions),
    assertion(member(foreign_predicates([category_parent/2]), OnOptions)),
    assertion(member(clojure_foreign_handlers(_), OnOptions)),
    assertion(OffOptions == [no_kernels(true)]).

test(collect_seeded_predicates) :-
    collect_wam_predicates(seeded, Predicates),
    assertion(member(user:dimension_n/1, Predicates)),
    assertion(member(user:max_depth/1, Predicates)),
    assertion(member(user:category_ancestor/4, Predicates)),
    assertion(member(user:power_sum_bound/4, Predicates)).

test(generate_seeded_kernels_on_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_on', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(sub_string(CoreCode, _, _, _, '(def foreign-handlers {')),
        assertion(sub_string(CoreCode, _, _, _, '"category_parent/2" (fn [_args] false)')),
        assertion(sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_parent" :arity 2}')),
        delete_directory_and_contents(TmpDir)
    )).

test(generate_seeded_kernels_off_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_off', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_off),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(sub_string(CoreCode, _, _, _, '(def foreign-handlers {')),
        assertion(\+ sub_string(CoreCode, _, _, _, '"category_parent/2" (fn [_args] false)')),
        assertion(\+ sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_parent" :arity 2}')),
        delete_directory_and_contents(TmpDir)
    )).

:- end_tests(wam_clojure_benchmark_generator).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).
