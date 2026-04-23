:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../examples/benchmark/generate_wam_clojure_optimized_benchmark').

:- begin_tests(wam_clojure_benchmark_generator).

test(kernel_mode_options) :-
    parse_kernel_mode(kernels_on, OnOptions),
    parse_kernel_mode(kernels_off, OffOptions),
    assertion(member(foreign_predicates([category_parent/2, category_ancestor/4]), OnOptions)),
    assertion(member(clojure_foreign_handlers(_), OnOptions)),
    assertion(OffOptions == [no_kernels(true)]).

test(data_mode_options) :-
    parse_benchmark_data_mode(sidecar, sidecar),
    parse_benchmark_data_mode(artifact, artifact),
    parse_benchmark_data_mode(inline, inline),
    setup_call_cleanup(
        load_files(user:'data/benchmark/dev/facts.pl', [silent(true)]),
        assertion(parse_benchmark_data_mode(auto, sidecar)),
        true
    ).

test(data_mode_predicate_override) :-
    setup_call_cleanup(
        ( load_files(user:'data/benchmark/dev/facts.pl', [silent(true)]),
          assertz(user:wam_clojure_benchmark_data_mode(artifact))
        ),
        assertion(parse_benchmark_data_mode(auto, artifact)),
        maybe_abolish_test_predicate(wam_clojure_benchmark_data_mode/1)
    ).

test(relation_data_mode_override_seeded_article_artifact) :-
    setup_call_cleanup(
        ( load_files(user:'data/benchmark/dev/facts.pl', [silent(true)]),
          assertz(user:wam_clojure_benchmark_relation_data_mode(article_category, artifact))
        ),
        once((
            unique_tmp_dir('tmp_wam_clojure_bench_rel_article', TmpDir),
            generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on, artifact),
            directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
            directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/manifest.edn', ManifestPath),
            read_file_to_string(CorePath, CoreCode, []),
            read_file_to_string(ManifestPath, Manifest, []),
            assertion(sub_string(CoreCode, _, _, _, 'article_category_by_article.tsv')),
            assertion(\+ sub_string(CoreCode, _, _, _, '(def benchmark-article-categories-delay')),
            assertion(sub_string(Manifest, _, _, _, '"article_category" {:mode "artifact"')),
            assertion(sub_string(Manifest, _, _, _, ':file "article_category_by_article.tsv"')),
            delete_directory_and_contents(TmpDir)
        )),
        maybe_abolish_test_predicate(wam_clojure_benchmark_relation_data_mode/2)
    ).

test(relation_data_mode_override_accumulated_parent_sidecar) :-
    setup_call_cleanup(
        ( load_files(user:'data/benchmark/dev/facts.pl', [silent(true)]),
          assertz(user:wam_clojure_benchmark_relation_data_mode(category_parent, sidecar))
        ),
        once((
            unique_tmp_dir('tmp_wam_clojure_bench_rel_parent', TmpDir),
            generate('data/benchmark/dev/facts.pl', TmpDir, accumulated, kernels_on, artifact),
            directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
            directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/manifest.edn', ManifestPath),
            read_file_to_string(CorePath, CoreCode, []),
            read_file_to_string(ManifestPath, Manifest, []),
            assertion(sub_string(CoreCode, _, _, _, 'category_parent.edn')),
            assertion(\+ sub_string(CoreCode, _, _, _, 'category_parent_by_child.tsv')),
            assertion(sub_string(CoreCode, _, _, _, '(def benchmark-category-parents-delay')),
            assertion(sub_string(CoreCode, _, _, _, '(def benchmark-parents-by-child-delay')),
            assertion(sub_string(Manifest, _, _, _, '"category_parent" {:mode "sidecar"')),
            assertion(sub_string(Manifest, _, _, _, ':file "category_parent.edn"')),
            delete_directory_and_contents(TmpDir)
        )),
        maybe_abolish_test_predicate(wam_clojure_benchmark_relation_data_mode/2)
    ).

test(collect_seeded_predicates) :-
    collect_wam_predicates(seeded, Predicates),
    assertion(member(user:dimension_n/1, Predicates)),
    assertion(member(user:max_depth/1, Predicates)),
    assertion(member(user:category_ancestor/4, Predicates)),
    assertion(member(user:power_sum_bound/4, Predicates)).

test(generate_seeded_kernels_on_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_on', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on, sidecar),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/category_parent.edn', CategoryParentPath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/article_category.edn', ArticleCategoryPath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/root_category.edn', RootCategoryPath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/manifest.edn', ManifestPath),
        read_file_to_string(CorePath, CoreCode, []),
        read_file_to_string(ManifestPath, Manifest, []),
        assertion(exists_file(CategoryParentPath)),
        assertion(exists_file(ArticleCategoryPath)),
        assertion(exists_file(RootCategoryPath)),
        assertion(exists_file(ManifestPath)),
        assertion(sub_string(CoreCode, _, _, _, '(def foreign-handlers {')),
        assertion(sub_string(CoreCode, _, _, _, 'slurp "/')),
        assertion(sub_string(CoreCode, _, _, _, '"category_ancestor/4" (let [parents-by-child-delay (delay (reduce')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-data-dir "/')),
        assertion(sub_string(CoreCode, _, _, _, '(slurp (str benchmark-data-dir "/category_parent.edn"))')),
        assertion(sub_string(Manifest, _, _, _, '"unifyweaver.clojure_benchmark_artifacts.v1"')),
        assertion(sub_string(Manifest, _, _, _, ':data_mode "sidecar"')),
        assertion(sub_string(Manifest, _, _, _, '"category_parent" {:mode "sidecar"')),
        assertion(sub_string(Manifest, _, _, _, ':row_count')),
        assertion(sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_parent" :arity 2}')),
        assertion(sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_ancestor" :arity 4}')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-use-traversal-kernel? true)')),
        assertion(sub_string(CoreCode, _, _, _, '(defn benchmark-foreign-category-root-hops [category root]')),
        assertion(sub_string(CoreCode, _, _, _, '(handler [category root {:var 9001} (benchmark-list-term [category])]')),
        assertion(sub_string(CoreCode, _, _, _, '(benchmark-foreign-category-root-hops category root)')),
        assertion(\+ sub_string(CoreCode, _, _, _, 'benchmark-ancestor-hops-index')),
        assertion(\+ sub_string(CoreCode, _, _, _, 'benchmark-build-ancestor-hops-index')),
        delete_directory_and_contents(TmpDir)
    )).

test(generate_seeded_kernels_off_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_off', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_off, sidecar),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/category_parent.edn', CategoryParentPath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(exists_file(CategoryParentPath)),
        assertion(sub_string(CoreCode, _, _, _, '(def foreign-handlers {')),
        assertion(\+ sub_string(CoreCode, _, _, _, '"category_parent/2" (let [edges-delay (delay')),
        assertion(\+ sub_string(CoreCode, _, _, _, '"category_ancestor/4" (let [parents-by-child-delay (delay')),
        assertion(\+ sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_parent" :arity 2}')),
        assertion(\+ sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_ancestor" :arity 4}')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-use-traversal-kernel? false)')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-data-dir "/')),
        assertion(sub_string(CoreCode, _, _, _, '(benchmark-ancestor-hops category root #{category})')),
        assertion(\+ sub_string(CoreCode, _, _, _, 'benchmark-ancestor-hops-index')),
        assertion(\+ sub_string(CoreCode, _, _, _, 'benchmark-build-ancestor-hops-index')),
        delete_directory_and_contents(TmpDir)
    )).

test(generate_seeded_inline_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_inline', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on, inline),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/category_parent.edn', CategoryParentPath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(\+ exists_file(CategoryParentPath)),
        assertion(sub_string(CoreCode, _, _, _, '"category_parent/2" (let [edges #{')),
        assertion(sub_string(CoreCode, _, _, _, '"category_ancestor/4" (let [parents-by-child {')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-article-categories-delay (delay [[')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-category-parents-delay (delay [[')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-roots-delay (delay [')),
        delete_directory_and_contents(TmpDir)
    )).

test(generate_seeded_artifact_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_artifact', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on, artifact),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/category_parent_by_child.tsv', ParentByChildPath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/article_category_by_article.tsv', ArticleByArticlePath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/manifest.edn', ManifestPath),
        read_file_to_string(CorePath, CoreCode, []),
        read_file_to_string(ManifestPath, Manifest, []),
        assertion(exists_file(ParentByChildPath)),
        assertion(exists_file(ArticleByArticlePath)),
        assertion(exists_file(ManifestPath)),
        assertion(sub_string(CoreCode, _, _, _, '"category_parent/2" (let [parents-by-child-delay (delay (into {}')),
        assertion(sub_string(CoreCode, _, _, _, '"category_ancestor/4" (let [parents-by-child-delay (delay (into {}')),
        assertion(sub_string(CoreCode, _, _, _, 'category_parent_by_child.tsv')),
        assertion(\+ sub_string(CoreCode, _, _, _, 'article_category_by_article.tsv')),
        assertion(sub_string(Manifest, _, _, _, ':data_mode "artifact"')),
        assertion(sub_string(Manifest, _, _, _, '"article_category" {:mode "sidecar"')),
        assertion(sub_string(Manifest, _, _, _, '"category_parent" {:mode "artifact"')),
        assertion(sub_string(Manifest, _, _, _, ':file "category_parent_by_child.tsv"')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-article-categories-delay')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-parents-by-child-delay')),
        assertion(\+ sub_string(CoreCode, _, _, _, '(def benchmark-category-parents-delay')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-article-categories-by-article-delay')),
        delete_directory_and_contents(TmpDir)
    )).

test(generate_accumulated_artifact_project) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_acc_artifact', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, accumulated, kernels_on, artifact),
        directory_file_path(TmpDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
        directory_file_path(TmpDir, 'data/generated/wam_clojure_optimized_bench/manifest.edn', ManifestPath),
        read_file_to_string(CorePath, CoreCode, []),
        read_file_to_string(ManifestPath, Manifest, []),
        assertion(sub_string(CoreCode, _, _, _, 'article_category_by_article.tsv')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-article-categories-by-article-delay')),
        assertion(\+ sub_string(CoreCode, _, _, _, '(def benchmark-article-categories-delay')),
        assertion(sub_string(CoreCode, _, _, _, '(def benchmark-parents-by-child-delay')),
        assertion(\+ sub_string(CoreCode, _, _, _, '(def benchmark-category-parents-delay')),
        assertion(sub_string(Manifest, _, _, _, '"article_category" {:mode "artifact"')),
        assertion(sub_string(Manifest, _, _, _, '"category_parent" {:mode "artifact"')),
        delete_directory_and_contents(TmpDir)
    )).

test(generated_category_parent_handler_executes, [condition(clojure_available)]) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_exec', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on, artifact),
        run_clojure_predicate(TmpDir, 'category_parent/2',
                              ['Abstraction', 'Thought'], "true"),
        run_clojure_predicate(TmpDir, 'category_parent/2',
                              ['Abstraction', 'NotAParent'], "false"),
        run_clojure_predicate(TmpDir, 'category_ancestor/4',
                              ['Abstraction', 'Thought', raw('{:var 1000}'), visited(['Abstraction'])],
                              "true"),
        run_clojure_predicate(TmpDir, 'category_ancestor/4',
                              ['Abstraction', 'Cognition', raw('{:var 1000}'), visited(['Abstraction'])],
                              "true"),
        run_clojure_predicate(TmpDir, 'category_ancestor/4',
                              ['Abstraction', 'NotAParent', raw('{:var 1000}'), visited(['Abstraction'])],
                              "false"),
        delete_directory_and_contents(TmpDir)
    )).

:- end_tests(wam_clojure_benchmark_generator).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).

run_clojure_predicate(ProjectDir, PredKey, Args, Output) :-
    find_clojure_classpath(ClassPath),
    maplist(clojure_edn_arg, Args, EdnArgs),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m',
                    'generated.wam_clojure_optimized_bench.core', PredKey|EdnArgs],
                   [ cwd(ProjectDir),
                     stdout(pipe(Out)),
                     stderr(pipe(Err)),
                     process(PID)
                   ]),
    process_wait(PID, Status, [timeout(20)]),
    (   Status == timeout
    ->  process_kill(PID)
    ;   true
    ),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    (   Status == exit(0)
    ->  true
    ;   throw(error(java_exit(PredKey, Args, Status, ErrStr), _))
    ),
    normalize_space(string(Output), OutStr0),
    (   ErrStr == ""
    ->  true
    ;   throw(error(java_stderr(PredKey, Args, ErrStr), _))
    ).

clojure_edn_arg(raw(Edn), Edn) :- !.
clojure_edn_arg(visited(Atoms), Edn) :- !,
    visited_list_edn(Atoms, Edn).
clojure_edn_arg(Atom, Edn) :-
    format(string(Edn), '"~w"', [Atom]).

visited_list_edn([], '"[]"').
visited_list_edn([Atom|Rest], Edn) :-
    visited_list_edn(Rest, Tail),
    format(string(Edn),
           '{:tag :struct :functor "[|]/2" :args ["~w" ~w]}',
           [Atom, Tail]).

clojure_available :-
    find_clojure_classpath(_).

find_clojure_classpath(ClassPath) :-
    findall(Path,
        ( member(Path,
              [ '/data/data/com.termux/files/home/.m2/repository/org/clojure/clojure/1.11.1/clojure-1.11.1.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/spec.alpha/0.3.218/spec.alpha-0.3.218.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/core.specs.alpha/0.2.62/core.specs.alpha-0.2.62.jar'
              ]),
          exists_file(Path)
        ),
        JarPaths),
    JarPaths \= [],
    atomic_list_concat(['src'|JarPaths], :, ClassPath).

maybe_abolish_test_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).
