:- module(generate_wam_clojure_optimized_benchmark,
          [ main/0,
            generate/3,
            generate/4,
            parse_kernel_mode/2,
            category_parent_handler_code/1,
            collect_wam_predicates/2,
            collect_wam_predicates/3
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_clojure_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module(library(filesex), [directory_file_path/3]).

%% generate_wam_clojure_optimized_benchmark.pl
%%
%% Generates a Clojure WAM project from the optimized effective-distance
%% Prolog workload. This is the Clojure counterpart to the optimized Go,
%% Python, Haskell, and Rust WAM benchmark generators.
%%
%% The generated project is intentionally a benchmark scaffold, not a large
%% Termux benchmark runner. JVM startup and memory use make large local runs
%% noisy; this generator gives the benchmark matrix a reproducible Clojure
%% project shape and explicit kernel controls.
%%
%% Usage:
%%   swipl -q -s generate_wam_clojure_optimized_benchmark.pl -- \
%%       <facts.pl> <output-dir> [accumulated|seeded] [kernels_on|kernels_off]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom, KernelModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom]
    ->  KernelModeAtom = kernels_on
    ;   Argv = [FactsPath, OutputDir]
    ->  VariantAtom = accumulated,
        KernelModeAtom = kernels_on
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [accumulated|seeded] [kernels_on|kernels_off]~n',
            []),
        halt(1)
    ),
    (   Argv == []
    ->  true
    ;   generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom),
        halt(0)
    ).

main :-
    format(user_error, 'Error: Clojure WAM benchmark generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, VariantAtom) :-
    generate(FactsPath, OutputDir, VariantAtom, kernels_on).

generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom) :-
    must_exist_file(FactsPath),
    benchmark_workload_path(WorkloadPath),
    load_files(user:FactsPath, [silent(true)]),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    load_files(user:WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    parse_variant(VariantAtom, OptimizationOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),

    maybe_load_optimized_predicates(VariantAtom, ScriptCode),

    maybe_assert_seeded_helper(VariantAtom),
    collect_wam_predicates(VariantAtom, KernelModeAtom, Predicates),
    append([ [ module_name('wam-clojure-optimized-bench'),
               namespace('generated.wam_clojure_optimized_bench')
             ],
             KernelOptions
           ],
           Options),
    write_wam_clojure_project(Predicates, Options, OutputDir),
    append_effective_distance_runner(OutputDir),
    format(user_error,
           '[WAM-Clojure-Optimized] facts=~w variant=~w kernels=~w output=~w~n',
           [FactsPath, VariantAtom, KernelModeAtom, OutputDir]).

parse_variant(seeded, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false)
]).
parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

parse_kernel_mode(kernels_on, [
    foreign_predicates([category_parent/2]),
    clojure_foreign_handlers([
        handler(category_parent/2, HandlerCode)
    ])
]) :-
    category_parent_handler_code(HandlerCode).
parse_kernel_mode(kernels_off, [
    no_kernels(true)
]).

category_parent_handler_code(HandlerCode) :-
    findall(Child-Parent,
            current_category_parent_fact(Child, Parent),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(category_parent_edge_literal, Pairs, EdgeLiterals),
    atomic_list_concat(EdgeLiterals, ' ', Edges),
    format(string(HandlerCode),
           "(let [edges #{~s}] (fn [args] (contains? edges [(nth args 0) (nth args 1)])))",
           [Edges]).

current_category_parent_fact(Child, Parent) :-
    current_predicate(user:category_parent/2),
    call(user:category_parent(Child, Parent)).

category_parent_edge_literal(Child-Parent, Literal) :-
    clj_string_literal_local(Child, ChildLit),
    clj_string_literal_local(Parent, ParentLit),
    format(atom(Literal), '[~w ~w]', [ChildLit, ParentLit]).

clj_string_literal_local(In, Literal) :-
    atom_string(In, InStr0),
    escape_clj_string_local(InStr0, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

escape_clj_string_local(In, Out) :-
    split_string(In, "\\", "", Parts1),
    atomic_list_concat(Parts1, "\\\\", Tmp1),
    split_string(Tmp1, "\"", "", Parts2),
    atomic_list_concat(Parts2, "\\\"", Out).

append_effective_distance_runner(OutputDir) :-
    directory_file_path(OutputDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
    read_file_to_string(CorePath, Core0, []),
    effective_distance_runner_code(RunnerCode),
    atomic_list_concat([Core0, RunnerCode], '\n\n', Core),
    setup_call_cleanup(
        open(CorePath, write, Stream),
        write(Stream, Core),
        close(Stream)
    ).

effective_distance_runner_code(Code) :-
    benchmark_article_category_literal(ArticleCategories),
    benchmark_category_parent_literal(CategoryParents),
    benchmark_root_category_literal(RootCategories),
    benchmark_dimension_n_literal(DimensionN),
    benchmark_max_depth_literal(MaxDepth),
    format(string(Code),
'
;; Effective-distance benchmark entrypoint generated from facts.pl.
;; Predicate calls are still available by passing a predicate key and args.
(def benchmark-article-categories [~s])
(def benchmark-category-parents [~s])
(def benchmark-roots [~s])
(def benchmark-dimension-n ~w)
(def benchmark-max-depth ~w)

(def benchmark-parents-by-child
  (reduce (fn [acc [child parent]]
            (update acc child (fnil conj []) parent))
          {}
          benchmark-category-parents))

(def benchmark-article-categories-by-article
  (reduce (fn [acc [article category]]
            (update acc article (fnil conj []) category))
          {}
          benchmark-article-categories))

(defn benchmark-ancestor-hops [category root visited]
  (let [parents (get benchmark-parents-by-child category [])]
    (vec
      (concat
        (for [parent parents
              :when (and (not (contains? visited parent))
                         (= parent root))]
          1)
        (when (< (count visited) benchmark-max-depth)
          (apply concat
            (for [mid parents
                  :when (not (contains? visited mid))
                  hop (benchmark-ancestor-hops mid root (conj visited mid))]
              [(inc hop)])))))))

(defn benchmark-article-root-weight [article root]
  (reduce
    +
    0.0
    (for [category (get benchmark-article-categories-by-article article [])
          weight (if (= category root)
                   [1.0]
                   (for [hops (benchmark-ancestor-hops category root #{category})]
                     (Math/pow (+ hops 1.0) (- benchmark-dimension-n))))]
      weight)))

(defn benchmark-effective-distance-rows []
  (let [inv-n (- (/ 1.0 benchmark-dimension-n))]
    (sort-by
      (fn [{:keys [distance article root]}] [distance article root])
      (for [root benchmark-roots
            article (sort (keys benchmark-article-categories-by-article))
            :let [weight-sum (benchmark-article-root-weight article root)]
            :when (pos? weight-sum)]
        {:article article
         :root root
         :distance (Math/pow weight-sum inv-n)}))))

(defn run-effective-distance-benchmark []
  (let [rows (benchmark-effective-distance-rows)]
    (binding [*out* *err*]
      (println "mode=clojure_wam_accumulated")
      (println (str "article_count=" (count rows))))
    (println "article\\troot_category\\teffective_distance")
    (doseq [{:keys [article root distance]} rows]
      (println (format "%s\\t%s\\t%.6f" article root distance)))))

(defn -main [& args]
  (if (seq args)
    (let [[pred-key & pred-args] args]
      (println (invoke-predicate pred-key (mapv edn/read-string pred-args))))
    (run-effective-distance-benchmark)))
',
           [ArticleCategories, CategoryParents, RootCategories, DimensionN, MaxDepth]).

benchmark_article_category_literal(Literal) :-
    findall(Article-Category,
            current_article_category_fact(Article, Category),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(edge_literal, Pairs, Literals),
    atomic_list_concat(Literals, ' ', Literal).

benchmark_category_parent_literal(Literal) :-
    findall(Child-Parent,
            current_category_parent_fact(Child, Parent),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(edge_literal, Pairs, Literals),
    atomic_list_concat(Literals, ' ', Literal).

benchmark_root_category_literal(Literal) :-
    findall(Root, current_root_category_fact(Root), Roots0),
    sort(Roots0, Roots),
    maplist(clj_string_literal_local, Roots, Literals),
    atomic_list_concat(Literals, ' ', Literal).

benchmark_dimension_n_literal(Literal) :-
    (   current_predicate(user:dimension_n/1),
        call(user:dimension_n(N))
    ->  format(atom(Literal), '~w.0', [N])
    ;   Literal = '5.0'
    ).

benchmark_max_depth_literal(Literal) :-
    (   current_predicate(user:max_depth/1),
        call(user:max_depth(MaxDepth))
    ->  Literal = MaxDepth
    ;   Literal = 10
    ).

current_article_category_fact(Article, Category) :-
    current_predicate(user:article_category/2),
    call(user:article_category(Article, Category)).

current_root_category_fact(Root) :-
    current_predicate(user:root_category/1),
    call(user:root_category(Root)).

edge_literal(Left-Right, Literal) :-
    clj_string_literal_local(Left, LeftLit),
    clj_string_literal_local(Right, RightLit),
    format(atom(Literal), '[~w ~w]', [LeftLit, RightLit]).

maybe_assert_seeded_helper(seeded) :- !,
    retractall(user:power_sum_bound(_, _, _, _)),
    assertz((user:power_sum_bound(Cat, Root, NegN, WeightSum) :-
        aggregate_all(sum(W),
            (user:category_ancestor(Cat, Root, Hops, [Cat]),
             H is Hops + 1,
             W is H ** NegN),
            WeightSum))).
maybe_assert_seeded_helper(_).

maybe_load_optimized_predicates(seeded, _) :- !.
maybe_load_optimized_predicates(accumulated, ScriptCode) :-
    strip_generated_initialization(ScriptCode, LoadableScript),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, LoadableScript),
    close(TmpStream),
    load_files(user:TmpPath, [silent(true)]),
    delete_file(TmpPath),
    cleanup_generated_script_entrypoint.

strip_generated_initialization(ScriptCode, LoadableScript) :-
    split_string(ScriptCode, "\n", "", Lines0),
    exclude(generated_initialization_line, Lines0, Lines),
    atomic_list_concat(Lines, '\n', LoadableScript).

generated_initialization_line(Line) :-
    sub_string(Line, _, _, _, "initialization(main").

cleanup_generated_script_entrypoint :-
    (   current_predicate(user:main/0)
    ->  abolish(user:main/0)
    ;   true
    ).

collect_wam_predicates(Variant, Predicates) :-
    collect_wam_predicates(Variant, kernels_on, Predicates).

collect_wam_predicates(seeded, kernels_on, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_parent/2,
    user:category_ancestor/4,
    user:power_sum_bound/4
]).
collect_wam_predicates(seeded, kernels_off, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:power_sum_bound/4
]).
collect_wam_predicates(accumulated, kernels_on, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_parent/2,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).
collect_wam_predicates(accumulated, kernels_off, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

must_exist_file(Path) :-
    (   exists_file(Path)
    ->  true
    ;   throw(error(existence_error(file, Path), _))
    ).

%% power_sum_bound/4 - needed for seeded variant.
power_sum_bound(Cat, Root, NegN, WeightSum) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, Hops, [Cat]),
         H is Hops + 1,
         W is H ** NegN),
        WeightSum).
