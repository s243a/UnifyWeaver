:- module(generate_wam_clojure_optimized_benchmark,
          [ main/0,
            generate/3,
            generate/4,
            generate/5,
            parse_kernel_mode/2,
            parse_benchmark_data_mode/2,
            category_parent_handler_code/1,
            category_ancestor_handler_code/1,
            collect_wam_predicates/2,
            collect_wam_predicates/3
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_clojure_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module('../../src/unifyweaver/core/predicate_preprocessing',
              [declared_preprocess/3,
               declared_preprocess_metadata/4]).
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(process), [process_create/3, process_wait/2]).
:- dynamic lmdb_artifact_builder_ready/1.
:- dynamic lmdb_artifact_target_dir/1.
:- discontiguous benchmark_article_category_by_article_artifact/1.
:- discontiguous benchmark_category_parent_by_child_artifact/1.
:- discontiguous category_parent_handler_code/3.
:- discontiguous category_parent_handler_code/4.
:- discontiguous category_ancestor_handler_code/3.
:- discontiguous category_ancestor_handler_code/4.

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
%%       [auto|sidecar|artifact|inline]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

benchmark_repo_root(Root) :-
    source_file(benchmark_repo_root(_), ThisFile),
    file_directory_name(ThisFile, BenchDir),
    file_directory_name(BenchDir, ExamplesDir),
    file_directory_name(ExamplesDir, Root).

benchmark_repo_path(RelativePath, AbsolutePath) :-
    benchmark_repo_root(Root),
    directory_file_path(Root, RelativePath, AbsolutePath).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom, KernelModeAtom, DataModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom, KernelModeAtom]
    ->  DataModeAtom = auto
    ;   Argv = [FactsPath, OutputDir, VariantAtom]
    ->  KernelModeAtom = kernels_on
        ,
        DataModeAtom = auto
    ;   Argv = [FactsPath, OutputDir]
    ->  VariantAtom = accumulated,
        KernelModeAtom = kernels_on,
        DataModeAtom = auto
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [accumulated|seeded] [kernels_on|kernels_off] [auto|sidecar|artifact|inline]~n',
            []),
        halt(1)
    ),
    (   Argv == []
    ->  true
    ;   generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom, DataModeAtom),
        halt(0)
    ).

main :-
    format(user_error, 'Error: Clojure WAM benchmark generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, VariantAtom) :-
    generate(FactsPath, OutputDir, VariantAtom, kernels_on, auto).

generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom) :-
    generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom, auto).

generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom, DataModeAtom) :-
    must_exist_file(FactsPath),
    benchmark_workload_path(WorkloadPath),
    reset_benchmark_predicates,
    load_files(user:WorkloadPath, [silent(true), if(true)]),
    load_files(user:FactsPath, [silent(true), if(true)]),
    parse_benchmark_data_mode(DataModeAtom, DataMode),
    parse_kernel_mode(OutputDir, VariantAtom, KernelModeAtom, DataMode, KernelOptions),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    parse_variant(VariantAtom, OptimizationOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),

    maybe_load_optimized_predicates(VariantAtom, ScriptCode),

    maybe_assert_seeded_helper(VariantAtom),
    collect_wam_predicates(VariantAtom, KernelModeAtom, Predicates),
    collect_benchmark_data(BenchmarkData),
    effective_distance_runner_code(OutputDir, VariantAtom, KernelModeAtom, DataMode, BenchmarkData, RunnerCode),
    append([ [ module_name('wam-clojure-optimized-bench'),
               namespace('generated.wam_clojure_optimized_bench'),
               clojure_benchmark_data_mode(DataMode)
             ],
             KernelOptions
           ],
           Options),
    write_wam_clojure_project(Predicates, Options, OutputDir),
    maybe_write_benchmark_sidecar_files(OutputDir, FactsPath, VariantAtom, KernelModeAtom, DataMode, BenchmarkData),
    append_effective_distance_runner(OutputDir, RunnerCode),
    format(user_error,
           '[WAM-Clojure-Optimized] facts=~w variant=~w kernels=~w data_mode=~w output=~w~n',
           [FactsPath, VariantAtom, KernelModeAtom, DataMode, OutputDir]).

reset_benchmark_predicates :-
    forall(member(Name/Arity,
                  [ article_category/2,
                    category_parent/2,
                    root_category/1,
                    dimension_n/1,
                    max_depth/1,
                    category_ancestor/4,
                    power_sum_bound/4,
                    'category_ancestor$power_sum_bound'/3,
                    'category_ancestor$power_sum_selected'/3,
                    'category_ancestor$effective_distance_sum_selected'/3,
                    'category_ancestor$effective_distance_sum_bound'/3
                  ]),
           maybe_abolish_user_predicate(Name/Arity)).

maybe_abolish_user_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).

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
    foreign_predicates([category_parent/2, category_ancestor/4]),
    clojure_foreign_handlers([
        handler(category_parent/2, ParentHandlerCode),
        handler(category_ancestor/4, AncestorHandlerCode)
    ])
]) :-
    DataMode = sidecar,
    category_parent_handler_code(DataMode, ParentHandlerCode),
    category_ancestor_handler_code(DataMode, AncestorHandlerCode).
parse_kernel_mode(kernels_off, [
    no_kernels(true)
]).

parse_kernel_mode(kernels_on, DataMode, [
    foreign_predicates([category_parent/2, category_ancestor/4]),
    clojure_foreign_handlers([
        handler(category_parent/2, ParentHandlerCode),
        handler(category_ancestor/4, AncestorHandlerCode)
    ])
]) :-
    category_parent_handler_code(DataMode, ParentHandlerCode),
    category_ancestor_handler_code(DataMode, AncestorHandlerCode).
parse_kernel_mode(kernels_off, _DataMode, [
    no_kernels(true)
]).

parse_kernel_mode(OutputDir, VariantAtom, kernels_on, DataMode, [
    foreign_predicates([category_parent/2, category_ancestor/4]),
    clojure_foreign_handlers([
        handler(category_parent/2, ParentHandlerCode),
        handler(category_ancestor/4, AncestorHandlerCode)
    ])
]) :-
    benchmark_effective_data_modes(VariantAtom, DataMode, _ArticleMode, ParentMode),
    benchmark_relation_cache_policy(category_parent, ParentMode, ParentCachePolicy),
    category_parent_handler_code(OutputDir, ParentMode, ParentCachePolicy, ParentHandlerCode),
    category_ancestor_handler_code(OutputDir, ParentMode, ParentCachePolicy, AncestorHandlerCode).
parse_kernel_mode(_OutputDir, _VariantAtom, kernels_off, _DataMode, [
    no_kernels(true)
]).

parse_benchmark_data_mode(sidecar, sidecar).
parse_benchmark_data_mode(artifact, artifact).
parse_benchmark_data_mode(inline, inline).
parse_benchmark_data_mode(auto, Mode) :-
    (   benchmark_declared_data_mode(DeclaredMode),
        DeclaredMode \== auto
    ->  parse_benchmark_data_mode(DeclaredMode, Mode)
    ;   benchmark_fact_volume(RowCount),
        (   RowCount >= 128
        ->  Mode = sidecar
        ;   Mode = inline
        )
    ).

benchmark_declared_data_mode(Mode) :-
    member(Name,
           [ wam_clojure_benchmark_data_mode,
             benchmark_data_mode
           ]),
    current_predicate(user:Name/1),
    Goal =.. [Name, DeclaredMode],
    once(call(user:Goal)),
    memberchk(DeclaredMode, [auto, sidecar, artifact, inline, lmdb]),
    Mode = DeclaredMode.

benchmark_relation_declared_data_mode(Relation, Mode) :-
    member(Name,
           [ wam_clojure_benchmark_relation_data_mode,
             benchmark_relation_data_mode
           ]),
    current_predicate(user:Name/2),
    Goal =.. [Name, Relation, DeclaredMode],
    once(call(user:Goal)),
    memberchk(DeclaredMode, [auto, sidecar, artifact, inline, lmdb]),
    Mode = DeclaredMode.

benchmark_relation_declared_cache_policy(Relation, Policy) :-
    member(Name,
           [ wam_clojure_benchmark_relation_cache_policy,
             benchmark_relation_cache_policy
           ]),
    current_predicate(user:Name/2),
    Goal =.. [Name, Relation, DeclaredPolicy],
    once(call(user:Goal)),
    memberchk(DeclaredPolicy, [none, memoize, shared, two_level]),
    Policy = DeclaredPolicy.

benchmark_relation_cache_policy(_Relation, Mode, none) :-
    Mode \== lmdb,
    !.
benchmark_relation_cache_policy(Relation, _Mode, Policy) :-
    (   benchmark_relation_declared_cache_policy(Relation, DeclaredPolicy)
    ->  Policy = DeclaredPolicy
    ;   Policy = none
    ).

benchmark_relation_declared_cache_debug(Relation, Debug) :-
    member(Name,
           [ wam_clojure_benchmark_relation_cache_debug,
             benchmark_relation_cache_debug
           ]),
    current_predicate(user:Name/2),
    Goal =.. [Name, Relation, DeclaredDebug],
    once(call(user:Goal)),
    memberchk(DeclaredDebug, [true, false]),
    Debug = DeclaredDebug.

benchmark_relation_cache_debug(_Relation, Mode, false) :-
    Mode \== lmdb,
    !.
benchmark_relation_cache_debug(Relation, _Mode, Debug) :-
    (   benchmark_relation_declared_cache_debug(Relation, DeclaredDebug)
    ->  Debug = DeclaredDebug
    ;   Debug = false
    ).

benchmark_fact_volume(RowCount) :-
    benchmark_fact_count(user:category_parent/2, ParentCount),
    benchmark_fact_count(user:article_category/2, ArticleCount),
    benchmark_fact_count(user:root_category/1, RootCount),
    RowCount is ParentCount + ArticleCount + RootCount.

benchmark_fact_count(Module:Name/Arity, Count) :-
    functor(Goal, Name, Arity),
    findall(Goal, Module:Goal, Goals),
    length(Goals, Count).

category_parent_handler_code(HandlerCode) :-
    parse_benchmark_data_mode(auto, DataMode),
    category_parent_handler_code(DataMode, HandlerCode).

category_parent_handler_code(OutputDir, sidecar, _CachePolicy, HandlerCode) :-
    category_parent_handler_code(OutputDir, sidecar, HandlerCode).
category_parent_handler_code(OutputDir, artifact, _CachePolicy, HandlerCode) :-
    category_parent_handler_code(OutputDir, artifact, HandlerCode).
category_parent_handler_code(OutputDir, inline, _CachePolicy, HandlerCode) :-
    category_parent_handler_code(OutputDir, inline, HandlerCode).
category_parent_handler_code(OutputDir, sidecar, HandlerCode) :-
    benchmark_sidecar_file_path_literal(OutputDir, 'category_parent.edn', CategoryParentPath),
    format(string(HandlerCode),
           "(let [edges-delay (delay (set (edn/read-string (slurp ~w))))] (fn [args] (contains? @edges-delay [(nth args 0) (nth args 1)])))",
           [CategoryParentPath]).
category_parent_handler_code(OutputDir, artifact, HandlerCode) :-
    benchmark_sidecar_file_path_literal(OutputDir, 'category_parent_by_child.tsv', CategoryParentMapPath),
    format(string(HandlerCode),
           "(let [parents-by-child-delay (delay (into {} (keep (fn [line] (when-not (.isEmpty ^String line) (let [parts (vec (.split ^String line \"\\\\t\")) child (first parts) parents (subvec parts 1)] [child parents]))) (.split ^String (slurp ~w) \"\\\\r?\\\\n\"))))] (fn [args] (let [child (nth args 0) parent (nth args 1)] (boolean (some #(= parent %) (get @parents-by-child-delay child []))))))",
           [CategoryParentMapPath]).
category_parent_handler_code(OutputDir, lmdb, CachePolicy, HandlerCode) :-
    benchmark_sidecar_subdir_literal(OutputDir, 'category_parent_lmdb', ArtifactDirPath),
    wam_clojure_target:clojure_lmdb_reader_open_expr(
        ArtifactDirPath,
        [clojure_lmdb_cache_policy(CachePolicy)],
        ReaderOpenExpr),
    benchmark_relation_cache_debug(category_parent, lmdb, CacheDebug),
    wam_clojure_target:clojure_lmdb_cache_log_snippet(
        'category_parent/2',
        [clojure_lmdb_cache_debug(CacheDebug)],
        ParentLogCode),
    format(string(HandlerCode),
           "(let [reader-delay (delay ~w)] (fn [args] (let [child (nth args 0) parent (nth args 1) rows (.lookupArg1 ^generated.lmdb.LmdbArtifactReader @reader-delay child) result (boolean (some (fn [^generated.lmdb.LmdbRow row] (= parent (.value row))) rows))] (do ~w result))))",
           [ReaderOpenExpr, ParentLogCode]).
category_parent_handler_code(_OutputDir, inline, HandlerCode) :-
    category_parent_handler_code(inline, HandlerCode).
category_parent_handler_code(sidecar, HandlerCode) :-
    format(string(HandlerCode),
           "(let [edges-delay (delay (set (edn/read-string (slurp \"data/generated/wam_clojure_optimized_bench/category_parent.edn\"))))] (fn [args] (contains? @edges-delay [(nth args 0) (nth args 1)])))",
           []).
category_parent_handler_code(artifact, HandlerCode) :-
    format(string(HandlerCode),
           "(let [parents-by-child-delay (delay (into {} (keep (fn [line] (when-not (.isEmpty ^String line) (let [parts (vec (.split ^String line \"\\\\t\")) child (first parts) parents (subvec parts 1)] [child parents]))) (.split ^String (slurp \"data/generated/wam_clojure_optimized_bench/category_parent_by_child.tsv\") \"\\\\r?\\\\n\"))))] (fn [args] (let [child (nth args 0) parent (nth args 1)] (boolean (some #(= parent %) (get @parents-by-child-delay child []))))))",
           []).
category_parent_handler_code(inline, HandlerCode) :-
    findall(Child-Parent,
            current_category_parent_fact(Child, Parent),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(category_parent_edge_literal, Pairs, EdgeLiterals),
    atomic_list_concat(EdgeLiterals, ' ', Edges),
    format(string(HandlerCode),
           "(let [edges #{~s}] (fn [args] (contains? edges [(nth args 0) (nth args 1)])))",
           [Edges]).

category_ancestor_handler_code(HandlerCode) :-
    parse_benchmark_data_mode(auto, DataMode),
    category_ancestor_handler_code(DataMode, HandlerCode).

category_ancestor_handler_code(OutputDir, sidecar, _CachePolicy, HandlerCode) :-
    category_ancestor_handler_code(OutputDir, sidecar, HandlerCode).
category_ancestor_handler_code(OutputDir, artifact, _CachePolicy, HandlerCode) :-
    category_ancestor_handler_code(OutputDir, artifact, HandlerCode).
category_ancestor_handler_code(OutputDir, inline, _CachePolicy, HandlerCode) :-
    category_ancestor_handler_code(OutputDir, inline, HandlerCode).
category_ancestor_handler_code(OutputDir, sidecar, HandlerCode) :-
    benchmark_max_depth_literal(MaxDepth),
    benchmark_sidecar_file_path_literal(OutputDir, 'category_parent.edn', CategoryParentPath),
    format(string(HandlerCode),
           "(let [parents-by-child-delay (delay (reduce (fn [acc [child parent]] (update acc child (fnil conj []) parent)) {} (edn/read-string (slurp ~w)))) max-depth ~w term-list-values (fn term-list-values [term] (if (and (map? term) (= \"[|]/2\" (:functor term))) (cons (first (:args term)) (term-list-values (second (:args term)))) [])) ancestor-hops (fn ancestor-hops [category target visited] (let [parents (get @parents-by-child-delay category [])] (vec (concat (for [parent parents :when (and (not (contains? visited parent)) (or (map? target) (= parent target)))] [parent 1]) (when (< (count visited) max-depth) (apply concat (for [mid parents :when (not (contains? visited mid)) [ancestor hops] (ancestor-hops mid target (conj visited mid))] [[ancestor (inc hops)]])))))))] (fn [args] (let [category (nth args 0) target (nth args 1) visited (set (term-list-values (nth args 3))) solutions (for [[ancestor hops] (ancestor-hops category target visited)] {:bindings {2 ancestor 3 hops}})] {:solutions (vec solutions)})))",
           [CategoryParentPath, MaxDepth]).
category_ancestor_handler_code(OutputDir, artifact, HandlerCode) :-
    benchmark_max_depth_literal(MaxDepth),
    benchmark_sidecar_file_path_literal(OutputDir, 'category_parent_by_child.tsv', CategoryParentMapPath),
    format(string(HandlerCode),
           "(let [parents-by-child-delay (delay (into {} (keep (fn [line] (when-not (.isEmpty ^String line) (let [parts (vec (.split ^String line \"\\\\t\")) child (first parts) parents (subvec parts 1)] [child parents]))) (.split ^String (slurp ~w) \"\\\\r?\\\\n\")))) max-depth ~w term-list-values (fn term-list-values [term] (if (and (map? term) (= \"[|]/2\" (:functor term))) (cons (first (:args term)) (term-list-values (second (:args term)))) [])) ancestor-hops (fn ancestor-hops [category target visited] (let [parents (get @parents-by-child-delay category [])] (vec (concat (for [parent parents :when (and (not (contains? visited parent)) (or (map? target) (= parent target)))] [parent 1]) (when (< (count visited) max-depth) (apply concat (for [mid parents :when (not (contains? visited mid)) [ancestor hops] (ancestor-hops mid target (conj visited mid))] [[ancestor (inc hops)]])))))))] (fn [args] (let [category (nth args 0) target (nth args 1) visited (set (term-list-values (nth args 3))) solutions (for [[ancestor hops] (ancestor-hops category target visited)] {:bindings {2 ancestor 3 hops}})] {:solutions (vec solutions)})))",
           [CategoryParentMapPath, MaxDepth]).
category_ancestor_handler_code(OutputDir, lmdb, CachePolicy, HandlerCode) :-
    benchmark_max_depth_literal(MaxDepth),
    benchmark_sidecar_subdir_literal(OutputDir, 'category_parent_lmdb', ArtifactDirPath),
    wam_clojure_target:clojure_lmdb_reader_open_expr(
        ArtifactDirPath,
        [clojure_lmdb_cache_policy(CachePolicy)],
        ReaderOpenExpr),
    benchmark_relation_cache_debug(category_parent, lmdb, CacheDebug),
    wam_clojure_target:clojure_lmdb_cache_log_snippet(
        'category_ancestor/4',
        [clojure_lmdb_cache_debug(CacheDebug)],
        AncestorLogCode),
    format(string(HandlerCode),
           "(let [reader-delay (delay ~w) max-depth ~w term-list-values (fn term-list-values [term] (if (and (map? term) (= \"[|]/2\" (:functor term))) (cons (first (:args term)) (term-list-values (second (:args term)))) [])) parent-values (fn [category] (mapv (fn [^generated.lmdb.LmdbRow row] (.value row)) (.lookupArg1 ^generated.lmdb.LmdbArtifactReader @reader-delay category))) ancestor-hops (fn ancestor-hops [category target visited] (let [parents (parent-values category)] (vec (concat (for [parent parents :when (and (not (contains? visited parent)) (or (map? target) (= parent target)))] [parent 1]) (when (< (count visited) max-depth) (apply concat (for [mid parents :when (not (contains? visited mid)) [ancestor hops] (ancestor-hops mid target (conj visited mid))] [[ancestor (inc hops)]])))))))] (fn [args] (let [category (nth args 0) target (nth args 1) visited (set (term-list-values (nth args 3))) solutions (for [[ancestor hops] (ancestor-hops category target visited)] {:bindings {2 ancestor 3 hops}})] (do ~w {:solutions (vec solutions)}))))",
           [ReaderOpenExpr, MaxDepth, AncestorLogCode]).
category_ancestor_handler_code(_OutputDir, inline, HandlerCode) :-
    category_ancestor_handler_code(inline, HandlerCode).
category_ancestor_handler_code(sidecar, HandlerCode) :-
    benchmark_max_depth_literal(MaxDepth),
    format(string(HandlerCode),
           "(let [parents-by-child-delay (delay (reduce (fn [acc [child parent]] (update acc child (fnil conj []) parent)) {} (edn/read-string (slurp \"data/generated/wam_clojure_optimized_bench/category_parent.edn\")))) max-depth ~w term-list-values (fn term-list-values [term] (if (and (map? term) (= \"[|]/2\" (:functor term))) (cons (first (:args term)) (term-list-values (second (:args term)))) [])) ancestor-hops (fn ancestor-hops [category target visited] (let [parents (get @parents-by-child-delay category [])] (vec (concat (for [parent parents :when (and (not (contains? visited parent)) (or (map? target) (= parent target)))] [parent 1]) (when (< (count visited) max-depth) (apply concat (for [mid parents :when (not (contains? visited mid)) [ancestor hops] (ancestor-hops mid target (conj visited mid))] [[ancestor (inc hops)]])))))))] (fn [args] (let [category (nth args 0) target (nth args 1) visited (set (term-list-values (nth args 3))) solutions (for [[ancestor hops] (ancestor-hops category target visited)] {:bindings {2 ancestor 3 hops}})] {:solutions (vec solutions)})))",
           [MaxDepth]).
category_ancestor_handler_code(artifact, HandlerCode) :-
    benchmark_max_depth_literal(MaxDepth),
    format(string(HandlerCode),
           "(let [parents-by-child-delay (delay (into {} (keep (fn [line] (when-not (.isEmpty ^String line) (let [parts (vec (.split ^String line \"\\\\t\")) child (first parts) parents (subvec parts 1)] [child parents]))) (.split ^String (slurp \"data/generated/wam_clojure_optimized_bench/category_parent_by_child.tsv\") \"\\\\r?\\\\n\")))) max-depth ~w term-list-values (fn term-list-values [term] (if (and (map? term) (= \"[|]/2\" (:functor term))) (cons (first (:args term)) (term-list-values (second (:args term)))) [])) ancestor-hops (fn ancestor-hops [category target visited] (let [parents (get @parents-by-child-delay category [])] (vec (concat (for [parent parents :when (and (not (contains? visited parent)) (or (map? target) (= parent target)))] [parent 1]) (when (< (count visited) max-depth) (apply concat (for [mid parents :when (not (contains? visited mid)) [ancestor hops] (ancestor-hops mid target (conj visited mid))] [[ancestor (inc hops)]])))))))] (fn [args] (let [category (nth args 0) target (nth args 1) visited (set (term-list-values (nth args 3))) solutions (for [[ancestor hops] (ancestor-hops category target visited)] {:bindings {2 ancestor 3 hops}})] {:solutions (vec solutions)})))",
           [MaxDepth]).
category_ancestor_handler_code(inline, HandlerCode) :-
    benchmark_max_depth_literal(MaxDepth),
    category_parent_map_literal(ParentsByChild),
    format(string(HandlerCode),
           "(let [parents-by-child {~s} max-depth ~w term-list-values (fn term-list-values [term] (if (and (map? term) (= \"[|]/2\" (:functor term))) (cons (first (:args term)) (term-list-values (second (:args term)))) [])) ancestor-hops (fn ancestor-hops [category target visited] (let [parents (get parents-by-child category [])] (vec (concat (for [parent parents :when (and (not (contains? visited parent)) (or (map? target) (= parent target)))] [parent 1]) (when (< (count visited) max-depth) (apply concat (for [mid parents :when (not (contains? visited mid)) [ancestor hops] (ancestor-hops mid target (conj visited mid))] [[ancestor (inc hops)]])))))))] (fn [args] (let [category (nth args 0) target (nth args 1) visited (set (term-list-values (nth args 3))) solutions (for [[ancestor hops] (ancestor-hops category target visited)] {:bindings {2 ancestor 3 hops}})] {:solutions (vec solutions)})))",
           [ParentsByChild, MaxDepth]).

current_category_parent_fact(Child, Parent) :-
    current_predicate(user:category_parent/2),
    call(user:category_parent(Child, Parent)).

category_parent_edge_literal(Child-Parent, Literal) :-
    clj_string_literal_local(Child, ChildLit),
    clj_string_literal_local(Parent, ParentLit),
    format(atom(Literal), '[~w ~w]', [ChildLit, ParentLit]).

category_parent_map_literal(Literal) :-
    findall(Child,
            current_category_parent_fact(Child, _),
            Children0),
    sort(Children0, Children),
    maplist(category_parent_bucket_literal, Children, BucketLiterals),
    atomic_list_concat(BucketLiterals, ' ', Literal).

category_parent_bucket_literal(Child, Literal) :-
    findall(Parent,
            current_category_parent_fact(Child, Parent),
            Parents0),
    sort(Parents0, Parents),
    clj_string_literal_local(Child, ChildLit),
    maplist(clj_string_literal_local, Parents, ParentLiterals),
    atomic_list_concat(ParentLiterals, ' ', ParentBody),
    format(atom(Literal), '~w [~w]', [ChildLit, ParentBody]).

clj_string_literal_local(In, Literal) :-
    atom_string(In, InStr0),
    escape_clj_string_local(InStr0, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

escape_clj_string_local(In, Out) :-
    split_string(In, "\\", "", Parts1),
    atomic_list_concat(Parts1, "\\\\", Tmp1),
    split_string(Tmp1, "\"", "", Parts2),
    atomic_list_concat(Parts2, "\\\"", Out).

append_effective_distance_runner(OutputDir, RunnerCode) :-
    directory_file_path(OutputDir, 'src/generated/wam_clojure_optimized_bench/core.clj', CorePath),
    read_file_to_string(CorePath, Core0, []),
    atomic_list_concat([Core0, RunnerCode], '\n\n', Core),
    setup_call_cleanup(
        open(CorePath, write, Stream),
        write(Stream, Core),
        close(Stream)
    ).

effective_distance_runner_code(OutputDir, VariantAtom, KernelModeAtom, DataMode, BenchmarkData, Code) :-
    benchmark_dimension_n_literal(DimensionN),
    benchmark_max_depth_literal(MaxDepth),
    benchmark_use_traversal_kernel_literal(KernelModeAtom, UseTraversalKernel),
    benchmark_sidecar_dir_literal(OutputDir, DataDirLiteral),
    benchmark_effective_data_modes(VariantAtom, DataMode, ArticleMode, ParentMode),
    benchmark_relation_cache_policy(category_parent, ParentMode, ParentCachePolicy),
    benchmark_article_categories_code(ArticleMode, DataDirLiteral, BenchmarkData, ArticleCategoriesCode),
    benchmark_category_parents_code(ParentMode, ParentCachePolicy, DataDirLiteral, BenchmarkData, CategoryParentsCode),
    benchmark_roots_code(DataMode, DataDirLiteral, BenchmarkData, RootsCode),
    benchmark_derived_state_code(ArticleMode, ParentMode, DerivedStateCode),
    format(string(Code),
'
;; Effective-distance benchmark entrypoint generated from facts.pl.
;; Predicate calls are still available by passing a predicate key and args.
(def benchmark-data-dir ~w)
~s
~s
~s
~s
(def benchmark-dimension-n ~w)
(def benchmark-max-depth ~w)
(def benchmark-use-traversal-kernel? ~w)

(defn benchmark-ancestor-hops [category root visited]
  (let [parents (get @benchmark-parents-by-child-delay category [])]
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

(defn benchmark-list-term [items]
  (reduce
    (fn [tail item]
      {:tag :struct :functor "[|]/2" :args [item tail]})
    "[]"
    (reverse items)))

(defn benchmark-foreign-category-root-hops [category root]
  (let [handler (get foreign-handlers "category_ancestor/4")
        result (when handler
                 (handler [category root {:var 9001} (benchmark-list-term [category])]))]
    (vec
      (keep (fn [solution]
              (get-in solution [:bindings 3]))
            (:solutions result)))))

(defn benchmark-category-root-hops [category root]
  (if benchmark-use-traversal-kernel?
    (benchmark-foreign-category-root-hops category root)
    (benchmark-ancestor-hops category root #{category})))

(defn benchmark-article-root-weight [article root]
  (reduce
    +
    0.0
    (for [category (get @benchmark-article-categories-by-article-delay article [])
          weight (if (= category root)
                   [1.0]
                   (for [hops (benchmark-category-root-hops category root)]
                     (Math/pow (+ hops 1.0) (- benchmark-dimension-n))))]
      weight)))

(defn benchmark-effective-distance-rows []
  (let [inv-n (- (/ 1.0 benchmark-dimension-n))]
    (sort-by
      (fn [{:keys [distance article root]}] [distance article root])
      (for [root @benchmark-roots-delay
            article (sort (keys @benchmark-article-categories-by-article-delay))
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
           [DataDirLiteral, ArticleCategoriesCode, CategoryParentsCode, RootsCode, DerivedStateCode,
            DimensionN, MaxDepth, UseTraversalKernel]).

benchmark_effective_data_modes(_VariantAtom, sidecar, sidecar, sidecar).
benchmark_effective_data_modes(_VariantAtom, inline, inline, inline).
benchmark_effective_data_modes(VariantAtom, artifact, ArticleMode, ParentMode) :-
    benchmark_effective_default_data_modes(VariantAtom, DefaultArticleMode, DefaultParentMode),
    benchmark_effective_relation_mode(article_category, DefaultArticleMode, ArticleMode),
    benchmark_effective_relation_mode(category_parent, DefaultParentMode, ParentMode).

benchmark_effective_default_data_modes(accumulated, artifact, artifact).
benchmark_effective_default_data_modes(seeded, sidecar, artifact).

benchmark_effective_relation_mode(Relation, DefaultMode, Mode) :-
    (   benchmark_relation_declared_data_mode(Relation, DeclaredMode),
        DeclaredMode \== auto
    ->  Mode = DeclaredMode
    ;   benchmark_relation_predicate(Relation, PredIndicator),
        declared_preprocess(PredIndicator, DeclaredMode, _Info)
    ->  Mode = DeclaredMode
    ;   Mode = DefaultMode
    ).

benchmark_relation_predicate(article_category, article_category/2).
benchmark_relation_predicate(category_parent, category_parent/2).

benchmark_derived_state_code(artifact, artifact, Code) :-
    Code = '(def benchmark-seed-categories-delay
  (delay (sort (set (mapcat identity (vals @benchmark-article-categories-by-article-delay))))))'.
benchmark_derived_state_code(artifact, lmdb, Code) :-
    benchmark_derived_state_code(artifact, artifact, Code).
benchmark_derived_state_code(sidecar, artifact, Code) :-
    Code = '(def benchmark-article-categories-by-article-delay
  (delay
    (reduce (fn [acc [article category]]
              (update acc article (fnil conj []) category))
            {}
            @benchmark-article-categories-delay)))

(def benchmark-seed-categories-delay
  (delay (sort (set (map second @benchmark-article-categories-delay)))))'.
benchmark_derived_state_code(sidecar, lmdb, Code) :-
    benchmark_derived_state_code(sidecar, artifact, Code).
benchmark_derived_state_code(_ArticleMode, _ParentMode, Code) :-
    Code = '(def benchmark-parents-by-child-delay
  (delay
    (reduce (fn [acc [child parent]]
              (update acc child (fnil conj []) parent))
            {}
            @benchmark-category-parents-delay)))

(def benchmark-article-categories-by-article-delay
  (delay
    (reduce (fn [acc [article category]]
              (update acc article (fnil conj []) category))
            {}
            @benchmark-article-categories-delay)))

(def benchmark-seed-categories-delay
  (delay (sort (set (map second @benchmark-article-categories-delay)))))'.

benchmark_article_categories_code(sidecar, _DataDirLiteral, _BenchmarkData, Code) :-
    Code = '(def benchmark-article-categories-delay
  (delay (vec (edn/read-string (slurp (str benchmark-data-dir "/article_category.edn"))))))'.
benchmark_article_categories_code(artifact, _DataDirLiteral, _BenchmarkData, Code) :-
    Code = '(def benchmark-article-categories-by-article-delay
  (delay
    (into {}
      (keep (fn [line]
              (when-not (.isEmpty ^String line)
                (let [parts (vec (.split ^String line "\\t"))
                      article (first parts)
                      categories (subvec parts 1)]
                  [article categories]))))
      (.split ^String (slurp (str benchmark-data-dir "/article_category_by_article.tsv")) "\\r?\\n"))))'.
benchmark_article_categories_code(inline, _DataDirLiteral, benchmark_data(ArticleCategories, _, _, _, _, _, _, _), Code) :-
    format(string(Code),
           "(def benchmark-article-categories-delay (delay [~s]))",
           [ArticleCategories]).

benchmark_category_parents_code(sidecar, _ParentCachePolicy, _DataDirLiteral, _BenchmarkData, Code) :-
    Code = '(def benchmark-category-parents-delay
  (delay (vec (edn/read-string (slurp (str benchmark-data-dir "/category_parent.edn"))))))'.
benchmark_category_parents_code(artifact, _ParentCachePolicy, _DataDirLiteral, _BenchmarkData, Code) :-
    Code = '(def benchmark-parents-by-child-delay
  (delay
    (into {}
      (keep (fn [line]
              (when-not (.isEmpty ^String line)
                (let [parts (vec (.split ^String line "\\t"))
                      child (first parts)
                      parents (subvec parts 1)]
                  [child parents]))))
      (.split ^String (slurp (str benchmark-data-dir "/category_parent_by_child.tsv")) "\\r?\\n"))))'.
benchmark_category_parents_code(lmdb, CachePolicy, _DataDirLiteral, _BenchmarkData, Code) :-
    lmdb_reader_open_expr_runtime(CachePolicy, ReaderOpenExpr),
    benchmark_relation_cache_debug(category_parent, lmdb, CacheDebug),
    lmdb_runner_cache_log_snippet(CacheDebug, RunnerLogCode),
    format(string(Code),
           "(def benchmark-parents-by-child-delay\n  (delay\n    (let [reader ~w\n          result (reduce (fn [acc ^generated.lmdb.LmdbRow row]\n                           (update acc (.key row) (fnil conj []) (.value row)))\n                         {}\n                         (.scan ^generated.lmdb.LmdbArtifactReader reader))]\n      ~w\n      result)))",
           [ReaderOpenExpr, RunnerLogCode]).
benchmark_category_parents_code(inline, _ParentCachePolicy, _DataDirLiteral, benchmark_data(_, CategoryParents, _, _, _, _, _, _), Code) :-
    format(string(Code),
           "(def benchmark-category-parents-delay (delay [~s]))",
           [CategoryParents]).

lmdb_reader_open_expr_runtime(memoize, Expr) :-
    wam_clojure_target:clojure_lmdb_reader_open_expr(
        '(str benchmark-data-dir "/category_parent_lmdb")',
        [clojure_lmdb_cache_policy(memoize)],
        Expr).
lmdb_reader_open_expr_runtime(shared, Expr) :-
    wam_clojure_target:clojure_lmdb_reader_open_expr(
        '(str benchmark-data-dir "/category_parent_lmdb")',
        [clojure_lmdb_cache_policy(shared)],
        Expr).
lmdb_reader_open_expr_runtime(two_level, Expr) :-
    wam_clojure_target:clojure_lmdb_reader_open_expr(
        '(str benchmark-data-dir "/category_parent_lmdb")',
        [clojure_lmdb_cache_policy(two_level)],
        Expr).
lmdb_reader_open_expr_runtime(none, Expr) :-
    wam_clojure_target:clojure_lmdb_reader_open_expr(
        '(str benchmark-data-dir "/category_parent_lmdb")',
        [clojure_lmdb_cache_policy(none)],
        Expr).

lmdb_runner_cache_log_snippet(false, "nil").
lmdb_runner_cache_log_snippet(true, Snippet) :-
    Snippet = '(binding [*out* *err*]
                 (println
                   (str "lmdb_cache_stats benchmark-category-parents "
                        (pr-str (.cacheStats ^generated.lmdb.LmdbArtifactReader reader)))))'.

benchmark_roots_code(sidecar, _DataDirLiteral, _BenchmarkData, Code) :-
    Code = '(def benchmark-roots-delay
  (delay (vec (edn/read-string (slurp (str benchmark-data-dir "/root_category.edn"))))))'.
benchmark_roots_code(artifact, _DataDirLiteral, _BenchmarkData, Code) :-
    Code = '(def benchmark-roots-delay
  (delay (vec (edn/read-string (slurp (str benchmark-data-dir "/root_category.edn"))))))'.
benchmark_roots_code(inline, _DataDirLiteral, benchmark_data(_, _, RootCategories, _, _, _, _, _), Code) :-
    format(string(Code),
           "(def benchmark-roots-delay (delay [~s]))",
           [RootCategories]).

maybe_write_benchmark_sidecar_files(OutputDir, FactsPath, VariantAtom, KernelModeAtom, sidecar, BenchmarkData) :-
    !,
    write_benchmark_sidecar_files(OutputDir, FactsPath, VariantAtom, KernelModeAtom, sidecar, BenchmarkData).
maybe_write_benchmark_sidecar_files(OutputDir, FactsPath, VariantAtom, KernelModeAtom, artifact, BenchmarkData) :-
    !,
    write_benchmark_sidecar_files(OutputDir, FactsPath, VariantAtom, KernelModeAtom, artifact, BenchmarkData).
maybe_write_benchmark_sidecar_files(_, _, _, _, inline, _).

write_benchmark_sidecar_files(OutputDir, FactsPath, VariantAtom, KernelModeAtom, DataMode, BenchmarkData) :-
    benchmark_effective_data_modes(VariantAtom, DataMode, _ArticleMode, ParentMode),
    benchmark_sidecar_dir(OutputDir, DataDir),
    make_directory_path(DataDir),
    write_benchmark_sidecar_file(DataDir, 'article_category.edn', BenchmarkData, benchmark_article_category_edn),
    write_benchmark_sidecar_file(DataDir, 'article_category_by_article.tsv', BenchmarkData, benchmark_article_category_by_article_artifact),
    write_benchmark_sidecar_file(DataDir, 'category_parent.edn', BenchmarkData, benchmark_category_parent_edn),
    write_benchmark_sidecar_file(DataDir, 'category_parent_by_child.tsv', BenchmarkData, benchmark_category_parent_by_child_artifact),
    write_benchmark_sidecar_file(DataDir, 'category_parent.tsv', BenchmarkData, benchmark_category_parent_rows_tsv),
    write_benchmark_sidecar_file(DataDir, 'root_category.edn', BenchmarkData, benchmark_root_category_edn),
    maybe_write_category_parent_lmdb_artifact(DataDir, ParentMode),
    benchmark_artifact_manifest(FactsPath, VariantAtom, KernelModeAtom, DataMode, Manifest),
    write_benchmark_sidecar_content(DataDir, 'manifest.edn', Manifest),
    maybe_prepare_clojure_lmdb_runtime_support(OutputDir, ParentMode).

benchmark_sidecar_dir(OutputDir, DataDir) :-
    directory_file_path(OutputDir, 'data', DataRoot),
    directory_file_path(DataRoot, 'generated', GeneratedDir),
    directory_file_path(GeneratedDir, 'wam_clojure_optimized_bench', DataDir).

benchmark_sidecar_dir_literal(OutputDir, Literal) :-
    benchmark_sidecar_dir(OutputDir, DataDir),
    absolute_file_name(DataDir, AbsoluteDir),
    clj_string_literal_local(AbsoluteDir, Literal).

benchmark_sidecar_file_path_literal(OutputDir, FileName, Literal) :-
    benchmark_sidecar_dir(OutputDir, DataDir),
    directory_file_path(DataDir, FileName, FilePath),
    absolute_file_name(FilePath, AbsolutePath),
    clj_string_literal_local(AbsolutePath, Literal).

benchmark_sidecar_subdir_literal(OutputDir, DirName, Literal) :-
    benchmark_sidecar_dir(OutputDir, DataDir),
    directory_file_path(DataDir, DirName, DirPath),
    absolute_file_name(DirPath, AbsolutePath),
    clj_string_literal_local(AbsolutePath, Literal).

write_benchmark_sidecar_file(DataDir, FileName, BenchmarkData, Builder) :-
    call(Builder, BenchmarkData, Content),
    write_benchmark_sidecar_content(DataDir, FileName, Content).

write_benchmark_sidecar_content(DataDir, FileName, Content) :-
    directory_file_path(DataDir, FileName, Path),
    setup_call_cleanup(
        open(Path, write, Stream),
        write(Stream, Content),
        close(Stream)
    ).

benchmark_artifact_manifest(FactsPath, VariantAtom, KernelModeAtom, DataMode, Manifest) :-
    benchmark_effective_data_modes(VariantAtom, DataMode, ArticleMode, ParentMode),
    benchmark_relation_cache_policy(category_parent, ParentMode, ParentCachePolicy),
    benchmark_fact_count(user:article_category/2, ArticleRows),
    benchmark_fact_count(user:category_parent/2, ParentRows),
    benchmark_fact_count(user:root_category/1, RootRows),
    clj_string_literal_local(FactsPath, FactsPathLit),
    clj_string_literal_local(VariantAtom, VariantLit),
    clj_string_literal_local(KernelModeAtom, KernelModeLit),
    clj_string_literal_local(DataMode, DataModeLit),
    relation_manifest_entry(article_category, ArticleMode, none, ArticleRows, ArticleEntry),
    relation_manifest_entry(category_parent, ParentMode, ParentCachePolicy, ParentRows, ParentEntry),
    relation_manifest_entry(root_category, sidecar, none, RootRows, RootEntry),
    format(atom(Manifest),
'{:format "unifyweaver.clojure_benchmark_artifacts.v1"
 :exactness :exact
 :source {:path ~w}
 :benchmark {:variant ~w :kernel_mode ~w :data_mode ~w}
 :relations {~w
             ~w
             ~w}
 :notes "Generated sidecar/artifact manifest for Clojure WAM benchmark validation."}
',
           [FactsPathLit, VariantLit, KernelModeLit, DataModeLit,
            ArticleEntry, ParentEntry, RootEntry]).

relation_manifest_entry(Relation, Mode, CachePolicy, RowCount, Entry) :-
    relation_manifest_shape(Relation, Mode, FileName, Format, Access),
    relation_manifest_preprocess_decl(Relation, DeclDecl),
    clj_string_literal_local(Relation, RelationLit),
    clj_string_literal_local(Mode, ModeLit),
    clj_string_literal_local(FileName, FileNameLit),
    clj_string_literal_local(Format, FormatLit),
    clj_string_literal_local(Access, AccessLit),
    preprocess_decl_field(DeclDecl, DeclField),
    cache_policy_field(CachePolicy, CacheField),
    format(atom(Entry),
           '~w {:mode ~w :file ~w :format ~w :row_count ~w :access ~w~w~w}',
           [RelationLit, ModeLit, FileNameLit, FormatLit, RowCount, AccessLit, DeclField, CacheField]).

relation_manifest_preprocess_decl(Relation, Decl) :-
    benchmark_relation_predicate(Relation, PredIndicator),
    declared_preprocess_metadata(PredIndicator, Mode,
                                 preprocess_info(Kind, Options),
                                 Metadata),
    !,
    Decl = preprocess_decl(Mode, Kind, Options, Metadata).
relation_manifest_preprocess_decl(_Relation, none).

preprocess_decl_field(none, '').
preprocess_decl_field(preprocess_decl(Mode, Kind, Options, Metadata), Field) :-
    clj_string_literal_local(shared_preprocess, SourceLit),
    clj_string_literal_local(Mode, ModeLit),
    clj_string_literal_local(Kind, KindLit),
    metadata_format_literal(Metadata, FormatLit),
    metadata_access_contracts_literal(Metadata, AccessLit),
    preprocess_options_literal(Options, OptionsLit),
    format(atom(Field),
           ' :declaration {:source ~w :mode ~w :kind ~w :format ~w :access_contracts ~w :options ~w}',
           [SourceLit, ModeLit, KindLit, FormatLit, AccessLit, OptionsLit]).

cache_policy_field(none, '').
cache_policy_field(CachePolicy, Field) :-
    clj_string_literal_local(CachePolicy, CacheLit),
    format(atom(Field), ' :cache_policy ~w', [CacheLit]).

metadata_format_literal(Metadata, Literal) :-
    clj_string_literal_local(Metadata.format, Literal).

metadata_access_contracts_literal(Metadata, Literal) :-
    maplist(term_string, Metadata.access_contracts, AccessStrings0),
    sort(AccessStrings0, AccessStrings),
    maplist(atom_string, AccessAtoms, AccessStrings),
    maplist(clj_string_literal_local, AccessAtoms, AccessLiterals),
    atomic_list_concat(AccessLiterals, ' ', AccessBody),
    format(atom(Literal), '[~w]', [AccessBody]).

preprocess_options_literal(Options, Literal) :-
    maplist(term_string, Options, OptionStrings0),
    maplist(atom_string, OptionAtoms, OptionStrings0),
    maplist(clj_string_literal_local, OptionAtoms, OptionLiterals),
    atomic_list_concat(OptionLiterals, ' ', OptionBody),
    format(atom(Literal), '[~w]', [OptionBody]).

relation_manifest_shape(article_category, artifact,
                        'article_category_by_article.tsv',
                        'grouped_tsv_arg1',
                        'scan,arg1_grouped_lookup') :- !.
relation_manifest_shape(article_category, _,
                        'article_category.edn',
                        'edn_rows',
                        'scan,arg1_group_at_startup') :- !.
relation_manifest_shape(category_parent, artifact,
                        'category_parent_by_child.tsv',
                        'grouped_tsv_arg1',
                        'arg1_lookup,scan_grouped') :- !.
relation_manifest_shape(category_parent, lmdb,
                        'category_parent_lmdb/manifest.json',
                        'lmdb_dupsort_btree',
                        'arg1_lookup,arg1_multi_lookup,scan') :- !.
relation_manifest_shape(category_parent, _,
                        'category_parent.edn',
                        'edn_rows',
                        'scan,arg1_group_at_startup') :- !.
relation_manifest_shape(root_category, _,
                        'root_category.edn',
                        'edn_rows',
                        'scan').

collect_benchmark_data(benchmark_data(ArticleCategories, CategoryParents, RootCategories,
                                      ArticleEdn, CategoryEdn, RootEdn,
                                      ArticleByArticleArtifact, ParentByChildArtifact)) :-
    benchmark_article_category_literal(ArticleCategories),
    benchmark_category_parent_literal(CategoryParents),
    benchmark_root_category_literal(RootCategories),
    benchmark_article_category_edn_from_literal(ArticleCategories, ArticleEdn),
    benchmark_category_parent_edn_from_literal(CategoryParents, CategoryEdn),
    benchmark_root_category_edn_from_literal(RootCategories, RootEdn),
    benchmark_article_category_by_article_artifact(ArticleByArticleArtifact),
    benchmark_category_parent_by_child_artifact(ParentByChildArtifact).

benchmark_use_traversal_kernel_literal(kernels_on, true).
benchmark_use_traversal_kernel_literal(kernels_off, false).

benchmark_article_category_literal(Literal) :-
    findall(Article-Category,
            current_article_category_fact(Article, Category),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(edge_literal, Pairs, Literals),
    atomic_list_concat(Literals, ' ', Literal).

benchmark_article_category_edn(benchmark_data(_, _, _, Content, _, _, _, _), Content).

benchmark_article_category_edn_from_literal(Literal, Content) :-
    format(atom(Content), '[~w]', [Literal]).

benchmark_category_parent_literal(Literal) :-
    findall(Child-Parent,
            current_category_parent_fact(Child, Parent),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(edge_literal, Pairs, Literals),
    atomic_list_concat(Literals, ' ', Literal).

benchmark_article_category_by_article_artifact(Content) :-
    findall(Article,
            current_article_category_fact(Article, _),
            Articles0),
    sort(Articles0, Articles),
    maplist(article_category_bucket_artifact_line, Articles, Lines),
    atomic_list_concat(Lines, '\n', Content).
benchmark_article_category_by_article_artifact(benchmark_data(_, _, _, _, _, _, Content, _), Content) :-
    nonvar(Content),
    !.
benchmark_article_category_by_article_artifact(_, Content) :-
    benchmark_article_category_by_article_artifact(Content).

article_category_bucket_artifact_line(Article, Line) :-
    findall(Category,
            current_article_category_fact(Article, Category),
            Categories0),
    sort(Categories0, Categories),
    atomic_list_concat([Article|Categories], '\t', Line).

benchmark_category_parent_edn(benchmark_data(_, _, _, _, Content, _, _, _), Content).

benchmark_category_parent_edn_from_literal(Literal, Content) :-
    format(atom(Content), '[~w]', [Literal]).

benchmark_root_category_literal(Literal) :-
    findall(Root, current_root_category_fact(Root), Roots0),
    sort(Roots0, Roots),
    maplist(clj_string_literal_local, Roots, Literals),
    atomic_list_concat(Literals, ' ', Literal).

benchmark_category_parent_by_child_artifact(Content) :-
    findall(Child,
            current_category_parent_fact(Child, _),
            Children0),
    sort(Children0, Children),
    maplist(category_parent_bucket_artifact_line, Children, Lines),
    atomic_list_concat(Lines, '\n', Content).
benchmark_category_parent_by_child_artifact(benchmark_data(_, _, _, _, _, _, _, Content), Content) :-
    nonvar(Content),
    !.
benchmark_category_parent_by_child_artifact(_, Content) :-
    benchmark_category_parent_by_child_artifact(Content).

category_parent_bucket_artifact_line(Child, Line) :-
    findall(Parent,
            current_category_parent_fact(Child, Parent),
            Parents0),
    sort(Parents0, Parents),
    atomic_list_concat([Child|Parents], '\t', Line).

benchmark_category_parent_rows_tsv(Content) :-
    findall(Child-Parent,
            current_category_parent_fact(Child, Parent),
            Pairs0),
    sort(Pairs0, Pairs),
    maplist(category_parent_row_tsv_line, Pairs, Lines),
    atomic_list_concat(Lines, '\n', Content).
benchmark_category_parent_rows_tsv(_, Content) :-
    benchmark_category_parent_rows_tsv(Content).

category_parent_row_tsv_line(Child-Parent, Line) :-
    atomic_list_concat([Child, Parent], '\t', Line).

benchmark_root_category_edn(benchmark_data(_, _, _, _, _, Content, _, _), Content).

benchmark_root_category_edn_from_literal(Literal, Content) :-
    format(atom(Content), '[~w]', [Literal]).

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

maybe_write_category_parent_lmdb_artifact(_DataDir, ParentMode) :-
    ParentMode \== lmdb,
    !.
maybe_write_category_parent_lmdb_artifact(DataDir, lmdb) :-
    directory_file_path(DataDir, 'category_parent.tsv', TsvPath),
    directory_file_path(DataDir, 'category_parent_lmdb', ArtifactDir),
    make_directory_path(ArtifactDir),
    benchmark_repo_path('examples/lmdb_relation_artifact/Cargo.toml', CargoManifest),
    ensure_lmdb_artifact_builder(CargoManifest, ArtifactBuilder),
    format(atom(ArtifactBuildCmd),
           '"~w" build "category_parent/2" "~w" "~w" --dupsort',
           [ArtifactBuilder, TsvPath, ArtifactDir]),
    process_create(path(sh),
                   ['-c', ArtifactBuildCmd],
                   [process(RunPID)]),
    process_wait(RunPID, RunStatus),
    (   RunStatus == exit(0)
    ->  true
    ;   throw(error(cargo_lmdb_artifact_build_failed(RunStatus, TsvPath, ArtifactDir), _))
    ).

ensure_lmdb_artifact_builder(_CargoManifest, ArtifactBuilder) :-
    lmdb_artifact_builder_ready(ArtifactBuilder),
    exists_file(ArtifactBuilder),
    !.
ensure_lmdb_artifact_builder(CargoManifest, ArtifactBuilder) :-
    lmdb_artifact_target_dir(CargoTargetDir),
    !,
    directory_file_path(CargoTargetDir, 'debug/lmdb_relation_artifact', ArtifactBuilder),
    build_lmdb_artifact_builder(CargoManifest, CargoTargetDir, ArtifactBuilder).
ensure_lmdb_artifact_builder(CargoManifest, ArtifactBuilder) :-
    get_time(T),
    Stamp is floor(T * 1000),
    benchmark_repo_root(RepoRoot),
    format(atom(TargetDirName), 'tmp_lmdb_relation_artifact_target_~w', [Stamp]),
    directory_file_path(RepoRoot, TargetDirName, CargoTargetDir),
    make_directory_path(CargoTargetDir),
    assertz(lmdb_artifact_target_dir(CargoTargetDir)),
    directory_file_path(CargoTargetDir, 'debug/lmdb_relation_artifact', ArtifactBuilder),
    build_lmdb_artifact_builder(CargoManifest, CargoTargetDir, ArtifactBuilder).

build_lmdb_artifact_builder(CargoManifest, CargoTargetDir, ArtifactBuilder) :-
    format(atom(BuildCmd),
           'CARGO_TARGET_DIR="~w" cargo build --quiet --manifest-path "~w"',
           [CargoTargetDir, CargoManifest]),
    process_create(path(sh),
                   ['-c', BuildCmd],
                   [process(PID)]),
    process_wait(PID, Status),
    (   Status == exit(0),
        exists_file(ArtifactBuilder)
    ->  retractall(lmdb_artifact_builder_ready(_)),
        assertz(lmdb_artifact_builder_ready(ArtifactBuilder))
    ;   throw(error(cargo_lmdb_artifact_build_failed(Status, CargoManifest, ArtifactBuilder), _))
    ).

maybe_prepare_clojure_lmdb_runtime_support(_OutputDir, ParentMode) :-
    ParentMode \== lmdb,
    !.
maybe_prepare_clojure_lmdb_runtime_support(OutputDir, lmdb) :-
    wam_clojure_target:prepare_wam_clojure_lmdb_runtime_support(
        OutputDir,
        [clojure_lmdb_runtime_support(true)]).
