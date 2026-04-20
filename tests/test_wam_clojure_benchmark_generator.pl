:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
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
        assertion(sub_string(CoreCode, _, _, _, '"category_parent/2" (let [edges #{')),
        assertion(sub_string(CoreCode, _, _, _, '["Abstraction" "Thought"]')),
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
        assertion(\+ sub_string(CoreCode, _, _, _, '"category_parent/2" (let [edges #{')),
        assertion(\+ sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "category_parent" :arity 2}')),
        delete_directory_and_contents(TmpDir)
    )).

test(generated_category_parent_handler_executes, [condition(clojure_available)]) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_bench_exec', TmpDir),
        generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on),
        run_clojure_predicate(TmpDir, 'category_parent/2',
                              ['Abstraction', 'Thought'], "true"),
        run_clojure_predicate(TmpDir, 'category_parent/2',
                              ['Abstraction', 'NotAParent'], "false"),
        delete_directory_and_contents(TmpDir)
    )).

:- end_tests(wam_clojure_benchmark_generator).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).

run_clojure_predicate(ProjectDir, PredKey, Args, Output) :-
    find_clojure_classpath(ClassPath),
    maplist(clojure_string_arg, Args, EdnArgs),
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

clojure_string_arg(Atom, Edn) :-
    format(string(Edn), '"~w"', [Atom]).

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
