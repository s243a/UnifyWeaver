:- begin_tests(wam_scala_effective_distance_generator).

:- use_module('../examples/benchmark/generate_wam_scala_effective_distance_benchmark').
:- use_module(library(filesex), [delete_directory_and_contents/1]).
:- use_module(library(readutil), [read_file_to_string/3]).

test(generate_kernels_on_project_emits_runner_and_fact_sidecar) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_scala_effective_distance_on', TmpDir),
        (   generate('data/benchmark/dev/facts.pl', TmpDir, accumulated, kernels_on),
            directory_file_path(TmpDir,
                                'src/main/scala/generated/wam_scala_effective_distance/core/EffectiveDistanceRunner.scala',
                                RunnerPath),
            directory_file_path(TmpDir, 'data/category_parent.csv', CsvPath),
            exists_file(RunnerPath),
            exists_file(CsvPath),
            read_file_to_string(RunnerPath, Runner, []),
            sub_string(Runner, _, _, _, 'object EffectiveDistanceRunner'),
            sub_string(Runner, _, _, _, 'article\\troot_category\\teffective_distance'),
            sub_string(Runner, _, _, _, 'category_ancestor/4'),
            !
        ),
        cleanup_tmp_dir(TmpDir)).

test(generate_kernels_off_project_omits_fact_sidecar) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_scala_effective_distance_off', TmpDir),
        (   generate('data/benchmark/dev/facts.pl', TmpDir, accumulated, kernels_off),
            directory_file_path(TmpDir,
                                'src/main/scala/generated/wam_scala_effective_distance/core/EffectiveDistanceRunner.scala',
                                RunnerPath),
            directory_file_path(TmpDir, 'data/category_parent.csv', CsvPath),
            exists_file(RunnerPath),
            \+ exists_file(CsvPath),
            !
        ),
        cleanup_tmp_dir(TmpDir)).

test(data_mode_options) :-
    parse_benchmark_data_mode(sidecar, sidecar),
    parse_benchmark_data_mode(inline, inline),
    parse_benchmark_data_mode(artifact, artifact),
    setup_call_cleanup(
        load_files(user:'data/benchmark/dev/facts.pl', [silent(true)]),
        assertion(parse_benchmark_data_mode(auto, sidecar)),
        true
    ).

test(data_mode_predicate_override) :-
    setup_call_cleanup(
        ( load_files(user:'data/benchmark/dev/facts.pl', [silent(true)]),
          assertz(user:wam_scala_benchmark_data_mode(sidecar))
        ),
        assertion(parse_benchmark_data_mode(auto, sidecar)),
        maybe_abolish_test_predicate(wam_scala_benchmark_data_mode/1)
    ).

test(relation_data_mode_override_parent_inline) :-
    setup_call_cleanup(
        assertz(user:wam_scala_benchmark_relation_data_mode(category_parent, inline)),
        setup_call_cleanup(
            unique_tmp_dir('tmp_wam_scala_effective_distance_rel_inline', TmpDir),
            (   generate('data/benchmark/dev/facts.pl', TmpDir, accumulated, kernels_on, sidecar),
                directory_file_path(TmpDir, 'data/category_parent.csv', CsvPath),
                \+ exists_file(CsvPath),
                directory_file_path(TmpDir,
                                    'src/main/scala/generated/wam_scala_effective_distance/core/GeneratedProgram.scala',
                                    GeneratedPath),
                read_file_to_string(GeneratedPath, Generated, []),
                sub_string(Generated, _, _, _, 'private val sols: Seq[Map[Int, WamTerm]] = Seq('),
                !
            ),
            cleanup_tmp_dir(TmpDir)),
        maybe_abolish_test_predicate(wam_scala_benchmark_relation_data_mode/2)
    ).

test(generate_inline_project_omits_fact_sidecar) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_scala_effective_distance_inline', TmpDir),
        (   generate('data/benchmark/dev/facts.pl', TmpDir, accumulated, kernels_on, inline),
            directory_file_path(TmpDir, 'data/category_parent.csv', CsvPath),
            \+ exists_file(CsvPath),
            directory_file_path(TmpDir,
                                'src/main/scala/generated/wam_scala_effective_distance/core/GeneratedProgram.scala',
                                GeneratedPath),
            read_file_to_string(GeneratedPath, Generated, []),
            sub_string(Generated, _, _, _, 'private val sols: Seq[Map[Int, WamTerm]] = Seq('),
            !
        ),
        cleanup_tmp_dir(TmpDir)).

test(generate_artifact_project_emits_distinct_file_backend) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_scala_effective_distance_artifact', TmpDir),
        (   generate('data/benchmark/dev/facts.pl', TmpDir, seeded, kernels_on, artifact),
            directory_file_path(TmpDir, 'data/category_parent_by_child.tsv', ParentArtifactPath),
            directory_file_path(TmpDir, 'data/article_category_by_article.tsv', ArticleArtifactPath),
            directory_file_path(TmpDir, 'data/category_parent.csv', SidecarPath),
            exists_file(ParentArtifactPath),
            exists_file(ArticleArtifactPath),
            \+ exists_file(SidecarPath),
            read_file_to_string(ParentArtifactPath, ParentArtifact, []),
            read_file_to_string(ArticleArtifactPath, ArticleArtifact, []),
            sub_string(ParentArtifact, _, _, _, '\t'),
            sub_string(ArticleArtifact, _, _, _, '\t'),
            directory_file_path(TmpDir,
                                'src/main/scala/generated/wam_scala_effective_distance/core/GeneratedProgram.scala',
                                GeneratedPath),
            directory_file_path(TmpDir,
                                'src/main/scala/generated/wam_scala_effective_distance/core/EffectiveDistanceRunner.scala',
                                RunnerPath),
            read_file_to_string(GeneratedPath, Generated, []),
            read_file_to_string(RunnerPath, Runner, []),
            sub_string(Generated, _, _, _, 'category_parent_by_child.tsv'),
            sub_string(Generated, _, _, _, 'parentsByChild'),
            sub_string(Generated, _, _, _, 'CallForeign("category_parent", 2)'),
            \+ sub_string(Generated, _, _, _, 'Call("category_parent", 2)'),
            sub_string(Generated, _, _, _, 'internTable.isInRange(id)'),
            sub_string(Generated, _, _, _, 'internTable.stringOf(id)'),
            \+ sub_string(Generated, _, _, _, 'internTable.idToString'),
            sub_string(Runner, _, _, _, 'article_category_by_article.tsv'),
            sub_string(Runner, _, _, _, 'article_source_mode='),
            sub_string(Runner, _, _, _, 'val hopsRef = Ref(1000000)'),
            sub_string(Runner, _, _, _, 'WamRuntime.deref(state.bindings, hopsRef)'),
            \+ sub_string(Runner, _, _, _, 'WamRuntime.deref(state.bindings, state.regs(3))'),
            sub_string(Runner, _, _, _, 'load_ms='),
            sub_string(Runner, _, _, _, 'query_ms='),
            sub_string(Runner, _, _, _, 'aggregation_ms='),
            sub_string(Runner, _, _, _, 'total_ms='),
            sub_string(Runner, _, _, _, 'loadGroupedPairs'),
            !
        ),
        cleanup_tmp_dir(TmpDir)).

unique_tmp_dir(Prefix, TmpDir) :-
    tmp_file(Prefix, TmpDir),
    catch(delete_directory_and_contents(TmpDir), _, true),
    make_directory_path(TmpDir).

cleanup_tmp_dir(TmpDir) :-
    catch(delete_directory_and_contents(TmpDir), _, true).

maybe_abolish_test_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).

:- end_tests(wam_scala_effective_distance_generator).
