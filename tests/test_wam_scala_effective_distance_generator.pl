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

unique_tmp_dir(Prefix, TmpDir) :-
    tmp_file(Prefix, TmpDir),
    catch(delete_directory_and_contents(TmpDir), _, true),
    make_directory_path(TmpDir).

cleanup_tmp_dir(TmpDir) :-
    catch(delete_directory_and_contents(TmpDir), _, true).

:- end_tests(wam_scala_effective_distance_generator).
