:- encoding(utf8).
% Test suite for the WAM-C effective-distance benchmark generator.
% Usage: swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl

:- use_module('../examples/benchmark/generate_wam_c_effective_distance_benchmark').
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

test_generate_and_run_kernels_on :-
    Test = 'WAM-C effective-distance: kernels_on generated runner emits expected result',
    (   run_generated_effective_distance(kernels_on, Output),
        sub_string(Output, _, _, _, "article\troot_category\teffective_distance"),
        sub_string(Output, _, _, _, "article_a\troot\t1.951123")
    ->  pass(Test)
    ;   fail_test(Test, 'kernels_on runner output mismatch')
    ).

test_generate_and_run_kernels_off :-
    Test = 'WAM-C effective-distance: kernels_off generated runner emits expected result',
    (   run_generated_effective_distance(kernels_off, Output),
        sub_string(Output, _, _, _, "article\troot_category\teffective_distance"),
        sub_string(Output, _, _, _, "article_a\troot\t1.951123")
    ->  pass(Test)
    ;   fail_test(Test, 'kernels_off runner output mismatch')
    ).

run_generated_effective_distance(KernelMode, Output) :-
    unique_tmp_dir(KernelMode, OutputDir),
    directory_file_path(OutputDir, 'facts.pl', FactsPath),
    write_text_file(FactsPath,
'article_category(article_a, leaf).
root_category(root).
category_parent(leaf, root).
category_parent(leaf, mid).
category_parent(mid, root).
'),
    generate_wam_c_effective_distance_benchmark:generate(FactsPath, OutputDir, KernelMode),
    compile_generated_project(OutputDir),
    run_generated_project(OutputDir, Output).

unique_tmp_dir(KernelMode, OutputDir) :-
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(OutputDir), '/tmp/unifyweaver_wam_c_effective_~w_~w', [KernelMode, Stamp]),
    make_directory_path(OutputDir).

compile_generated_project(OutputDir) :-
    directory_file_path(OutputDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(OutputDir, 'lib.c', LibPath),
    directory_file_path(OutputDir, 'main.c', MainPath),
    directory_file_path(OutputDir, 'wam_c_effective_distance', ExePath),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
        ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
         RuntimePath, LibPath, MainPath, '-lm', '-o', ExePath],
        [stdout(null), stderr(pipe(Err)), process(Pid)]),
    read_string(Err, _, ErrText),
    close(Err),
    process_wait(Pid, Status),
    (   Status = exit(0)
    ->  true
    ;   format(user_error, 'gcc failed: ~w~n', [ErrText]),
        fail
    ).

run_generated_project(OutputDir, Output) :-
    directory_file_path(OutputDir, 'wam_c_effective_distance', ExePath),
    process_create(ExePath, [],
        [cwd(OutputDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, Output),
    read_string(Err, _, ErrText),
    close(Out),
    close(Err),
    process_wait(Pid, Status),
    (   Status = exit(0)
    ->  true
    ;   format(user_error, 'generated runner failed: ~w~n', [ErrText]),
        fail
    ).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C Effective Distance Benchmark Tests ===~n~n'),
    test_generate_and_run_kernels_on,
    test_generate_and_run_kernels_off,
    format('~n=== WAM-C Effective Distance Benchmark Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
