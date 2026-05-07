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
    (   run_generated_effective_distance(kernels_on, facts_tsv, Output),
        sub_string(Output, _, _, _, "article\troot_category\teffective_distance"),
        sub_string(Output, _, _, _, "article_a\troot\t1.951123")
    ->  pass(Test)
    ;   fail_test(Test, 'kernels_on runner output mismatch')
    ).

test_generate_and_run_kernels_off :-
    Test = 'WAM-C effective-distance: kernels_off generated runner emits expected result',
    (   run_generated_effective_distance(kernels_off, facts_tsv, Output),
        sub_string(Output, _, _, _, "article\troot_category\teffective_distance"),
        sub_string(Output, _, _, _, "article_a\troot\t1.951123")
    ->  pass(Test)
    ;   fail_test(Test, 'kernels_off runner output mismatch')
    ).

test_generate_lmdb_mode_files :-
    Test = 'WAM-C effective-distance: facts_lmdb mode emits seeder and LMDB loader',
    (   unique_tmp_dir(facts_lmdb_files, OutputDir),
        write_test_facts(OutputDir, FactsPath),
        generate_wam_c_effective_distance_benchmark:generate(FactsPath, OutputDir, kernels_on, facts_lmdb),
        directory_file_path(OutputDir, 'seed_category_parent_lmdb.c', SeederPath),
        directory_file_path(OutputDir, 'main.c', MainPath),
        directory_file_path(OutputDir, 'README.md', ReadmePath),
        read_file_to_string(SeederPath, Seeder, []),
        read_file_to_string(MainPath, Main, []),
        read_file_to_string(ReadmePath, Readme, []),
        sub_string(Seeder, _, _, _, 'MDB_CREATE | MDB_DUPSORT'),
        sub_string(Seeder, _, _, _, 'LMDB category_parent artifact validation failed'),
        sub_string(Main, _, _, _, 'wam_fact_source_load_lmdb(&state, &source, "category_parent.lmdb", NULL)'),
        sub_string(Readme, _, _, _, '-DWAM_C_ENABLE_LMDB'),
        sub_string(Readme, _, _, _, '-llmdb')
    ->  pass(Test)
    ;   fail_test(Test, 'facts_lmdb generated files mismatch')
    ).

test_generate_and_run_lmdb_if_available :-
    Test = 'WAM-C effective-distance: facts_lmdb generated runner emits expected result',
    (   lmdb_toolchain_available
    ->  (   run_generated_effective_distance(kernels_on, facts_lmdb, Output),
            sub_string(Output, _, _, _, "article\troot_category\teffective_distance"),
            sub_string(Output, _, _, _, "article_a\troot\t1.951123")
        ->  pass(Test)
        ;   fail_test(Test, 'facts_lmdb runner output mismatch')
        )
    ;   pass('WAM-C effective-distance: facts_lmdb generated runner skipped (LMDB toolchain unavailable)')
    ).

run_generated_effective_distance(KernelMode, FactStorage, Output) :-
    unique_tmp_dir(KernelMode, OutputDir),
    write_test_facts(OutputDir, FactsPath),
    generate_wam_c_effective_distance_benchmark:generate(FactsPath, OutputDir, KernelMode, FactStorage),
    compile_generated_project(OutputDir, FactStorage),
    run_generated_project(OutputDir, Output).

write_test_facts(OutputDir, FactsPath) :-
    directory_file_path(OutputDir, 'facts.pl', FactsPath),
    write_text_file(FactsPath,
'article_category(article_a, leaf).
root_category(root).
category_parent(leaf, root).
category_parent(leaf, mid).
category_parent(mid, root).
').

unique_tmp_dir(KernelMode, OutputDir) :-
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(OutputDir), '/tmp/unifyweaver_wam_c_effective_~w_~w', [KernelMode, Stamp]),
    make_directory_path(OutputDir).

compile_generated_project(OutputDir, FactStorage) :-
    maybe_seed_lmdb_project(OutputDir, FactStorage),
    directory_file_path(OutputDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(OutputDir, 'lib.c', LibPath),
    directory_file_path(OutputDir, 'main.c', MainPath),
    directory_file_path(OutputDir, 'wam_c_effective_distance', ExePath),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    compile_flags(FactStorage, Flags),
    link_flags(FactStorage, Links),
    append(['-std=c11', '-Wall', '-Wextra'|Flags],
           ['-I', IncludeDir, RuntimePath, LibPath, MainPath, '-lm'|Links],
           Args0),
    append(Args0, ['-o', ExePath], Args),
    process_create(path(gcc),
        Args,
        [stdout(null), stderr(pipe(Err)), process(Pid)]),
    read_string(Err, _, ErrText),
    close(Err),
    process_wait(Pid, Status),
    (   Status = exit(0)
    ->  true
    ;   format(user_error, 'gcc failed: ~w~n', [ErrText]),
        fail
    ).

maybe_seed_lmdb_project(_OutputDir, facts_tsv).
maybe_seed_lmdb_project(OutputDir, facts_lmdb) :-
    directory_file_path(OutputDir, 'seed_category_parent_lmdb.c', SeederPath),
    directory_file_path(OutputDir, 'seed_category_parent_lmdb', SeederExe),
    process_create(path(gcc),
        ['-std=c11', '-Wall', '-Wextra', SeederPath, '-llmdb', '-o', SeederExe],
        [stdout(null), stderr(pipe(CompileErr)), process(CompilePid)]),
    read_string(CompileErr, _, CompileErrText),
    close(CompileErr),
    process_wait(CompilePid, CompileStatus),
    (   CompileStatus = exit(0)
    ->  true
    ;   format(user_error, 'LMDB seeder gcc failed: ~w~n', [CompileErrText]),
        fail
    ),
    process_create(SeederExe, [],
        [cwd(OutputDir), stdout(null), stderr(pipe(RunErr)), process(RunPid)]),
    read_string(RunErr, _, RunErrText),
    close(RunErr),
    process_wait(RunPid, RunStatus),
    (   RunStatus = exit(0)
    ->  true
    ;   format(user_error, 'LMDB seeder failed: ~w~n', [RunErrText]),
        fail
    ).

compile_flags(facts_tsv, []).
compile_flags(facts_lmdb, ['-DWAM_C_ENABLE_LMDB']).

link_flags(facts_tsv, []).
link_flags(facts_lmdb, ['-llmdb']).

lmdb_toolchain_available :-
    catch(
        (   process_create(path('pkg-config'), ['--exists', 'lmdb'],
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _,
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
    test_generate_lmdb_mode_files,
    test_generate_and_run_lmdb_if_available,
    format('~n=== WAM-C Effective Distance Benchmark Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
