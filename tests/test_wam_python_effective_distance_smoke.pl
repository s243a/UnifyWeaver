:- begin_tests(wam_python_effective_distance_smoke).

:- use_module(library(filesex), [delete_directory_and_contents/1, make_directory_path/1]).
:- use_module(library(process)).

test(lowered_optimized_project_runs_tiny_chain) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_python_effective_distance_smoke', RootDir),
        (   directory_file_path(RootDir, data, DataDir),
            directory_file_path(RootDir, project, ProjectDir),
            make_directory_path(DataDir),
            make_directory_path(ProjectDir),
            write_tiny_chain_fixture(DataDir, FactsPath),
            generate_project(FactsPath, ProjectDir),
            run_generated_project(ProjectDir, DataDir, Output),
            once(sub_string(Output, _, _, _, "seeds=2")),
            once(sub_string(Output, _, _, _, "solutions=2"))
        ),
        cleanup_tmp_dir(RootDir)).

write_tiny_chain_fixture(DataDir, FactsPath) :-
    directory_file_path(DataDir, 'facts.pl', FactsPath),
    directory_file_path(DataDir, 'category_parent.tsv', CategoryParentPath),
    directory_file_path(DataDir, 'article_category.tsv', ArticleCategoryPath),
    directory_file_path(DataDir, 'root_categories.tsv', RootCategoriesPath),
    write_text_file(FactsPath,
'category_parent(cat_001, cat_002).
category_parent(cat_002, cat_003).
category_parent(cat_003, cat_004).
category_parent(cat_004, cat_005).
category_parent(cat_005, cat_006).
category_parent(cat_006, cat_007).
category_parent(cat_007, cat_008).
root_category(cat_008).
article_category(art_001, cat_001).
article_category(art_002, cat_002).
'),
    write_text_file(CategoryParentPath,
'child\tparent
cat_001\tcat_002
cat_002\tcat_003
cat_003\tcat_004
cat_004\tcat_005
cat_005\tcat_006
cat_006\tcat_007
cat_007\tcat_008
'),
    write_text_file(ArticleCategoryPath,
'article\tcategory
art_001\tcat_001
art_002\tcat_002
'),
    write_text_file(RootCategoriesPath,
'category
cat_008
').

generate_project(FactsPath, ProjectDir) :-
    generator_path(GeneratorPath),
    process_create(path(swipl),
        ['-q', '-s', GeneratorPath, '--', FactsPath, ProjectDir, accumulated],
        [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, _Output),
    read_string(Err, _, ErrText),
    close(Out),
    close(Err),
    process_wait(Pid, Status),
    (   Status = exit(0)
    ->  true
    ;   format(user_error, 'WAM Python generator failed: ~w~n', [ErrText]),
        fail
    ).

run_generated_project(ProjectDir, DataDir, Output) :-
    process_create(path(python), ['main.py', DataDir, '1'],
        [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, Output),
    read_string(Err, _, ErrText),
    close(Out),
    close(Err),
    process_wait(Pid, Status),
    (   Status = exit(0)
    ->  true
    ;   format(user_error, 'generated WAM Python benchmark failed: ~w~n', [ErrText]),
        fail
    ).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).

unique_tmp_dir(Prefix, TmpDir) :-
    tmp_file(Prefix, TmpDir),
    catch(delete_directory_and_contents(TmpDir), _, true),
    make_directory_path(TmpDir).

cleanup_tmp_dir(TmpDir) :-
    catch(delete_directory_and_contents(TmpDir), _, true).

generator_path(GeneratorPath) :-
    source_file(generator_path(_), ThisFile),
    file_directory_name(ThisFile, TestsDir),
    file_directory_name(TestsDir, RepoRoot),
    directory_file_path(RepoRoot, 'examples/benchmark/generate_wam_python_optimized_benchmark.pl', GeneratorPath).

:- end_tests(wam_python_effective_distance_smoke).
