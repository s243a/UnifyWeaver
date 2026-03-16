:- module(test_typr_toolchain, [run_all_tests/0]).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/type_declarations').
:- use_module('../src/unifyweaver/targets/typr_target').

run_all_tests :-
    run_tests([typr_toolchain]).

:- begin_tests(typr_toolchain).

test(toolchain_smoke, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            overwrite_project_source(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            run_typr(ProjectDir, ['build']),
            maybe_run_generated_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ).

:- end_tests(typr_toolchain).

typr_cli_available :-
    process_create(path(sh), ['-c', 'command -v typr >/dev/null 2>&1'], [process(Pid)]),
    process_wait(Pid, exit(0)).

create_smoke_project(ProjectDir) :-
    tmp_file(typr_smoke, RootDir),
    make_directory(RootDir),
    run_typr(RootDir, ['new', 'smoke_project']),
    directory_file_path(RootDir, 'smoke_project', CreatedDir),
    exists_directory(CreatedDir),
    !,
    ProjectDir = CreatedDir.
create_smoke_project(ProjectDir) :-
    tmp_file(typr_smoke, ProjectDir),
    make_directory(ProjectDir).

overwrite_project_source(ProjectDir, Code) :-
    find_typr_source_file(ProjectDir, SourceFile),
    setup_call_cleanup(
        open(SourceFile, write, Stream),
        write(Stream, Code),
        close(Stream)
    ).

find_typr_source_file(Dir, File) :-
    directory_files(Dir, Entries),
    member(Entry, Entries),
    Entry \= '.',
    Entry \= '..',
    directory_file_path(Dir, Entry, Path),
    (   exists_directory(Path)
    ->  find_typr_source_file(Path, File)
    ;   file_name_extension(_, Ext, Path),
        member(Ext, ['typr', 'tr']),
        File = Path
    ),
    !.

run_typr(ProjectDir, Args) :-
    process_create(
        path(typr),
        Args,
        [ cwd(ProjectDir),
          stdout(pipe(Stdout)),
          stderr(pipe(Stderr)),
          process(Pid)
        ]
    ),
    read_string(Stdout, _, _),
    read_string(Stderr, _, _),
    close(Stdout),
    close(Stderr),
    process_wait(Pid, exit(0)).

maybe_run_generated_r(ProjectDir) :-
    (   rscript_available,
        find_generated_r_file(ProjectDir, RFile)
    ->  process_create(path('Rscript'), [RFile], [process(Pid)]),
        process_wait(Pid, exit(0))
    ;   true
    ).

rscript_available :-
    process_create(path(sh), ['-c', 'command -v Rscript >/dev/null 2>&1'], [process(Pid)]),
    process_wait(Pid, exit(0)).

find_generated_r_file(Dir, File) :-
    directory_files(Dir, Entries),
    member(Entry, Entries),
    Entry \= '.',
    Entry \= '..',
    directory_file_path(Dir, Entry, Path),
    (   exists_directory(Path)
    ->  find_generated_r_file(Path, File)
    ;   file_name_extension(_, 'R', Path),
        File = Path
    ),
    !.
