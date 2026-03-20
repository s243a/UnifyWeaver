:- module(test_typr_toolchain, [run_all_tests/0]).

:- use_module(library(filesex)).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/type_declarations').
:- use_module('../src/unifyweaver/targets/typr_target').
:- use_module('../src/unifyweaver/core/recursive_compiler').

run_all_tests :-
    run_tests([typr_toolchain]).

:- begin_tests(typr_toolchain).

test(generated_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:simple_fact(hello)),
    assertz(type_declarations:uw_type(simple_fact/1, 1, atom)),
    once(compile_predicate_to_typr(simple_fact/1, [typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:simple_fact(_)).

test(transitive_closure_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(type_declarations:uw_type(edge/2, 1, atom)),
    assertz(type_declarations:uw_type(edge/2, 2, atom)),
    once(compile_predicate_to_typr(tc/2, [base_pred(edge), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:edge(_, _)).

test(tail_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:factorial_acc(0, Acc, Acc)),
    assertz(user:(factorial_acc(N, Acc, Result) :-
        N > 0,
        N1 is N - 1,
        Acc1 is Acc * N,
        factorial_acc(N1, Acc1, Result)
    )),
    assertz(type_declarations:uw_type(factorial_acc/3, 1, integer)),
    assertz(type_declarations:uw_type(factorial_acc/3, 2, integer)),
    assertz(type_declarations:uw_type(factorial_acc/3, 3, integer)),
    assertz(type_declarations:uw_return_type(factorial_acc/3, integer)),
    once(recursive_compiler:compile_recursive(factorial_acc/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:factorial_acc(_, _, _)).

test(linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:factorial_linear(0, 1)),
    assertz(user:(factorial_linear(N, Result) :-
        N > 0,
        N1 is N - 1,
        factorial_linear(N1, Prev),
        Result is N * Prev
    )),
    assertz(type_declarations:uw_type(factorial_linear/2, 1, integer)),
    assertz(type_declarations:uw_type(factorial_linear/2, 2, integer)),
    assertz(type_declarations:uw_return_type(factorial_linear/2, integer)),
    once(recursive_compiler:compile_recursive(factorial_linear/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:factorial_linear(_, _)).

test(list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:list_length([], 0)),
    assertz(user:(list_length([_|T], N) :-
        list_length(T, N1),
        N is N1 + 1
    )),
    assertz(type_declarations:uw_type(list_length/2, 1, list(any))),
    assertz(type_declarations:uw_type(list_length/2, 2, integer)),
    assertz(type_declarations:uw_return_type(list_length/2, integer)),
    once(recursive_compiler:compile_recursive(list_length/2, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:list_length(_, _)).

test(nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:power(_Base, 0, 1)),
    assertz(user:(power(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        power(Base, N1, Prev),
        Result is Base * Prev
    )),
    assertz(type_declarations:uw_type(power/3, 1, integer)),
    assertz(type_declarations:uw_type(power/3, 2, integer)),
    assertz(type_declarations:uw_type(power/3, 3, integer)),
    assertz(type_declarations:uw_return_type(power/3, integer)),
    once(recursive_compiler:compile_recursive(power/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:power(_, _, _)).

test(nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:list_length_from(Base, [], Base)),
    assertz(user:(list_length_from(Base, [_|T], N) :-
        list_length_from(Base, T, N1),
        N is N1 + 1
    )),
    assertz(type_declarations:uw_type(list_length_from/3, 1, integer)),
    assertz(type_declarations:uw_type(list_length_from/3, 2, list(any))),
    assertz(type_declarations:uw_type(list_length_from/3, 3, integer)),
    assertz(type_declarations:uw_return_type(list_length_from/3, integer)),
    once(recursive_compiler:compile_recursive(list_length_from/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:list_length_from(_, _, _)).

:- end_tests(typr_toolchain).

typr_cli_available :-
    process_create(path(sh), ['-c', 'command -v typr >/dev/null 2>&1'], [process(Pid)]),
    process_wait(Pid, exit(0)).

create_smoke_project(ProjectDir) :-
    tmp_file(typr_smoke, RootDir),
    make_directory(RootDir),
    run_typr(RootDir, ['new', 'smoke_project']),
    directory_file_path(RootDir, 'smoke_project', ProjectDir),
    exists_directory(ProjectDir).

write_generated_typr_program(ProjectDir, Code) :-
    directory_file_path(ProjectDir, 'TypR/main.ty', MainFile),
    setup_call_cleanup(
        open(MainFile, write, Stream),
        write(Stream, Code),
        close(Stream)
    ).

maybe_build_with_r(ProjectDir) :-
    (   rscript_available
    ->  run_typr(ProjectDir, ['build'])
    ;   true
    ).

rscript_available :-
    process_create(path(sh), ['-c', 'command -v Rscript >/dev/null 2>&1'], [process(Pid)]),
    process_wait(Pid, exit(0)).

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
