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

test(guarded_nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:power_if(_Base, 0, 1)),
    assertz(user:(power_if(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        power_if(Base, N1, Prev),
        ( Base > 1 -> Result is Base * Prev ; Result is Prev )
    )),
    assertz(type_declarations:uw_type(power_if/3, 1, integer)),
    assertz(type_declarations:uw_type(power_if/3, 2, integer)),
    assertz(type_declarations:uw_type(power_if/3, 3, integer)),
    assertz(type_declarations:uw_return_type(power_if/3, integer)),
    once(recursive_compiler:compile_recursive(power_if/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:power_if(_, _, _)).

test(guarded_nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:count_occ(_, [], 0)),
    assertz(user:(count_occ(X, [Y|T], N) :-
        count_occ(X, T, N1),
        ( X == Y -> N is N1 + 1 ; N is N1 )
    )),
    assertz(type_declarations:uw_type(count_occ/3, 1, integer)),
    assertz(type_declarations:uw_type(count_occ/3, 2, list(integer))),
    assertz(type_declarations:uw_type(count_occ/3, 3, integer)),
    assertz(type_declarations:uw_return_type(count_occ/3, integer)),
    once(recursive_compiler:compile_recursive(count_occ/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:count_occ(_, _, _)).

test(multistate_nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:power_multistate(_Base, 0, 1)),
    assertz(user:(power_multistate(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        power_multistate(Base, N1, Prev),
        ( Base > 1 ->
            Step is Base * Prev,
            Offset is Step + 1
        ;   Step is Prev,
            Offset is Step + 2
        ),
        Result is Offset + Step
    )),
    assertz(type_declarations:uw_type(power_multistate/3, 1, integer)),
    assertz(type_declarations:uw_type(power_multistate/3, 2, integer)),
    assertz(type_declarations:uw_type(power_multistate/3, 3, integer)),
    assertz(type_declarations:uw_return_type(power_multistate/3, integer)),
    once(recursive_compiler:compile_recursive(power_multistate/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:power_multistate(_, _, _)).

test(multistate_nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:count_weighted(_, [], 0)),
    assertz(user:(count_weighted(X, [Y|T], N) :-
        count_weighted(X, T, N1),
        ( X == Y ->
            Delta is N1 + 1,
            Adjust is Delta + 1
        ;   Delta is N1,
            Adjust is Delta + 2
        ),
        N is Adjust + Delta
    )),
    assertz(type_declarations:uw_type(count_weighted/3, 1, integer)),
    assertz(type_declarations:uw_type(count_weighted/3, 2, list(integer))),
    assertz(type_declarations:uw_type(count_weighted/3, 3, integer)),
    assertz(type_declarations:uw_return_type(count_weighted/3, integer)),
    once(recursive_compiler:compile_recursive(count_weighted/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:count_weighted(_, _, _)).

test(asymmetric_nary_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:asym_rec_a(_Base, 0, 1)),
    assertz(user:(asym_rec_a(Base, N, Result) :-
        N > 0,
        N1 is N - 1,
        asym_rec_a(Base, N1, Prev),
        ( Base > 1 ->
            Step is Base * Prev,
            Result is Step + 1
        ;   Temp is Prev + 2,
            Result is Temp + Prev
        )
    )),
    assertz(type_declarations:uw_type(asym_rec_a/3, 1, integer)),
    assertz(type_declarations:uw_type(asym_rec_a/3, 2, integer)),
    assertz(type_declarations:uw_type(asym_rec_a/3, 3, integer)),
    assertz(type_declarations:uw_return_type(asym_rec_a/3, integer)),
    once(recursive_compiler:compile_recursive(asym_rec_a/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:asym_rec_a(_, _, _)).

test(asymmetric_nary_list_linear_recursive_output_checks_with_typr, [condition(typr_cli_available)]) :-
    clear_type_declarations,
    assertz(user:asym_rec_c(_, [], 0)),
    assertz(user:(asym_rec_c(X, [Y|T], N) :-
        asym_rec_c(X, T, N1),
        ( X == Y ->
            Delta is N1 + 1,
            N is Delta + 1
        ;   Extra is N1 + 2,
            N is Extra + N1
        )
    )),
    assertz(type_declarations:uw_type(asym_rec_c/3, 1, integer)),
    assertz(type_declarations:uw_type(asym_rec_c/3, 2, list(integer))),
    assertz(type_declarations:uw_type(asym_rec_c/3, 3, integer)),
    assertz(type_declarations:uw_return_type(asym_rec_c/3, integer)),
    once(recursive_compiler:compile_recursive(asym_rec_c/3, [target(typr), typed_mode(explicit)], Code)),
    setup_call_cleanup(
        create_smoke_project(ProjectDir),
        (
            write_generated_typr_program(ProjectDir, Code),
            run_typr(ProjectDir, ['check']),
            maybe_build_with_r(ProjectDir)
        ),
        delete_directory_and_contents(ProjectDir)
    ),
    retractall(user:asym_rec_c(_, _, _)).

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
