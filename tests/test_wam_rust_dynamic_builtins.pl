:- encoding(utf8).
% Rust WAM runtime dynamic database builtins.
% Usage: swipl -q -s tests/test_wam_rust_dynamic_builtins.pl -g run_tests -t halt

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).
:- use_module(library(process)).
:- use_module(library(filesex)).

:- dynamic rust_dyn_dummy/0.
rust_dyn_dummy.

:- dynamic rust_assert_alias_demo/0.
rust_assert_alias_demo :- assert(dyn(alias)).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :-
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_dynamic_builtins).

test(assert_retract_dynamic_db,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_wam_rust_dynamic_builtins'))]) :-
    Dir = 'output/test_wam_rust_dynamic_builtins',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:rust_dyn_dummy/0, user:rust_assert_alias_demo/0],
        [module_name(dynrt), no_kernels(true), runtime_parser(compiled)],
        Dir)),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/dynamic_builtins.rs', TestPath),
    read_file_to_string('tests/fixtures/wam_rust_dynamic_builtins.rs', TestSrc, []),
    setup_call_cleanup(open(TestPath, write, S), format(S, '~s', [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test dynamic_builtins -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[WAM Rust dynamic builtins FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_dynamic_builtins).
