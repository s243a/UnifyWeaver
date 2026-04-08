:- encoding(utf8).
% Runtime execution tests for WAM-to-Rust transpilation
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_runtime.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/rust_target').
:- use_module(library(process)).
:- use_module(library(filesex)).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Test predicates
:- dynamic fact/1.
fact(alice).
fact(bob).

:- dynamic path/2.
path(a, b).
path(b, c).
path(c, d).

:- dynamic tc_parent/2.
tc_parent(tom, bob).
tc_parent(tom, liz).
tc_parent(bob, ann).
tc_parent(bob, pat).
tc_parent(pat, jim).

:- dynamic tc_ancestor/2.
tc_ancestor(X, Y) :- tc_parent(X, Y).
tc_ancestor(X, Y) :- tc_parent(X, Z), tc_ancestor(Z, Y).

%% Integration test
test_runtime_execution :-
    Test = 'WAM-Rust Runtime: end-to-end execution',
    TmpDir = 'output/test_wam_rust_runtime_e2e',
    (   % Skip if cargo is not available
        \+ cargo_available
    ->  format('[SKIP] ~w (cargo not found)~n', [Test]),
        pass(Test)
    ;   (exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true),
        % 1. Generate project
        write_wam_rust_project([user:tc_ancestor/2],
            [module_name('runtime_test'), wam_fallback(true), foreign_lowering(true)], TmpDir),
        
        % 2. Add a test file
        make_directory_path('output/test_wam_rust_runtime_e2e/tests'),
        directory_file_path(TmpDir, 'tests/integration_test.rs', TestPath),
        TestContent = '
use runtime_test::value::Value;
use runtime_test::state::WamState;
use runtime_test::tc_ancestor;

#[test]
fn test_generated_predicates() {
    // Test tc_ancestor/2 foreign lowering end-to-end
    let mut vm_tc = WamState::new(vec![], std::collections::HashMap::new());
    let ok1 = tc_ancestor(&mut vm_tc,
        Value::Atom("tom".to_string()),
        Value::Unbound("Desc".to_string()));
    assert!(ok1, "tc_ancestor(tom, Desc) first solution should succeed");

    let mut descendants: Vec<String> = Vec::new();
    if let Some(Value::Atom(desc)) = vm_tc.bindings.get("Desc").cloned() {
        descendants.push(desc);
    } else {
        panic!("expected first tc_ancestor result in Desc");
    }

    while vm_tc.backtrack() {
        if let Some(Value::Atom(desc)) = vm_tc.bindings.get("Desc").cloned() {
            descendants.push(desc);
        } else {
            panic!("expected backtracked tc_ancestor result in Desc");
        }
    }

    descendants.sort();
    assert_eq!(descendants, vec![
        "ann".to_string(),
        "bob".to_string(),
        "jim".to_string(),
        "liz".to_string(),
        "pat".to_string(),
    ]);

    let mut vm_tc_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok2 = tc_ancestor(&mut vm_tc_check,
        Value::Atom("tom".to_string()),
        Value::Atom("jim".to_string()));
    assert!(ok2, "tc_ancestor(tom, jim) should succeed");

    let mut vm_tc_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok3 = tc_ancestor(&mut vm_tc_fail,
        Value::Atom("liz".to_string()),
        Value::Atom("jim".to_string()));
    assert!(!ok3, "tc_ancestor(liz, jim) should fail");
}
',
        setup_call_cleanup(open(TestPath, write, S), format(S, "~w", [TestContent]), close(S)),
        
        % 3. Run cargo test
        format(atom(Cmd), 'cd "~w" && cargo test -- --nocapture 2>&1', [TmpDir]),
        (   catch(
                (process_create(path(sh), ['-c', Cmd], [stdout(pipe(Out)), process(Pid)]),
                 read_string(Out, _, OutStr),
                 close(Out),
                 process_wait(Pid, exit(ExitCode))),
                _, (ExitCode = 1, OutStr = ""))
        ->  % 4. Verify results
            (   ExitCode == 0,
                sub_string(OutStr, _, _, _, 'test_generated_predicates ... ok')
            ->  pass(Test)
            ;   fail_test(Test, 'Runtime execution failed or output mismatch'),
                format('Cargo output:~n~w~n', [OutStr])
            )
        ;   fail_test(Test, 'Failed to execute cargo test')
        ),
        
        % Clean up
        (exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true)
    ).

cargo_available :-
    catch(
        (process_create(path(cargo), ['--version'], [stdout(null), stderr(null), process(Pid)]),
         process_wait(Pid, exit(0))),
        _, fail).

run_tests :-
    format('~n========================================~n'),
    format('WAM-Rust Runtime Integration Tests~n'),
    format('========================================~n~n'),
    
    test_runtime_execution,
    
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
