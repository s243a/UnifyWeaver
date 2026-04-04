:- encoding(utf8).
% Runtime execution tests for WAM-to-Rust transpilation
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_runtime.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
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

:- dynamic test_member/1.
test_member(X) :- member(X, [1, 2, 3]).

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
        write_wam_rust_project([user:fact/1, user:path/2, user:test_member/1], 
            [module_name('runtime_test'), wam_fallback(true)], TmpDir),
        
        % 2. Add a test file
        make_directory_path('output/test_wam_rust_runtime_e2e/tests'),
        directory_file_path(TmpDir, 'tests/integration_test.rs', TestPath),
        TestContent = '
use runtime_test::value::Value;
use runtime_test::state::WamState;
use runtime_test::{fact, path, test_member};

#[test]
fn test_generated_predicates() {
    // Test fact/1
    let mut vm_fact = WamState::new(vec![], std::collections::HashMap::new());
    let ok1 = fact(&mut vm_fact, Value::Atom("alice".to_string()));
    assert!(ok1, "fact(alice) should succeed");
    
    let ok2 = fact(&mut vm_fact, Value::Atom("charlie".to_string()));
    assert!(!ok2, "fact(charlie) should fail");
    
    // Test path/2
    let mut vm_path = WamState::new(vec![], std::collections::HashMap::new());
    let ok3 = path(&mut vm_path, Value::Atom("a".to_string()), Value::Atom("b".to_string()));
    assert!(ok3, "path(a, b) should succeed");

    // Test relational member/2 directly
    let mut vm_direct = WamState::new(vec![], std::collections::HashMap::new());
    let list = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
    vm_direct.set_reg("A1", Value::Unbound("X".to_string()));
    vm_direct.set_reg("A2", list);
    
    // First solution: X=1
    let mut ok_mem = vm_direct.execute_builtin("member/2", 2);
    assert!(ok_mem, "member first call should succeed");
    assert_eq!(vm_direct.get_reg("X"), Some(Value::Integer(1)));
    
    // Backtrack for second solution: X=2
    ok_mem = vm_direct.backtrack();
    assert!(ok_mem, "member second solution should succeed");
    assert_eq!(vm_direct.get_reg("X"), Some(Value::Integer(2)));
    
    // Backtrack for third solution: X=3
    ok_mem = vm_direct.backtrack();
    assert!(ok_mem, "member third solution should succeed");
    assert_eq!(vm_direct.get_reg("X"), Some(Value::Integer(3)));
    
    // Fourth backtrack: should fail
    ok_mem = vm_direct.backtrack();
    assert!(!ok_mem, "member fourth solution should fail");

    // Test test_member/1 which calls member/2 via WAM instructions
    let mut vm_nested = WamState::new(vec![], std::collections::HashMap::new());
    let ok4 = test_member(&mut vm_nested, Value::Unbound("Y".to_string()));
    assert!(ok4, "test_member(Y) first solution should succeed");
    assert_eq!(vm_nested.get_reg("Y"), Some(Value::Integer(1)));

    let ok5 = vm_nested.backtrack();
    assert!(ok5, "test_member(Y) second solution should succeed");
    assert_eq!(vm_nested.get_reg("Y"), Some(Value::Integer(2)));

    let ok6 = vm_nested.backtrack();
    assert!(ok6, "test_member(Y) third solution should succeed");
    assert_eq!(vm_nested.get_reg("Y"), Some(Value::Integer(3)));

    let ok7 = vm_nested.backtrack();
    assert!(!ok7, "test_member(Y) fourth solution should fail");
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
