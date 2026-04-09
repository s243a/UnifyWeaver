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

:- dynamic tc_descendant/2.
tc_descendant(X, Y) :- tc_parent(Y, X).
tc_descendant(X, Y) :- tc_parent(Z, X), tc_descendant(Z, Y).

:- dynamic tc_distance/3.
tc_distance(X, Y, 1) :- tc_parent(X, Y).
tc_distance(X, Y, D) :- tc_parent(X, Z), tc_distance(Z, Y, D1), D is D1 + 1.

:- dynamic tc_parent_distance/4.
tc_parent_distance(X, Y, X, 1) :- tc_parent(X, Y).
tc_parent_distance(X, Y, Parent, D) :-
    tc_parent(X, Z),
    tc_parent_distance(Z, Y, Parent, D1),
    D is D1 + 1.

:- dynamic tc_step_parent_distance/5.
tc_step_parent_distance(X, Y, Y, X, 1) :- tc_parent(X, Y).
tc_step_parent_distance(X, Y, Step, Parent, D) :-
    tc_parent(X, Step),
    tc_step_parent_distance(Step, Y, _Inner, Parent, D1),
    D is D1 + 1.

:- dynamic tri_sum/2.
tri_sum(0, 0).
tri_sum(N, Sum) :-
    N > 0,
    N1 is N - 1,
    tri_sum(N1, Prev),
    Sum is Prev + N.

:- dynamic tail_suffix/2.
tail_suffix(S, S).
tail_suffix([_|T], S) :- tail_suffix(T, S).

:- dynamic tail_suffixes/2.
tail_suffixes([], [[]]).
tail_suffixes([H|T], [[H|T]|Rest]) :- tail_suffixes(T, Rest).

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
        write_wam_rust_project([user:tc_ancestor/2, user:tc_descendant/2, user:tc_distance/3, user:tc_parent_distance/4, user:tc_step_parent_distance/5, user:tri_sum/2, user:tail_suffix/2, user:tail_suffixes/2],
            [module_name('runtime_test'), wam_fallback(true), foreign_lowering(true)], TmpDir),
        
        % 2. Add a test file
        make_directory_path('output/test_wam_rust_runtime_e2e/tests'),
        directory_file_path(TmpDir, 'tests/integration_test.rs', TestPath),
        TestContent = '
use runtime_test::value::Value;
use runtime_test::state::WamState;
use runtime_test::tc_ancestor;
use runtime_test::tc_descendant;
use runtime_test::tc_distance;
use runtime_test::tc_parent_distance;
use runtime_test::tc_step_parent_distance;
use runtime_test::tri_sum;
use runtime_test::tail_suffix;
use runtime_test::tail_suffixes;

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

    // Test tc_descendant/2 reverse foreign lowering end-to-end
    let mut vm_desc = WamState::new(vec![], std::collections::HashMap::new());
    let ok4 = tc_descendant(&mut vm_desc,
        Value::Atom("jim".to_string()),
        Value::Unbound("Anc".to_string()));
    assert!(ok4, "tc_descendant(jim, Anc) first solution should succeed");

    let mut ancestors: Vec<String> = Vec::new();
    if let Some(Value::Atom(anc)) = vm_desc.bindings.get("Anc").cloned() {
        ancestors.push(anc);
    } else {
        panic!("expected first tc_descendant result in Anc");
    }

    while vm_desc.backtrack() {
        if let Some(Value::Atom(anc)) = vm_desc.bindings.get("Anc").cloned() {
            ancestors.push(anc);
        } else {
            panic!("expected backtracked tc_descendant result in Anc");
        }
    }

    ancestors.sort();
    assert_eq!(ancestors, vec![
        "bob".to_string(),
        "pat".to_string(),
        "tom".to_string(),
    ]);

    let mut vm_desc_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok5 = tc_descendant(&mut vm_desc_check,
        Value::Atom("jim".to_string()),
        Value::Atom("tom".to_string()));
    assert!(ok5, "tc_descendant(jim, tom) should succeed");

    let mut vm_desc_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok6 = tc_descendant(&mut vm_desc_fail,
        Value::Atom("liz".to_string()),
        Value::Atom("jim".to_string()));
    assert!(!ok6, "tc_descendant(liz, jim) should fail");

    // Test tc_distance/3 foreign lowering end-to-end
    let mut vm_dist = WamState::new(vec![], std::collections::HashMap::new());
    let ok7 = tc_distance(&mut vm_dist,
        Value::Atom("tom".to_string()),
        Value::Unbound("Target".to_string()),
        Value::Unbound("Dist".to_string()));
    assert!(ok7, "tc_distance(tom, Target, Dist) first solution should succeed");

    let mut distances: Vec<String> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Integer(dist))) =
        (vm_dist.bindings.get("Target").cloned(), vm_dist.bindings.get("Dist").cloned()) {
        distances.push(format!("{}:{}", target, dist));
    } else {
        panic!("expected first tc_distance result in Target/Dist");
    }

    while vm_dist.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Integer(dist))) =
            (vm_dist.bindings.get("Target").cloned(), vm_dist.bindings.get("Dist").cloned()) {
            distances.push(format!("{}:{}", target, dist));
        } else {
            panic!("expected backtracked tc_distance result in Target/Dist");
        }
    }

    distances.sort();
    assert_eq!(distances, vec![
        "ann:2".to_string(),
        "bob:1".to_string(),
        "jim:3".to_string(),
        "liz:1".to_string(),
        "pat:2".to_string(),
    ]);

    let mut vm_dist_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok8 = tc_distance(&mut vm_dist_check,
        Value::Atom("tom".to_string()),
        Value::Atom("jim".to_string()),
        Value::Integer(3));
    assert!(ok8, "tc_distance(tom, jim, 3) should succeed");

    let mut vm_dist_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok9 = tc_distance(&mut vm_dist_fail,
        Value::Atom("liz".to_string()),
        Value::Atom("jim".to_string()),
        Value::Unbound("D".to_string()));
    assert!(!ok9, "tc_distance(liz, jim, D) should fail");

    // Test tc_parent_distance/4 foreign lowering end-to-end
    let mut vm_parent_dist = WamState::new(vec![], std::collections::HashMap::new());
    let ok_pd1 = tc_parent_distance(&mut vm_parent_dist,
        Value::Atom("tom".to_string()),
        Value::Unbound("Target2".to_string()),
        Value::Unbound("Parent2".to_string()),
        Value::Unbound("Dist2".to_string()));
    assert!(ok_pd1, "tc_parent_distance(tom, Target, Parent, Dist) first solution should succeed");

    let mut parent_distances: Vec<String> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Atom(parent)), Some(Value::Integer(dist))) =
        (vm_parent_dist.bindings.get("Target2").cloned(),
         vm_parent_dist.bindings.get("Parent2").cloned(),
         vm_parent_dist.bindings.get("Dist2").cloned()) {
        parent_distances.push(format!("{}:{}:{}", target, parent, dist));
    } else {
        panic!("expected first tc_parent_distance result in Target2/Parent2/Dist2");
    }

    while vm_parent_dist.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Atom(parent)), Some(Value::Integer(dist))) =
            (vm_parent_dist.bindings.get("Target2").cloned(),
             vm_parent_dist.bindings.get("Parent2").cloned(),
             vm_parent_dist.bindings.get("Dist2").cloned()) {
            parent_distances.push(format!("{}:{}:{}", target, parent, dist));
        } else {
            panic!("expected backtracked tc_parent_distance result in Target2/Parent2/Dist2");
        }
    }

    parent_distances.sort();
    assert_eq!(parent_distances, vec![
        "ann:bob:2".to_string(),
        "bob:tom:1".to_string(),
        "jim:pat:3".to_string(),
        "liz:tom:1".to_string(),
        "pat:bob:2".to_string(),
    ]);

    let mut vm_parent_dist_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok_pd2 = tc_parent_distance(&mut vm_parent_dist_check,
        Value::Atom("tom".to_string()),
        Value::Atom("jim".to_string()),
        Value::Atom("pat".to_string()),
        Value::Integer(3));
    assert!(ok_pd2, "tc_parent_distance(tom, jim, pat, 3) should succeed");

    let mut vm_parent_dist_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok_pd3 = tc_parent_distance(&mut vm_parent_dist_fail,
        Value::Atom("liz".to_string()),
        Value::Atom("jim".to_string()),
        Value::Unbound("Parent3".to_string()),
        Value::Unbound("Dist3".to_string()));
    assert!(!ok_pd3, "tc_parent_distance(liz, jim, Parent, Dist) should fail");

    // Test tc_step_parent_distance/5 foreign lowering end-to-end
    let mut vm_step_parent = WamState::new(vec![], std::collections::HashMap::new());
    let ok_sp1 = tc_step_parent_distance(&mut vm_step_parent,
        Value::Atom("tom".to_string()),
        Value::Unbound("Target4".to_string()),
        Value::Unbound("Step4".to_string()),
        Value::Unbound("Parent4".to_string()),
        Value::Unbound("Dist4".to_string()));
    assert!(ok_sp1, "tc_step_parent_distance(tom, Target, Step, Parent, Dist) first solution should succeed");

    let mut step_parent_distances: Vec<String> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Atom(step)), Some(Value::Atom(parent)), Some(Value::Integer(dist))) =
        (vm_step_parent.bindings.get("Target4").cloned(),
         vm_step_parent.bindings.get("Step4").cloned(),
         vm_step_parent.bindings.get("Parent4").cloned(),
         vm_step_parent.bindings.get("Dist4").cloned()) {
        step_parent_distances.push(format!("{}:{}:{}:{}", target, step, parent, dist));
    } else {
        panic!("expected first tc_step_parent_distance result in Target4/Step4/Parent4/Dist4");
    }

    while vm_step_parent.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Atom(step)), Some(Value::Atom(parent)), Some(Value::Integer(dist))) =
            (vm_step_parent.bindings.get("Target4").cloned(),
             vm_step_parent.bindings.get("Step4").cloned(),
             vm_step_parent.bindings.get("Parent4").cloned(),
             vm_step_parent.bindings.get("Dist4").cloned()) {
            step_parent_distances.push(format!("{}:{}:{}:{}", target, step, parent, dist));
        } else {
            panic!("expected backtracked tc_step_parent_distance result in Target4/Step4/Parent4/Dist4");
        }
    }

    step_parent_distances.sort();
    assert_eq!(step_parent_distances, vec![
        "ann:bob:bob:2".to_string(),
        "bob:bob:tom:1".to_string(),
        "jim:bob:pat:3".to_string(),
        "liz:liz:tom:1".to_string(),
        "pat:bob:bob:2".to_string(),
    ]);

    let mut vm_step_parent_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok_sp2 = tc_step_parent_distance(&mut vm_step_parent_check,
        Value::Atom("tom".to_string()),
        Value::Atom("jim".to_string()),
        Value::Atom("bob".to_string()),
        Value::Atom("pat".to_string()),
        Value::Integer(3));
    assert!(ok_sp2, "tc_step_parent_distance(tom, jim, bob, pat, 3) should succeed");

    let mut vm_step_parent_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok_sp3 = tc_step_parent_distance(&mut vm_step_parent_fail,
        Value::Atom("liz".to_string()),
        Value::Atom("jim".to_string()),
        Value::Unbound("Step".to_string()),
        Value::Unbound("Parent".to_string()),
        Value::Unbound("Dist".to_string()));
    assert!(!ok_sp3, "tc_step_parent_distance(liz, jim, Step, Parent, Dist) should fail");

    // Test tri_sum/2 foreign lowering end-to-end
    let mut vm_sum = WamState::new(vec![], std::collections::HashMap::new());
    let ok10 = tri_sum(&mut vm_sum,
        Value::Integer(4),
        Value::Unbound("Sum".to_string()));
    assert!(ok10, "tri_sum(4, Sum) should succeed");
    assert_eq!(vm_sum.bindings.get("Sum").cloned(), Some(Value::Integer(10)));

    let mut vm_sum_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok11 = tri_sum(&mut vm_sum_check,
        Value::Integer(4),
        Value::Integer(10));
    assert!(ok11, "tri_sum(4, 10) should succeed");

    let mut vm_sum_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok12 = tri_sum(&mut vm_sum_fail,
        Value::Integer(4),
        Value::Integer(11));
    assert!(!ok12, "tri_sum(4, 11) should fail");

    // Test tail_suffix/2 foreign lowering end-to-end
    let list_abc = Value::List(vec![
        Value::Atom("a".to_string()),
        Value::Atom("b".to_string()),
        Value::Atom("c".to_string()),
    ]);
    let mut vm_suffix = WamState::new(vec![], std::collections::HashMap::new());
    let ok13 = tail_suffix(&mut vm_suffix,
        list_abc.clone(),
        Value::Unbound("Suffix".to_string()));
    assert!(ok13, "tail_suffix([a,b,c], Suffix) first solution should succeed");

    let mut suffixes: Vec<String> = Vec::new();
    if let Some(value) = vm_suffix.bindings.get("Suffix").cloned() {
        suffixes.push(format!("{}", value));
    } else {
        panic!("expected first tail_suffix result in Suffix");
    }

    while vm_suffix.backtrack() {
        if let Some(value) = vm_suffix.bindings.get("Suffix").cloned() {
            suffixes.push(format!("{}", value));
        } else {
            panic!("expected backtracked tail_suffix result in Suffix");
        }
    }

    assert_eq!(suffixes, vec![
        "[a, b, c]".to_string(),
        "[b, c]".to_string(),
        "[c]".to_string(),
        "[]".to_string(),
    ]);

    let mut vm_suffix_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok14 = tail_suffix(&mut vm_suffix_check,
        list_abc.clone(),
        Value::List(vec![
            Value::Atom("b".to_string()),
            Value::Atom("c".to_string()),
        ]));
    assert!(ok14, "tail_suffix([a,b,c], [b,c]) should succeed");

    let mut vm_suffix_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok15 = tail_suffix(&mut vm_suffix_fail,
        list_abc,
        Value::List(vec![Value::Atom("d".to_string())]));
    assert!(!ok15, "tail_suffix([a,b,c], [d]) should fail");

    // Test tail_suffixes/2 deterministic collection foreign lowering end-to-end
    let mut vm_suffixes = WamState::new(vec![], std::collections::HashMap::new());
    let ok16 = tail_suffixes(&mut vm_suffixes,
        Value::List(vec![
            Value::Atom("a".to_string()),
            Value::Atom("b".to_string()),
            Value::Atom("c".to_string()),
        ]),
        Value::Unbound("Suffixes".to_string()));
    assert!(ok16, "tail_suffixes([a,b,c], Suffixes) should succeed");
    match vm_suffixes.bindings.get("Suffixes").cloned() {
        Some(Value::List(items)) => assert_eq!(items, vec![
            Value::List(vec![
                Value::Atom("a".to_string()),
                Value::Atom("b".to_string()),
                Value::Atom("c".to_string()),
            ]),
            Value::List(vec![
                Value::Atom("b".to_string()),
                Value::Atom("c".to_string()),
            ]),
            Value::List(vec![Value::Atom("c".to_string())]),
            Value::List(vec![]),
        ]),
        other => panic!("expected deterministic suffix collection in Suffixes, got {:?}", other),
    }
    assert!(!vm_suffixes.backtrack(), "tail_suffixes/2 should not leave backtracking results behind");
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
