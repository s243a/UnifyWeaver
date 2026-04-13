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

:- dynamic weighted_edge/3.
weighted_edge(s, a, 1.0).
weighted_edge(s, b, 4.0).
weighted_edge(a, b, 2.0).
weighted_edge(a, c, 5.0).
weighted_edge(b, c, 1.0).
weighted_edge(c, d, 3.0).

:- dynamic direct_semantic_dist/3.
direct_semantic_dist(s, a, 1.0).
direct_semantic_dist(s, b, 3.0).
direct_semantic_dist(s, c, 4.0).
direct_semantic_dist(s, d, 7.0).
direct_semantic_dist(a, b, 2.0).
direct_semantic_dist(a, c, 3.0).
direct_semantic_dist(a, d, 6.0).
direct_semantic_dist(b, c, 1.0).
direct_semantic_dist(b, d, 4.0).
direct_semantic_dist(c, d, 3.0).

:- dynamic weighted_path/3.
weighted_path(X, Y, W) :- weighted_edge(X, Y, W).
weighted_path(X, Y, Cost) :-
    weighted_edge(X, Z, W),
    weighted_path(Z, Y, RestCost),
    Cost is W + RestCost.

:- dynamic astar_weighted_path/4.
astar_weighted_path(X, Y, _Dim, W) :- weighted_edge(X, Y, W).
astar_weighted_path(X, Y, Dim, Cost) :-
    weighted_edge(X, Z, W),
    direct_semantic_dist(Z, Y, _Heuristic),
    astar_weighted_path(Z, Y, Dim, RestCost),
    Cost is W + RestCost.

:- dynamic min_semantic_dist/3.
min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Cost), weighted_path(Start, Target, Cost), MinDist).

:- dynamic min_semantic_dist_astar/4.
min_semantic_dist_astar(Start, Target, Dim, MinDist) :-
    aggregate_all(min(Cost), astar_weighted_path(Start, Target, Dim, Cost), MinDist).

:- dynamic grouped_min_semantic_dist/3.
grouped_min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Cost), weighted_path(Start, Target, Cost), Target, MinDist).

:- dynamic grouped_min_semantic_dist_astar/4.
grouped_min_semantic_dist_astar(Start, Target, Dim, MinDist) :-
    aggregate_all(min(Cost), astar_weighted_path(Start, Target, Dim, Cost), Target, MinDist).

:- dynamic filtered_adjusted_min_semantic_dist/3.
filtered_adjusted_min_semantic_dist(Start, Target, MinDist) :-
    aggregate_all(min(Adjusted),
        (weighted_path(Start, Target, Cost), Cost > 2, Adjusted is Cost + 1),
        MinDist).

:- dynamic filtered_adjusted_min_semantic_dist_astar/4.
filtered_adjusted_min_semantic_dist_astar(Start, Target, Dim, MinDist) :-
    aggregate_all(min(Adjusted),
        (astar_weighted_path(Start, Target, Dim, Cost), Cost > 2, Adjusted is Cost + 1),
        MinDist).

:- dynamic filtered_adjusted_weighted_path/3.
filtered_adjusted_weighted_path(Start, Target, Adjusted) :-
    weighted_path(Start, Target, Cost),
    Cost > 2,
    Adjusted is Cost + 1.

:- dynamic filtered_adjusted_astar_weighted_path/4.
filtered_adjusted_astar_weighted_path(Start, Target, Dim, Adjusted) :-
    astar_weighted_path(Start, Target, Dim, Cost),
    Cost > 2,
    Adjusted is Cost + 1.

:- dynamic target_label/2.
target_label(b, branch_b).
target_label(c, branch_c).
target_label(d, goal_d1).
target_label(d, goal_d2).

:- dynamic label_bucket/2.
label_bucket(branch_b, branch_bucket).
label_bucket(branch_c, branch_bucket).
label_bucket(goal_d1, goal_bucket).
label_bucket(goal_d2, goal_bucket).

:- dynamic labeled_adjusted_weighted_path/3.
labeled_adjusted_weighted_path(Start, Label, Adjusted) :-
    weighted_path(Start, Target, Cost),
    target_label(Target, Label),
    Cost > 2,
    Adjusted is Cost + 1.

:- dynamic labeled_adjusted_astar_weighted_path/4.
labeled_adjusted_astar_weighted_path(Start, Label, Dim, Adjusted) :-
    astar_weighted_path(Start, Target, Dim, Cost),
    target_label(Target, Label),
    Cost > 2,
    Adjusted is Cost + 1.

:- dynamic bucketed_adjusted_weighted_path/3.
bucketed_adjusted_weighted_path(Start, Bucket, Adjusted) :-
    weighted_path(Start, Target, Cost),
    target_label(Target, Label),
    label_bucket(Label, Bucket),
    Cost > 2,
    Adjusted is Cost + 1.

:- dynamic bucketed_adjusted_astar_weighted_path/4.
bucketed_adjusted_astar_weighted_path(Start, Bucket, Dim, Adjusted) :-
    astar_weighted_path(Start, Target, Dim, Cost),
    target_label(Target, Label),
    label_bucket(Label, Bucket),
    Cost > 2,
    Adjusted is Cost + 1.

:- dynamic bucketed_min_semantic_dist/3.
bucketed_min_semantic_dist(Start, Bucket, MinDist) :-
    aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        MinDist).

:- dynamic bucketed_min_semantic_dist_astar/4.
bucketed_min_semantic_dist_astar(Start, Bucket, Dim, MinDist) :-
    aggregate_all(min(Adjusted),
        ( astar_weighted_path(Start, Target, Dim, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        MinDist).

:- dynamic grouped_bucketed_min_semantic_dist/3.
grouped_bucketed_min_semantic_dist(Start, Bucket, MinDist) :-
    aggregate_all(min(Adjusted),
        ( weighted_path(Start, Target, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        Bucket,
        MinDist).

:- dynamic grouped_bucketed_min_semantic_dist_astar/4.
grouped_bucketed_min_semantic_dist_astar(Start, Bucket, Dim, MinDist) :-
    aggregate_all(min(Adjusted),
        ( astar_weighted_path(Start, Target, Dim, Cost),
          target_label(Target, Label),
          label_bucket(Label, Bucket),
          Cost > 2,
          Adjusted is Cost + 1
        ),
        Bucket,
        MinDist).

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
        write_wam_rust_project([user:tc_ancestor/2, user:tc_descendant/2, user:tc_distance/3, user:tc_parent_distance/4, user:tc_step_parent_distance/5, user:tri_sum/2, user:tail_suffix/2, user:tail_suffixes/2, user:weighted_path/3, user:astar_weighted_path/4, user:min_semantic_dist/3, user:min_semantic_dist_astar/4, user:grouped_min_semantic_dist/3, user:grouped_min_semantic_dist_astar/4, user:filtered_adjusted_min_semantic_dist/3, user:filtered_adjusted_min_semantic_dist_astar/4, user:filtered_adjusted_weighted_path/3, user:filtered_adjusted_astar_weighted_path/4, user:labeled_adjusted_weighted_path/3, user:labeled_adjusted_astar_weighted_path/4, user:bucketed_adjusted_weighted_path/3, user:bucketed_adjusted_astar_weighted_path/4, user:bucketed_min_semantic_dist/3, user:bucketed_min_semantic_dist_astar/4, user:grouped_bucketed_min_semantic_dist/3, user:grouped_bucketed_min_semantic_dist_astar/4],
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
use runtime_test::weighted_path;
use runtime_test::astar_weighted_path;
use runtime_test::min_semantic_dist;
use runtime_test::min_semantic_dist_astar;
use runtime_test::grouped_min_semantic_dist;
use runtime_test::grouped_min_semantic_dist_astar;
use runtime_test::filtered_adjusted_min_semantic_dist;
use runtime_test::filtered_adjusted_min_semantic_dist_astar;
use runtime_test::filtered_adjusted_weighted_path;
use runtime_test::filtered_adjusted_astar_weighted_path;
use runtime_test::labeled_adjusted_weighted_path;
use runtime_test::labeled_adjusted_astar_weighted_path;
use runtime_test::bucketed_adjusted_weighted_path;
use runtime_test::bucketed_adjusted_astar_weighted_path;
use runtime_test::bucketed_min_semantic_dist;
use runtime_test::bucketed_min_semantic_dist_astar;
use runtime_test::grouped_bucketed_min_semantic_dist;
use runtime_test::grouped_bucketed_min_semantic_dist_astar;

#[test]
fn test_generated_predicates() {
    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "expected {}, got {}", expected, actual);
    }

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

    // Test weighted_path/3 foreign lowering end-to-end
    let mut vm_weighted = WamState::new(vec![], std::collections::HashMap::new());
    let ok17 = weighted_path(&mut vm_weighted,
        Value::Atom("s".to_string()),
        Value::Unbound("TargetW".to_string()),
        Value::Unbound("CostW".to_string()));
    assert!(ok17, "weighted_path(s, Target, Cost) first solution should succeed");

    let mut weighted_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_weighted.bindings.get("TargetW").cloned(), vm_weighted.bindings.get("CostW").cloned()) {
        weighted_results.push((target, cost));
    } else {
        panic!("expected first weighted_path result in TargetW/CostW");
    }

    while vm_weighted.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_weighted.bindings.get("TargetW").cloned(), vm_weighted.bindings.get("CostW").cloned()) {
            weighted_results.push((target, cost));
        } else {
            panic!("expected backtracked weighted_path result in TargetW/CostW");
        }
    }

    weighted_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(weighted_results.len(), 4);
    assert_eq!(weighted_results[0].0, "a".to_string());
    assert_eq!(weighted_results[1].0, "b".to_string());
    assert_eq!(weighted_results[2].0, "c".to_string());
    assert_eq!(weighted_results[3].0, "d".to_string());
    assert_close(weighted_results[0].1, 1.0);
    assert_close(weighted_results[1].1, 3.0);
    assert_close(weighted_results[2].1, 4.0);
    assert_close(weighted_results[3].1, 7.0);

    let mut vm_weighted_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok18 = weighted_path(&mut vm_weighted_check,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Unbound("CostD".to_string()));
    assert!(ok18, "weighted_path(s, d, Cost) should succeed");
    match vm_weighted_check.bindings.get("CostD").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 7.0),
        other => panic!("expected weighted_path(s, d, CostD) to bind float cost, got {:?}", other),
    }

    let mut vm_weighted_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok19 = weighted_path(&mut vm_weighted_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Float(4.0));
    assert!(ok19, "weighted_path(s, c, 4.0) should succeed");

    let mut vm_weighted_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok20 = weighted_path(&mut vm_weighted_fail,
        Value::Atom("d".to_string()),
        Value::Atom("s".to_string()),
        Value::Unbound("CostFail".to_string()));
    assert!(!ok20, "weighted_path(d, s, Cost) should fail");

    // Test astar_weighted_path/4 foreign lowering end-to-end
    let mut vm_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok_astar1 = astar_weighted_path(&mut vm_astar,
        Value::Atom("s".to_string()),
        Value::Unbound("ATarget".to_string()),
        Value::Integer(5),
        Value::Unbound("ACost".to_string()));
    assert!(ok_astar1, "astar_weighted_path(s, Target, 5, Cost) first solution should succeed");

    let mut astar_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_astar.bindings.get("ATarget").cloned(), vm_astar.bindings.get("ACost").cloned()) {
        astar_results.push((target, cost));
    } else {
        panic!("expected first astar_weighted_path result in ATarget/ACost");
    }

    while vm_astar.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_astar.bindings.get("ATarget").cloned(), vm_astar.bindings.get("ACost").cloned()) {
            astar_results.push((target, cost));
        } else {
            panic!("expected backtracked astar_weighted_path result in ATarget/ACost");
        }
    }

    astar_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(astar_results.len(), 4);
    assert_eq!(astar_results[0].0, "a".to_string());
    assert_eq!(astar_results[1].0, "b".to_string());
    assert_eq!(astar_results[2].0, "c".to_string());
    assert_eq!(astar_results[3].0, "d".to_string());
    assert_close(astar_results[0].1, 1.0);
    assert_close(astar_results[1].1, 3.0);
    assert_close(astar_results[2].1, 4.0);
    assert_close(astar_results[3].1, 7.0);

    let mut vm_astar_check = WamState::new(vec![], std::collections::HashMap::new());
    let ok_astar2 = astar_weighted_path(&mut vm_astar_check,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Integer(5),
        Value::Unbound("ACostD".to_string()));
    assert!(ok_astar2, "astar_weighted_path(s, d, 5, Cost) should succeed");
    match vm_astar_check.bindings.get("ACostD").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 7.0),
        other => panic!("expected astar_weighted_path(s, d, 5, Cost) to bind float cost, got {:?}", other),
    }

    let mut vm_astar_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok_astar3 = astar_weighted_path(&mut vm_astar_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Integer(5),
        Value::Float(4.0));
    assert!(ok_astar3, "astar_weighted_path(s, c, 5, 4.0) should succeed");

    let mut vm_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok_astar4 = astar_weighted_path(&mut vm_astar_fail,
        Value::Atom("d".to_string()),
        Value::Atom("s".to_string()),
        Value::Integer(5),
        Value::Unbound("ACostFail".to_string()));
    assert!(!ok_astar4, "astar_weighted_path(d, s, 5, Cost) should fail");

    // Test min_semantic_dist/3 aggregate wrapper over weighted_path/3
    let mut vm_min = WamState::new(vec![], std::collections::HashMap::new());
    let ok21 = min_semantic_dist(&mut vm_min,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Unbound("MinCost".to_string()));
    assert!(ok21, "min_semantic_dist(s, d, MinCost) should succeed");
    match vm_min.bindings.get("MinCost").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 7.0),
        other => panic!("expected min_semantic_dist(s, d, MinCost) to bind float cost, got {:?}", other),
    }

    let mut vm_min_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok22 = min_semantic_dist(&mut vm_min_any,
        Value::Atom("s".to_string()),
        Value::Unbound("AnyTarget".to_string()),
        Value::Unbound("GlobalMin".to_string()));
    assert!(ok22, "min_semantic_dist(s, Target, GlobalMin) should succeed");
    assert!(vm_min_any.bindings.get("AnyTarget").is_none(), "aggregate wrapper should not bind existential Target");
    match vm_min_any.bindings.get("GlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 1.0),
        other => panic!("expected min_semantic_dist(s, Target, GlobalMin) to bind float cost, got {:?}", other),
    }

    let mut vm_min_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok23 = min_semantic_dist(&mut vm_min_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Float(4.0));
    assert!(ok23, "min_semantic_dist(s, c, 4.0) should succeed");

    let mut vm_min_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok24 = min_semantic_dist(&mut vm_min_fail,
        Value::Atom("d".to_string()),
        Value::Atom("s".to_string()),
        Value::Unbound("NoPath".to_string()));
    assert!(!ok24, "min_semantic_dist(d, s, NoPath) should fail");

    // Test min_semantic_dist_astar/4 aggregate wrapper over astar_weighted_path/4
    let mut vm_astar_min = WamState::new(vec![], std::collections::HashMap::new());
    let ok25 = min_semantic_dist_astar(&mut vm_astar_min,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarMinCost".to_string()));
    assert!(ok25, "min_semantic_dist_astar(s, d, 5, MinCost) should succeed");
    match vm_astar_min.bindings.get("AStarMinCost").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 7.0),
        other => panic!("expected min_semantic_dist_astar(s, d, 5, MinCost) to bind float cost, got {:?}", other),
    }

    let mut vm_astar_min_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok26 = min_semantic_dist_astar(&mut vm_astar_min_any,
        Value::Atom("s".to_string()),
        Value::Unbound("AStarAnyTarget".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarGlobalMin".to_string()));
    assert!(ok26, "min_semantic_dist_astar(s, Target, 5, GlobalMin) should succeed");
    assert!(vm_astar_min_any.bindings.get("AStarAnyTarget").is_none(), "A* aggregate wrapper should not bind existential Target");
    match vm_astar_min_any.bindings.get("AStarGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 1.0),
        other => panic!("expected min_semantic_dist_astar(s, Target, 5, GlobalMin) to bind float cost, got {:?}", other),
    }

    let mut vm_astar_min_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok27 = min_semantic_dist_astar(&mut vm_astar_min_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Integer(5),
        Value::Float(4.0));
    assert!(ok27, "min_semantic_dist_astar(s, c, 5, 4.0) should succeed");

    let mut vm_astar_min_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok28 = min_semantic_dist_astar(&mut vm_astar_min_fail,
        Value::Atom("d".to_string()),
        Value::Atom("s".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarNoPath".to_string()));
    assert!(!ok28, "min_semantic_dist_astar(d, s, 5, NoPath) should fail");

    // Test grouped_min_semantic_dist/3 streaming grouped minima over weighted_path/3
    let mut vm_grouped = WamState::new(vec![], std::collections::HashMap::new());
    let ok29 = grouped_min_semantic_dist(&mut vm_grouped,
        Value::Atom("s".to_string()),
        Value::Unbound("GroupedTarget".to_string()),
        Value::Unbound("GroupedMin".to_string()));
    assert!(ok29, "grouped_min_semantic_dist(s, Target, Min) first grouped result should succeed");
    let mut grouped_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_grouped.bindings.get("GroupedTarget").cloned(), vm_grouped.bindings.get("GroupedMin").cloned()) {
        grouped_results.push((target, cost));
    } else {
        panic!("expected first grouped_min_semantic_dist result");
    }
    while vm_grouped.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_grouped.bindings.get("GroupedTarget").cloned(), vm_grouped.bindings.get("GroupedMin").cloned()) {
            grouped_results.push((target, cost));
        } else {
            panic!("expected grouped_min_semantic_dist backtrack result");
        }
    }
    grouped_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(grouped_results, vec![
        ("a".to_string(), 1.0),
        ("b".to_string(), 3.0),
        ("c".to_string(), 4.0),
        ("d".to_string(), 7.0),
    ]);

    let mut vm_grouped_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok30 = grouped_min_semantic_dist(&mut vm_grouped_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Unbound("GroupedC".to_string()));
    assert!(ok30, "grouped_min_semantic_dist(s, c, Min) should succeed");
    match vm_grouped_exact.bindings.get("GroupedC").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected grouped_min_semantic_dist(s, c, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_grouped_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok31 = grouped_min_semantic_dist(&mut vm_grouped_fail,
        Value::Atom("d".to_string()),
        Value::Unbound("GroupedNoTarget".to_string()),
        Value::Unbound("GroupedNoMin".to_string()));
    assert!(!ok31, "grouped_min_semantic_dist(d, Target, Min) should fail");

    // Test grouped_min_semantic_dist_astar/4 streaming grouped minima over astar_weighted_path/4
    let mut vm_grouped_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok32 = grouped_min_semantic_dist_astar(&mut vm_grouped_astar,
        Value::Atom("s".to_string()),
        Value::Unbound("GroupedAStarTarget".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarMin".to_string()));
    assert!(ok32, "grouped_min_semantic_dist_astar(s, Target, 5, Min) first grouped result should succeed");
    let mut grouped_astar_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_grouped_astar.bindings.get("GroupedAStarTarget").cloned(), vm_grouped_astar.bindings.get("GroupedAStarMin").cloned()) {
        grouped_astar_results.push((target, cost));
    } else {
        panic!("expected first grouped_min_semantic_dist_astar result");
    }
    while vm_grouped_astar.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_grouped_astar.bindings.get("GroupedAStarTarget").cloned(), vm_grouped_astar.bindings.get("GroupedAStarMin").cloned()) {
            grouped_astar_results.push((target, cost));
        } else {
            panic!("expected grouped_min_semantic_dist_astar backtrack result");
        }
    }
    grouped_astar_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(grouped_astar_results, vec![
        ("a".to_string(), 1.0),
        ("b".to_string(), 3.0),
        ("c".to_string(), 4.0),
        ("d".to_string(), 7.0),
    ]);

    let mut vm_grouped_astar_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok33 = grouped_min_semantic_dist_astar(&mut vm_grouped_astar_exact,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarD".to_string()));
    assert!(ok33, "grouped_min_semantic_dist_astar(s, d, 5, Min) should succeed");
    match vm_grouped_astar_exact.bindings.get("GroupedAStarD").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 7.0),
        other => panic!("expected grouped_min_semantic_dist_astar(s, d, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_grouped_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok34 = grouped_min_semantic_dist_astar(&mut vm_grouped_astar_fail,
        Value::Atom("d".to_string()),
        Value::Unbound("GroupedAStarNoTarget".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarNoMin".to_string()));
    assert!(!ok34, "grouped_min_semantic_dist_astar(d, Target, 5, Min) should fail");

    // Test filtered_adjusted_min_semantic_dist/3 mixed-goal wrapper
    let mut vm_filtered = WamState::new(vec![], std::collections::HashMap::new());
    let ok35 = filtered_adjusted_min_semantic_dist(&mut vm_filtered,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Unbound("FilteredD".to_string()));
    assert!(ok35, "filtered_adjusted_min_semantic_dist(s, d, Min) should succeed");
    match vm_filtered.bindings.get("FilteredD").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 8.0),
        other => panic!("expected filtered_adjusted_min_semantic_dist(s, d, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_filtered_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok36 = filtered_adjusted_min_semantic_dist(&mut vm_filtered_any,
        Value::Atom("s".to_string()),
        Value::Unbound("FilteredAnyTarget".to_string()),
        Value::Unbound("FilteredGlobalMin".to_string()));
    assert!(ok36, "filtered_adjusted_min_semantic_dist(s, Target, Min) should succeed");
    assert!(vm_filtered_any.bindings.get("FilteredAnyTarget").is_none(), "filtered aggregate wrapper should not bind existential Target");
    match vm_filtered_any.bindings.get("FilteredGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected filtered_adjusted_min_semantic_dist(s, Target, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_filtered_reverse_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok36b = filtered_adjusted_min_semantic_dist(&mut vm_filtered_reverse_any,
        Value::Unbound("FilteredAnyStart".to_string()),
        Value::Atom("d".to_string()),
        Value::Unbound("FilteredReverseGlobalMin".to_string()));
    assert!(ok36b, "filtered_adjusted_min_semantic_dist(Start, d, Min) should succeed");
    assert!(vm_filtered_reverse_any.bindings.get("FilteredAnyStart").is_none(), "filtered aggregate wrapper should not bind existential Start");
    match vm_filtered_reverse_any.bindings.get("FilteredReverseGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected filtered_adjusted_min_semantic_dist(Start, d, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_filtered_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok37 = filtered_adjusted_min_semantic_dist(&mut vm_filtered_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Float(5.0));
    assert!(ok37, "filtered_adjusted_min_semantic_dist(s, c, 5.0) should succeed");

    let mut vm_filtered_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok38 = filtered_adjusted_min_semantic_dist(&mut vm_filtered_fail,
        Value::Atom("s".to_string()),
        Value::Atom("a".to_string()),
        Value::Unbound("FilteredNoPath".to_string()));
    assert!(!ok38, "filtered_adjusted_min_semantic_dist(s, a, Min) should fail");

    // Test filtered_adjusted_min_semantic_dist_astar/4 mixed-goal wrapper
    let mut vm_filtered_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok39 = filtered_adjusted_min_semantic_dist_astar(&mut vm_filtered_astar,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Integer(5),
        Value::Unbound("FilteredAStarD".to_string()));
    assert!(ok39, "filtered_adjusted_min_semantic_dist_astar(s, d, 5, Min) should succeed");
    match vm_filtered_astar.bindings.get("FilteredAStarD").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 8.0),
        other => panic!("expected filtered_adjusted_min_semantic_dist_astar(s, d, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_filtered_astar_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok40 = filtered_adjusted_min_semantic_dist_astar(&mut vm_filtered_astar_any,
        Value::Atom("s".to_string()),
        Value::Unbound("FilteredAStarAnyTarget".to_string()),
        Value::Integer(5),
        Value::Unbound("FilteredAStarGlobalMin".to_string()));
    assert!(ok40, "filtered_adjusted_min_semantic_dist_astar(s, Target, 5, Min) should succeed");
    assert!(vm_filtered_astar_any.bindings.get("FilteredAStarAnyTarget").is_none(), "filtered A* aggregate wrapper should not bind existential Target");
    match vm_filtered_astar_any.bindings.get("FilteredAStarGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected filtered_adjusted_min_semantic_dist_astar(s, Target, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_filtered_astar_reverse_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok40b = filtered_adjusted_min_semantic_dist_astar(&mut vm_filtered_astar_reverse_any,
        Value::Unbound("FilteredAStarAnyStart".to_string()),
        Value::Atom("d".to_string()),
        Value::Integer(5),
        Value::Unbound("FilteredAStarReverseGlobalMin".to_string()));
    assert!(ok40b, "filtered_adjusted_min_semantic_dist_astar(Start, d, 5, Min) should succeed");
    assert!(vm_filtered_astar_reverse_any.bindings.get("FilteredAStarAnyStart").is_none(), "filtered A* aggregate wrapper should not bind existential Start");
    match vm_filtered_astar_reverse_any.bindings.get("FilteredAStarReverseGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected filtered_adjusted_min_semantic_dist_astar(Start, d, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_filtered_astar_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok41 = filtered_adjusted_min_semantic_dist_astar(&mut vm_filtered_astar_exact,
        Value::Atom("s".to_string()),
        Value::Atom("c".to_string()),
        Value::Integer(5),
        Value::Float(5.0));
    assert!(ok41, "filtered_adjusted_min_semantic_dist_astar(s, c, 5, 5.0) should succeed");

    let mut vm_filtered_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok42 = filtered_adjusted_min_semantic_dist_astar(&mut vm_filtered_astar_fail,
        Value::Atom("s".to_string()),
        Value::Atom("a".to_string()),
        Value::Integer(5),
        Value::Unbound("FilteredAStarNoPath".to_string()));
    assert!(!ok42, "filtered_adjusted_min_semantic_dist_astar(s, a, 5, Min) should fail");

    // Test filtered_adjusted_weighted_path/3 mixed-goal non-aggregate wrapper
    let mut vm_weighted_stream = WamState::new(vec![], std::collections::HashMap::new());
    let ok43 = filtered_adjusted_weighted_path(&mut vm_weighted_stream,
        Value::Atom("s".to_string()),
        Value::Unbound("WeightedStreamTarget".to_string()),
        Value::Unbound("WeightedStreamAdjusted".to_string()));
    assert!(ok43, "filtered_adjusted_weighted_path(s, Target, Adjusted) first solution should succeed");
    let mut weighted_stream_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_weighted_stream.bindings.get("WeightedStreamTarget").cloned(), vm_weighted_stream.bindings.get("WeightedStreamAdjusted").cloned()) {
        weighted_stream_results.push((target, cost));
    } else {
        panic!("expected first filtered_adjusted_weighted_path result");
    }
    while vm_weighted_stream.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_weighted_stream.bindings.get("WeightedStreamTarget").cloned(), vm_weighted_stream.bindings.get("WeightedStreamAdjusted").cloned()) {
            weighted_stream_results.push((target, cost));
        } else {
            panic!("expected filtered_adjusted_weighted_path backtrack result");
        }
    }
    weighted_stream_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(weighted_stream_results, vec![
        ("b".to_string(), 4.0),
        ("c".to_string(), 5.0),
        ("d".to_string(), 8.0),
    ]);

    let mut vm_weighted_stream_any_start = WamState::new(vec![], std::collections::HashMap::new());
    let ok43b = filtered_adjusted_weighted_path(&mut vm_weighted_stream_any_start,
        Value::Unbound("WeightedStreamStart".to_string()),
        Value::Unbound("WeightedStreamAnyTarget".to_string()),
        Value::Unbound("WeightedStreamAnyAdjusted".to_string()));
    assert!(ok43b, "filtered_adjusted_weighted_path(Start, Target, Adjusted) first solution should succeed");
    let mut weighted_stream_any_start_results: Vec<(String, String, f64)> = Vec::new();
    if let (Some(Value::Atom(start)), Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_weighted_stream_any_start.bindings.get("WeightedStreamStart").cloned(),
         vm_weighted_stream_any_start.bindings.get("WeightedStreamAnyTarget").cloned(),
         vm_weighted_stream_any_start.bindings.get("WeightedStreamAnyAdjusted").cloned()) {
        weighted_stream_any_start_results.push((start, target, cost));
    } else {
        panic!("expected first filtered_adjusted_weighted_path(Start, Target, Adjusted) result");
    }
    while vm_weighted_stream_any_start.backtrack() {
        if let (Some(Value::Atom(start)), Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_weighted_stream_any_start.bindings.get("WeightedStreamStart").cloned(),
             vm_weighted_stream_any_start.bindings.get("WeightedStreamAnyTarget").cloned(),
             vm_weighted_stream_any_start.bindings.get("WeightedStreamAnyAdjusted").cloned()) {
            weighted_stream_any_start_results.push((start, target, cost));
        } else {
            panic!("expected filtered_adjusted_weighted_path(Start, Target, Adjusted) backtrack result");
        }
    }
    weighted_stream_any_start_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.partial_cmp(&b.2).unwrap()));
    assert_eq!(weighted_stream_any_start_results, vec![
        ("a".to_string(), "c".to_string(), 4.0),
        ("a".to_string(), "d".to_string(), 7.0),
        ("b".to_string(), "d".to_string(), 5.0),
        ("c".to_string(), "d".to_string(), 4.0),
        ("s".to_string(), "b".to_string(), 4.0),
        ("s".to_string(), "c".to_string(), 5.0),
        ("s".to_string(), "d".to_string(), 8.0),
    ]);

    let mut vm_weighted_stream_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok44 = filtered_adjusted_weighted_path(&mut vm_weighted_stream_exact,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Float(8.0));
    assert!(ok44, "filtered_adjusted_weighted_path(s, d, 8.0) should succeed");

    let mut vm_weighted_stream_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok45 = filtered_adjusted_weighted_path(&mut vm_weighted_stream_fail,
        Value::Atom("s".to_string()),
        Value::Atom("a".to_string()),
        Value::Unbound("WeightedStreamNoPath".to_string()));
    assert!(!ok45, "filtered_adjusted_weighted_path(s, a, Adjusted) should fail");

    // Test filtered_adjusted_astar_weighted_path/4 mixed-goal non-aggregate wrapper
    let mut vm_astar_stream = WamState::new(vec![], std::collections::HashMap::new());
    let ok46 = filtered_adjusted_astar_weighted_path(&mut vm_astar_stream,
        Value::Atom("s".to_string()),
        Value::Unbound("AStarStreamTarget".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarStreamAdjusted".to_string()));
    assert!(ok46, "filtered_adjusted_astar_weighted_path(s, Target, 5, Adjusted) first solution should succeed");
    let mut astar_stream_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_astar_stream.bindings.get("AStarStreamTarget").cloned(), vm_astar_stream.bindings.get("AStarStreamAdjusted").cloned()) {
        astar_stream_results.push((target, cost));
    } else {
        panic!("expected first filtered_adjusted_astar_weighted_path result");
    }
    while vm_astar_stream.backtrack() {
        if let (Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_astar_stream.bindings.get("AStarStreamTarget").cloned(), vm_astar_stream.bindings.get("AStarStreamAdjusted").cloned()) {
            astar_stream_results.push((target, cost));
        } else {
            panic!("expected filtered_adjusted_astar_weighted_path backtrack result");
        }
    }
    astar_stream_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(astar_stream_results, vec![
        ("b".to_string(), 4.0),
        ("c".to_string(), 5.0),
        ("d".to_string(), 8.0),
    ]);

    let mut vm_astar_stream_any_start = WamState::new(vec![], std::collections::HashMap::new());
    let ok46b = filtered_adjusted_astar_weighted_path(&mut vm_astar_stream_any_start,
        Value::Unbound("AStarStreamStart".to_string()),
        Value::Unbound("AStarStreamAnyTarget".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarStreamAnyAdjusted".to_string()));
    assert!(ok46b, "filtered_adjusted_astar_weighted_path(Start, Target, 5, Adjusted) first solution should succeed");
    let mut astar_stream_any_start_results: Vec<(String, String, f64)> = Vec::new();
    if let (Some(Value::Atom(start)), Some(Value::Atom(target)), Some(Value::Float(cost))) =
        (vm_astar_stream_any_start.bindings.get("AStarStreamStart").cloned(),
         vm_astar_stream_any_start.bindings.get("AStarStreamAnyTarget").cloned(),
         vm_astar_stream_any_start.bindings.get("AStarStreamAnyAdjusted").cloned()) {
        astar_stream_any_start_results.push((start, target, cost));
    } else {
        panic!("expected first filtered_adjusted_astar_weighted_path(Start, Target, 5, Adjusted) result");
    }
    while vm_astar_stream_any_start.backtrack() {
        if let (Some(Value::Atom(start)), Some(Value::Atom(target)), Some(Value::Float(cost))) =
            (vm_astar_stream_any_start.bindings.get("AStarStreamStart").cloned(),
             vm_astar_stream_any_start.bindings.get("AStarStreamAnyTarget").cloned(),
             vm_astar_stream_any_start.bindings.get("AStarStreamAnyAdjusted").cloned()) {
            astar_stream_any_start_results.push((start, target, cost));
        } else {
            panic!("expected filtered_adjusted_astar_weighted_path(Start, Target, 5, Adjusted) backtrack result");
        }
    }
    astar_stream_any_start_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.partial_cmp(&b.2).unwrap()));
    assert_eq!(astar_stream_any_start_results, weighted_stream_any_start_results);

    let mut vm_astar_stream_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok47 = filtered_adjusted_astar_weighted_path(&mut vm_astar_stream_exact,
        Value::Atom("s".to_string()),
        Value::Atom("d".to_string()),
        Value::Integer(5),
        Value::Float(8.0));
    assert!(ok47, "filtered_adjusted_astar_weighted_path(s, d, 5, 8.0) should succeed");

    let mut vm_astar_stream_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok48 = filtered_adjusted_astar_weighted_path(&mut vm_astar_stream_fail,
        Value::Atom("s".to_string()),
        Value::Atom("a".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarStreamNoPath".to_string()));
    assert!(!ok48, "filtered_adjusted_astar_weighted_path(s, a, 5, Adjusted) should fail");

    // Test labeled_adjusted_weighted_path/3 relational join wrapper
    let mut vm_labeled_weighted = WamState::new(vec![], std::collections::HashMap::new());
    let ok49 = labeled_adjusted_weighted_path(&mut vm_labeled_weighted,
        Value::Atom("s".to_string()),
        Value::Unbound("WeightedLabel".to_string()),
        Value::Unbound("WeightedAdjusted".to_string()));
    assert!(ok49, "labeled_adjusted_weighted_path(s, Label, Adjusted) first solution should succeed");
    let mut labeled_weighted_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(label)), Some(Value::Float(cost))) =
        (vm_labeled_weighted.bindings.get("WeightedLabel").cloned(), vm_labeled_weighted.bindings.get("WeightedAdjusted").cloned()) {
        labeled_weighted_results.push((label, cost));
    } else {
        panic!("expected first labeled_adjusted_weighted_path result");
    }
    while vm_labeled_weighted.backtrack() {
        if let (Some(Value::Atom(label)), Some(Value::Float(cost))) =
            (vm_labeled_weighted.bindings.get("WeightedLabel").cloned(), vm_labeled_weighted.bindings.get("WeightedAdjusted").cloned()) {
            labeled_weighted_results.push((label, cost));
        } else {
            panic!("expected labeled_adjusted_weighted_path backtrack result");
        }
    }
    labeled_weighted_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(labeled_weighted_results, vec![
        ("branch_b".to_string(), 4.0),
        ("branch_c".to_string(), 5.0),
        ("goal_d1".to_string(), 8.0),
        ("goal_d2".to_string(), 8.0),
    ]);

    let mut vm_labeled_weighted_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok50 = labeled_adjusted_weighted_path(&mut vm_labeled_weighted_exact,
        Value::Atom("s".to_string()),
        Value::Atom("goal_d2".to_string()),
        Value::Float(8.0));
    assert!(ok50, "labeled_adjusted_weighted_path(s, goal_d2, 8.0) should succeed");

    let mut vm_labeled_weighted_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok51 = labeled_adjusted_weighted_path(&mut vm_labeled_weighted_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing".to_string()),
        Value::Unbound("WeightedMissing".to_string()));
    assert!(!ok51, "labeled_adjusted_weighted_path(s, missing, Adjusted) should fail");

    // Test labeled_adjusted_astar_weighted_path/4 relational join wrapper
    let mut vm_labeled_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok52 = labeled_adjusted_astar_weighted_path(&mut vm_labeled_astar,
        Value::Atom("s".to_string()),
        Value::Unbound("AStarLabel".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarAdjusted".to_string()));
    assert!(ok52, "labeled_adjusted_astar_weighted_path(s, Label, 5, Adjusted) first solution should succeed");
    let mut labeled_astar_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(label)), Some(Value::Float(cost))) =
        (vm_labeled_astar.bindings.get("AStarLabel").cloned(), vm_labeled_astar.bindings.get("AStarAdjusted").cloned()) {
        labeled_astar_results.push((label, cost));
    } else {
        panic!("expected first labeled_adjusted_astar_weighted_path result");
    }
    while vm_labeled_astar.backtrack() {
        if let (Some(Value::Atom(label)), Some(Value::Float(cost))) =
            (vm_labeled_astar.bindings.get("AStarLabel").cloned(), vm_labeled_astar.bindings.get("AStarAdjusted").cloned()) {
            labeled_astar_results.push((label, cost));
        } else {
            panic!("expected labeled_adjusted_astar_weighted_path backtrack result");
        }
    }
    labeled_astar_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(labeled_astar_results, vec![
        ("branch_b".to_string(), 4.0),
        ("branch_c".to_string(), 5.0),
        ("goal_d1".to_string(), 8.0),
        ("goal_d2".to_string(), 8.0),
    ]);

    let mut vm_labeled_astar_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok53 = labeled_adjusted_astar_weighted_path(&mut vm_labeled_astar_exact,
        Value::Atom("s".to_string()),
        Value::Atom("goal_d1".to_string()),
        Value::Integer(5),
        Value::Float(8.0));
    assert!(ok53, "labeled_adjusted_astar_weighted_path(s, goal_d1, 5, 8.0) should succeed");

    let mut vm_labeled_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok54 = labeled_adjusted_astar_weighted_path(&mut vm_labeled_astar_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarMissing".to_string()));
    assert!(!ok54, "labeled_adjusted_astar_weighted_path(s, missing, 5, Adjusted) should fail");

    // Test bucketed_adjusted_weighted_path/3 multi-stage relational wrapper
    let mut vm_bucketed_weighted = WamState::new(vec![], std::collections::HashMap::new());
    let ok55 = bucketed_adjusted_weighted_path(&mut vm_bucketed_weighted,
        Value::Atom("s".to_string()),
        Value::Unbound("WeightedBucket".to_string()),
        Value::Unbound("WeightedBucketAdjusted".to_string()));
    assert!(ok55, "bucketed_adjusted_weighted_path(s, Bucket, Adjusted) first solution should succeed");
    let mut bucketed_weighted_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_bucketed_weighted.bindings.get("WeightedBucket").cloned(), vm_bucketed_weighted.bindings.get("WeightedBucketAdjusted").cloned()) {
        bucketed_weighted_results.push((bucket, cost));
    } else {
        panic!("expected first bucketed_adjusted_weighted_path result");
    }
    while vm_bucketed_weighted.backtrack() {
        if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_bucketed_weighted.bindings.get("WeightedBucket").cloned(), vm_bucketed_weighted.bindings.get("WeightedBucketAdjusted").cloned()) {
            bucketed_weighted_results.push((bucket, cost));
        } else {
            panic!("expected bucketed_adjusted_weighted_path backtrack result");
        }
    }
    bucketed_weighted_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(bucketed_weighted_results, vec![
        ("branch_bucket".to_string(), 4.0),
        ("branch_bucket".to_string(), 5.0),
        ("goal_bucket".to_string(), 8.0),
        ("goal_bucket".to_string(), 8.0),
    ]);

    let mut vm_bucketed_weighted_any_start = WamState::new(vec![], std::collections::HashMap::new());
    let ok55b = bucketed_adjusted_weighted_path(&mut vm_bucketed_weighted_any_start,
        Value::Unbound("WeightedBucketStart".to_string()),
        Value::Unbound("WeightedAnyBucket".to_string()),
        Value::Unbound("WeightedAnyBucketAdjusted".to_string()));
    assert!(ok55b, "bucketed_adjusted_weighted_path(Start, Bucket, Adjusted) first solution should succeed");
    let mut bucketed_weighted_any_start_results: Vec<(String, String, f64)> = Vec::new();
    if let (Some(Value::Atom(start)), Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_bucketed_weighted_any_start.bindings.get("WeightedBucketStart").cloned(),
         vm_bucketed_weighted_any_start.bindings.get("WeightedAnyBucket").cloned(),
         vm_bucketed_weighted_any_start.bindings.get("WeightedAnyBucketAdjusted").cloned()) {
        bucketed_weighted_any_start_results.push((start, bucket, cost));
    } else {
        panic!("expected first bucketed_adjusted_weighted_path(Start, Bucket, Adjusted) result");
    }
    while vm_bucketed_weighted_any_start.backtrack() {
        if let (Some(Value::Atom(start)), Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_bucketed_weighted_any_start.bindings.get("WeightedBucketStart").cloned(),
             vm_bucketed_weighted_any_start.bindings.get("WeightedAnyBucket").cloned(),
             vm_bucketed_weighted_any_start.bindings.get("WeightedAnyBucketAdjusted").cloned()) {
            bucketed_weighted_any_start_results.push((start, bucket, cost));
        } else {
            panic!("expected bucketed_adjusted_weighted_path(Start, Bucket, Adjusted) backtrack result");
        }
    }
    bucketed_weighted_any_start_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.partial_cmp(&b.2).unwrap()));
    assert_eq!(bucketed_weighted_any_start_results, vec![
        ("a".to_string(), "branch_bucket".to_string(), 4.0),
        ("a".to_string(), "goal_bucket".to_string(), 7.0),
        ("a".to_string(), "goal_bucket".to_string(), 7.0),
        ("b".to_string(), "goal_bucket".to_string(), 5.0),
        ("b".to_string(), "goal_bucket".to_string(), 5.0),
        ("c".to_string(), "goal_bucket".to_string(), 4.0),
        ("c".to_string(), "goal_bucket".to_string(), 4.0),
        ("s".to_string(), "branch_bucket".to_string(), 4.0),
        ("s".to_string(), "branch_bucket".to_string(), 5.0),
        ("s".to_string(), "goal_bucket".to_string(), 8.0),
        ("s".to_string(), "goal_bucket".to_string(), 8.0),
    ]);

    let mut vm_bucketed_weighted_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok56 = bucketed_adjusted_weighted_path(&mut vm_bucketed_weighted_exact,
        Value::Atom("s".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Float(8.0));
    assert!(ok56, "bucketed_adjusted_weighted_path(s, goal_bucket, 8.0) should succeed");

    let mut vm_bucketed_weighted_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok57 = bucketed_adjusted_weighted_path(&mut vm_bucketed_weighted_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing_bucket".to_string()),
        Value::Unbound("WeightedBucketMissing".to_string()));
    assert!(!ok57, "bucketed_adjusted_weighted_path(s, missing_bucket, Adjusted) should fail");

    // Test bucketed_adjusted_astar_weighted_path/4 multi-stage relational wrapper
    let mut vm_bucketed_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok58 = bucketed_adjusted_astar_weighted_path(&mut vm_bucketed_astar,
        Value::Atom("s".to_string()),
        Value::Unbound("AStarBucket".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarBucketAdjusted".to_string()));
    assert!(ok58, "bucketed_adjusted_astar_weighted_path(s, Bucket, 5, Adjusted) first solution should succeed");
    let mut bucketed_astar_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_bucketed_astar.bindings.get("AStarBucket").cloned(), vm_bucketed_astar.bindings.get("AStarBucketAdjusted").cloned()) {
            bucketed_astar_results.push((bucket, cost));
    } else {
        panic!("expected first bucketed_adjusted_astar_weighted_path result");
    }
    while vm_bucketed_astar.backtrack() {
        if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_bucketed_astar.bindings.get("AStarBucket").cloned(), vm_bucketed_astar.bindings.get("AStarBucketAdjusted").cloned()) {
            bucketed_astar_results.push((bucket, cost));
        } else {
            panic!("expected bucketed_adjusted_astar_weighted_path backtrack result");
        }
    }
    bucketed_astar_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
    assert_eq!(bucketed_astar_results, vec![
        ("branch_bucket".to_string(), 4.0),
        ("branch_bucket".to_string(), 5.0),
        ("goal_bucket".to_string(), 8.0),
        ("goal_bucket".to_string(), 8.0),
    ]);

    let mut vm_bucketed_astar_any_start = WamState::new(vec![], std::collections::HashMap::new());
    let ok58b = bucketed_adjusted_astar_weighted_path(&mut vm_bucketed_astar_any_start,
        Value::Unbound("AStarBucketStart".to_string()),
        Value::Unbound("AStarAnyBucket".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarAnyBucketAdjusted".to_string()));
    assert!(ok58b, "bucketed_adjusted_astar_weighted_path(Start, Bucket, 5, Adjusted) first solution should succeed");
    let mut bucketed_astar_any_start_results: Vec<(String, String, f64)> = Vec::new();
    if let (Some(Value::Atom(start)), Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_bucketed_astar_any_start.bindings.get("AStarBucketStart").cloned(),
         vm_bucketed_astar_any_start.bindings.get("AStarAnyBucket").cloned(),
         vm_bucketed_astar_any_start.bindings.get("AStarAnyBucketAdjusted").cloned()) {
        bucketed_astar_any_start_results.push((start, bucket, cost));
    } else {
        panic!("expected first bucketed_adjusted_astar_weighted_path(Start, Bucket, 5, Adjusted) result");
    }
    while vm_bucketed_astar_any_start.backtrack() {
        if let (Some(Value::Atom(start)), Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_bucketed_astar_any_start.bindings.get("AStarBucketStart").cloned(),
             vm_bucketed_astar_any_start.bindings.get("AStarAnyBucket").cloned(),
             vm_bucketed_astar_any_start.bindings.get("AStarAnyBucketAdjusted").cloned()) {
            bucketed_astar_any_start_results.push((start, bucket, cost));
        } else {
            panic!("expected bucketed_adjusted_astar_weighted_path(Start, Bucket, 5, Adjusted) backtrack result");
        }
    }
    bucketed_astar_any_start_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.partial_cmp(&b.2).unwrap()));
    assert_eq!(bucketed_astar_any_start_results, bucketed_weighted_any_start_results);

    let mut vm_bucketed_astar_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok59 = bucketed_adjusted_astar_weighted_path(&mut vm_bucketed_astar_exact,
        Value::Atom("s".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Integer(5),
        Value::Float(8.0));
    assert!(ok59, "bucketed_adjusted_astar_weighted_path(s, goal_bucket, 5, 8.0) should succeed");

    let mut vm_bucketed_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok60 = bucketed_adjusted_astar_weighted_path(&mut vm_bucketed_astar_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing_bucket".to_string()),
        Value::Integer(5),
        Value::Unbound("AStarBucketMissing".to_string()));
    assert!(!ok60, "bucketed_adjusted_astar_weighted_path(s, missing_bucket, 5, Adjusted) should fail");

    // Test bucketed_min_semantic_dist/3 scalar aggregate over multi-stage weighted wrapper shape
    let mut vm_bucketed_min_weighted = WamState::new(vec![], std::collections::HashMap::new());
    let ok61 = bucketed_min_semantic_dist(&mut vm_bucketed_min_weighted,
        Value::Atom("s".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Unbound("BucketedWeightedMin".to_string()));
    assert!(ok61, "bucketed_min_semantic_dist(s, goal_bucket, Min) should succeed");
    match vm_bucketed_min_weighted.bindings.get("BucketedWeightedMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 8.0),
        other => panic!("expected bucketed_min_semantic_dist(s, goal_bucket, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_bucketed_min_weighted_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok62 = bucketed_min_semantic_dist(&mut vm_bucketed_min_weighted_any,
        Value::Atom("s".to_string()),
        Value::Unbound("AnyBucket".to_string()),
        Value::Unbound("BucketedGlobalMin".to_string()));
    assert!(ok62, "bucketed_min_semantic_dist(s, Bucket, Min) should succeed");
    assert!(vm_bucketed_min_weighted_any.bindings.get("AnyBucket").is_none(), "scalar aggregate wrapper should not bind existential Bucket");
    match vm_bucketed_min_weighted_any.bindings.get("BucketedGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected bucketed_min_semantic_dist(s, Bucket, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_bucketed_min_weighted_reverse_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok62b = bucketed_min_semantic_dist(&mut vm_bucketed_min_weighted_reverse_any,
        Value::Unbound("AnyBucketedStart".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Unbound("BucketedReverseGlobalMin".to_string()));
    assert!(ok62b, "bucketed_min_semantic_dist(Start, goal_bucket, Min) should succeed");
    assert!(vm_bucketed_min_weighted_reverse_any.bindings.get("AnyBucketedStart").is_none(), "scalar aggregate wrapper should not bind existential Start");
    match vm_bucketed_min_weighted_reverse_any.bindings.get("BucketedReverseGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected bucketed_min_semantic_dist(Start, goal_bucket, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_bucketed_min_weighted_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok63 = bucketed_min_semantic_dist(&mut vm_bucketed_min_weighted_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing_bucket".to_string()),
        Value::Unbound("MissingBucketMin".to_string()));
    assert!(!ok63, "bucketed_min_semantic_dist(s, missing_bucket, Min) should fail");

    // Test bucketed_min_semantic_dist_astar/4 scalar aggregate over multi-stage A* wrapper shape
    let mut vm_bucketed_min_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok64 = bucketed_min_semantic_dist_astar(&mut vm_bucketed_min_astar,
        Value::Atom("s".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Integer(5),
        Value::Unbound("BucketedAStarMin".to_string()));
    assert!(ok64, "bucketed_min_semantic_dist_astar(s, goal_bucket, 5, Min) should succeed");
    match vm_bucketed_min_astar.bindings.get("BucketedAStarMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 8.0),
        other => panic!("expected bucketed_min_semantic_dist_astar(s, goal_bucket, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_bucketed_min_astar_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok65 = bucketed_min_semantic_dist_astar(&mut vm_bucketed_min_astar_any,
        Value::Atom("s".to_string()),
        Value::Unbound("AnyAStarBucket".to_string()),
        Value::Integer(5),
        Value::Unbound("BucketedAStarGlobalMin".to_string()));
    assert!(ok65, "bucketed_min_semantic_dist_astar(s, Bucket, 5, Min) should succeed");
    assert!(vm_bucketed_min_astar_any.bindings.get("AnyAStarBucket").is_none(), "scalar A* aggregate wrapper should not bind existential Bucket");
    match vm_bucketed_min_astar_any.bindings.get("BucketedAStarGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected bucketed_min_semantic_dist_astar(s, Bucket, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_bucketed_min_astar_reverse_any = WamState::new(vec![], std::collections::HashMap::new());
    let ok65b = bucketed_min_semantic_dist_astar(&mut vm_bucketed_min_astar_reverse_any,
        Value::Unbound("AnyAStarBucketedStart".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Integer(5),
        Value::Unbound("BucketedAStarReverseGlobalMin".to_string()));
    assert!(ok65b, "bucketed_min_semantic_dist_astar(Start, goal_bucket, 5, Min) should succeed");
    assert!(vm_bucketed_min_astar_reverse_any.bindings.get("AnyAStarBucketedStart").is_none(), "scalar A* aggregate wrapper should not bind existential Start");
    match vm_bucketed_min_astar_reverse_any.bindings.get("BucketedAStarReverseGlobalMin").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 4.0),
        other => panic!("expected bucketed_min_semantic_dist_astar(Start, goal_bucket, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_bucketed_min_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok66 = bucketed_min_semantic_dist_astar(&mut vm_bucketed_min_astar_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing_bucket".to_string()),
        Value::Integer(5),
        Value::Unbound("MissingAStarBucketMin".to_string()));
    assert!(!ok66, "bucketed_min_semantic_dist_astar(s, missing_bucket, 5, Min) should fail");

    // Test grouped_bucketed_min_semantic_dist/3 grouped aggregate over multi-stage weighted wrapper shape
    let mut vm_grouped_bucketed_weighted = WamState::new(vec![], std::collections::HashMap::new());
    let ok67 = grouped_bucketed_min_semantic_dist(&mut vm_grouped_bucketed_weighted,
        Value::Atom("s".to_string()),
        Value::Unbound("GroupedBucket".to_string()),
        Value::Unbound("GroupedBucketMin".to_string()));
    assert!(ok67, "grouped_bucketed_min_semantic_dist(s, Bucket, Min) first grouped result should succeed");
    let mut grouped_bucketed_weighted_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_grouped_bucketed_weighted.bindings.get("GroupedBucket").cloned(), vm_grouped_bucketed_weighted.bindings.get("GroupedBucketMin").cloned()) {
        grouped_bucketed_weighted_results.push((bucket, cost));
    } else {
        panic!("expected first grouped_bucketed_min_semantic_dist result");
    }
    while vm_grouped_bucketed_weighted.backtrack() {
        if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_grouped_bucketed_weighted.bindings.get("GroupedBucket").cloned(), vm_grouped_bucketed_weighted.bindings.get("GroupedBucketMin").cloned()) {
            grouped_bucketed_weighted_results.push((bucket, cost));
        } else {
            panic!("expected grouped_bucketed_min_semantic_dist backtrack result");
        }
    }
    grouped_bucketed_weighted_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(grouped_bucketed_weighted_results, vec![
        ("branch_bucket".to_string(), 4.0),
        ("goal_bucket".to_string(), 8.0),
    ]);

    let mut vm_grouped_bucketed_weighted_reverse = WamState::new(vec![], std::collections::HashMap::new());
    let ok67b = grouped_bucketed_min_semantic_dist(&mut vm_grouped_bucketed_weighted_reverse,
        Value::Unbound("GroupedAnyStart".to_string()),
        Value::Unbound("GroupedAnyStartBucket".to_string()),
        Value::Unbound("GroupedAnyStartMin".to_string()));
    assert!(ok67b, "grouped_bucketed_min_semantic_dist(Start, Bucket, Min) first grouped result should succeed");
    assert!(vm_grouped_bucketed_weighted_reverse.bindings.get("GroupedAnyStart").is_none(), "grouped aggregate wrapper should not bind existential Start");
    let mut grouped_bucketed_weighted_reverse_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_grouped_bucketed_weighted_reverse.bindings.get("GroupedAnyStartBucket").cloned(), vm_grouped_bucketed_weighted_reverse.bindings.get("GroupedAnyStartMin").cloned()) {
        grouped_bucketed_weighted_reverse_results.push((bucket, cost));
    } else {
        panic!("expected first grouped_bucketed_min_semantic_dist(Start, Bucket, Min) result");
    }
    while vm_grouped_bucketed_weighted_reverse.backtrack() {
        assert!(vm_grouped_bucketed_weighted_reverse.bindings.get("GroupedAnyStart").is_none(), "grouped aggregate wrapper should not bind existential Start on backtracking");
        if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_grouped_bucketed_weighted_reverse.bindings.get("GroupedAnyStartBucket").cloned(), vm_grouped_bucketed_weighted_reverse.bindings.get("GroupedAnyStartMin").cloned()) {
            grouped_bucketed_weighted_reverse_results.push((bucket, cost));
        } else {
            panic!("expected grouped_bucketed_min_semantic_dist(Start, Bucket, Min) backtrack result");
        }
    }
    grouped_bucketed_weighted_reverse_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(grouped_bucketed_weighted_reverse_results, vec![
        ("branch_bucket".to_string(), 4.0),
        ("goal_bucket".to_string(), 4.0),
    ]);

    let mut vm_grouped_bucketed_weighted_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok68 = grouped_bucketed_min_semantic_dist(&mut vm_grouped_bucketed_weighted_exact,
        Value::Atom("s".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Unbound("GroupedGoalBucket".to_string()));
    assert!(ok68, "grouped_bucketed_min_semantic_dist(s, goal_bucket, Min) should succeed");
    match vm_grouped_bucketed_weighted_exact.bindings.get("GroupedGoalBucket").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 8.0),
        other => panic!("expected grouped_bucketed_min_semantic_dist(s, goal_bucket, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_grouped_bucketed_weighted_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok69 = grouped_bucketed_min_semantic_dist(&mut vm_grouped_bucketed_weighted_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing_bucket".to_string()),
        Value::Unbound("GroupedMissingBucket".to_string()));
    assert!(!ok69, "grouped_bucketed_min_semantic_dist(s, missing_bucket, Min) should fail");

    // Test grouped_bucketed_min_semantic_dist_astar/4 grouped aggregate over multi-stage A* wrapper shape
    let mut vm_grouped_bucketed_astar = WamState::new(vec![], std::collections::HashMap::new());
    let ok70 = grouped_bucketed_min_semantic_dist_astar(&mut vm_grouped_bucketed_astar,
        Value::Atom("s".to_string()),
        Value::Unbound("GroupedAStarBucket".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarBucketMin".to_string()));
    assert!(ok70, "grouped_bucketed_min_semantic_dist_astar(s, Bucket, 5, Min) first grouped result should succeed");
    let mut grouped_bucketed_astar_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_grouped_bucketed_astar.bindings.get("GroupedAStarBucket").cloned(), vm_grouped_bucketed_astar.bindings.get("GroupedAStarBucketMin").cloned()) {
        grouped_bucketed_astar_results.push((bucket, cost));
    } else {
        panic!("expected first grouped_bucketed_min_semantic_dist_astar result");
    }
    while vm_grouped_bucketed_astar.backtrack() {
        if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_grouped_bucketed_astar.bindings.get("GroupedAStarBucket").cloned(), vm_grouped_bucketed_astar.bindings.get("GroupedAStarBucketMin").cloned()) {
            grouped_bucketed_astar_results.push((bucket, cost));
        } else {
            panic!("expected grouped_bucketed_min_semantic_dist_astar backtrack result");
        }
    }
    grouped_bucketed_astar_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(grouped_bucketed_astar_results, vec![
        ("branch_bucket".to_string(), 4.0),
        ("goal_bucket".to_string(), 8.0),
    ]);

    let mut vm_grouped_bucketed_astar_reverse = WamState::new(vec![], std::collections::HashMap::new());
    let ok70b = grouped_bucketed_min_semantic_dist_astar(&mut vm_grouped_bucketed_astar_reverse,
        Value::Unbound("GroupedAStarAnyStart".to_string()),
        Value::Unbound("GroupedAStarAnyStartBucket".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarAnyStartMin".to_string()));
    assert!(ok70b, "grouped_bucketed_min_semantic_dist_astar(Start, Bucket, 5, Min) first grouped result should succeed");
    assert!(vm_grouped_bucketed_astar_reverse.bindings.get("GroupedAStarAnyStart").is_none(), "grouped A* aggregate wrapper should not bind existential Start");
    let mut grouped_bucketed_astar_reverse_results: Vec<(String, f64)> = Vec::new();
    if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
        (vm_grouped_bucketed_astar_reverse.bindings.get("GroupedAStarAnyStartBucket").cloned(), vm_grouped_bucketed_astar_reverse.bindings.get("GroupedAStarAnyStartMin").cloned()) {
        grouped_bucketed_astar_reverse_results.push((bucket, cost));
    } else {
        panic!("expected first grouped_bucketed_min_semantic_dist_astar(Start, Bucket, 5, Min) result");
    }
    while vm_grouped_bucketed_astar_reverse.backtrack() {
        assert!(vm_grouped_bucketed_astar_reverse.bindings.get("GroupedAStarAnyStart").is_none(), "grouped A* aggregate wrapper should not bind existential Start on backtracking");
        if let (Some(Value::Atom(bucket)), Some(Value::Float(cost))) =
            (vm_grouped_bucketed_astar_reverse.bindings.get("GroupedAStarAnyStartBucket").cloned(), vm_grouped_bucketed_astar_reverse.bindings.get("GroupedAStarAnyStartMin").cloned()) {
            grouped_bucketed_astar_reverse_results.push((bucket, cost));
        } else {
            panic!("expected grouped_bucketed_min_semantic_dist_astar(Start, Bucket, 5, Min) backtrack result");
        }
    }
    grouped_bucketed_astar_reverse_results.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(grouped_bucketed_astar_reverse_results, vec![
        ("branch_bucket".to_string(), 4.0),
        ("goal_bucket".to_string(), 4.0),
    ]);

    let mut vm_grouped_bucketed_astar_exact = WamState::new(vec![], std::collections::HashMap::new());
    let ok71 = grouped_bucketed_min_semantic_dist_astar(&mut vm_grouped_bucketed_astar_exact,
        Value::Atom("s".to_string()),
        Value::Atom("goal_bucket".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarGoalBucket".to_string()));
    assert!(ok71, "grouped_bucketed_min_semantic_dist_astar(s, goal_bucket, 5, Min) should succeed");
    match vm_grouped_bucketed_astar_exact.bindings.get("GroupedAStarGoalBucket").cloned() {
        Some(Value::Float(cost)) => assert_close(cost, 8.0),
        other => panic!("expected grouped_bucketed_min_semantic_dist_astar(s, goal_bucket, 5, Min) to bind float cost, got {:?}", other),
    }

    let mut vm_grouped_bucketed_astar_fail = WamState::new(vec![], std::collections::HashMap::new());
    let ok72 = grouped_bucketed_min_semantic_dist_astar(&mut vm_grouped_bucketed_astar_fail,
        Value::Atom("s".to_string()),
        Value::Atom("missing_bucket".to_string()),
        Value::Integer(5),
        Value::Unbound("GroupedAStarMissingBucket".to_string()));
    assert!(!ok72, "grouped_bucketed_min_semantic_dist_astar(s, missing_bucket, 5, Min) should fail");
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
