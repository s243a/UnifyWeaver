:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_rust_bidirectional_e2e.pl - End-to-end test for the Rust
% bidirectional ancestor kernel (F# parity port).
%
% Verifies the full pipeline: category_ancestor detection ->
% kernel_mode(bidirectional) upgrade -> 5-ary wrapper + dispatch arm +
% calibration/A*-pruned explore codegen -> cargo build -> execution
% with hand-checked path enumerations.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_bidirectional_e2e.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module(library(process)).
:- use_module(library(filesex)).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Kernel-shaped predicate: same clause shape the shared detector keys on.
:- dynamic category_ancestor/4.
:- dynamic category_parent/2.
:- dynamic max_depth/1.

category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).
category_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD), length(Visited, D), D < MaxD, !,
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

max_depth(10).

cargo_available :-
    catch(
        (process_create(path(cargo), ['--version'],
                        [stdout(null), stderr(null), process(Pid)]),
         process_wait(Pid, exit(0))),
        _, fail).

%% Codegen-level test: no cargo needed.
test_bidirectional_codegen :-
    Test = 'WAM-Rust bidirectional: codegen emits upgraded kernel + 5-ary wrapper',
    TmpDir = 'output/test_wam_rust_bidir_codegen',
    (   (exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true),
        write_wam_rust_project([user:category_ancestor/4],
            [module_name('bidir_codegen_test'), kernel_mode(bidirectional)],
            TmpDir),
        directory_file_path(TmpDir, 'src/lib.rs', LibPath),
        read_file_to_string(LibPath, LibCode, []),
        % Registration upgraded to the 5-ary key and bidirectional kind
        sub_string(LibCode, _, _, _, 'vm.register_foreign_native_kind("category_ancestor/5", "bidirectional_ancestor")'),
        sub_string(LibCode, _, _, _, 'vm.register_foreign_result_layout("category_ancestor/5", "tuple(3)")'),
        sub_string(LibCode, _, _, _, 'vm.register_foreign_string_config("category_ancestor/5", "edge_pred", "category_parent")'),
        % Public wrapper is 5-ary and dispatches to the kernel
        sub_string(LibCode, _, _, _, 'pub fn category_ancestor(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value, a5: Value) -> bool'),
        sub_string(LibCode, _, _, _, 'vm.execute_foreign_predicate("category_ancestor", 5)'),
        directory_file_path(TmpDir, 'src/state.rs', StatePath),
        read_file_to_string(StatePath, StateCode, []),
        % Kernel machinery present in the runtime
        sub_string(StateCode, _, _, _, 'pub fn calibrate_bidirectional_graph'),
        sub_string(StateCode, _, _, _, 'pub fn collect_native_bidirectional_ancestor_hops'),
        sub_string(StateCode, _, _, _, '"bidirectional_ancestor" =>'),
        sub_string(StateCode, _, _, _, 'pub fn ensure_reverse_edge_index'),
        sub_string(StateCode, _, _, _, 'pub fn register_foreign_f64_config')
    ->  pass(Test)
    ;   fail_test(Test, 'expected bidirectional registration/wrapper/kernel code missing')
    ).

%% Default-mode regression guard: without the option, nothing changes.
test_no_upgrade_without_option :-
    Test = 'WAM-Rust bidirectional: no upgrade without kernel_mode(bidirectional)',
    TmpDir = 'output/test_wam_rust_bidir_default',
    (   (exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true),
        write_wam_rust_project([user:category_ancestor/4],
            [module_name('bidir_default_test')],
            TmpDir),
        directory_file_path(TmpDir, 'src/lib.rs', LibPath),
        read_file_to_string(LibPath, LibCode, []),
        sub_string(LibCode, _, _, _, 'vm.register_foreign_native_kind("category_ancestor/4", "category_ancestor")'),
        \+ sub_string(LibCode, _, _, _, 'bidirectional_ancestor')
    ->  pass(Test)
    ;   fail_test(Test, 'default mode no longer emits the plain category_ancestor kernel')
    ).

%% Execution test (cargo-gated): hand-checked path enumeration.
%%
%% Graph (category_parent):
%%   c1 -> p1, c1 -> p2, c2 -> p1, c3 -> p2, p1 -> root, p2 -> root
%% Derived child edges: root -> {p1,p2}, p1 -> {c1,c2}, p2 -> {c1,c3}
%% min_dist from root (BFS via children): root 0, p1/p2 1, c1/c2/c3 2.
%%
%% Query (c1, root), costs parent=1 child=3 budget=10:
%%   c1->p1->root and c1->p2->root are the only arrivals: two solutions,
%%   both (total=2, parent=2, child=0). Continuations from root revisit
%%   only visited nodes or dead-end (c2/c3 have a single, visited parent).
%%
%% Query (p1, p2): the upward route via root is pruned by the A* bound
%%   (root is unreachable from p2 via child edges, so min_dist has no
%%   entry); the only arrival is p1 ->child c1 ->parent p2:
%%   one solution (total=2, parent=1, child=1) — exercises the mixed
%%   direction and the lower-bound elimination.
test_bidirectional_execution :-
    Test = 'WAM-Rust bidirectional: end-to-end execution (cargo test)',
    TmpDir = 'output/test_wam_rust_bidir_e2e',
    (   \+ cargo_available
    ->  format('[SKIP] ~w (cargo not found)~n', [Test]),
        pass(Test)
    ;   (exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true),
        write_wam_rust_project([user:category_ancestor/4],
            [module_name('bidir_e2e_test'), kernel_mode(bidirectional)],
            TmpDir),
        directory_file_path(TmpDir, 'tests', TestsDir),
        make_directory_path(TestsDir),
        directory_file_path(TestsDir, 'integration_test.rs', TestPath),
        TestContent = '
use bidir_e2e_test::state::WamState;
use bidir_e2e_test::value::Value;
use bidir_e2e_test::category_ancestor;

fn solutions(vm: &mut WamState, cat: &str, root: &str) -> Vec<(i64, i64, i64)> {
    let mut out = Vec::new();
    let ok = category_ancestor(vm,
        Value::Atom(cat.to_string()),
        Value::Atom(root.to_string()),
        Value::Unbound("T".to_string()),
        Value::Unbound("P".to_string()),
        Value::Unbound("C".to_string()));
    if !ok {
        return out;
    }
    loop {
        let t = match vm.bindings.get("T").map(|v| vm.deref_var(v)) {
            Some(Value::Integer(t)) => t,
            other => panic!("expected integer T, got {:?}", other),
        };
        let p = match vm.bindings.get("P").map(|v| vm.deref_var(v)) {
            Some(Value::Integer(p)) => p,
            other => panic!("expected integer P, got {:?}", other),
        };
        let c = match vm.bindings.get("C").map(|v| vm.deref_var(v)) {
            Some(Value::Integer(c)) => c,
            other => panic!("expected integer C, got {:?}", other),
        };
        out.push((t, p, c));
        if !vm.backtrack() {
            break;
        }
    }
    out.sort();
    out
}

fn graph_vm() -> WamState {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    vm.register_ffi_fact_pairs("category_parent", &[
        ("c1", "p1"),
        ("c1", "p2"),
        ("c2", "p1"),
        ("c3", "p2"),
        ("p1", "root"),
        ("p2", "root"),
    ]);
    vm
}

#[test]
fn test_two_upward_paths() {
    let mut vm = graph_vm();
    let sols = solutions(&mut vm, "c1", "root");
    assert_eq!(sols, vec![(2, 2, 0), (2, 2, 0)],
        "c1->root must yield exactly the two pure-parent paths");
}

#[test]
fn test_mixed_direction_with_astar_prune() {
    let mut vm = graph_vm();
    let sols = solutions(&mut vm, "p1", "p2");
    assert_eq!(sols, vec![(2, 1, 1)],
        "p1->p2 must be reached only via the child+parent route");
}

#[test]
fn test_unreachable_fails() {
    let mut vm = graph_vm();
    let ok = category_ancestor(&mut vm,
        Value::Atom("root".to_string()),
        Value::Atom("c9".to_string()),
        Value::Unbound("T".to_string()),
        Value::Unbound("P".to_string()),
        Value::Unbound("C".to_string()));
    assert!(!ok, "query to an unknown root must fail");
}
',
        setup_call_cleanup(
            open(TestPath, write, Out, [encoding(utf8)]),
            write(Out, TestContent),
            close(Out)),
        format(atom(Cmd), 'cd "~w" && cargo test -- --nocapture 2>&1', [TmpDir]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Stream)), process(Pid)]),
        read_string(Stream, _, Output),
        close(Stream),
        process_wait(Pid, exit(ExitCode)),
        (   ExitCode == 0,
            sub_string(Output, _, _, _, "test result: ok")
        ->  pass(Test)
        ;   format('--- cargo test output ---~n~w~n--- end ---~n', [Output]),
            fail_test(Test, 'cargo test failed')
        )
    ).

run_tests :-
    format('=== WAM-Rust bidirectional kernel tests ===~n'),
    test_bidirectional_codegen,
    test_no_upgrade_without_option,
    test_bidirectional_execution,
    (   test_failed
    ->  format('~n=== SOME TESTS FAILED ===~n'), halt(1)
    ;   format('~n=== ALL TESTS PASSED ===~n')
    ).
