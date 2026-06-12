% test_wam_rust_aggregate_exec.pl
%
% T7 slice 2b.2 — acceptance harness (test-first). Builds and RUNS a generated
% Rust project containing aggregate predicates (findall/aggregate_all) and
% asserts the collected results are correct end-to-end. This is the first exec
% coverage of the WAM-Rust aggregate path, and the baseline any parallel fork
% must preserve: parallel result-set == this sequential result-set.
%
% cargo-gated (skips cleanly if cargo is absent).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic agf/1.
agf(1). agf(2). agf(3).

ag_count(N)   :- aggregate_all(count, agf(_), N).      % -> 3
ag_sum(S)     :- aggregate_all(sum(X), agf(X), S).     % -> 6
ag_collect(L) :- findall(X, agf(X), L).                % -> [1,2,3]

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_aggregate_exec).

test(aggregate_path_executes_correctly,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_agg_exec'))]) :-
    Dir = 'output/test_agg_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:ag_count/1, user:ag_sum/1, user:ag_collect/1, user:agf/1],
        [module_name(agg_exec), parallel_aggregates(true)], Dir)),
    % embed an integration test that runs each aggregate and checks the result
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/agg_test.rs', TestPath),
    TestSrc = '
use agg_exec::value::Value;
use agg_exec::state::WamState;
use agg_exec::{ag_count_1, ag_sum_1, ag_collect_1};

fn fresh() -> WamState { WamState::new(vec![], std::collections::HashMap::new()) }

#[test]
fn aggregates_execute() {
    // count -> 3
    let mut vm = fresh();
    assert!(ag_count_1(&mut vm, Value::Unbound("N".to_string())), "ag_count should succeed");
    assert_eq!(vm.bindings.get("N").cloned(), Some(Value::Integer(3)), "count");

    // sum -> 6
    let mut vm = fresh();
    assert!(ag_sum_1(&mut vm, Value::Unbound("S".to_string())), "ag_sum should succeed");
    assert_eq!(vm.bindings.get("S").cloned(), Some(Value::Integer(6)), "sum");

    // collect -> [1,2,3]
    let mut vm = fresh();
    assert!(ag_collect_1(&mut vm, Value::Unbound("L".to_string())), "ag_collect should succeed");
    match vm.bindings.get("L").cloned() {
        Some(Value::List(xs)) => {
            let got: Vec<i64> = xs.iter().filter_map(|v| match v {
                Value::Integer(n) => Some(*n), _ => None }).collect();
            assert_eq!(got, vec![1, 2, 3], "collect order/contents");
        }
        other => panic!("expected list in L, got {:?}", other),
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    % run it
    format(atom(Cmd), 'cd "~w" && cargo test --test agg_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[aggregate exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_aggregate_exec).
