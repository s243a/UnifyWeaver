% test_wam_rust_parallel_reduce_exec.pl
%
% T7: end-to-end exec coverage of the non-collect reduce types (max/min/set/sum)
% in the parallel-aggregate injection. Generates a project with four
% parallel-eligible aggregates over a recursive numeric body, builds, runs, and
% asserts each reduces correctly through the native par_collect wrappers.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic red_fact/1.
red_fact(1). red_fact(2). red_fact(3).
% recursive numeric body: red_rsum(X,S) = X*(X+1)/2  -> 1,3,6
red_rsum(0, 0).
red_rsum(N, S) :- N > 0, M is N - 1, red_rsum(M, S0), S is S0 + N.

red_max(M) :- aggregate_all(max(S), (red_fact(X), red_rsum(X, S)), M).   % 6
red_min(M) :- aggregate_all(min(S), (red_fact(X), red_rsum(X, S)), M).   % 1
red_set(L) :- aggregate_all(set(S), (red_fact(X), red_rsum(X, S)), L).   % [1,3,6]
red_sum(S) :- aggregate_all(sum(S0), (red_fact(X), red_rsum(X, S0)), S). % 10

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_parallel_reduce_exec).

test(reduce_types_run_correctly,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_reduce_exec'))]) :-
    Dir = 'output/test_reduce_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:red_max/1, user:red_min/1, user:red_set/1, user:red_sum/1,
         user:red_fact/1, user:red_rsum/2],
        [module_name(red), parallel_aggregates(true)], Dir)),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/reduce_test.rs', TestPath),
    TestSrc = '
use red::value::Value;
use red::state::WamState;
use red::{red_max_1, red_min_1, red_set_1, red_sum_1};

fn fresh() -> WamState { WamState::new(vec![], std::collections::HashMap::new()) }
fn ints(v: &Value) -> Vec<i64> {
    match v { Value::List(xs) => xs.iter().filter_map(|e| match e {
        Value::Integer(n) => Some(*n), _ => None }).collect(), _ => vec![] }
}

#[test]
fn reduce_types() {
    let mut vm = fresh();
    assert!(red_max_1(&mut vm, Value::Unbound("M".to_string())));
    assert_eq!(vm.bindings.get("M").cloned(), Some(Value::Integer(6)), "max");

    let mut vm = fresh();
    assert!(red_min_1(&mut vm, Value::Unbound("M".to_string())));
    assert_eq!(vm.bindings.get("M").cloned(), Some(Value::Integer(1)), "min");

    let mut vm = fresh();
    assert!(red_sum_1(&mut vm, Value::Unbound("S".to_string())));
    assert_eq!(vm.bindings.get("S").cloned(), Some(Value::Integer(10)), "sum");

    let mut vm = fresh();
    assert!(red_set_1(&mut vm, Value::Unbound("L".to_string())));
    match vm.bindings.get("L").cloned() {
        Some(ref l @ Value::List(_)) => assert_eq!(ints(l), vec![1,3,6], "set sorted+deduped"),
        other => panic!("expected set list, got {:?}", other),
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test reduce_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[reduce exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_parallel_reduce_exec).
