% test_wam_rust_parallel_injection_exec.pl
%
% T7 R1.2c end-to-end: a USER predicate whose body is a parallel-eligible
% aggregate (findall over a recursive body) is compiled with
% parallel_aggregates(true), which injects enum/body helpers + a native
% par_collect wrapper. Builds and RUNS the project, asserting the user predicate
% returns the correct collected result (== the known sequential answer).
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic uinj_fact/1.
uinj_fact(1). uinj_fact(2). uinj_fact(3). uinj_fact(4). uinj_fact(5). uinj_fact(6).
uinj_down(0, []).
uinj_down(N, [N|T]) :- N > 0, M is N - 1, uinj_down(M, T).

% the user-facing predicate: a single forkable, recursive-body aggregate
uinj_collect(L) :- findall(D, (uinj_fact(X), uinj_down(X, D)), L).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_parallel_injection_exec).

test(user_findall_compiles_to_parallel_and_runs,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_inj_exec'))]) :-
    Dir = 'output/test_inj_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:uinj_collect/1, user:uinj_fact/1, user:uinj_down/2],
        [module_name(uinj), parallel_aggregates(true)], Dir)),
    % the injection must have emitted a native par_collect wrapper for uinj_collect
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "par_collect")),
    assertion(sub_string(Src, _, _, _, "parallel-aggregate wrappers")),
    % embed an integration test that runs the user predicate
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/inj_test.rs', TestPath),
    TestSrc = '
use uinj::value::Value;
use uinj::state::WamState;
use uinj::uinj_collect_1;

fn ints(v: &Value) -> Vec<i64> {
    match v { Value::List(xs) => xs.iter().filter_map(|e| match e {
        Value::Integer(n) => Some(*n), _ => None }).collect(), _ => vec![] }
}

#[test]
fn user_findall_parallel() {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    assert!(uinj_collect_1(&mut vm, Value::Unbound("L".to_string())), "uinj_collect should succeed");
    match vm.bindings.get("L").cloned() {
        Some(Value::List(items)) => {
            assert_eq!(items.len(), 6, "one list per fact");
            assert_eq!(ints(&items[0]), vec![1], "first = [1]");
            assert_eq!(ints(items.last().unwrap()), vec![6,5,4,3,2,1], "last = [6..1]");
        }
        other => panic!("expected list in L, got {:?}", other),
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test inj_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[injection exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_parallel_injection_exec).
