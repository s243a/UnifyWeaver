% test_wam_rust_par_aggregate_exec.pl
%
% T7 R1.2b — exec test of the parallel-aggregate runtime. Generates a project
% with the two helper predicates a transform produces (an enumerator and a
% per-branch body), and runs the native orchestrator `par_collect` (which clones
% the machine per input and runs the body across worker threads) against the
% sequential reference `seq_collect`. Proves parallel result-set == sequential
% end-to-end on real generated predicate functions -- the acceptance criterion
% for the fork.
%
% cargo-gated (skips cleanly if cargo is absent).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic pae_fact/1.
pae_fact(1). pae_fact(2). pae_fact(3). pae_fact(4). pae_fact(5). pae_fact(6).

% recursive per-branch body (the expensive work the gate would flag)
pae_down(0, []).
pae_down(N, [N|T]) :- N > 0, M is N - 1, pae_down(M, T).

% the two helper predicates a parallel_aggregate_transform produces:
%   enumerator yields each input; body maps input -> collected value
pae_enum(X)    :- pae_fact(X).
pae_body(X, L) :- pae_down(X, L).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_par_aggregate_exec).

test(par_collect_equals_seq_collect,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_par_agg_exec'))]) :-
    Dir = 'output/test_par_agg_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:pae_enum/1, user:pae_body/2, user:pae_fact/1, user:pae_down/2],
        [module_name(parexec)], Dir)),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/par_agg_test.rs', TestPath),
    TestSrc = '
use parexec::value::Value;
use parexec::state::WamState;
use parexec::par_aggregate::{par_collect, seq_collect};
use parexec::{pae_enum_1, pae_body_2};

#[test]
fn parallel_equals_sequential() {
    let base = WamState::new(vec![], std::collections::HashMap::new());
    let par = par_collect(&base, pae_enum_1, pae_body_2);
    let seq = seq_collect(&base, pae_enum_1, pae_body_2);

    // the acceptance criterion: parallel result-set == sequential, in order
    assert_eq!(par, seq, "parallel result must equal sequential");
    assert_eq!(seq.len(), 6, "expected one collected value per fact (6)");

    // spot-check the shape: first value is the list [1]
    match &seq[0] {
        Value::List(xs) => {
            let got: Vec<i64> = xs.iter().filter_map(|v| match v {
                Value::Integer(n) => Some(*n), _ => None }).collect();
            assert_eq!(got, vec![1], "pae_body(1) should yield [1]");
        }
        other => panic!("expected a list, got {:?}", other),
    }
    // and the last is [6,5,4,3,2,1]
    match seq.last() {
        Some(Value::List(xs)) => {
            let got: Vec<i64> = xs.iter().filter_map(|v| match v {
                Value::Integer(n) => Some(*n), _ => None }).collect();
            assert_eq!(got, vec![6,5,4,3,2,1], "pae_body(6) should yield [6..1]");
        }
        other => panic!("expected a list last, got {:?}", other),
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test par_agg_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[par_aggregate exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_par_aggregate_exec).
