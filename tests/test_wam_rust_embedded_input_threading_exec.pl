% test_wam_rust_embedded_input_threading_exec.pl
%
% T7 route-2 input threading: an EMBEDDED aggregate whose enumerator reads an
% enclosing-clause input (e.g. findall(D,(eg_link(N,X),eg_dn(X,D)),R) where N is a
% head arg). Earlier this was parallelised with N UNBOUND -> it enumerated every
% eg_link, returning wrong results (eg_p(1,R) gave both links' bodies); then it
% was made to DECLINE (compile sequentially) for correctness. Now route 2 THREADS
% the external inputs: the helpers take them as leading params and the
% ParAggregate instruction captures their values from the container registers, so
% the aggregate parallelises AND honours the input (eg_p(1,R) = [[3,2,1]] only).
%
% Asserts (a) a ParAggregate instruction IS spliced (parallelised, not declined),
% and (b) eg_p(1,R) honours N=1.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic eg_link/2.
eg_guard(_).
eg_link(1, 3). eg_link(2, 4).          % distinct keys; N selects which
eg_dn(0, []).
eg_dn(N, [N|T]) :- N > 0, M is N - 1, eg_dn(M, T).
% embedded aggregate whose inner goal reads the head-arg input N
eg_p(N, R) :- eg_guard(N), findall(D, (eg_link(N, X), eg_dn(X, D)), R).

% Two external inputs (A, B), both head args, feeding the inner goal in order.
% tw_link(A, B, X): A=1,B=2 selects X=3; the OTHER order (B,A)=(2,1) matches no
% link, so a correct result proves the helper params line up with the captured
% registers in the right order.
:- dynamic tw_link/3.
tw_guard(_).
tw_link(1, 2, 3). tw_link(1, 9, 4). tw_link(5, 2, 5).
tw_dn(0, []).
tw_dn(N, [N|T]) :- N > 0, M is N - 1, tw_dn(M, T).
tw_p(A, B, R) :- tw_guard(A), findall(D, (tw_link(A, B, X), tw_dn(X, D)), R).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_embedded_input_threading_exec).

test(input_taking_embedded_threads_and_is_correct,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_emb_thread_exec'))]) :-
    Dir = 'output/test_emb_thread_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:eg_p/2, user:eg_guard/1, user:eg_link/2, user:eg_dn/2],
        [module_name(eg), parallel_aggregates(true)], Dir)),
    % parallelised: a ParAggregate IS spliced for this input-taking embedded
    % aggregate (it carries the container input register on the line).
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "ParAggregate")),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/eg_test.rs', TestPath),
    TestSrc = '
use eg::value::Value;
use eg::state::WamState;
use eg::eg_p_2;

fn ii(v: &Value) -> Vec<Vec<i64>> {
    match v { Value::List(xs) => xs.iter().map(|e| match e {
        Value::List(ys) => ys.iter().filter_map(|z| if let Value::Integer(n)=z {Some(*n)} else {None}).collect(),
        _ => vec![] }).collect(), _ => vec![] }
}

#[test]
fn embedded_input_correct() {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    assert!(eg_p_2(&mut vm, Value::Integer(1), Value::Unbound("R".to_string())));
    // only eg_link(1,3) -> eg_dn(3) = [3,2,1]; NOT eg_link(2,4) too.
    assert_eq!(ii(&vm.bindings.get("R").cloned().unwrap()), vec![vec![3,2,1]],
        "embedded aggregate must honour the head-arg input N=1");
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test eg_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[embedded-threading exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

test(two_external_inputs_thread_in_order,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_emb_thread2_exec'))]) :-
    Dir = 'output/test_emb_thread2_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:tw_p/3, user:tw_guard/1, user:tw_link/3, user:tw_dn/2],
        [module_name(tw), parallel_aggregates(true)], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "ParAggregate")),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/tw_test.rs', TestPath),
    TestSrc = '
use tw::value::Value;
use tw::state::WamState;
use tw::tw_p_3;

fn ii(v: &Value) -> Vec<Vec<i64>> {
    match v { Value::List(xs) => xs.iter().map(|e| match e {
        Value::List(ys) => ys.iter().filter_map(|z| if let Value::Integer(n)=z {Some(*n)} else {None}).collect(),
        _ => vec![] }).collect(), _ => vec![] }
}

#[test]
fn two_inputs_in_order() {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    // tw_p(1, 2, R): tw_link(1,2,3) -> tw_dn(3) = [3,2,1]. The swapped order
    // (2,1) matches no link, so [[3,2,1]] proves A,B map to the right registers.
    assert!(tw_p_3(&mut vm, Value::Integer(1), Value::Integer(2), Value::Unbound("R".to_string())));
    assert_eq!(ii(&vm.bindings.get("R").cloned().unwrap()), vec![vec![3,2,1]],
        "two external inputs must thread in helper-parameter order");
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test tw_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[two-input threading exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_embedded_input_threading_exec).
