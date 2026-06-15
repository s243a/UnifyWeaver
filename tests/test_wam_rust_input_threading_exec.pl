% test_wam_rust_input_threading_exec.pl
%
% T7 input-threading fix (step 2): a WHOLE-BODY parallel aggregate whose
% enumerator reads a HEAD-ARG input. Before the fix, par_collect ran the
% enumerator with that input unbound (enumerating over everything); now the
% wrapper threads the head arg into the enum/body helpers via closures. Builds +
% runs a generated project and asserts the input is honoured.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

% wbi_link(N, X): distinct first-arg keys (avoids an unrelated first-arg-indexing
% issue with repeated keys). N=1->X=3, N=2->X=4, N=3->X=5. The aggregate below
% must use only the N it is CALLED with, not all N.
:- dynamic wbi_link/2.
wbi_link(1, 3). wbi_link(2, 4). wbi_link(3, 5).
wbi_fib(0, 0).
wbi_fib(1, 1).
wbi_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2,
                 wbi_fib(N1, F1), wbi_fib(N2, F2), F is F1 + F2.
% whole-body aggregate with a head-arg INPUT (N) feeding the enumerator
wbi_q(N, L) :- findall(F, (wbi_link(N, X), wbi_fib(X, F)), L).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_input_threading_exec).

test(head_arg_input_is_threaded,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_inthread_exec'))]) :-
    Dir = 'output/test_inthread_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:wbi_q/2, user:wbi_link/2, user:wbi_fib/2],
        [module_name(wbi), parallel_aggregates(true)], Dir)),
    % the generated wrapper threads the head arg into the enum closure
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "par_collect")),
    assertion(sub_string(Src, _, _, _, "a1.clone()")),   % N threaded into the closures
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/inthread_test.rs', TestPath),
    TestSrc = '
use wbi::value::Value;
use wbi::state::WamState;
use wbi::wbi_q_2;

fn ints(v: &Value) -> Vec<i64> {
    match v { Value::List(xs) => xs.iter().filter_map(|e| match e {
        Value::Integer(n) => Some(*n), _ => None }).collect(), _ => vec![] }
}

#[test]
fn input_is_honoured() {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    // wbi_q(2, L): link of 2 is X=4 -> fib(4)=3. Must be [3], NOT all links
    // [fib(3),fib(4),fib(5)] = [2,3,5] (which is what the unthreaded bug gave).
    assert!(wbi_q_2(&mut vm, Value::Integer(2), Value::Unbound("L".to_string())));
    match vm.bindings.get("L").cloned() {
        Some(ref l @ Value::List(_)) => {
            assert_eq!(ints(l), vec![3],
                "must use only N=2's link (fib 4 = 3), not all N");
        }
        other => panic!("expected list, got {:?}", other),
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test inthread_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[input-threading exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_input_threading_exec).
