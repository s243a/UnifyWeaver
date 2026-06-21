% test_wam_rust_fact_table_callsite_exec.pl
%
% T9 call-site integration: a WAM-compiled predicate that calls a fact-table
% predicate through ordinary WAM dispatch (`call` for a non-last goal, `execute`
% for a tail call) must enumerate the fact predicate's solutions correctly,
% including backtracking into the fact-table choice point from a downstream goal.
% The Call/Execute handlers route an unresolved predicate to the generated
% crate::fact_table_call dispatcher.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic edge/2.
edge(a, 1). edge(a, 2). edge(b, 3). edge(a, 4). edge(c, 5). edge(b, 6).
qlast(X) :- edge(a, X).            % tail call -> execute edge/2
qmid(X) :- edge(a, X), X > 1.      % non-last  -> call edge/2, then filter
qall(K, L) :- findall(X, edge(K, X), L).   % aggregate enumerates via call edge/2

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_fact_table_callsite_exec).

test(callers_dispatch_to_fact_table,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_t9_callsite_exec'))]) :-
    Dir = 'output/test_t9_callsite_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:qlast/1, user:qmid/1, user:qall/2, user:edge/2],
        [module_name(c), fact_table_inline(true), t9_min_rows(4)], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    % edge/2 is a fact table; callers reach it via Call/Execute -> fact_table_call
    assertion(sub_string(Src, _, _, _, "Strategy: fact_table")),
    assertion(sub_string(Src, _, _, _, "fact_table_call")),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/c_test.rs', TestPath),
    TestSrc = '
use c::value::Value;
use c::state::WamState;
use c::{qlast_1, qmid_1, qall_2, shared_wam_program};

fn fresh() -> WamState {
    let (code, labels) = shared_wam_program();
    WamState::new(code, labels)
}
fn enum1(call: impl Fn(&mut WamState) -> bool, var: &str) -> Vec<i64> {
    let mut vm = fresh();
    let mut out = Vec::new();
    if call(&mut vm) {
        loop {
            if let Some(Value::Integer(n)) = vm.bindings.get(var).cloned() { out.push(n); }
            if !vm.backtrack() || !vm.run() { break; }
        }
    }
    out
}
fn ints(v: &Value) -> Vec<i64> {
    match v { Value::List(xs) => xs.iter().filter_map(|e| if let Value::Integer(n)=e {Some(*n)} else {None}).collect(), _ => vec![] }
}

#[test]
fn execute_tailcall() {
    // qlast(X) :- edge(a, X).  -> execute edge/2 ; X in {1,2,4}
    assert_eq!(enum1(|vm| qlast_1(vm, Value::Unbound("X".into())), "X"), vec![1,2,4]);
}
#[test]
fn call_nonlast_with_filter() {
    // qmid(X) :- edge(a, X), X > 1.  -> call edge/2, backtrack into fact CP
    assert_eq!(enum1(|vm| qmid_1(vm, Value::Unbound("X".into())), "X"), vec![2,4]);
}
#[test]
fn findall_over_fact() {
    // qall(a, L) :- findall(X, edge(a,X), L). -> L = [1,2,4]
    let mut vm = fresh();
    assert!(qall_2(&mut vm, Value::Atom("a".into()), Value::Unbound("L".into())));
    assert_eq!(ints(&vm.bindings.get("L").cloned().unwrap()), vec![1,2,4]);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test c_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[T9 call-site exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_fact_table_callsite_exec).
