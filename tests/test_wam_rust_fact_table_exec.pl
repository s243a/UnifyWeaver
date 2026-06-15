% test_wam_rust_fact_table_exec.pl
%
% T9 fact-table inline (Rust) — query-mode exec matrix. A large all-ground-facts
% predicate compiled with fact_table_inline(true) must enumerate exactly the same
% solutions, in the same order, as the ordinary path across the four arg modes:
%   (+,-) key bound        -> all values for that key, source order
%   (-,+) value bound      -> the matching key(s)
%   (-,-) both unbound     -> every row, source order
%   (+,+) both bound       -> succeed iff the row exists
% Backtracking is driven by vm.backtrack() resuming the fact_table choice point.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic edge/2.
% 6 rows; t9_min_rows(4) makes this eligible. Distinct + repeated first-arg keys.
edge(a, 1). edge(a, 2). edge(b, 3). edge(a, 4). edge(c, 5). edge(b, 6).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_fact_table_exec).

test(query_mode_matrix,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_t9_exec'))]) :-
    Dir = 'output/test_t9_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:edge/2],
        [module_name(eg), fact_table_inline(true), t9_min_rows(4)], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "Strategy: fact_table")),
    assertion(sub_string(Src, _, _, _, "fact_table_attempt")),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/eg_test.rs', TestPath),
    TestSrc = '
use eg::value::Value;
use eg::state::WamState;
use eg::edge_2;

// Enumerate all solutions of `edge(a1, a2)`, reading the listed var bindings per
// solution. Drives backtrack() (which resumes the fact_table choice point).
fn solutions(a1: Value, a2: Value, vars: &[&str]) -> Vec<Vec<Value>> {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    let mut out = Vec::new();
    if edge_2(&mut vm, a1, a2) {
        loop {
            out.push(vars.iter()
                .map(|v| vm.bindings.get(*v).cloned().unwrap_or(Value::Unbound(v.to_string())))
                .collect());
            if !vm.backtrack() { break; }
        }
    }
    out
}
fn at(s: &str) -> Value { Value::Atom(s.to_string()) }
fn ity(n: i64) -> Value { Value::Integer(n) }
fn ub(s: &str) -> Value { Value::Unbound(s.to_string()) }

#[test]
fn plus_minus_key_bound() {
    // edge(a, X): values 1,2,4 in source order
    let s = solutions(at("a"), ub("X"), &["X"]);
    assert_eq!(s, vec![vec![ity(1)], vec![ity(2)], vec![ity(4)]]);
}
#[test]
fn minus_plus_value_bound() {
    // edge(K, 3): only b
    let s = solutions(ub("K"), ity(3), &["K"]);
    assert_eq!(s, vec![vec![at("b")]]);
}
#[test]
fn minus_minus_all_rows() {
    // edge(K, X): every row, source order
    let s = solutions(ub("K"), ub("X"), &["K", "X"]);
    assert_eq!(s, vec![
        vec![at("a"), ity(1)], vec![at("a"), ity(2)], vec![at("b"), ity(3)],
        vec![at("a"), ity(4)], vec![at("c"), ity(5)], vec![at("b"), ity(6)],
    ]);
}
#[test]
fn plus_plus_membership() {
    assert_eq!(solutions(at("a"), ity(2), &[]).len(), 1, "edge(a,2) exists");
    assert_eq!(solutions(at("a"), ity(3), &[]).len(), 0, "edge(a,3) absent");
    assert_eq!(solutions(at("z"), ity(9), &[]).len(), 0, "unknown key fails");
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
    ;   format(user_error, "~n[T9 fact-table exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_fact_table_exec).
