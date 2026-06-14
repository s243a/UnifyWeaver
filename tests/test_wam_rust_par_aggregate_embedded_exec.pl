% test_wam_rust_par_aggregate_embedded_exec.pl
%
% T7 route 2 end-to-end: an aggregate EMBEDDED in a larger clause body (a guard
% goal before the findall) is compiled with parallel_aggregates(true). The
% containing predicate stays a WAM predicate, but its begin_aggregate..
% end_aggregate block is rewritten to a single `par_aggregate` instruction that
% drives synthesised __par_enum/__par_body helpers in parallel. Builds and RUNS
% the project, asserting the user predicate returns the correct collected result
% (== the known sequential answer).
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic eemb_ready/0.
:- dynamic eemb_fact/1.
eemb_ready.
eemb_fact(1). eemb_fact(2). eemb_fact(3). eemb_fact(4). eemb_fact(5). eemb_fact(6).
eemb_down(0, []).
eemb_down(N, [N|T]) :- N > 0, M is N - 1, eemb_down(M, T).

% Embedded aggregate: the findall is NOT the whole body (eemb_ready guards it),
% so route-1 (whole-body native wrapper) does not apply; route-2 must.
eemb_collect(L) :- eemb_ready, findall(D, (eemb_fact(X), eemb_down(X, D)), L).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_par_aggregate_embedded_exec).

test(embedded_findall_compiles_to_par_aggregate_and_runs,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_paragg_embed_exec'))]) :-
    Dir = 'output/test_paragg_embed_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:eemb_collect/1, user:eemb_ready/0, user:eemb_fact/1, user:eemb_down/2],
        [module_name(pemb), parallel_aggregates(true)], Dir)),
    % the rewrite must have spliced a ParAggregate instruction into the shared
    % WAM table, with the synthesised helper predicates present.
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "ParAggregate")),
    assertion(sub_string(Src, _, _, _, "__par_enum_eemb_collect/1")),
    assertion(sub_string(Src, _, _, _, "__par_body_eemb_collect/2")),
    % embed an integration test that runs the user predicate
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/pemb_test.rs', TestPath),
    % The result var L is permanent (head arg surviving the eemb_ready call), so
    % the aggregate binds it in a Y register. The aggregate finalisation binds
    % through the Y-aware accessors, so the collected list surfaces to the clause
    % variable in bindings["L"] — same as a whole-body aggregate. The point of the
    % test is parallel result == the known sequential answer.
    TestSrc = '
use pemb::value::Value;
use pemb::state::WamState;
use pemb::eemb_collect_1;

fn ints(v: &Value) -> Vec<i64> {
    match v { Value::List(xs) => xs.iter().filter_map(|e| match e {
        Value::Integer(n) => Some(*n), _ => None }).collect(), _ => vec![] }
}

#[test]
fn user_embedded_findall_parallel() {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    assert!(eemb_collect_1(&mut vm, Value::Unbound("L".to_string())), "eemb_collect should succeed");
    match vm.bindings.get("L").cloned() {
        Some(Value::List(items)) => {
            assert_eq!(items.len(), 6, "one list per fact");
            assert_eq!(ints(&items[0]), vec![1], "first = [1]");
            assert_eq!(ints(items.last().unwrap()), vec![6,5,4,3,2,1], "last = [6..1]");
        }
        other => panic!("expected list in bindings[L], got {:?}", other),
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test pemb_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[embedded exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_par_aggregate_embedded_exec).
