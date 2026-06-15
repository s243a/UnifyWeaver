% test_wam_rust_fact_table_throughput.pl
%
% T9 capstone: a large fact predicate compiled with fact_table_inline(true) must
% enumerate correctly AT SCALE (the kind of many-key workload where the WAM
% shared-table first-arg index drops keys) and do so with low per-query latency.
% Generates a 200-unique-key fact project, point-looks-up every key many times
% through the generated code, asserts each lookup returns exactly its one row
% (catching any dropped key), and prints the measured per-query latency.
%
% Release build so the timing is meaningful. cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic edge/2.
mk :- forall(between(0, 199, N), ( atom_concat(k, N, Key), assertz(edge(Key, N)) )).
:- initialization(mk).
% point lookup: each key has exactly one row
lookup(K, X) :- edge(K, X).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_fact_table_throughput).

test(point_lookup_throughput,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_t9_throughput'))]) :-
    Dir = 'output/test_t9_throughput',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:lookup/2, user:edge/2],
        [module_name(ft), fact_table_inline(true), t9_min_rows(10)], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "Strategy: fact_table")),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/tp_test.rs', TestPath),
    TestSrc = '
use ft::value::Value;
use ft::state::WamState;
use ft::edge_2;

#[test]
fn point_lookup_throughput() {
    let iters = 500;
    let keys: Vec<Value> = (0..200).map(|k| Value::Atom(format!("k{}", k))).collect();
    // warm
    { let mut vm = WamState::new(vec![], std::collections::HashMap::new());
      edge_2(&mut vm, keys[0].clone(), Value::Unbound("X".into())); }
    let mut rows: usize = 0;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for (i, key) in keys.iter().enumerate() {
            let mut vm = WamState::new(vec![], std::collections::HashMap::new());
            if edge_2(&mut vm, key.clone(), Value::Unbound("X".into())) {
                // exactly one row per key, and it must be the right one
                if let Some(Value::Integer(n)) = vm.bindings.get("X").cloned() {
                    assert_eq!(n as usize, i, "key k{} must map to row {}", i, i);
                    rows += 1;
                }
                assert!(!vm.backtrack(), "point lookup must leave no extra solutions");
            } else {
                panic!("key k{} not found (dropped)", i);
            }
        }
    }
    let n = iters * 200;
    assert_eq!(rows, n, "every point lookup returns its one row");
    let us = t0.elapsed().as_secs_f64() / (n as f64) * 1e6;
    println!("T9_THROUGHPUT queries={} per_query_us={:.3}", n, us);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd),
        'cd "~w" && cargo test --release --test tp_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( sub_string(OutS, B0, _, _, "T9_THROUGHPUT"),
      sub_string(OutS, B0, 80, _, Line)
    ->  format(user_error, "~n[T9 ~w]~n", [Line])
    ;   true ),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[T9 throughput FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_fact_table_throughput).
