% test_wam_rust_parallel_speedup.pl
%
% T7 capstone: prove the compiled parallel path delivers a real WALL-CLOCK
% speedup (not just correctness) through the actual generated code. Generates a
% project for a parallel-eligible aggregate with an expensive recursive body
% (naive fib per branch), then times seq_collect vs par_collect over the real
% generated predicate functions and asserts par < seq (and par == seq results).
% Built in --release so the timing is meaningful.
%
% cargo-gated; tolerant of slow machines (asserts a speedup only with >= 2 cores).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bench_fact/1.
% 32 independent branches
bench_fact(1).  bench_fact(2).  bench_fact(3).  bench_fact(4).
bench_fact(5).  bench_fact(6).  bench_fact(7).  bench_fact(8).
bench_fact(9).  bench_fact(10). bench_fact(11). bench_fact(12).
bench_fact(13). bench_fact(14). bench_fact(15). bench_fact(16).
bench_fact(17). bench_fact(18). bench_fact(19). bench_fact(20).
bench_fact(21). bench_fact(22). bench_fact(23). bench_fact(24).
bench_fact(25). bench_fact(26). bench_fact(27). bench_fact(28).
bench_fact(29). bench_fact(30). bench_fact(31). bench_fact(32).

% naive fib — expensive recursive per-branch work
bench_fib(0, 0).
bench_fib(1, 1).
bench_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2,
                   bench_fib(N1, F1), bench_fib(N2, F2), F is F1 + F2.

% per-branch body depends on the enumerator var X (frontier = {X})
bench_body(X, F) :- N is 20 + (X mod 3), bench_fib(N, F).

% the user aggregate (whole-body) -> parallelised
bench_collect(L) :- findall(F, (bench_fact(X), bench_body(X, F)), L).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_parallel_speedup).

test(parallel_beats_sequential,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_par_speedup'))]) :-
    Dir = 'output/test_par_speedup',
    safe_rmdir(Dir),
    once(write_wam_rust_project(
        [user:bench_collect/1, user:bench_fact/1, user:bench_body/2, user:bench_fib/2],
        [module_name(bench), parallel_aggregates(true)], Dir)),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bench_test.rs', TestPath),
    TestSrc = '
use bench::value::Value;
use bench::state::WamState;
use bench::par_aggregate::{par_collect, seq_collect};
use bench::{__par_enum_bench_collect_1, __par_body_bench_collect_2};

#[test]
fn parallel_speedup() {
    let base = WamState::new(vec![], std::collections::HashMap::new());
    let e = __par_enum_bench_collect_1;
    let b = __par_body_bench_collect_2;
    let _ = seq_collect(&base, e, b);                 // warm up

    let t0 = std::time::Instant::now();
    let seq = seq_collect(&base, e, b);
    let t_seq = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let par = par_collect(&base, e, b);
    let t_par = t1.elapsed().as_secs_f64();

    let cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    println!("SPEEDUP_LINE seq={:.4}s par={:.4}s speedup={:.2}x n={} cores={}",
             t_seq, t_par, t_seq / t_par, seq.len(), cores);

    assert_eq!(par, seq, "parallel result must equal sequential");
    assert!(seq.len() == 32, "expected 32 collected values, got {}", seq.len());
    if cores >= 2 {
        assert!(t_par < t_seq,
            "parallel ({:.4}s) should beat sequential ({:.4}s)", t_par, t_seq);
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd),
        'cd "~w" && cargo test --release --test bench_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    % surface the measured speedup line
    ( sub_string(OutS, B0, _, _, "SPEEDUP_LINE"),
      sub_string(OutS, B0, 120, _, Line)
    ->  format(user_error, "~n[T7 ~w]~n", [Line])
    ;   true ),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[speedup test FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_parallel_speedup).
