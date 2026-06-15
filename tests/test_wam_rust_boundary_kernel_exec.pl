% test_wam_rust_boundary_kernel_exec.pl
%
% P2c parity: the boundary-spliced ancestor kernel
% (collect_native_category_ancestor_boundary_hist) produces the SAME weighted-
% power aggregate as the PRODUCTION kernel (collect_native_category_ancestor_hops)
% — validated end-to-end in a generated crate, through the real edge-access path.
% This closes the gap P2a left open (P2a compared against an in-module oracle;
% this compares against the actual emitted production kernel).
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bk_fact/2.
bk_fact(a, b).   % keeps the project non-trivial; the graph for the kernels is
bk_fact(b, c).   % registered inside the cargo test via register_ffi_fact_pairs.

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_boundary_kernel_exec).

test(boundary_kernel_matches_production_kernel,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bk_exec'))]) :-
    Dir = 'output/test_bk_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bk_fact/2], [module_name(bk)], Dir)),
    % the runtime always emits the production ancestor kernel
    atom_concat(Dir, '/src/state.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    ( sub_string(Src, _, _, _, "collect_native_category_ancestor_hops")
    -> true ; sub_string(Src, _, _, _, "collect_native_category_ancestor") ),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bk_test.rs', TestPath),
    TestSrc = '
use bk::state::WamState;
use std::collections::HashMap;

// weighted_power(N=2) aggregate from the production kernel (hop stream).
fn ws_production(vm: &WamState, seed: u32, root: u32, budget: usize, n: f64) -> f64 {
    let acc = vm.resolve_edge_accessor("category_parent");
    let mut hops: Vec<i64> = Vec::new();
    let mut vis = vec![seed];
    vm.collect_native_category_ancestor_hops(seed, root, &mut vis, budget, &acc, 0, &mut hops);
    hops.iter().map(|&h| (h as f64).powf(-n)).sum()
}

// weighted_power(N=2) aggregate from the boundary-spliced kernel (histogram).
fn ws_boundary(vm: &WamState, seed: u32, root: u32, budget: usize, n: f64) -> f64 {
    let acc = vm.resolve_edge_accessor("category_parent");
    let mut hist: Vec<u64> = Vec::new();
    let mut vis = vec![seed];
    vm.collect_native_category_ancestor_boundary_hist(seed, root, &mut vis, budget, &acc, 0, &mut hist);
    hist.iter().enumerate().filter(|(l, _)| *l > 0)
        .map(|(l, &c)| c as f64 * (l as f64).powf(-n)).sum()
}

#[test]
fn boundary_matches_production() {
    let mut vm = WamState::new(vec![], HashMap::new());
    // child -> parent diamond DAG toward root "0", plus a couple deeper seeds.
    vm.register_ffi_fact_pairs("category_parent", &[
        ("5","4"),("5","2"),("4","3"),("4","1"),("3","1"),("3","2"),
        ("1","0"),("2","0"),("6","5"),("7","4"),("8","6"),("8","7"),
    ]);
    let root = vm.intern_atom("0");
    let budget = 10usize;
    let n = 2.0f64;
    // boundary band near the root
    let bnodes: Vec<u32> = ["1","2"].iter().map(|s| vm.intern_atom(s)).collect();
    vm.build_boundary_suffix(&bnodes, root, budget, "category_parent");
    let mut checked = 0;
    for sname in ["3","4","5","6","7","8"] {
        let seed = vm.intern_atom(sname);
        let wp = ws_production(&vm, seed, root, budget, n);
        let wb = ws_boundary(&vm, seed, root, budget, n);
        assert!((wp - wb).abs() < 1e-12, "seed {}: production {} != boundary {}", sname, wp, wb);
        assert!(wp > 0.0, "seed {} should reach root", sname);
        checked += 1;
    }
    assert_eq!(checked, 6);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bk_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary kernel exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_kernel_exec).
