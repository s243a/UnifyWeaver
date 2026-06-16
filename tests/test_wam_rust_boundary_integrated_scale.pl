% test_wam_rust_boundary_integrated_scale.pl
%
% §6 exact-match invariant, guarded deterministically: the FULL integrated
% boundary path — min_dist -> boundary_band_root_near(D_pre) ->
% build_boundary_suffix_sweep -> collect_native_category_ancestor_boundary_hist —
% reproduces the production kernel's weighted-power aggregate EXACTLY, on a
% moderate dense-core synthetic graph, across a D_pre sweep. This is the
% correctness counterpart of the wall-time measurement
% (examples/benchmark/wam_rust_boundary_measurement.pl) and the bug-guard for the
% integrated path on the real emitted kernels (debug build, deterministic).
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bis_fact/2.
bis_fact(a, b).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_boundary_integrated_scale).

test(integrated_path_matches_production,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bis'))]) :-
    Dir = 'output/test_bis',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bis_fact/2], [module_name(bis)], Dir)),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bis_test.rs', TestPath),
    TestSrc = '
use bis::state::WamState;
use std::collections::{HashMap, VecDeque};

fn rnd(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); *s >> 33 }

fn build(core: u32, periph: u32, cp: usize, seed: u64) -> (Vec<(u32, u32)>, Vec<u32>) {
    let mut s = seed;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for i in 1..core {
        let k = (cp as u32).min(i).max(1);
        let mut ps: Vec<u32> = Vec::new();
        for _ in 0..k { ps.push((rnd(&mut s) as u32) % i); }
        ps.sort(); ps.dedup();
        for p in ps { edges.push((i, p)); }
    }
    let w = 20u32.min(core - 1);
    let mut seeds: Vec<u32> = Vec::new();
    for i in core..core + periph {
        let mut ps: Vec<u32> = Vec::new();
        for _ in 0..2 { ps.push(core - w + (rnd(&mut s) as u32) % w); }
        ps.sort(); ps.dedup();
        for p in ps { edges.push((i, p)); }
        seeds.push(i);
    }
    (edges, seeds)
}

fn min_dist_to_root(edges: &[(u32, u32)], root: u32) -> HashMap<u32, i32> {
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(c, p) in edges { children.entry(p).or_default().push(c); }
    let mut dist: HashMap<u32, i32> = HashMap::new();
    dist.insert(root, 0);
    let mut q = VecDeque::new();
    q.push_back(root);
    while let Some(node) = q.pop_front() {
        let d = dist[&node];
        if let Some(cs) = children.get(&node) {
            for &c in cs { if !dist.contains_key(&c) { dist.insert(c, d + 1); q.push_back(c); } }
        }
    }
    dist
}

fn wpow_hops(h: &[i64], n: f64) -> f64 { h.iter().map(|&x| (x as f64).powf(-n)).sum() }
fn wpow_hist(h: &[u64], n: f64) -> f64 {
    h.iter().enumerate().filter(|(l, _)| *l > 0).map(|(l, &c)| c as f64 * (l as f64).powf(-n)).sum()
}

#[test]
fn integrated_boundary_equals_production() {
    let budget = 8usize;
    let n = 2.0f64;
    let (edges, seed_ids) = build(80, 200, 3, 42);
    let md_num = min_dist_to_root(&edges, 0);
    let mut vm = WamState::new(vec![], HashMap::new());
    let owned: Vec<(String, String)> =
        edges.iter().map(|&(c, p)| (c.to_string(), p.to_string())).collect();
    let refs: Vec<(&str, &str)> = owned.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    vm.register_ffi_fact_pairs("category_parent", &refs);
    let root = vm.intern_atom("0");
    let mut md: HashMap<i32, i32> = HashMap::new();
    for (&node, &d) in &md_num { md.insert(vm.intern_atom(&node.to_string()) as i32, d); }
    vm.set_min_dist(&md);
    let seeds: Vec<u32> = seed_ids.iter().map(|&i| vm.intern_atom(&i.to_string())).collect();
    // production aggregate per seed (reference)
    let prod: Vec<f64> = {
        let acc = vm.resolve_edge_accessor("category_parent");
        seeds.iter().map(|&s| {
            let mut hops: Vec<i64> = Vec::new();
            let mut vis = vec![s];
            vm.collect_native_category_ancestor_hops(s, root, &mut vis, budget, &acc, 0, &mut hops);
            wpow_hops(&hops, n)
        }).collect()
    };
    assert!(prod.iter().any(|&x| x > 0.0), "seeds should reach root");
    // sweep the integrated boundary path at several D_pre and demand exact match
    for dpre in [1usize, 2, 3] {
        let band = vm.boundary_band_root_near(dpre);
        vm.build_boundary_suffix_sweep(&band, root, budget, "category_parent", 0, 0).unwrap();
        let acc = vm.resolve_edge_accessor("category_parent");
        for (i, &s) in seeds.iter().enumerate() {
            let mut h: Vec<u64> = Vec::new();
            let mut vis = vec![s];
            vm.collect_native_category_ancestor_boundary_hist(s, root, &mut vis, budget, &acc, 0, &mut h);
            let wb = wpow_hist(&h, n);
            assert!((prod[i] - wb).abs() < 1e-9 * prod[i].abs().max(1.0),
                "D_pre {} seed#{}: production {} != boundary {}", dpre, i, prod[i], wb);
        }
    }
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bis_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary integrated scale FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_integrated_scale).
