% wam_rust_boundary_measurement.pl
%
% P3 / spec §6 measurement: does the boundary distribution cache add measurable
% wall-time ON TOP OF the (warm) edge cache, and from what D_pre — measured on the
% REAL emitted kernels in a generated crate (not a std-only re-implementation):
%
%   baseline  : collect_native_category_ancestor_hops      (edge-cached = eager
%               HashMap edges; parent lookups are warm-cache hits)
%   boundary  : build_boundary_suffix_sweep(root-near band at D_pre)  [precompute]
%               + collect_native_category_ancestor_boundary_hist        [splice]
%
% on a DENSE-core synthetic graph (diamonds -> exponentially many seed->root paths
% behind a thin boundary cut). Reports, per (scale, D_pre): production ms, boundary
% query ms, precompute ms, speedup, retained band size, sweep peak_resident, and
% the exact-match correctness invariant (boundary aggregate == production).
%
% Run:  swipl examples/benchmark/wam_rust_boundary_measurement.pl
% (cargo-gated; builds the generated crate in --release.)

:- use_module('../../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic m_fact/2.
m_fact(a, b).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

bench_rust('
use bm::state::WamState;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

fn rnd(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); *s >> 33 }

// Dense core (node i in 1..core has up to cp parents of smaller index, toward
// root 0) behind a thin boundary band (top w core nodes); periphery seeds attach
// only to the band, so every seed->root path crosses the cut.
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
    let w = 40u32.min(core - 1);
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

fn main() {
    let budget = 8usize;
    let n = 2.0f64;
    let edge_pred = "category_parent";
    println!("{:<16} {:>4} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>4}",
        "config", "Dpre", "prod_ms", "bound_ms", "pre_ms", "speedup", "region", "front", "eq");
    for (core, periph, cp) in [(120u32, 500u32, 3usize), (180, 500, 3), (240, 500, 3)] {
        let (edges, seed_ids) = build(core, periph, cp, 42);
        let md_num = min_dist_to_root(&edges, 0);
        let mut vm = WamState::new(vec![], HashMap::new());
        let owned: Vec<(String, String)> =
            edges.iter().map(|&(c, p)| (c.to_string(), p.to_string())).collect();
        let refs: Vec<(&str, &str)> = owned.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        vm.register_ffi_fact_pairs(edge_pred, &refs);
        let root = vm.intern_atom("0");
        let mut md: HashMap<i32, i32> = HashMap::new();
        for (&node, &d) in &md_num { md.insert(vm.intern_atom(&node.to_string()) as i32, d); }
        vm.set_min_dist(&md);
        let seeds: Vec<u32> = seed_ids.iter().map(|&i| vm.intern_atom(&i.to_string())).collect();
        // production baseline (acc borrows vm immutably -> scope it before any &mut sweep).
        let (ta, sa) = {
            let acc = vm.resolve_edge_accessor(edge_pred);
            let t = Instant::now();
            let mut sa = 0.0;
            for &s in &seeds {
                let mut hops: Vec<i64> = Vec::new();
                let mut vis = vec![s];
                vm.collect_native_category_ancestor_hops(s, root, &mut vis, budget, &acc, 0, &mut hops);
                sa += wpow_hops(&hops, n);
            }
            (t.elapsed().as_secs_f64() * 1e3, sa)
        };
        for dpre in [1usize, 2, 3, 4] {
            // entry-frontier band (thin cut) vs the whole root-near region.
            let region_sz = vm.boundary_band_root_near(dpre).len();
            let band = vm.boundary_band_entry_frontier(dpre, edge_pred);
            let front_sz = band.len();
            let pre = Instant::now();
            vm.build_boundary_suffix_sweep(&band, root, budget, edge_pred, 0, 0).unwrap();
            let tpre = pre.elapsed().as_secs_f64() * 1e3;
            let (tb, sb) = {
                let acc = vm.resolve_edge_accessor(edge_pred);
                let t = Instant::now();
                let mut sb = 0.0;
                for &s in &seeds {
                    let mut h: Vec<u64> = Vec::new();
                    let mut vis = vec![s];
                    vm.collect_native_category_ancestor_boundary_hist(s, root, &mut vis, budget, &acc, 0, &mut h);
                    sb += wpow_hist(&h, n);
                }
                (t.elapsed().as_secs_f64() * 1e3, sb)
            };
            let eq = (sa - sb).abs() < 1e-6 * sa.abs().max(1.0);
            println!("core={:<4} cp={:<2}    {:>4} {:>9.2} {:>9.3} {:>9.3} {:>8.1}x {:>7} {:>7} {:>4}",
                core, cp, dpre, ta, tb, tpre, ta / tb, region_sz, front_sz,
                if eq { "yes" } else { "NO!" });
        }
        // LAZY (on-demand warmup) vs EAGER (precompute the whole band), apples to
        // apples on the SAME min_dist<=d_pre band, exposing the two axes that
        // decide it: dataset size (the eager band grows; lazy computes only what is
        // touched) and query count (steady state is splice-only, identical for
        // both, so it amortizes the same way).
        let dpre = 2usize;
        let mut mk = |take: usize| -> (WamState, u32, Vec<u32>) {
            let mut v = WamState::new(vec![], HashMap::new());
            v.register_ffi_fact_pairs(edge_pred, &refs);
            let r = v.intern_atom("0");
            let mut m: HashMap<i32, i32> = HashMap::new();
            for (&node, &d) in &md_num { m.insert(v.intern_atom(&node.to_string()) as i32, d); }
            v.set_min_dist(&m);
            let ss: Vec<u32> = seed_ids.iter().take(take).map(|&i| v.intern_atom(&i.to_string())).collect();
            (v, r, ss)
        };
        // helper: time one steady-state query round (plain kernel over a warm cache).
        fn query_round(vm: &WamState, seeds: &[u32], root: u32, budget: usize, edge_pred: &str, n: f64) -> (f64, f64) {
            let acc = vm.resolve_edge_accessor(edge_pred);
            let t = Instant::now();
            let mut s = 0.0;
            for &x in seeds {
                let mut h: Vec<u64> = Vec::new();
                let mut vis = vec![x];
                vm.collect_native_category_ancestor_boundary_hist(x, root, &mut vis, budget, &acc, 0, &mut h);
                s += wpow_hist(&h, n);
            }
            (t.elapsed().as_secs_f64() * 1e3, s)
        }
        // EAGER precomputes the WHOLE region band ONCE — it does not know which
        // seeds will be queried, so it must materialise the full potential band.
        let (mut evm, eroot, _) = mk(seed_ids.len());
        let region = evm.boundary_band_root_near(dpre);
        let region_n = region.len();
        let pe = Instant::now();
        evm.build_boundary_suffix(&region, eroot, budget, edge_pred);
        let eager_pre = pe.elapsed().as_secs_f64() * 1e3;
        // Compare two WORKLOADS on the same band: dense (all seeds, touches most of
        // the band) and sparse (20 seeds, touches little). This is the axis that
        // decides lazy vs eager.
        for (wlabel, wsize) in [("dense ", seed_ids.len()), ("sparse", 20usize)] {
            let wseeds: Vec<u32> = seed_ids.iter().take(wsize).map(|&i| evm.intern_atom(&i.to_string())).collect();
            let (eq_round, esum) = query_round(&evm, &wseeds, eroot, budget, edge_pred, n);
            // LAZY: fresh cache, warm it on demand over THIS workload only.
            let (mut lvm, lroot, _) = mk(seed_ids.len());
            let lws: Vec<u32> = seed_ids.iter().take(wsize).map(|&i| lvm.intern_atom(&i.to_string())).collect();
            let pl = Instant::now();
            let _ = lvm.lazy_boundary_weightsum(&lws, lroot, budget, edge_pred, dpre, n);
            let lazy_warm = pl.elapsed().as_secs_f64() * 1e3;
            let touched_n = lvm.boundary_suffix.len();
            let (lz_round, lsum) = query_round(&lvm, &lws, lroot, budget, edge_pred, n);
            let eq = (esum - lsum).abs() < 1e-9 * esum.abs().max(1.0);
            println!("  [{}|{} seeds] eager: band={} pre={:.3}ms | lazy: touched={} warm={:.3}ms | steady eager={:.3}ms lazy={:.3}ms eq={}",
                wlabel, wsize, region_n, eager_pre, touched_n, lazy_warm, eq_round, lz_round, if eq {"y"} else {"N"});
            for k in [1usize, 10, 100] {
                let etot = eager_pre + k as f64 * eq_round;       // eager: full precompute + K rounds
                let ltot = lazy_warm + (k as f64 - 1.0) * lz_round; // lazy: warm (round 1) + (K-1) rounds
                println!("       K={:<4} eager={:.2}ms lazy={:.2}ms winner={}",
                    k, etot, ltot, if ltot < etot { "LAZY" } else { "eager" });
            }
        }
    }
}').

main :-
    Dir = 'output/boundary_measure',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:m_fact/2], [module_name(bm)], Dir)),
    atom_concat(Dir, '/examples', ExDir),
    make_directory_path(ExDir),
    atom_concat(Dir, '/examples/boundary_measure.rs', ExPath),
    bench_rust(Src),
    setup_call_cleanup(open(ExPath, write, S), format(S, "~w", [Src]), close(S)),
    format(atom(Cmd),
        'cd "~w" && cargo run --release --example boundary_measure 2>&1', [Dir]),
    format("Building + running (release)...~n", []),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, OutS), close(Out)),
    process_wait(Pid, Status),
    format("~n=== wam-rust boundary measurement (status ~w) ===~n~w~n", [Status, OutS]),
    halt.

:- initialization(main).
