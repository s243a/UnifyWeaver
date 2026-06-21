% wam_rust_boundary_lazy_edge_measurement.pl
%
% Closes the §6 "to be confirmed on the LMDB run" caveat: does the boundary win
% hold (or grow) when each parent lookup is an LMDB seek (the two-tier edge cache)
% instead of an in-memory HashMap hit? Measures production vs boundary on the LAZY
% (LMDB-backed) edge path AND the eager (in-memory) path, SAME synthetic graph,
% side by side. Int-native mode (raw integer node ids == LMDB keys) keeps the lazy
% setup minimal (no s2i intern table).
%
% Run:  swipl examples/benchmark/wam_rust_boundary_lazy_edge_measurement.pl
% (cargo-gated; builds an lmdb_zero crate in --release.)

:- use_module('../../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic m_fact/2.
m_fact(a, b).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

bench_rust('
use bm::state::{WamState, LookupSource};
use bm::lmdb_fact_source::LmdbFactSource;
use lmdb_zero as lmdb;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

// Minimal LookupSource over the LMDB parent table (int-native: the kernel node id
// IS the LMDB key).
struct LazyParents { src: LmdbFactSource }
impl LookupSource for LazyParents {
    fn lookup_key_for_atom(&self, _atom: &str) -> Option<i32> { None }
    fn lookup_parents(&self, key: i32) -> Vec<i32> { self.src.lookup_parents(key).unwrap_or_default() }
    fn atom_for_key(&self, _key: i32) -> Option<String> { None }
}

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

// Write the synthetic graph into a fresh LMDB env (Phase-1 layout: empty s2i/i2s,
// category_parent + category_child as DUPSORT int32_le).
fn write_lmdb(path: &str, edges: &[(u32, u32)]) {
    let env = unsafe {
        let mut b = lmdb::EnvBuilder::new().unwrap();
        b.set_maxdbs(16).unwrap();
        b.set_mapsize(256 * 1024 * 1024).unwrap();
        b.open(path, lmdb::open::Flags::empty(), 0o600).unwrap()
    };
    let env = Arc::new(env);
    for name in ["s2i", "i2s"] {
        let _ = lmdb::Database::open(Arc::clone(&env), Some(name),
            &lmdb::DatabaseOptions::new(lmdb::db::CREATE)).unwrap();
    }
    let cp = lmdb::Database::open(Arc::clone(&env), Some("category_parent"),
        &lmdb::DatabaseOptions::new(lmdb::db::CREATE | lmdb::db::DUPSORT)).unwrap();
    let cc = lmdb::Database::open(Arc::clone(&env), Some("category_child"),
        &lmdb::DatabaseOptions::new(lmdb::db::CREATE | lmdb::db::DUPSORT)).unwrap();
    let txn = lmdb::WriteTransaction::new(&*env).unwrap();
    {
        let mut acc = txn.access();
        for &(c, p) in edges {
            let cb = (c as i32).to_le_bytes();
            let pb = (p as i32).to_le_bytes();
            acc.put(&cp, &cb[..], &pb[..], lmdb::put::Flags::empty()).unwrap();
            acc.put(&cc, &pb[..], &cb[..], lmdb::put::Flags::empty()).unwrap();
        }
    }
    txn.commit().unwrap();
}

fn main() {
    let budget = 8usize;
    let n = 2.0f64;
    let edge_pred = "category_parent";
    let dpre = 2usize;
    println!("{:<10} | {:>9} {:>9} {:>8} | {:>9} {:>9} {:>8} | {:>4}",
        "config", "L_prod_ms", "L_bnd_ms", "L_speed", "E_prod_ms", "E_bnd_ms", "E_speed", "eq");
    for (core, periph, cp) in [(120u32, 500u32, 3usize), (180, 500, 3), (240, 500, 3)] {
        let (edges, seed_ids) = build(core, periph, cp, 42);
        let md = min_dist_to_root(&edges, 0);

        // ---------- LAZY (LMDB-backed) edge path ----------
        // Run in a FRESH thread: LmdbFactSource caches a per-thread read txn, so
        // opening a second env on the same thread would reuse the first env stale
        // snapshot. A new thread gets a fresh (None) txn slot bound to this env.
        let dir = std::env::temp_dir().join(format!("bnd_lazy_{}_{}", std::process::id(), core));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_lmdb(dir.to_str().unwrap(), &edges);
        let (lazy_prod, lazy_bound) = std::thread::scope(|sc| {
            sc.spawn(|| {
                let lmdb_src = LmdbFactSource::open(dir.to_str().unwrap()).unwrap();
                let mut lvm = WamState::new(vec![], HashMap::new());
                lvm.register_lazy_lookup("category_parent/2", Arc::new(LazyParents { src: lmdb_src }));
                lvm.set_int_native_edges(true);
                let lmd: HashMap<i32, i32> = md.iter().map(|(&k, &v)| (k as i32, v)).collect();
                lvm.set_min_dist(&lmd);
                let lazy_prod = {
                    let acc = lvm.resolve_edge_accessor(edge_pred);
                    let t = Instant::now();
                    let mut s = 0.0;
                    for &seed in &seed_ids {
                        let mut hops: Vec<i64> = Vec::new();
                        let mut vis = vec![seed];
                        lvm.collect_native_category_ancestor_hops(seed, 0, &mut vis, budget, &acc, 0, &mut hops);
                        s += wpow_hops(&hops, n);
                    }
                    (t.elapsed().as_secs_f64() * 1e3, s)
                };
                let band = lvm.boundary_band_entry_frontier(dpre, edge_pred);
                lvm.build_boundary_suffix(&band, 0, budget, edge_pred);
                let lazy_bound = {
                    let acc = lvm.resolve_edge_accessor(edge_pred);
                    let t = Instant::now();
                    let mut s = 0.0;
                    for &seed in &seed_ids {
                        let mut h: Vec<u64> = Vec::new();
                        let mut vis = vec![seed];
                        lvm.collect_native_category_ancestor_boundary_hist(seed, 0, &mut vis, budget, &acc, 0, &mut h);
                        s += wpow_hist(&h, n);
                    }
                    (t.elapsed().as_secs_f64() * 1e3, s)
                };
                (lazy_prod, lazy_bound)
            }).join().unwrap()
        });
        let _ = std::fs::remove_dir_all(&dir);

        // ---------- EAGER (in-memory) edge path, same graph ----------
        let mut evm = WamState::new(vec![], HashMap::new());
        let owned: Vec<(String, String)> = edges.iter().map(|&(c, p)| (c.to_string(), p.to_string())).collect();
        let refs: Vec<(&str, &str)> = owned.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        evm.register_ffi_fact_pairs(edge_pred, &refs);
        let eroot = evm.intern_atom("0");
        let mut emd: HashMap<i32, i32> = HashMap::new();
        for (&node, &d) in &md { emd.insert(evm.intern_atom(&node.to_string()) as i32, d); }
        evm.set_min_dist(&emd);
        let eseeds: Vec<u32> = seed_ids.iter().map(|&i| evm.intern_atom(&i.to_string())).collect();
        let eager_prod = {
            let acc = evm.resolve_edge_accessor(edge_pred);
            let t = Instant::now();
            let mut s = 0.0;
            for &seed in &eseeds {
                let mut hops: Vec<i64> = Vec::new();
                let mut vis = vec![seed];
                evm.collect_native_category_ancestor_hops(seed, eroot, &mut vis, budget, &acc, 0, &mut hops);
                s += wpow_hops(&hops, n);
            }
            (t.elapsed().as_secs_f64() * 1e3, s)
        };
        let eband = evm.boundary_band_entry_frontier(dpre, edge_pred);
        evm.build_boundary_suffix(&eband, eroot, budget, edge_pred);
        let eager_bound = {
            let acc = evm.resolve_edge_accessor(edge_pred);
            let t = Instant::now();
            let mut s = 0.0;
            for &seed in &eseeds {
                let mut h: Vec<u64> = Vec::new();
                let mut vis = vec![seed];
                evm.collect_native_category_ancestor_boundary_hist(seed, eroot, &mut vis, budget, &acc, 0, &mut h);
                s += wpow_hist(&h, n);
            }
            (t.elapsed().as_secs_f64() * 1e3, s)
        };

        let eq = (lazy_prod.1 - lazy_bound.1).abs() < 1e-6 * lazy_prod.1.abs().max(1.0)
            && (eager_prod.1 - eager_bound.1).abs() < 1e-6 * eager_prod.1.abs().max(1.0)
            && (lazy_prod.1 - eager_prod.1).abs() < 1e-6 * eager_prod.1.abs().max(1.0);
        println!("core={:<5} | {:>9.2} {:>9.3} {:>7.1}x | {:>9.2} {:>9.3} {:>7.1}x | {:>4}",
            core, lazy_prod.0, lazy_bound.0, lazy_prod.0 / lazy_bound.0,
            eager_prod.0, eager_bound.0, eager_prod.0 / eager_bound.0,
            if eq { "yes" } else { "NO!" });
    }
}').

main :-
    Dir = 'output/boundary_lazy_edge',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:m_fact/2],
        [module_name(bm), lmdb_mode(cursor), lmdb_crate(lmdb_zero)], Dir)),
    atom_concat(Dir, '/examples', ExDir),
    make_directory_path(ExDir),
    atom_concat(Dir, '/examples/lazy_edge.rs', ExPath),
    bench_rust(Src),
    setup_call_cleanup(open(ExPath, write, S), format(S, "~w", [Src]), close(S)),
    format(atom(Cmd),
        'cd "~w" && cargo run --release --example lazy_edge 2>&1', [Dir]),
    format("Building + running (release, LMDB)...~n", []),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, OutS), close(Out)),
    process_wait(Pid, Status),
    format("~n=== wam-rust boundary LAZY-EDGE measurement (status ~w) ===~n~w~n", [Status, OutS]),
    halt.

:- initialization(main).
